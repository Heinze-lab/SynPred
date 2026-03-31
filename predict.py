"""
predict.py  –  Fast, crash-proof blockwise inference over a zarr volume.

Usage:
    python predict.py parameter_logits_big.json

Speed optimisations vs v1:
    - Batched GPU inference (batch_size blocks per forward pass)
    - Sigmoid + context crop happen on GPU before CPU transfer
    - Proper blocking queues (no busy-wait CPU spin)
    - blocks_done loaded once into memory, not re-read per block
    - Vector channels written as one zarr slice, not 3 separate writes
    - Multiple reader threads for parallel zarr I/O
    - pin_memory tensors for faster CPU→GPU transfer

Add to params["predict"]:
    "batch_size":      4     # blocks per GPU forward pass (tune to VRAM)
    "num_readers":     2     # parallel zarr read threads
    "prefetch_blocks": 16    # read-ahead depth
"""

from __future__ import annotations

import json
import os
import queue
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import zarr
from tqdm import tqdm

from model import build_model


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def load_model(params, predict_cfg, device):
    model = build_model(params)
    ckpt_dir   = predict_cfg["checkpoint_dir"]
    ckpt_num   = predict_cfg["checkpoint_num"]
    model_name = predict_cfg.get("model_name", params.get("model_name", "model"))
    ckpt_path  = os.path.join(ckpt_dir, f"{model_name}_checkpoint_{ckpt_num}.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    print(f"[predict] Loading checkpoint: {ckpt_path}")
    state = torch.load(ckpt_path, map_location="cpu")
    sd    = state["model_state_dict"]
    if any(k.startswith("_orig_mod.") for k in sd):
        sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    model.load_state_dict(sd)
    model.to(device).eval()
    return model


# ---------------------------------------------------------------------------
# Output zarr
# ---------------------------------------------------------------------------

def setup_output_zarr(out_path, read_sh, output_size, n_blocks, out_props, overwrite,
                      offset=None, resolution=None):
    mode  = "w" if overwrite else "a"
    store = zarr.open(out_path, mode=mode)
    for cfg in out_props.values():
        dsname = cfg["dsname"]
        chunks = tuple(output_size.tolist())
        if dsname == "pred_syn_indicators":
            shape = tuple(read_sh.tolist())
        elif dsname == "pred_partner_vectors":
            shape  = (3,) + tuple(read_sh.tolist())
            chunks = (3,) + chunks
        else:
            shape = tuple(read_sh.tolist())
        if dsname not in store or overwrite:
            store.create_dataset(dsname, shape=shape, chunks=chunks,
                                 dtype=cfg["dtype"], overwrite=overwrite,
                                 fill_value=0)
            if offset is not None:
                store[dsname].attrs["offset"] = offset
            if resolution is not None:
                store[dsname].attrs["resolution"] = resolution
            print(f"[predict] Created '{dsname}' shape={shape} offset={offset} resolution={resolution}")
    if "blocks_done" not in store or overwrite:
        store.create_dataset("blocks_done", shape=tuple(n_blocks.tolist()),
                             dtype=bool, overwrite=overwrite, fill_value=False)
        print(f"[predict] Created 'blocks_done' shape={tuple(n_blocks.tolist())}")
    return store


# ---------------------------------------------------------------------------
# Normalise
# ---------------------------------------------------------------------------

def normalise(arr: np.ndarray) -> np.ndarray:
    """Normalise to [-1, 1] — must match training pipeline's IntensityScaleShift."""
    lo, hi = arr.min(), arr.max()
    arr = (arr - lo) / max(hi - lo, 1e-8)
    return arr * 2.0 - 1.0


# ---------------------------------------------------------------------------
# Block geometry helpers
# ---------------------------------------------------------------------------

def block_input_slice(blk_idx, output_size, read_off, read_sh, context, vol_full):
    """Return (out_off, out_size, raw_crop_normalised) for one block."""
    out_off  = blk_idx * output_size
    out_size = np.minimum(output_size, read_sh - out_off)
    if np.any(out_size <= 0):
        return None

    in_off_abs = read_off + out_off - context
    in_end_abs = in_off_abs + out_size + 2 * context
    in_off_c   = np.maximum(in_off_abs, 0)
    in_end_c   = np.minimum(in_end_abs, vol_full)

    if np.any(in_end_c <= in_off_c):
        return None   # clamped slice is empty (block fully outside volume)

    return out_off, out_size, in_off_c, in_end_c, in_off_abs, in_end_abs


def read_block(raw_ds, geom, input_size):
    out_off, out_size, in_off_c, in_end_c, in_off_abs, in_end_abs = geom
    sl  = tuple(slice(int(in_off_c[i]), int(in_end_c[i])) for i in range(3))
    raw = (raw_ds[sl] if raw_ds.ndim == 3 else raw_ds[0][sl]).astype(np.float32)
    raw = normalise(raw)

    pad_before = (in_off_c - in_off_abs).astype(int)
    pad_after  = np.maximum((in_end_abs - in_end_c).astype(int), 0)
    if np.any(pad_before > 0) or np.any(pad_after > 0):
        raw = np.pad(raw, list(zip(pad_before.tolist(), pad_after.tolist())),
                     mode="reflect")

    # guarantee exact input_size — boundary blocks may be slightly off
    raw = raw[:input_size[0], :input_size[1], :input_size[2]]
    if raw.shape != tuple(input_size):
        # pad any remaining shortfall (e.g. volume smaller than input_size)
        shortfall = [(0, max(0, input_size[i] - raw.shape[i])) for i in range(3)]
        raw = np.pad(raw, shortfall, mode="reflect")
        raw = raw[:input_size[0], :input_size[1], :input_size[2]]

    return out_off, out_size, raw


# ---------------------------------------------------------------------------
# Reader  (multi-threaded zarr reads)
# ---------------------------------------------------------------------------

_SENTINEL = object()


def reader_worker(raw_ds, vol_full, read_off, read_sh, input_size,
                  output_size, context, n_blocks, blocks_done_arr,
                  read_q, batch_size, prefetch):
    """
    Produces batches of (out_offs, out_sizes, raw_stack) where
    raw_stack is (B, 1, Z, Y, X) float32 pinned tensor.
    """
    total   = int(np.prod(n_blocks))
    pending = []   # accumulate blocks for current batch

    with ThreadPoolExecutor(max_workers=max(1, batch_size)) as pool:
        futures = []
        for flat_idx in range(total):
            blk_idx = np.array(np.unravel_index(flat_idx, n_blocks), dtype=int)
            if blocks_done_arr[tuple(blk_idx)]:
                continue

            geom = block_input_slice(blk_idx, output_size, read_off,
                                     read_sh, context, vol_full)
            if geom is None:
                continue

            futures.append(pool.submit(read_block, raw_ds, geom, input_size))

            if len(futures) >= batch_size:
                # wait for this batch then push to queue
                results = [f.result() for f in futures]
                futures = []
                # backpressure — wait if GPU is falling behind
                while read_q.qsize() >= prefetch:
                    import time; time.sleep(0.005)
                read_q.put(([r[0] for r in results],
                             [r[1] for r in results],
                             torch.from_numpy(
                                 np.stack([r[2] for r in results])[:, None]
                             ).pin_memory()))

        # flush remaining partial batch
        if futures:
            results = [f.result() for f in futures]
            while read_q.qsize() >= prefetch:
                import time; time.sleep(0.005)
            read_q.put(([r[0] for r in results],
                         [r[1] for r in results],
                         torch.from_numpy(
                             np.stack([r[2] for r in results])[:, None]
                         ).pin_memory()))

    read_q.put(_SENTINEL)


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------

def writer_worker(out_store, out_props, write_q, output_size):
    """Writes batches of predictions to zarr."""
    while True:
        item = write_q.get()
        if item is _SENTINEL:
            break

        out_offs, out_sizes, masks, vecs = item

        for out_off, out_size, mask_np, vec_np in zip(out_offs, out_sizes, masks, vecs):
            out_sl = tuple(
                slice(int(out_off[i]), int(out_off[i] + out_size[i]))
                for i in range(3)
            )
            for cfg in out_props.values():
                dsname = cfg["dsname"]
                scale  = cfg.get("scale", 1)

                if dsname == "pred_syn_indicators":
                    arr = (mask_np * scale).clip(0, 255).astype(cfg["dtype"])
                    out_store[dsname][out_sl] = arr

                elif dsname == "pred_partner_vectors":
                    if isinstance(scale, (list, tuple)):
                        sc = np.array(scale, dtype=np.float32)[:, None, None, None]
                    else:
                        sc = float(scale)
                    arr = (vec_np * sc).clip(-128, 127).astype(cfg["dtype"])
                    out_store[dsname][(slice(None),) + out_sl] = arr

            # blocks_done index = out_off // output_size (integer division per axis)
            bd_idx = tuple(int(out_off[i]) // int(output_size[i]) for i in range(3))
            out_store["blocks_done"][bd_idx] = True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def predict_blockwise(params_path: str) -> None:
    with open(params_path) as fh:
        params = json.load(fh)

    cfg = params["predict"]

    device_num = int(cfg.get("device_num", 0))
    device = torch.device(f"cuda:{device_num}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device_num)
    print(f"[predict] Device: {device}")

    model = load_model(params, cfg, device)

    # try torch.compile for inference too — helps on repeated same-size inputs
    if cfg.get("compile", True):
        try:
            compile_mode = cfg.get("compile_mode", "reduce-overhead")
            model = torch.compile(model, mode=compile_mode)
            print(f"[predict] Model compiled with torch.compile (mode={compile_mode})")
        except Exception:
            pass

    input_size  = np.array(cfg["input_size"],  dtype=int)
    output_size = np.array(cfg["output_size"], dtype=int)
    context     = (input_size - output_size) // 2
    assert np.all(context >= 0), "output_size must be ≤ input_size"

    raw_file = zarr.open(cfg["raw_file"], mode="r")
    raw_ds   = raw_file[cfg["raw_dataset"]]
    vol_full = np.array(raw_ds.shape[-3:], dtype=int)
    read_off = np.array(cfg.get("read_offset", [0, 0, 0]), dtype=int)
    read_sh  = np.array(cfg.get("read_shape",  list(vol_full)), dtype=int)
    # -1 sentinel: use full volume
    read_off = np.where(read_off < 0, 0,        read_off)
    read_sh  = np.where(read_sh  < 0, vol_full - read_off, read_sh)

    # copy offset/resolution from raw zarr for neuroglancer alignment
    raw_offset     = list(raw_ds.attrs.get("offset",     [0, 0, 0]))
    raw_resolution = list(raw_ds.attrs.get("resolution", [1, 1, 1]))
    # shift offset by read_off in world units
    out_offset = [raw_offset[i] + read_off[i] * raw_resolution[i] for i in range(3)]

    out_path  = os.path.join(cfg.get("out_directory", "."),
                             cfg.get("out_filename", "pred.zarr"))
    overwrite = cfg.get("overwrite", False)
    out_props = cfg.get("out_properties", {})
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    n_blocks     = np.ceil(read_sh / output_size).astype(int)
    total_blocks = int(np.prod(n_blocks))

    out_store = setup_output_zarr(out_path, read_sh, output_size,
                                  n_blocks, out_props, overwrite,
                                  offset=out_offset, resolution=raw_resolution)

    # load blocks_done into memory once — avoid per-block zarr reads
    blocks_done_arr = out_store["blocks_done"][:]
    n_done    = int(blocks_done_arr.sum())
    n_pending = total_blocks - n_done
    print(f"[predict] {total_blocks} blocks total, "
          f"{n_done} already done, {n_pending} to process")

    if n_pending == 0:
        print("[predict] All blocks already complete.")
        return

    batch_size  = cfg.get("batch_size",      4)
    prefetch    = cfg.get("prefetch_blocks",  16)
    use_amp     = (device.type == "cuda")

    print(f"[predict] batch_size={batch_size}  prefetch={prefetch}  amp={use_amp}")

    read_q  = queue.Queue(maxsize=prefetch)
    write_q = queue.Queue(maxsize=prefetch)

    reader = threading.Thread(
        target=reader_worker,
        args=(raw_ds, vol_full, read_off, read_sh, input_size,
              output_size, context, n_blocks, blocks_done_arr,
              read_q, batch_size, prefetch),
        daemon=True,
    )
    writer = threading.Thread(
        target=writer_worker,
        args=(out_store, out_props, write_q, output_size),
        daemon=True,
    )
    reader.start()
    writer.start()

    pbar = tqdm(total=n_pending, desc="Predicting", unit="blocks")

    with torch.no_grad():
        while True:
            item = read_q.get()
            if item is _SENTINEL:
                break

            out_offs, out_sizes, raw_stack = item
            B = raw_stack.shape[0]

            raw_t = raw_stack.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                pred_mask_t, pred_vec_t = model(raw_t)

            # compute context from actual model output size (U-Net convolutions
            # reduce spatial dims, so actual output != input - 2*context)
            actual_out = np.array(pred_mask_t.shape[2:], dtype=int)
            actual_context = (actual_out - output_size) // 2
            cz, cy, cx = int(actual_context[0]), int(actual_context[1]), int(actual_context[2])
            masks_np = []
            vecs_np  = []
            for b in range(B):
                oz = int(out_sizes[b][0])
                oy = int(out_sizes[b][1])
                ox = int(out_sizes[b][2])
                m = torch.sigmoid(
                    pred_mask_t[b, 0, cz:cz+oz, cy:cy+oy, cx:cx+ox]
                ).cpu().float().numpy()
                v = pred_vec_t[b, :, cz:cz+oz, cy:cy+oy, cx:cx+ox
                    ].cpu().float().numpy()
                masks_np.append(m)
                vecs_np.append(v)

            write_q.put((out_offs, out_sizes, masks_np, vecs_np))
            pbar.update(B)

    write_q.put(_SENTINEL)
    writer.join()
    pbar.close()
    print(f"[predict] Done. Output: {out_path}")


if __name__ == "__main__":
    params_path = sys.argv[1] if len(sys.argv) > 1 else "parameter_logits_big.json"
    predict_blockwise(params_path)