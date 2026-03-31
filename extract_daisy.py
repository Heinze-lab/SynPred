"""
extract_daisy.py  –  Block-wise synapse extraction using daisy.

Usage:
    python extract_daisy.py parameter.json

Reads params["extract_configs"] — same keys as extract.py, plus:
    block_size_zyx   : inner block size in voxels  (default [48, 536, 536])
    context_zyx      : halo on each side in voxels (default [20, 40, 40])
    num_workers      : daisy worker processes       (default 4)

Strategy:
    Each daisy block reads indicators + vectors with a halo (context).
    Threshold → CC → extract detections whose centroid falls inside the
    inner (non-halo) block.  Because the halo is larger than any synapse
    blob, every blob is fully captured in exactly one block.
    After all blocks finish, global NMS is applied and CSV/JSON written.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import zarr
from scipy.ndimage import (
    distance_transform_edt,
    label as nd_label,
    find_objects,
)

import daisy
from funlib.geometry import Coordinate, Roi

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# NMS (identical to extract.py)
# ---------------------------------------------------------------------------

def nms(detections: list[dict], radius: float) -> list[dict]:
    if not detections:
        return []
    from scipy.spatial import cKDTree
    dets = sorted(detections, key=lambda d: d["score"], reverse=True)
    pts = np.array([[d["post_z"], d["post_y"], d["post_x"]] for d in dets])
    tree = cKDTree(pts)
    suppressed = np.zeros(len(dets), dtype=bool)
    kept = []
    for i, d in enumerate(dets):
        if suppressed[i]:
            continue
        kept.append(d)
        neighbors = tree.query_ball_point(pts[i], radius)
        for j in neighbors:
            if j > i:
                suppressed[j] = True
    return kept


# ---------------------------------------------------------------------------
# Per-block extraction worker
# ---------------------------------------------------------------------------

def extract_block(
    block: daisy.Block,
    ind_ds,
    vec_ds,
    cc_threshold: float,
    loc_type: str,
    score_thr: float,
    score_type: str,
    size_thr: int,
    flipprepost: bool,
    post_offset_scale: float,
    pre_offset_scale: float,
    vec_scale: np.ndarray,
    voxel_size: np.ndarray,
    zarr_offset: np.ndarray,
    read_offset_nm: np.ndarray,
    tmp_dir: str,
) -> None:
    """Run inside a daisy worker process. Writes per-block results to tmp_dir."""

    # ---- read arrays for the full read_roi (inner + halo) ------------------
    read_roi  = block.read_roi
    write_roi = block.write_roi

    # convert daisy Roi (world/nm) → zarr voxel slices
    # zarr array is indexed from 0, so subtract the world origin (zarr_offset + read_offset_nm)
    world_origin_nm = zarr_offset + read_offset_nm
    vs = voxel_size  # ZYX nm/voxel

    def roi_to_slices(roi):
        offset_vx = ((np.array(roi.offset) - world_origin_nm) / vs).astype(int)
        shape_vx  = (np.array(roi.shape) / vs).astype(int)
        # clamp to valid array bounds
        offset_vx = np.maximum(offset_vx, 0)
        return tuple(slice(int(o), int(o + s)) for o, s in zip(offset_vx, shape_vx))

    read_sl  = roi_to_slices(read_roi)
    write_sl = roi_to_slices(write_roi)

    # halo offset inside the read crop (write_roi.offset - read_roi.offset)
    halo_vx = ((np.array(write_roi.offset) - np.array(read_roi.offset)) / vs).astype(int)

    block_id = "_".join(str(int(o)) for o in block.write_roi.offset)
    log.info(f"Block {block_id}: read_sl={read_sl}")

    # load only indicators for the full read region — vectors read lazily per detection
    try:
        ind_crop = np.array(ind_ds[read_sl]).astype(np.float32) / 255.0
    except Exception as exc:
        log.warning(f"Block {block_id} read failed: {exc}")
        _write_block_result(tmp_dir, block_id, [])
        return

    # ---- threshold + CC ----------------------------------------------------
    binary = ind_crop >= cc_threshold
    if not binary.any():
        log.info(f"Block {block_id}: no foreground")
        _write_block_result(tmp_dir, block_id, [])
        return

    labeled, n_cc = nd_label(binary)
    if n_cc == 0:
        _write_block_result(tmp_dir, block_id, [])
        return

    bboxes = find_objects(labeled)

    # write_roi shape in voxels (for centroid filtering)
    write_shape_vx = (np.array(write_roi.shape) / vs).astype(int)

    # absolute zarr origin of this read crop (for lazy vec slicing)
    read_origin_vx = np.array([s.start for s in read_sl])

    detections = []

    for lbl, bbox in enumerate(bboxes, start=1):
        if bbox is None:
            continue

        lab_crop  = labeled[bbox]
        mask_crop = lab_crop == lbl

        size = int(mask_crop.sum())
        if size < size_thr:
            continue

        ind_sub = ind_crop[bbox]
        if score_type == "mean":
            score = float(ind_sub[mask_crop].mean())
        else:
            score = float(ind_sub[mask_crop].max())

        if score < score_thr:
            continue

        # location within the read crop
        if loc_type == "centroid":
            zz, yy, xx = np.where(mask_crop)
            local_loc = np.array([zz.mean(), yy.mean(), xx.mean()])
        elif loc_type == "edt":
            edt = distance_transform_edt(mask_crop)
            local_loc = np.array(np.unravel_index(edt.argmax(), edt.shape), dtype=float)
        else:  # peak
            masked = ind_sub * mask_crop
            local_loc = np.array(np.unravel_index(masked.argmax(), masked.shape), dtype=float)

        # position in the read crop coords
        bbox_origin = np.array([s.start for s in bbox])
        post_in_crop = local_loc + bbox_origin   # ZYX within read crop

        # check centroid is inside the write (inner) roi
        pos_in_write = post_in_crop - halo_vx
        if np.any(pos_in_write < 0) or np.any(pos_in_write >= write_shape_vx):
            continue

        # lazy vector read: bbox in absolute zarr voxel coords
        abs_bbox = tuple(
            slice(int(s.start + read_origin_vx[i]), int(s.stop + read_origin_vx[i]))
            for i, s in enumerate(bbox)
        )
        vec_crop_raw = np.array(vec_ds[(slice(None),) + abs_bbox]).astype(np.float32)
        vec_crop_bb  = vec_crop_raw / vec_scale[:, None, None, None]

        vz = float(vec_crop_bb[0][mask_crop].mean())
        vy = float(vec_crop_bb[1][mask_crop].mean())
        vx = float(vec_crop_bb[2][mask_crop].mean())
        vec = np.array([vz, vy, vx])

        post_zyx = post_in_crop
        pre_zyx  = post_zyx + vec

        if post_offset_scale != 0.0:
            post_zyx = post_zyx + vec * post_offset_scale
        if pre_offset_scale != 0.0:
            pre_zyx = pre_zyx + vec * pre_offset_scale

        # convert crop-local voxel coords → world nm
        post_world = (post_zyx + read_origin_vx) * vs + world_origin_nm
        pre_world  = (pre_zyx  + read_origin_vx) * vs + world_origin_nm

        if flipprepost:
            post_world, pre_world = pre_world, post_world

        detections.append({
            "post_z": float(post_world[0]),
            "post_y": float(post_world[1]),
            "post_x": float(post_world[2]),
            "pre_z":  float(pre_world[0]),
            "pre_y":  float(pre_world[1]),
            "pre_x":  float(pre_world[2]),
            "score":  score,
            "size":   size,
        })

    log.info(f"Block {block_id}: {len(detections)} detections")
    _write_block_result(tmp_dir, block_id, detections)


def _write_block_result(tmp_dir: str, block_id: str, detections: list) -> None:
    import pickle
    path = os.path.join(tmp_dir, f"block_{block_id}.pkl")
    with open(path, "wb") as f:
        pickle.dump(detections, f)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def extract(params_path: str) -> None:
    with open(params_path) as fh:
        params = json.load(fh)

    cfg = params["extract_configs"]

    # ---- parameters --------------------------------------------------------
    cc_threshold      = float(cfg.get("cc_threshold",      0.5))
    loc_type          = cfg.get("loc_type",                "edt")
    score_thr         = float(cfg.get("score_thr",         0.5))
    score_type        = cfg.get("score_type",              "mean")
    size_thr          = int(cfg.get("size_thr",            0))
    nms_radius        = float(cfg.get("nms_radius",        0))
    flipprepost       = bool(cfg.get("flipprepost",        False))
    post_offset_scale = float(cfg.get("post_offset_scale", 0.0))
    pre_offset_scale  = float(cfg.get("pre_offset_scale",  0.0))
    vec_scale_cfg     = np.array(cfg.get("vector_scale",   [1, 1, 1]), dtype=np.float32)

    block_size_zyx = list(cfg.get("block_size_zyx", [48, 536, 536]))
    context_zyx    = list(cfg.get("context_zyx",    [20,  40,  40]))
    num_workers    = int(cfg.get("num_workers",     4))

    log.info(f"cc_threshold={cc_threshold}  loc_type={loc_type}")
    log.info(f"score_thr={score_thr}  size_thr={size_thr}  nms_radius={nms_radius}")
    log.info(f"block_size_zyx={block_size_zyx}  context_zyx={context_zyx}  num_workers={num_workers}")

    # ---- open prediction zarr ----------------------------------------------
    inf_path = os.path.join(cfg.get("inference_dir", "."), cfg.get("inference_file", "pred.zarr"))
    log.info(f"Reading: {inf_path}")

    store = zarr.open(inf_path, mode="r")
    if "pred_syn_indicators" not in store:
        raise KeyError(f"'pred_syn_indicators' not found in {inf_path}")
    if "pred_partner_vectors" not in store:
        raise KeyError(f"'pred_partner_vectors' not found in {inf_path}")

    ind_ds = store["pred_syn_indicators"]
    vec_ds = store["pred_partner_vectors"]

    vol_shape = np.array(ind_ds.shape)   # ZYX
    log.info(f"Volume shape: {vol_shape}  (indicators uint8 {vol_shape.prod()/1e9:.2f} GB)")

    # ---- world offset / voxel size -----------------------------------------
    raw_file = cfg.get("raw_file") or params.get("predict", {}).get("raw_file", "")
    raw_ds   = cfg.get("raw_dataset") or params.get("predict", {}).get("raw_dataset", "RAW")
    zarr_offset = np.zeros(3, dtype=float)
    voxel_size  = np.ones(3,  dtype=float)

    if raw_file and os.path.exists(raw_file):
        try:
            rz = zarr.open(raw_file, mode="r")
            zarr_offset = np.array(rz[raw_ds].attrs.get("offset",     [0,0,0]), dtype=float)
            voxel_size  = np.array(rz[raw_ds].attrs.get("resolution", [1,1,1]), dtype=float)
            log.info(f"Zarr offset:     {zarr_offset.tolist()}")
            log.info(f"Voxel size (nm): {voxel_size.tolist()}")
        except Exception as exc:
            log.warning(f"Could not read zarr attrs: {exc}")

    read_offset = np.array(
        params.get("predict", {}).get("read_offset", [0, 0, 0]), dtype=float
    )
    read_offset_nm = read_offset * voxel_size

    # ---- build daisy ROIs --------------------------------------------------
    # daisy works in world (nm) units
    vs = voxel_size  # ZYX

    total_roi = Roi(
        offset=tuple((zarr_offset + read_offset_nm).tolist()),
        shape=tuple((vol_shape * vs).tolist()),
    )
    block_size_nm = tuple(float(b * v) for b, v in zip(block_size_zyx, vs))
    context_nm    = tuple(float(c * v) for c, v in zip(context_zyx,    vs))

    write_roi = Roi(offset=(0,)*3, shape=block_size_nm)
    read_roi  = write_roi.grow(context_nm, context_nm)

    log.info(f"Total ROI: {total_roi}")
    log.info(f"Block size (nm): {block_size_nm}  Context (nm): {context_nm}")

    # ---- temp dir for per-block results ------------------------------------
    import tempfile, pickle
    tmp_dir = tempfile.mkdtemp(prefix="extract_daisy_")
    log.info(f"Temp dir for block results: {tmp_dir}")

    # ---- daisy task --------------------------------------------------------
    task = daisy.Task(
        task_id="extract",
        total_roi=total_roi,
        read_roi=read_roi,
        write_roi=write_roi,
        process_function=lambda block: extract_block(
            block,
            ind_ds,
            vec_ds,
            cc_threshold,
            loc_type,
            score_thr,
            score_type,
            size_thr,
            flipprepost,
            post_offset_scale,
            pre_offset_scale,
            vec_scale_cfg,
            vs,
            zarr_offset,
            read_offset_nm,
            tmp_dir,
        ),
        num_workers=num_workers,
        fit="shrink",
    )

    log.info("Starting daisy scheduler ...")
    daisy.run_blockwise([task])
    log.info("All blocks done.")

    # ---- collect results ---------------------------------------------------
    import glob
    all_detections = []
    for pkl_path in glob.glob(os.path.join(tmp_dir, "block_*.pkl")):
        with open(pkl_path, "rb") as f:
            all_detections.extend(pickle.load(f))
    import shutil
    shutil.rmtree(tmp_dir)

    log.info(f"After size/score filter: {len(all_detections)} detections")

    # ---- global NMS --------------------------------------------------------
    if nms_radius > 0:
        all_detections = nms(all_detections, nms_radius)
        log.info(f"After NMS (r={nms_radius}): {len(all_detections)} detections")

    # re-index IDs
    for i, d in enumerate(all_detections):
        d["id"] = i

    # ---- write output ------------------------------------------------------
    to_json  = params.get("to_json_config", {})
    out_name = to_json.get("output_name",
               cfg.get("inference_dir", ".") + "/synapses.json")
    os.makedirs(os.path.dirname(os.path.abspath(out_name)), exist_ok=True)

    with open(out_name, "w") as fh:
        json.dump({"synapses": all_detections, "n": len(all_detections)}, fh, indent=2)
    log.info(f"Wrote {len(all_detections)} synapses → {out_name}")

    csv_path = out_name.replace(".json", ".csv")
    with open(csv_path, "w") as fh:
        fh.write("id,post_z,post_y,post_x,pre_z,pre_y,pre_x,score,size\n")
        for d in all_detections:
            fh.write(
                f"{d['id']},{d['post_z']:.1f},{d['post_y']:.1f},{d['post_x']:.1f},"
                f"{d['pre_z']:.1f},{d['pre_y']:.1f},{d['pre_x']:.1f},"
                f"{d['score']:.4f},{d['size']}\n"
            )
    log.info(f"Wrote CSV → {csv_path}")


if __name__ == "__main__":
    params_path = sys.argv[1] if len(sys.argv) > 1 else "parameter.json"
    extract(params_path)
