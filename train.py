"""
train.py  –  Multi-task training for synaptic partner detection (setup03).

Usage:
    python train.py parameter_logits_big.json
"""

from __future__ import annotations

import json
import math
import os
import sys

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# Performance flags — matches train_gb.py
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")

from dataset import build_dataset
from model import build_model


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def center_crop(t: torch.Tensor, target_shape: torch.Size) -> torch.Tensor:
    """Center-crop tensor t (B,C,Z,Y,X) to match target_shape's spatial dims."""
    for dim in range(2, t.dim()):
        diff  = t.shape[dim] - target_shape[dim]
        start = diff // 2
        t     = t.narrow(dim, start, target_shape[dim])
    return t


def mask_loss(pred, target, gamma=2.0, pos_weight=None):
    """Sigmoid focal loss — automatically down-weights easy examples.
    pos_weight: scalar upweight for positive (synapse) examples, e.g. 50.0
    """
    target = center_crop(target, pred.shape)
    pw = torch.tensor([pos_weight], device=pred.device, dtype=pred.dtype) if pos_weight else None
    p   = torch.sigmoid(pred)
    ce  = nn.functional.binary_cross_entropy_with_logits(pred, target, pos_weight=pw, reduction='none')
    p_t = target * p + (1.0 - target) * (1.0 - p)
    return (((1.0 - p_t) ** gamma) * ce).mean()


def direction_loss(pred, target, weight_mask, channel_weights=None, normalize_by_magnitude=False):
    """MSE restricted to synapse blobs, with optional per-channel weighting
    and optional normalization by GT vector magnitude (makes loss scale-invariant)."""
    target      = center_crop(target,      pred.shape)
    weight_mask = center_crop(weight_mask, pred.shape)
    diff2 = (pred - target).pow(2) * weight_mask
    if channel_weights is not None:
        diff2 = diff2 * channel_weights.view(1, -1, 1, 1, 1)
    if normalize_by_magnitude:
        gt_mag = target.pow(2).sum(dim=1, keepdim=True).sqrt()
        diff2 = diff2 / (gt_mag + 1.0)
    n = weight_mask.sum() * pred.shape[1] + 1e-7
    return diff2.sum() / n


def combined_loss(pred_mask, pred_vec, t_mask, t_vec, d_weight,
                  m_scale, d_scale, comb_type, focal_gamma, channel_weights=None,
                  normalize_by_magnitude=False, pos_weight=None):
    m_loss = mask_loss(pred_mask, t_mask, focal_gamma, pos_weight)
    d_loss = direction_loss(pred_vec, t_vec, d_weight, channel_weights, normalize_by_magnitude)
    if comb_type == "sum":
        total = m_scale * m_loss + d_scale * d_loss
    elif comb_type == "mean":
        total = (m_scale * m_loss + d_scale * d_loss) / 2.0
    else:
        raise ValueError(f"Unknown loss_comb_type: {comb_type!r}")
    return total, m_loss, d_loss


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _ckpt_path(directory, model_name, iteration):
    return os.path.join(directory, f"{model_name}_checkpoint_{iteration}.pt")


def save_checkpoint(model, optimizer, scaler, scheduler, iteration, directory, model_name):
    os.makedirs(directory, exist_ok=True)
    path = _ckpt_path(directory, model_name, iteration)
    torch.save({
        "iteration":            iteration,
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict":    scaler.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
    }, path)
    print(f"[train] Saved checkpoint: {path}")


def load_latest_checkpoint(model, optimizer, scaler, scheduler, directory, model_name):
    if not os.path.isdir(directory):
        return 0
    from pathlib import Path
    ckpts = list(Path(directory).glob(f"{model_name}_checkpoint_*.pt"))
    if not ckpts:
        return 0
    latest = max(ckpts, key=lambda p: int(p.stem.split("_")[-1]))
    print(f"[train] Resuming from {latest}")
    state = torch.load(latest, map_location="cpu")
    sd    = state["model_state_dict"]
    if any(k.startswith("_orig_mod.") for k in sd):
        sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    # load into uncompiled model (strips _orig_mod prefix above), then compiled wrapper accepts it
    try:
        model.load_state_dict(sd)
    except RuntimeError:
        # checkpoint saved from compiled model, loading into uncompiled — add prefix
        sd = {"_orig_mod." + k: v for k, v in sd.items()}
        model.load_state_dict(sd)
    optimizer.load_state_dict(state["optimizer_state_dict"])
    if "scaler_state_dict" in state:
        scaler.load_state_dict(state["scaler_state_dict"])
    if "scheduler_state_dict" in state and scheduler is not None:
        scheduler.load_state_dict(state["scheduler_state_dict"])
    return state["iteration"]


# ---------------------------------------------------------------------------
# Snapshot  (matches gunpowder Snapshot output format)
# ---------------------------------------------------------------------------

def save_snapshot(
    batch:      dict,
    pred_mask:  torch.Tensor,
    pred_vec:   torch.Tensor,
    iteration:  int,
    directory:  str,
) -> None:
    """
    Save a training snapshot to HDF5, matching gunpowder's Snapshot format.

    Datasets written:
        volumes/raw                  (Z, Y, X)  float32  [-1, 1]
        volumes/gt_post_indicator    (Z, Y, X)  float32  binary
        volumes/gt_postpre_vectors   (3, Z, Y, X) float32 unit vectors
        volumes/pred_post_indicator  (Z, Y, X)  float32  sigmoid probs
        volumes/pred_postpre_vectors (3, Z, Y, X) float32
    """
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f"batch_{iteration:08d}.hdf")

    p_mask  = torch.sigmoid(pred_mask[0, 0]).detach().cpu().float().numpy()
    p_vec   = pred_vec[0].detach().cpu().float().numpy()
    pred_sh = p_mask.shape

    def center_crop_np(arr, target):
        """Crop numpy array (or 3D/4D) to target spatial shape from centre."""
        slices = [slice(None)] * (arr.ndim - 3)
        for a, t in zip(arr.shape[-3:], target):
            start = (a - t) // 2
            slices.append(slice(start, start + t))
        return arr[tuple(slices)]

    raw_np  = center_crop_np(batch["raw"][0, 0].cpu().numpy(),               pred_sh)
    gt_mask = center_crop_np(batch["indicator_mask"][0, 0].cpu().numpy(),    pred_sh)
    gt_vec  = center_crop_np(batch["direction_vectors"][0].cpu().numpy(),    pred_sh)

    with h5py.File(path, "w") as f:
        f.create_dataset("volumes/raw",                  data=raw_np,  compression="gzip")
        f.create_dataset("volumes/gt_post_indicator",    data=gt_mask, compression="gzip")
        f.create_dataset("volumes/gt_postpre_vectors",   data=gt_vec,  compression="gzip")
        f.create_dataset("volumes/pred_post_indicator",  data=p_mask,  compression="gzip")
        f.create_dataset("volumes/pred_postpre_vectors", data=p_vec,   compression="gzip")

    print(f"[train] Snapshot saved: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def train(params_path: str) -> None:
    with open(params_path) as fh:
        params = json.load(fh)

    # ---- device -----------------------------------------------------------
    device_num = params.get("device_num", 0)
    device = torch.device(
        f"cuda:{device_num}" if torch.cuda.is_available() else "cpu"
    )
    if device.type == "cuda":
        torch.cuda.set_device(device_num)
    print(f"[train] Device: {device}")

    # ---- model ------------------------------------------------------------
    model = build_model(params).to(device)

    # torch.compile — matches train_gb.py
    print("[train] Compiling model with torch.compile ...")
    model = torch.compile(model)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[train] Model parameters: {n_params:,}")

    # ---- optimiser (Adam beta1=0.95, matching original) -------------------
    optimizer = optim.Adam(
        model.parameters(),
        lr=params["learning_rate"],
        betas=(0.95, 0.999),
        eps=1e-8,
    )

    # ---- scheduler — fixed warmup + cosine decay (no total_steps dependency) -----
    max_iter         = params.get("max_iteration", 1_000_000)
    warmup_steps     = int(params.get("warmup_steps",  8_000))
    cosine_period    = int(params.get("cosine_period", 2_000_000))
    final_div_factor = float(params.get("final_div_factor", 100.0))
    lr_min_factor    = 1.0 / final_div_factor

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return max(step / max(warmup_steps, 1), 1e-6)
        progress = (step - warmup_steps) / cosine_period
        cosine   = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
        return lr_min_factor + (1.0 - lr_min_factor) * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ---- AMP --------------------------------------------------------------
    use_amp = params.get("use_amp", True) and device.type == "cuda"
    scaler  = torch.amp.GradScaler("cuda", enabled=use_amp)

    # ---- names / dirs -----------------------------------------------------
    model_name   = params.get("model_name",   "model")
    snapshot_dir = params.get("snapshot_dir", "snapshots")

    # ---- resume -----------------------------------------------------------
    start_iter = load_latest_checkpoint(
        model, optimizer, scaler, scheduler, snapshot_dir, model_name
    )

    # ---- dataset & loader -------------------------------------------------
    samples_per_epoch = max(1000, min(10_000, max_iter // 100))

    dataset = build_dataset(params, samples_per_epoch=samples_per_epoch)
    loader  = DataLoader(
        dataset,
        batch_size=params.get("batch_size", 1),
        num_workers=params.get("num_data_workers", 8),
        pin_memory=(device.type == "cuda"),
        prefetch_factor=4,
        persistent_workers=True,
    )

    # ---- loss hyper-params ------------------------------------------------
    m_scale     = float(params.get("m_loss_scale", 1.0))
    d_scale     = float(params.get("d_loss_scale", 1.0))
    comb_type   = params.get("loss_comb_type", "sum")
    focal_gamma = float(params.get("focal_gamma", 2.0))
    _cw = params.get("vec_channel_weights", [1.0, 1.0, 1.0])
    channel_weights = torch.tensor(_cw, dtype=torch.float32).to(device)
    normalize_by_magnitude = bool(params.get("vec_normalize_by_magnitude", False))
    pos_weight = params.get("mask_pos_weight", None)
    if pos_weight is not None:
        pos_weight = float(pos_weight)

    # ---- tensorboard ------------------------------------------------------
    writer = SummaryWriter(params.get("tensorboard_dir", "tensorboard"))

    save_every     = params.get("save_every",     10_000)
    log_every      = params.get("log_every",      100)
    snapshot_every = params.get("snapshot_every", 25_000)

    # ---- loop -------------------------------------------------------------
    iteration = start_iter
    model.train()
    data_iter = iter(loader)

    print(f"[train] Starting from iteration {iteration} / {max_iter}")

    while iteration < max_iter:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        raw      = batch["raw"].to(device, non_blocking=True)
        t_mask   = batch["indicator_mask"].to(device, non_blocking=True)
        t_vec    = batch["direction_vectors"].to(device, non_blocking=True)
        d_weight = batch["d_weight_mask"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=use_amp):
            pred_mask, pred_vec = model(raw)
            loss, m_loss, d_loss = combined_loss(
                pred_mask, pred_vec, t_mask, t_vec, d_weight,
                m_scale, d_scale, comb_type, focal_gamma, channel_weights,
                normalize_by_magnitude, pos_weight,
            )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=params.get("grad_clip", 1.0))
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        iteration += 1

        if iteration % log_every == 0:
            l, ml, dl = loss.item(), m_loss.item(), d_loss.item()
            lr = scheduler.get_last_lr()[0]
            print(f"[{iteration:>8}/{max_iter}]  loss={l:.5f}  mask={ml:.5f}  vec={dl:.5f}  lr={lr:.2e}")
            writer.add_scalar("loss/total",     l,  iteration)
            writer.add_scalar("loss/mask",      ml, iteration)
            writer.add_scalar("loss/direction", dl, iteration)
            writer.add_scalar("lr",             lr, iteration)

        if iteration % snapshot_every == 0:
            save_snapshot(batch, pred_mask, pred_vec, iteration, snapshot_dir)

        if iteration % save_every == 0:
            save_checkpoint(model, optimizer, scaler, scheduler, iteration, snapshot_dir, model_name)

    save_checkpoint(model, optimizer, scaler, scheduler, iteration, snapshot_dir, model_name)
    writer.close()
    print("[train] Done.")


if __name__ == "__main__":
    params_path = sys.argv[1] if len(sys.argv) > 1 else "parameter_logits_big.json"
    train(params_path)