# Synaptic Partner Detection — rewrite of [Synful](https://github.com/funkelab/synful)

PyTorch reimplementation of the dual-headed U-Net architecture from Synful, with a pure PyTorch training stack (no gunpowder dependency). Tested with zarr volumes.

Training, prediction, and extraction pipeline for synapse detection and partner vector prediction.

## Environment Setup

Requires CUDA 12.4 and conda.

```bash
conda create -n synpred python=3.10 -y
conda activate synpred
```

### PyTorch (CUDA 12.4)

```bash
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```

### Core dependencies

```bash
pip install \
    numpy==1.24.3 \
    zarr==2.18.4 \
    h5py \
    scipy \
    pandas \
    tqdm \
    tensorboard \
    neuroglancer==2.41.2
```

### funlib stack

```bash
pip install \
    funlib-geometry==0.3.0 \
    funlib-math==0.1 \
    funlib-persistence==0.5.4 \
    funlib.segment \
    daisy==1.0
```

## Pipeline

```
train.py → predict.py → extract_daisy.py
```

## Key files

| File | Purpose |
|---|---|
| `dataset.py` | Data loading + GT rendering |
| `model.py` | Dual-head U-Net architecture (DHUNet) |
| `augment.py` | Augmentation pipeline (elastic, intensity, defect, etc.) |
| `train.py` | Training loop |
| `predict.py` | Blockwise inference over zarr |
| `extract_daisy.py` | Daisy-based chunked synapse extraction |
| `extract.py` | Single-machine extraction (small volumes, no daisy required) |

## Utilities

| File | Purpose |
|---|---|
| `view_snap.ipynb` | Viewer for snapshots produced during training |
| `pred_view.ipynb` | Output viewer for predicted volumes |
| `profiling.py` | Single-sample speed/compile test using a parameter file |

## Usage

### Training

```bash
python train.py parameter_myexp.json
```

Copy `parameter.json` as a starting point, fill in your `zarr_locs`, `csv_dir`, `model_name`, and `snapshot_dir`/`tensorboard_dir`. All options are documented in that file.

```bash
tensorboard --logdir tensorboard_myexp
```

TensorBoard logs include: loss (total, mask, direction, EMA), learning rate, gradient norms, AMP scale factor, throughput, positive voxel fraction, per-axis vector magnitudes, weight/gradient histograms (every 2000 steps), PR curve for the indicator head (at snapshot steps), model graph, and a linked hparams tab.

### Prediction

Add a `"predict"` block to your parameter JSON (see `parameter.json` for all fields), then:

```bash
python predict.py parameter_myexp.json
```

### Extraction

```bash
python extract_daisy.py parameter_myexp.json
```

## Model

DHUNet: shared 3D U-Net encoder → two independent decoders.

- **Indicator head** — 1-channel sigmoid, predicts synapse probability mask
- **Vector head** — 3-channel linear, predicts pre→post partner direction vectors

Key parameter JSON fields:

| Key | Description |
|---|---|
| `fmap_num` | Base feature map count |
| `fmap_inc_factor` | Feature map multiplier per level |
| `downsample_factors` | List of `[z, y, x]` pool factors per level |
| `norm_type` | `"group"` (recommended) or `"batch"` |
| `grad_checkpoint` | Enable gradient checkpointing to trade compute for memory |
| `gpu_elastic` | Run elastic deformation on GPU (much faster than CPU scipy) |

## Ground truth data format

Training requires one zarr volume and two CSV files per training sample. The zarr volume should be at least 25 voxels larger than `input_size` in z on both sides, and 100 voxels larger in y and x on both sides (may vary with your input size).

### Zarr volumes

Each volume must have a `RAW` dataset (3D or 4D with a leading channel dim):

```
{name}.zarr/
    RAW          # uint8 or float32, shape (Z, Y, X) or (1, Z, Y, X)
        attrs:
            offset:     [1550, 18500, 20050]  # world-space origin in voxels (Z, Y, X)
            resolution: [1, 1, 1]             # voxel size per axis (Z, Y, X)
```

All coordinates throughout the pipeline are in **Z, Y, X order** — zarr `offset`/`resolution` attrs, CSV synapse coordinates, and all shape/size parameters in the JSON (`input_size`, `blob_radius`, `voxel_size`, etc.).

The `offset` attribute is required for correct coordinate mapping between CSV world coordinates and local voxel positions. If absent, `[0, 0, 0]` is assumed and CSV coordinates must be volume-local.

### CSV files

Two files per zarr, placed in `csv_dir`, named by the zarr stem:

```
csv_dir/
    {name}_pre.csv    # presynaptic (axon) site coordinates
    {name}_post.csv   # postsynaptic (dendrite) site coordinates
```

Format — no header, three columns, comma-delimited, **absolute world voxel coordinates in Z, Y, X order**:

```
1600,18739,20170
1600,18739,20173
1600,18743,20171
```

Row `i` in `_pre.csv` is the presynaptic partner of row `i` in `_post.csv`. Files must have equal row counts. Both files must exist even if empty (empty = volume has no annotated synapses and will not be used for positive sampling).

### Naming convention

Zarr stem must follow `{species}_{region}_{index}` (e.g. `megalopta_FB_1`). The first underscore-delimited token is used as the species identifier for the optional `species` filter in the parameter JSON.

### Parameter JSON

See `parameter.json` for a fully annotated template. Key training fields:

```json
"zarr_locs": ["/path/to/species_region_1.zarr", ...],
"csv_dir":   "/path/to/csvs",
"model_name": "myexp",
"snapshot_dir": "snapshots_myexp",
"tensorboard_dir": "tensorboard_myexp"
```

`json_dir` is accepted but currently unused — ROIs always default to the full zarr extent.
