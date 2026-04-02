# Synaptic Partner Detection - rewrite of [Synful](https://github.com/funkelab/synful)
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
predict.py → extract_daisy.py 
```

## Key files

| File | Purpose |
|---|---|
| `dataset.py` | Data loading + GT rendering |
| `model.py` | U-Net architecture |
| `train.py` | Training loop |
| `predict.py` | Blockwise inference over zarr |
| `extract_daisy.py` | Daisy-based chunked synapse extraction (faster and with daisy) |
| `extract.py` | Single-machine extraction (small volumes, slower but no daisy requirement) |

## Testing files

| File | Purpose |
|---|---|
| `view_snap.ipynb` | Viewer for snapshots produced during training|
| `pred_view.ipynb` | Output viewer for predicted volumes |
| `profiling.py` | Profiler, runs a single test with paramter file defined in it, to check for speed and compiling on GPU |

## Parameter files

- `param_template.json` — Template parameter JSON with all options and placeholder paths

---

## Ground truth data format

Training requires one zarr volume and two CSV files per training sample.

### Zarr volumes

Each volume must have a `RAW` dataset (3D or 4D with a leading channel dim):

```
{name}.zarr/
    RAW          # uint8 or float32, shape (Z, Y, X) or (1, Z, Y, X)
        attrs:
            offset:     [z, y, x]   # world-space origin of this volume in voxels
            resolution: [z, y, x]   # voxel size per axis; offset is divided by this if != 1
```

All coordinates throughout the pipeline are in **Z, Y, X order** — this applies to the zarr `offset` and `resolution` attrs, the CSV synapse coordinates, and all shape/size parameters in the JSON (`input_size`, `blob_radius`, `voxel_size`, etc.).

The `offset` attribute is required for correct coordinate mapping between the CSV world coordinates and the local voxel positions within each crop. If absent, `[0, 0, 0]` is assumed and CSV coordinates must be volume-local.

### CSV files

Two files per zarr, placed in the same directory (`csv_dir` in the parameter JSON), named by the zarr stem:

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

Row `i` in `_pre.csv` is the presynaptic partner of row `i` in `_post.csv`. The files must have equal row counts. Both files must exist even if empty (empty = volume has no annotated synapses and will not be used for positive sampling).

### Naming convention

Zarr stem must follow `{species}_{region}_{index}` (e.g. `megalopta_FB_1`). The first underscore-delimited token is used as the species identifier for the optional `species` filter in the parameter JSON.

### Parameter JSON fields

```json
"zarr_locs": ["/path/to/megalopta_FB_1.zarr", ...],
"csv_dir":   "/path/to/csvs",
"voxel_size": [1, 1, 1]
```

`json_dir` is accepted but currently unused — ROIs always default to the full zarr extent.
