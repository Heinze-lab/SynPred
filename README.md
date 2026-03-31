# Synaptic Partner Detection — ogogog

Training, prediction, and extraction pipeline for synapse detection and partner vector prediction.

## Environment Setup

Requires CUDA 12.4 and conda.

```bash
conda create -n synful python=3.10 -y
conda activate synful
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

### Connectomics / CATMAID

```bash
pip install \
    python-catmaid==2.4.2 \
    navis==1.5.0 \
    caveclient==5.17.4
```

### SegToPCG (for match_to_roots)

Clone and install from source — not on PyPI:

```bash
git clone https://github.com/seung-lab/SegToPCG /path/to/SegToPCG
pip install -e /path/to/SegToPCG
```

Update `match_to_roots/` imports to point to your local clone if needed.

## Pipeline

```
predict.py → extract_daisy.py → match_to_roots/get_roots_synapses.py
```

Or use the shell script:

```bash
bash run_pipeline.sh parameter.json
```

## Key files

| File | Purpose |
|---|---|
| `train.py` | Training loop |
| `dataset.py` | Data loading + GT rendering |
| `model.py` | U-Net architecture |
| `predict.py` | Blockwise inference over zarr |
| `extract_daisy.py` | Daisy-based chunked synapse extraction |
| `extract.py` | Single-machine extraction (small volumes) |
| `match_to_roots/get_roots_synapses.py` | Map synapses to neuron IDs via CAVE/CATMAID |
| `view_snap.ipynb` | Snapshot viewer for training QC |
| `view_out.ipynb` | Output viewer for predicted synapses |
| `parameter_small_blob_3.json` | Current best parameter file |

## Parameter files

- `parameter.json` — megalopta PB2 full prediction
- `parameter_small_blob_1/2/3.json` — training experiments (3 is current)
