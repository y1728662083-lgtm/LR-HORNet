# Section 4.8.2: attention and Mamba adaptation

The formal comparison contains four variants:

| Variant | Controlled replacement | Protocol |
|---|---|---|
| SATT | Sparse attention over electrodes after PatchLSTM and before the spatial convolution | seed 42, 10-fold LOSO |
| STATT | Sparse attention over time and electrodes at the same insertion point | seed 42, 10-fold LOSO |
| Full-sequence Mamba + LSTM offset | Replace PatchLSTM with one 1153-step Mamba per electrode group | repeat 1 across ten folds; five runs on fold 1 |
| PatchLSTM + Mamba offset | Retain PatchLSTM and replace the offset generator with Mamba | repeat 1 across ten folds; five runs on fold 1 |

## Formal attention checkpoint contract

The reported SATT/STATT checkpoints use the vector-token implementation: one
token is the complete eight-dimensional feature vector at an
`(electrode, time)` location. Their state dictionaries therefore contain
`sparse_attention.route.*` and `feature_readout.*`. They are incompatible with
the earlier exploratory scalar-route implementation whose keys contain
`sparse_attention.routes.0..7.*`.

The checked-in `VectorTokenMSDeformSparseAttention` implementation must not be
replaced by the scalar-route variant. `tests/test_4_8_package.py` locks this
contract.

The earlier exploratory FilterNFC SATT/STATT models remain restorable through
the separate `FeatureWiseMSDeformSparseAttention` scalar-route class. Portable
copies of their original hyperparameters are retained under `configs/legacy/`;
only `data.root` and `training.output_root` were changed from the archived
server paths. These legacy configurations are not part of the formal 4.8.2
comparison, but preserving them prevents the historical
`sparse_attention.routes.*.scalar_readout.*` checkpoints from becoming
orphaned.

## Attention backend

The formal YAML files request `attention_backend: cuda`. On Linux with a
matching PyTorch/CUDA toolchain, build the vendored Deformable-DETR operator:

```bash
cd src/Deformable-DETR-main/models/ops
python setup.py build_ext --inplace
cd ../../../..
```

The vendored operator retains its upstream Apache-2.0 license in
`src/Deformable-DETR-main/LICENSE`.

For CPU-only checkpoint compatibility checks, override only
`attention_backend` to `pytorch`. This reference backend has the same parameter
names and shapes, so `load_state_dict(..., strict=True)` remains meaningful;
it is not a numerical reproduction of the formal CUDA run.

Run the attention experiments from the repository root:

```bash
DATA_ROOT=/absolute/path/to/ds003626 \
OUTPUT_ROOT=outputs/4.8/4.8.2 \
GPU_ID=0 \
bash experiments/4.8.2/run_attention.sh
```

## Mamba variants

Install the CUDA-compatible dependencies in `requirements-mamba.txt`, then run:

```bash
DATA_ROOT=/absolute/path/to/ds003626 \
OUTPUT_ROOT=outputs/4.8/4.8.2 \
GPU_ID=0 MAMBA_MODE=all \
bash experiments/4.8.2/run_mamba.sh
```

`MAMBA_MODE=primary` runs repeat 1 across all ten folds;
`MAMBA_MODE=fold1-repeats` runs the five fold-1 repeats; `all` combines both
sets without duplicating fold 1/repeat 1. The two Mamba models require a
working `mamba_ssm` installation and cannot be treated as runtime-validated on
a machine where that extension is unavailable.

Formal aggregate records and checkpoint hashes are under `results/4.8.2/`.
The exploratory `filternfc_*` attention variants are outside this section's
formal scope.

To verify every formal Section 4.8 checkpoint against its recorded size and
SHA-256 and then perform strict restoration and a forward pass, run:

```bash
python scripts/smoke_test_4_8_checkpoints.py \
  --experiment-root /absolute/path/to/original/4.8 \
  --device cpu --attention-backend pytorch
```

Use `--skip-mamba` only when `mamba_ssm` is unavailable; the summary then
reports 28 skipped checkpoints and is a partial, not full, validation.
