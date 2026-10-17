# LaBraM reproduction bundle

This folder contains only LaBraM code and its shared data/protocol utilities.
It intentionally contains no implementation of any other comparison model.

## Provenance

- Official repository: https://github.com/935963004/LaBraM.git
- Pinned commit: `c431221e6cfd23dbfa9950e0180682fb322b0548`
- Checkpoint source: https://huggingface.co/eeg-telecom-paris/labram-base-official
- Checkpoint revision: `99fffc632f3e9a9e8c500eefbc75d74777cb7d6d`
- Expected checkpoint SHA-256: `f41dcf13b2ad84bfca4f0ce24da041608c8ca197ff174b2e16a2f19dd245b476`

The released source snapshot is under `addition/official_sources/LaBraM`.
Download `labram-base-official.safetensors` into `addition/weights/` and verify its hash before use.
Weights are not duplicated in this code-only folder.

## Included experiment configurations

- `ds003626_official_baselines.yaml`: exact hyperparameters used by the completed
  ds003626 result currently archived in experiment 4.4.1.
- Russian, Spanish and Bimodal YAMLs: the current multidataset rerun settings.
- `bci2020_official_baselines.yaml`: the hash-locked completed BCI2020 setting.
- All splits use calibrated LOSO. Russian and Spanish independently use their
  within-language EEG-channel intersection.

The package-local dispatcher schedules the original four-dataset batch. Run
BCI2020 from the repository root with `run_bci2020_all_models_4gpu.sh`; that
launcher preserves the archived YAML and writes a portable runtime copy with
the requested dataset and output roots.

Relative `data.root` values are resolved against the top-level LR-HORNet
repository, independent of the launcher's working directory. Place datasets
under the root README's `data/` layout, or set `data.root` to an absolute path.
Bimodal training expects the official-code-derived FIF files under
`derivatives/paper4/bimodal` within its dataset root.

## Environment and checkpoint

Install the dependencies into a compatible PyTorch/CUDA environment:

```bash
python -m pip install -r requirements.txt
sha256sum addition/weights/labram-base-official.safetensors
```

## Validate and run

```bash
export PYTHONPATH="$PWD/src:$PWD:$PWD/addition:${PYTHONPATH:-}"
python -m compileall -q addition src
python -u addition/dispatch_4gpu.py --plan
bash run_4gpu.sh
```

Run one fold directly:

```bash
CUDA_VISIBLE_DEVICES=0 python -u addition/run_official_baselines.py \
  --config addition/configs/ds003626_official_baselines.yaml \
  --models labram --folds 1
```

The four-GPU dispatcher starts at most one training process per GPU.
