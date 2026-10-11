# EEGMamba reproduction bundle

This folder contains only EEGMamba code and its shared data/protocol utilities.
It intentionally contains no implementation of any other comparison model.

## Provenance

- Official repository: https://github.com/wjq-learning/EEGMamba.git
- Pinned commit: `dbc83fa072744201e8897aeb9f65007b952ad323`
- Checkpoint source: https://huggingface.co/weighting666/EEGMamba
- Checkpoint revision: `0b060d87acd6f23bf1d0b852bf1726064f335f97`
- Expected checkpoint SHA-256: `b452bb29ecf1d6131ba82a50c6e13823ec1d660d9009d013e691d19b2916f4fe`

The released source snapshot is under `addition/official_sources/EEGMamba`.
Download `pretrained_EEGMamba.pth` into `addition/weights/` and verify its hash before use.
Weights are not duplicated in this code-only folder.

## Included experiment configurations

- `ds003626_official_baselines.yaml`: exact hyperparameters used by the completed
  ds003626 result currently archived in experiment 4.4.1.
- Russian, Spanish and Bimodal YAMLs: the current multidataset rerun settings.
- All splits use calibrated LOSO. Russian and Spanish independently use their
  within-language EEG-channel intersection.

Edit each YAML's `data.root` before running. Bimodal training expects the
official-code-derived FIF files under `derivatives/paper4/bimodal`.

## Environment and checkpoint

Install the dependencies into a compatible PyTorch/CUDA environment:

```bash
python -m pip install -r requirements.txt
sha256sum addition/weights/pretrained_EEGMamba.pth
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
  --models eegmamba --folds 1
```

The four-GPU dispatcher starts at most one training process per GPU.
