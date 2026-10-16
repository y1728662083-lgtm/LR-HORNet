# LR-HORNet

Official reproduction package for **LR-HORNet: a cross-dataset stable compact inner-speech EEG decoding network based on local recurrence and history-driven offset resampling**.

LR-HORNet combines a patch-wise local recurrent encoder with a history-conditioned offset resampler. The repository contains the training and preprocessing pipeline, paper configurations, comparison-model reproduction archives, section-level ablations, validated result tables, editable plotting sources, and the 36 validation-selected LR-HORNet checkpoints from the original four-dataset experiment. BCI2020 is included as a fifth dataset through its configuration, launchers, lightweight result evidence, and external-checkpoint validation support. Raw EEG data, the 223 local BCI2020 checkpoints, and large third-party pretrained weights are not redistributed.

## Repository contents

    .
    ├── baselines/             # Finalized comparison-model reproduction archives
    ├── checkpoints/main/      # 36 validation-selected LR-HORNet state dictionaries
    ├── configs/main/          # Five cross-dataset LR-HORNet configurations
    ├── experiments/          # Section 4.4.2–4.8.2 experiment packages
    ├── figures/              # Plot data, scripts, PNG, and SVG outputs
    ├── results/              # Audited aggregate tables used by the paper
    ├── src/eeg_pipeline/     # Data, model, training, and evaluation code
    ├── tests/                # Architecture and mechanism checks
    ├── docs/                 # Protocol, experiment, figure, and audit notes
    ├── main.py               # Training entry point
    └── preprocessing.py      # Dataset preprocessing entry point

## Reported LR-HORNet accuracy

Accuracy is the unweighted mean across held-out-subject folds; uncertainty is the sample standard deviation across folds. For BCI2020, each LR-HORNet fold value is first averaged over fixed-seed repeats 1–9, and the reported uncertainty is the sample standard deviation of the 15 fold means.

| Dataset | Folds | Accuracy (%) | Rank among the six compared models |
|---|---:|---:|---:|
| ds003626 | 10 | 30.90 ± 3.88 | 1 |
| Bimodal Inner Speech | 4 | 21.68 ± 3.43 | 1 |
| Directional Words–Russian | 12 | 32.64 ± 13.08 | 3 |
| Directional Words–Spanish | 10 | 38.10 ± 9.72 | 2 |
| BCI2020 | 15 | 21.68 ± 3.45 | 1 |

The original four-dataset machine-readable source is [results/4.4.1/model_summary_all_datasets.csv](results/4.4.1/model_summary_all_datasets.csv). BCI2020 lightweight evidence is stored under [results/4.4.1/bci2020/](results/4.4.1/bci2020/); the five baselines contain one complete result for each of 15 folds, whereas the LR-HORNet paper summary contains 135 completed fits from repeats 1–9. The incomplete tenth repeat is excluded.

## Installation

The reference environment used Python 3.12, PyTorch 2.3.0+cu121, and CUDA 12.1.

    python -m venv .venv
    source .venv/bin/activate
    # Windows PowerShell: .venv\Scripts\Activate.ps1
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt

For figure regeneration, install requirements-figures.txt. Section 4.8.2 Mamba variants and the BCI2020 EEGMamba comparison additionally require CUDA-compatible builds listed in requirements-mamba.txt; their compatibility depends on the installed PyTorch/CUDA ABI.

## Data layout

The repository does not redistribute public EEG datasets. Place or link datasets under the paths below, or edit only data.root in the selected YAML file.

    data/
    ├── ds003626/
    ├── bimodal_inner_speech/
    ├── bci2020/
    └── directional_words/
        ├── russian/
        └── spanish/

Dataset-specific organization and split behavior are documented in [docs/DATASETS.md](docs/DATASETS.md).

## Reproduce the main experiments

Run commands from the repository root:

    python main.py --config configs/main/ds003626.yaml
    python main.py --config configs/main/bimodal_inner_speech.yaml
    python main.py --config configs/main/directional_words_russian.yaml
    python main.py --config configs/main/directional_words_spanish.yaml

The archived BCI2020 YAML retains the original server paths for provenance. Run it through the portable launcher shown below, which generates an ignored runtime configuration with the requested local paths.

The portable BCI2020 launchers read the repository and dataset locations from environment variables. From Bash, Git Bash, or WSL:

    export PAPER4_ROOT="$(pwd)"
    export BCI2020_ROOT="/absolute/path/to/BCI_2020"
    bash run_bci2020_all_models_4gpu.sh

To run only the fixed-seed LR-HORNet repeat launcher, retain the same variables and run:

    bash run_bci2020_lr_hornet_repeat10_4gpu.sh

The six-model BCI2020 suite also needs `requirements-mamba.txt` plus the EEGMamba and LaBraM pretrained files documented in their baseline READMEs. Its preflight verifies their recorded SHA-256 hashes before training.

For Bimodal Inner Speech, validate preprocessing inputs first:

    python preprocessing.py --config configs/main/bimodal_inner_speech.yaml --dry-run
    python preprocessing.py --config configs/main/bimodal_inner_speech.yaml

The main configurations implement calibrated leave-one-subject-out evaluation. Each fold holds out one test subject and the next subject cyclically for validation; class-stratified calibration halves from both held-out subjects are added to training. See [docs/DATASETS.md](docs/DATASETS.md) before comparing results with a different split.

LR right-boundary handling is explicit in each main configuration. Setting `pad_incomplete_patch: true` zero-pads only the incomplete final Patch; setting it to `false` crops an uncovered suffix. The exact per-dataset sample counts are reported in [docs/DATASETS.md](docs/DATASETS.md).

## Reproduce ablations and figures

Each subdirectory under experiments/ is self-contained at the configuration/script level and uses the shared root main.py and src/ code. For example:

    bash experiments/4.5/run_4_5.sh /absolute/path/to/ds003626 /absolute/path/to/output

Run python experiments/<section>/scripts/<runner>.py --help when using PowerShell or when explicit arguments are preferred. The section map and protocol caveats are in [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md).

Figures whose source tables pass the result-provenance audit can be regenerated from their checked-in source records:

    python scripts/reproduce_figures.py

## Reproducibility boundary

Aggregate CSV files, figure-level source tables, source metrics for Sections 4.5, 4.6.3, and 4.7.1, and the 36 selected LR-HORNet checkpoints for the original four datasets are included. Raw EEG, large third-party pretrained weights, non-selected exploratory checkpoints, and all 223 local BCI2020 checkpoints remain outside Git history. The BCI2020 lightweight summaries are included, and the strict checkpoint smoke test accepts an external result tree:

    python scripts/smoke_test_bci2020_checkpoints.py --results-root /absolute/path/to/BCI2020/results --device cpu

Download locations, pinned revisions, and hashes for required EEGMamba and LaBraM pretrained weights are recorded under `baselines/`. Known protocol constraints are disclosed in [docs/RESULTS_AUDIT.md](docs/RESULTS_AUDIT.md). Section 4.5 reuses the Section 4.4.1 LR-HORNet main result for its final stage to avoid duplicating the full ten-fold training cost; the figure builder verifies the reused file hashes and metrics against the main result table.

## Validation

    python scripts/validate_repository.py
    python -m compileall -q main.py preprocessing.py src experiments figures scripts tests

This repository currently has no software license. Reuse beyond inspection and reproduction therefore requires permission from the copyright holder.
