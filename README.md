# LR-HORNet

Official reproduction package for **LR-HORNet: a cross-dataset stable compact inner-speech EEG decoding network based on local recurrence and history-driven offset resampling**.

LR-HORNet combines a patch-wise local recurrent encoder with a history-conditioned offset resampler. The repository contains the training pipeline, paper configurations, section-level ablations, processed result tables, and editable plotting sources. Raw EEG data, checkpoints, and per-run logs are intentionally excluded.

## Repository contents

    .
    ├── configs/main/          # Four cross-dataset LR-HORNet configurations
    ├── experiments/          # Section 4.4.2–4.8.2 experiment packages
    ├── figures/              # Plot data, scripts, PNG, and SVG outputs
    ├── results/              # Audited aggregate tables used by the paper
    ├── src/eeg_pipeline/     # Data, model, training, and evaluation code
    ├── tests/                # Architecture and mechanism checks
    ├── docs/                 # Protocol, experiment, figure, and audit notes
    ├── main.py               # Training entry point
    └── preprocessing.py      # Dataset preprocessing entry point

## Reported LR-HORNet accuracy

Accuracy is the unweighted mean across held-out-subject folds; uncertainty is the sample standard deviation across folds.

| Dataset | Folds | Accuracy (%) | Rank among the six compared models |
|---|---:|---:|---:|
| ds003626 | 10 | 30.90 ± 3.88 | 1 |
| Bimodal Inner Speech | 4 | 21.68 ± 3.43 | 1 |
| Directional Words–Russian | 12 | 32.64 ± 13.08 | 3 |
| Directional Words–Spanish | 10 | 38.10 ± 9.72 | 2 |

The machine-readable source is [results/4.4.1/model_summary_all_datasets.csv](results/4.4.1/model_summary_all_datasets.csv).

## Installation

The reference environment used Python 3.12, PyTorch 2.3.0+cu121, and CUDA 12.1.

    python -m venv .venv
    source .venv/bin/activate
    # Windows PowerShell: .venv\Scripts\Activate.ps1
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt

For figure regeneration, install requirements-figures.txt. Section 4.8.2 Mamba variants additionally require CUDA-compatible builds listed in requirements-mamba.txt; their compatibility depends on the installed PyTorch/CUDA ABI.

## Data layout

The repository does not redistribute public EEG datasets. Place or link datasets under the paths below, or edit only data.root in the selected YAML file.

    data/
    ├── ds003626/
    ├── bimodal_inner_speech/
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

For Bimodal Inner Speech, validate preprocessing inputs first:

    python preprocessing.py --config configs/main/bimodal_inner_speech.yaml --dry-run
    python preprocessing.py --config configs/main/bimodal_inner_speech.yaml

The main configurations implement calibrated leave-one-subject-out evaluation. Each fold holds out one test subject and the next subject cyclically for validation; class-stratified calibration halves from both held-out subjects are added to training. See [docs/DATASETS.md](docs/DATASETS.md) before comparing results with a different split.

## Reproduce ablations and figures

Each subdirectory under experiments/ is self-contained at the configuration/script level and uses the shared root main.py and src/ code. For example:

    bash experiments/4.5/run_4_5.sh /absolute/path/to/ds003626 /absolute/path/to/output

Run python experiments/<section>/scripts/<runner>.py --help when using PowerShell or when explicit arguments are preferred. The section map and protocol caveats are in [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md).

Figures whose source tables pass the result-provenance audit can be regenerated from their local CSV sources:

    python scripts/reproduce_figures.py

## Reproducibility boundary

Aggregate CSV files and figure-level source tables are included. Checkpoints, raw EEG, and large per-run logs are excluded from Git history. Several archived exploratory experiments have known incompleteness or provenance constraints; these are disclosed in [docs/RESULTS_AUDIT.md](docs/RESULTS_AUDIT.md) and must not be interpreted as complete repeated-seed benchmarks. Sections 4.5, 4.6.3, and 4.7.1 are intentionally excluded from the one-command figure rebuild until their legacy plot tables are reconciled with the formal result summaries.

## Validation

    python scripts/validate_repository.py
    python -m compileall -q main.py preprocessing.py src experiments figures scripts tests

This repository currently has no software license. Reuse beyond inspection and reproduction therefore requires permission from the copyright holder.
