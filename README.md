# LR-HORNet

Official reproduction package for **LR-HORNet: a cross-dataset stable compact inner-speech EEG decoding network based on local recurrence and history-driven offset resampling**.

LR-HORNet combines a patch-wise local recurrent encoder with a history-conditioned offset resampler. The repository contains the training and preprocessing pipeline, paper configurations, comparison-model reproduction archives, section-level ablations, validated result tables, editable plotting sources, and 51 released LR-HORNet checkpoints covering all five datasets. Raw EEG data, non-selected checkpoints, and large third-party pretrained weights are not redistributed.

## Repository contents

    .
    ├── baselines/             # Finalized comparison-model reproduction archives
    ├── checkpoints/main/      # 51 released LR-HORNet state dictionaries
    ├── configs/main/          # Five cross-dataset LR-HORNet configurations
    ├── experiments/          # Section 4.4.2–4.8.2 experiment packages
    ├── figures/              # Plot data, scripts, PNG, and SVG outputs
    ├── results/              # Audited aggregate tables used by the paper
    ├── src/eeg_pipeline/     # Data, model, training, and evaluation code
    ├── src/Deformable-DETR-main/ # Vendored 4.8.2 sparse-attention operator
    ├── tests/                # Architecture and mechanism checks
    ├── docs/                 # Protocol, experiment, figure, and audit notes
    ├── main.py               # Training entry point
    └── preprocessing.py      # Dataset preprocessing entry point

## Reported LR-HORNet accuracy

Accuracy is the unweighted mean across held-out-subject folds; uncertainty is the sample standard deviation across folds. For BCI2020 only, the released artifact follows the requested post-hoc rule of selecting the maximum test accuracy within each fold over fixed-seed repeats 1–9 (lowest repeat number breaks exact ties). This test-set selection is optimistic and is not an unbiased estimate of generalization.

| Dataset | Folds | Accuracy (%) | Rank among the six compared models |
|---|---:|---:|---:|
| ds003626 | 10 | 30.90 ± 3.88 | 1 |
| Bimodal Inner Speech | 4 | 21.68 ± 3.43 | 1 |
| Directional Words–Russian | 12 | 32.64 ± 13.08 | 3 |
| Directional Words–Spanish | 10 | 38.10 ± 9.72 | 2 |
| BCI2020 | 15 | 22.48 ± 3.02 | 1 |

The original four-dataset machine-readable source is [results/4.4.1/model_summary_all_datasets.csv](results/4.4.1/model_summary_all_datasets.csv). BCI2020 evidence is stored under [results/4.4.1/bci2020/](results/4.4.1/bci2020/): the five baselines contain one complete result per fold, and LR-HORNet has 135 complete candidates from repeats 1–9. The 15 selected checkpoints and their hashes are recorded in `selected_checkpoints.csv`; the incomplete tenth repeat is excluded in full.

## Installation

The reference environment used Python 3.12, PyTorch 2.3.0+cu121, and CUDA 12.1.

    python -m venv .venv
    source .venv/bin/activate
    # Windows PowerShell: .venv\Scripts\Activate.ps1
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt

For figure regeneration, install requirements-figures.txt. Section 4.8.2 Mamba variants and the BCI2020 EEGMamba comparison additionally require CUDA-compatible builds listed in requirements-mamba.txt; their compatibility depends on the installed PyTorch/CUDA ABI. The formal SATT/STATT configurations use the vendored Deformable-DETR CUDA operator; build instructions and the vector-token checkpoint contract are documented in [experiments/4.8.2/README.md](experiments/4.8.2/README.md).

## Data layout

The repository does not redistribute public EEG datasets. Place or link datasets under the paths below, or edit only data.root in the selected YAML file.

    data/
    ├── ds003626/
    ├── bimodal_inner_speech/
    ├── bci2020/
    └── directional_words/
        └── preprocessed/
            ├── russian/sub*/sub*_epochs.fif
            └── spanish/sub*/sub*_epochs.fif

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

After repeats 1–9 are complete, reproduce the disclosed fold-wise checkpoint
selection and its provenance table with:

    python scripts/select_bci2020_checkpoints.py --results-root /absolute/path/to/bci2020_lr_hornet_fixed_seed_repeat10
    python scripts/build_checkpoint_manifest.py

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

Aggregate CSV files, figure-level source tables, source metrics for Sections 4.5, 4.6.3, and 4.7.1, and 51 released LR-HORNet checkpoints are included. These comprise 36 validation-selected checkpoints from the original four datasets and 15 post-hoc test-selected BCI2020 checkpoints. Raw EEG, large third-party pretrained weights, and the other 208 local BCI2020 checkpoints remain outside Git history. The external-tree smoke test can additionally audit all 223 local BCI2020 checkpoints:

    python scripts/smoke_test_main_checkpoints.py --device cpu
    python scripts/smoke_test_bci2020_checkpoints.py --results-root /absolute/path/to/BCI2020/results --device cpu
    python scripts/smoke_test_4_8_checkpoints.py --experiment-root /absolute/path/to/4.8 --device cpu

On a machine without `mamba_ssm`, add `--skip-mamba` to the Section 4.8 command. This explicitly validates 45/73 formal checkpoints (all 4.8.1 and SATT/STATT checkpoints) and reports the 28 Mamba checkpoints as skipped; it must not be described as full 4.8 validation.

Download locations, pinned revisions, and hashes for required EEGMamba and LaBraM pretrained weights are recorded under `baselines/`. Known protocol constraints are disclosed in [docs/RESULTS_AUDIT.md](docs/RESULTS_AUDIT.md), and executed-versus-blocked smoke-test coverage is recorded in [docs/VALIDATION.md](docs/VALIDATION.md). Section 4.5 reuses the Section 4.4.1 LR-HORNet main result for its final stage to avoid duplicating the full ten-fold training cost; the figure builder verifies the reused file hashes and metrics against the main result table.

## Validation

    python scripts/run_test_suites.py
    python scripts/smoke_test_main_checkpoints.py --device cpu
    python scripts/validate_repository.py
    python -m compileall -q main.py preprocessing.py src experiments figures scripts tests

The main-checkpoint smoke command executes all 51 released weights by default.
For a faster CPU-only structural regression, add `--representative-forward`;
all 51 states are still hash-checked and restored strictly, while forward and
backward execution is performed once per dataset shape. Add `--skip-backward`
for the CI-style contract used by the unit tests: every state is still restored
strictly and one inference is executed for each of the five real input shapes.

The LR-HORNet repository currently has no project-wide software license. Reuse beyond inspection and reproduction therefore requires permission from the copyright holder. The vendored Deformable-DETR operator is separately covered by the Apache-2.0 license retained at `src/Deformable-DETR-main/LICENSE`.
