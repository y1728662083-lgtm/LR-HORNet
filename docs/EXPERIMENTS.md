# Experiment map

| Paper section | Repository path | Purpose | Stored output |
|---|---|---|---|
| 4.4.1 | configs/main/, results/4.4.1/ | Cross-dataset model comparison | Five dataset summaries; BCI2020 uses lightweight evidence |
| 4.4.2 | experiments/4.4.2/, figures/4.4.2/ | Validation-subject and associated training-composition sensitivity | Figure-level tables and plots |
| 4.5 | experiments/4.5/, results/4.5/ | Progressive component ablation; final stage reuses the 4.4.1 main result | Ten-fold summary |
| 4.6.1 | experiments/4.6.1/, results/4.6.1/ | Local feature extractor comparison | Stored summary |
| 4.6.2 | experiments/4.6.2/, results/4.6.2/ | Patch-length analysis | Fold-1 repeated-run summary |
| 4.6.3 | experiments/4.6.3/, results/4.6.3/ | Electrode-group count analysis | 5 conditions × 5 verified runs |
| 4.7.1 | experiments/4.7.1/, results/4.7.1/ | Offset-generation strategy comparison | 4 strategies × 10 verified folds |
| 4.7.2 | experiments/4.7.2/, results/4.7.2/ | Offset mechanism and controlled perturbation | Four completed ranked checkpoints |
| 4.8.1 | experiments/4.8.1/, results/4.8.1/ | Recurrent-operator adaptation | Repeated-run and supplemental tables |
| 4.8.2 | experiments/4.8.2/, results/4.8.2/ | Attention/Mamba adaptation | Ten-fold primary and fold-1 repeat summaries |

## Running a section package

The Bash launchers accept or read a dataset root and an output root, create run-specific YAML files under the chosen output directory, and invoke the shared root main.py. Inspect the Python runner's help before starting a long job:

    python experiments/4.5/scripts/run_4_5.py --help
    python experiments/4.6.3/scripts/run_4_6_3.py --help
    python experiments/4.8.2/scripts/run_4_8_2.py --help

Most YAML files checked into Git use repository-relative data/ and outputs/ paths. The seven archived BCI2020 source configurations retain their original server paths byte-for-byte so their SHA-256 values remain tied to the completed runs; the portable launchers generate ignored runtime YAML files with explicit local data and output roots. Generated run configurations and non-selected checkpoints belong under outputs/, which is ignored by Git. The 36 selected state dictionaries released under checkpoints/main/ cover the original four datasets; the 223 local BCI2020 checkpoints are not committed.

## BCI2020 launch and validation

BCI2020 is a five-class, 64-channel, 256 Hz dataset evaluated with 15-fold calibrated LOSO. The 795-sample input uses a 128-sample Patch and 64-sample stride, with 37 zeros appended only to complete the final Patch. The portable launchers take their locations from `PAPER4_ROOT` and `BCI2020_ROOT`:

    export PAPER4_ROOT="$(pwd)"
    export BCI2020_ROOT="/absolute/path/to/BCI_2020"
    bash run_bci2020_all_models_4gpu.sh

The all-model launcher preprocesses or verifies the dataset, preflights the six configurations, plans the jobs, runs 15 folds for each model, and writes summaries outside the repository source tree. The fixed-seed LR-HORNet launcher can be invoked with the same environment variables:

    bash run_bci2020_lr_hornet_repeat10_4gpu.sh

The all-model run also requires the CUDA-compatible packages in `requirements-mamba.txt` and the two pretrained files documented in `baselines/eegmamba/README.md` and `baselines/labram/README.md`. The preflight command verifies both file hashes before starting any fit.

For archived checkpoints stored outside Git, perform strict state restoration and a forward pass with:

    python scripts/smoke_test_bci2020_checkpoints.py --results-root /absolute/path/to/BCI2020/results --device cpu

The paper summary does not treat the tenth LR-HORNet repeat as complete. Repeats 1–9 provide 135 completed fits. In repeat 10, folds 1–9 are complete, folds 10–13 have checkpoints and training logs but lack completed test-stage results, and folds 14–15 are absent; the entire partial repeat is excluded from the paper aggregate.

To regenerate the reported aggregate from that partially populated external result tree, select the nine complete repeat identifiers explicitly:

    python addition/bci2020_suite/summarize_lr_hornet_fixed_seed_repeats.py --root /absolute/path/to/bci2020_lr_hornet_fixed_seed_repeat10 --expected-folds 15 --include-repeats 1 2 3 4 5 6 7 8 9 --expected-seed 42

## Interpretation

The sub-experiments do not all use the same statistical unit. Some are complete held-out-subject fold comparisons; others are repeated training runs on fold 1, mechanism probes, or exploratory adaptations. Never combine their standard deviations without checking RESULTS_AUDIT.md and the CSV columns describing folds, repeats, and completion.
