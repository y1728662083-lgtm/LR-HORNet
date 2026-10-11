# Experiment map

| Paper section | Repository path | Purpose | Stored output |
|---|---|---|---|
| 4.4.1 | configs/main/, results/4.4.1/ | Cross-dataset model comparison | Four complete dataset summaries |
| 4.4.2 | experiments/4.4.2/, figures/4.4.2/ | Validation-subject sensitivity | Figure-level tables and plots |
| 4.5 | experiments/4.5/, results/4.5/ | Progressive component ablation | Ten-fold summary |
| 4.6.1 | experiments/4.6.1/, results/4.6.1/ | Local feature extractor comparison | Stored summary |
| 4.6.2 | experiments/4.6.2/, results/4.6.2/ | Patch-length analysis | Fold-1 repeated-run summary |
| 4.6.3 | experiments/4.6.3/, results/4.6.3/ | Electrode-group count analysis | Partial summary with completion status |
| 4.7.1 | experiments/4.7.1/, results/4.7.1/ | Offset-generation strategy comparison | Fold table, contrasts, and summary |
| 4.7.2 | experiments/4.7.2/, results/4.7.2/ | Offset mechanism and controlled perturbation | Descriptive/mechanism summaries |
| 4.8.1 | experiments/4.8.1/, results/4.8.1/ | Recurrent-operator adaptation | Repeated-run and supplemental tables |
| 4.8.2 | experiments/4.8.2/, results/4.8.2/ | Attention/Mamba adaptation | Ten-fold primary and fold-1 repeat summaries |

## Running a section package

The Bash launchers accept or read a dataset root and an output root, create run-specific YAML files under the chosen output directory, and invoke the shared root main.py. Inspect the Python runner's help before starting a long job:

    python experiments/4.5/scripts/run_4_5.py --help
    python experiments/4.6.3/scripts/run_4_6_3.py --help
    python experiments/4.8.2/scripts/run_4_8_2.py --help

The YAML files checked into Git use repository-relative data/ and outputs/ paths. The runners override these fields when explicit paths are supplied. Generated run configurations and checkpoints belong under outputs/, which is ignored by Git.

## Interpretation

The sub-experiments do not all use the same statistical unit. Some are complete held-out-subject fold comparisons; others are repeated training runs on fold 1, mechanism probes, or exploratory adaptations. Never combine their standard deviations without checking RESULTS_AUDIT.md and the CSV columns describing folds, repeats, and completion.
