# LR-HORNet main-experiment checkpoints

This directory contains one released LR-HORNet checkpoint for every
held-out-subject fold reported in Section 4.4.1:

- ds003626: 10 folds;
- Bimodal Inner Speech: 4 folds;
- Directional Words–Russian: 12 folds;
- Directional Words–Spanish: 10 folds;
- BCI2020: 15 folds.

The original four-dataset files are validation-selected. The BCI2020 files use
the separately disclosed post-hoc rule of selecting the maximum test accuracy
within each fold from complete repeats 1–9; this result is optimistic and is not
an unbiased generalization estimate. Repeat 10 is excluded in full.

All files are small PyTorch state dictionaries. `manifest.csv` records each
checkpoint's dataset, fold, selection provenance, validation and test accuracy,
source-metrics path and digest, checkpoint byte size, and checkpoint SHA-256
digest. The model configuration must match the corresponding YAML file under
`../../configs/main/`.
