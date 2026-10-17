# LR-HORNet main-experiment checkpoints

This directory contains one released LR-HORNet checkpoint for every
held-out-subject fold reported in Section 4.4.1:

- ds003626: 10 folds;
- Bimodal Inner Speech: 4 folds;
- Directional Words–Russian: 12 folds;
- Directional Words–Spanish: 10 folds;
- BCI2020: 15 folds.

All files are small PyTorch state dictionaries. `manifest.csv` records each
checkpoint's dataset, fold, selection provenance, validation and test accuracy,
source-metrics path and digest, checkpoint byte size, and checkpoint SHA-256
digest. The model configuration must match the corresponding YAML file under
`../../configs/main/`.
