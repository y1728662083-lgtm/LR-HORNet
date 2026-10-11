# Section 4.4.1 dataset-wise result summary

All values are regenerated from the archived, non-checkpoint `test_metrics.json` files. Accuracy summaries are unweighted arithmetic means across LOSO folds; `±` denotes the sample standard deviation across folds.

Validated 216 fold-level results across 4 datasets and 6 models.

## ds003626

LOSO folds: 10; classes: 4; chance level: 25.00%.

| Rank | Model | Accuracy (%) | Balanced Acc (%) | Macro-F1 (%) |
|---:|:---|---:|---:|---:|
| 1 | LR-HORNet | 30.90 ± 3.88 | 30.91 ± 3.88 | 28.77 ± 5.49 |
| 2 | ATCNet | 28.09 ± 4.37 | 28.11 ± 4.37 | 27.39 ± 3.82 |
| 3 | EEGNet | 28.03 ± 4.20 | 28.02 ± 4.21 | 27.14 ± 4.27 |
| 4 | EEGMamba | 25.79 ± 4.84 | 25.78 ± 4.85 | 24.46 ± 4.75 |
| 5 | FAST | 25.58 ± 3.87 | 25.58 ± 3.86 | 23.83 ± 4.01 |
| 6 | LaBraM | 24.14 ± 2.39 | 24.12 ± 2.42 | 16.85 ± 4.64 |

## Directional Words–Russian

LOSO folds: 12; classes: 6; chance level: 16.67%.

| Rank | Model | Accuracy (%) | Balanced Acc (%) | Macro-F1 (%) |
|---:|:---|---:|---:|---:|
| 1 | FAST | 53.35 ± 26.57 | 53.19 ± 26.65 | 52.28 ± 27.69 |
| 2 | LaBraM | 35.10 ± 24.67 | 35.16 ± 24.91 | 34.15 ± 24.33 |
| 3 | LR-HORNet | 32.64 ± 13.08 | 32.31 ± 12.76 | 30.29 ± 13.27 |
| 4 | EEGMamba | 28.94 ± 20.98 | 28.89 ± 20.92 | 27.92 ± 20.86 |
| 5 | EEGNet | 26.99 ± 10.69 | 26.92 ± 10.43 | 24.27 ± 10.50 |
| 6 | ATCNet | 24.23 ± 14.12 | 24.51 ± 14.40 | 21.55 ± 13.84 |

## Directional Words–Spanish

LOSO folds: 10; classes: 6; chance level: 16.67%.

| Rank | Model | Accuracy (%) | Balanced Acc (%) | Macro-F1 (%) |
|---:|:---|---:|---:|---:|
| 1 | FAST | 63.14 ± 9.65 | 63.11 ± 9.72 | 63.08 ± 9.68 |
| 2 | LR-HORNet | 38.10 ± 9.72 | 37.73 ± 10.14 | 35.03 ± 11.13 |
| 3 | LaBraM | 37.22 ± 9.88 | 36.86 ± 10.26 | 36.63 ± 10.29 |
| 4 | EEGNet | 35.27 ± 10.51 | 35.68 ± 11.22 | 33.53 ± 10.60 |
| 5 | ATCNet | 33.24 ± 11.72 | 32.66 ± 11.41 | 31.48 ± 11.88 |
| 6 | EEGMamba | 20.10 ± 2.56 | 19.70 ± 2.51 | 18.25 ± 3.64 |

## Bimodal Inner Speech

LOSO folds: 4; classes: 8; chance level: 12.50%.

| Rank | Model | Accuracy (%) | Balanced Acc (%) | Macro-F1 (%) |
|---:|:---|---:|---:|---:|
| 1 | LR-HORNet | 21.68 ± 3.43 | 21.67 ± 3.43 | 20.98 ± 4.31 |
| 2 | EEGNet | 17.78 ± 4.74 | 17.78 ± 4.74 | 17.13 ± 4.54 |
| 3 | ATCNet | 16.85 ± 4.14 | 16.85 ± 4.14 | 15.72 ± 3.96 |
| 4 | LaBraM | 15.91 ± 4.46 | 15.89 ± 4.44 | 13.73 ± 4.21 |
| 5 | EEGMamba | 15.60 ± 1.38 | 15.61 ± 1.37 | 14.96 ± 1.65 |
| 6 | FAST | 14.51 ± 1.32 | 14.53 ± 1.29 | 7.98 ± 3.43 |

## Files

- `fold_level_results.csv`: every dataset/model/fold result with source path and SHA-256 hash.
- `model_summary_all_datasets.csv`: complete long-format summary.
- `accuracy_mean_sd_wide.csv`: paper-table-friendly cross-dataset accuracy view.
- `<dataset>_model_summary.csv`: one ranked table per dataset.

Run `python generate_dataset_summaries.py` to regenerate all tables from the raw experiment outputs.
