# Section 4.4.1 dataset-wise result summary

All values are regenerated from the archived, non-checkpoint `test_metrics.json` files. Accuracy summaries are unweighted arithmetic means across LOSO folds; `±` denotes the sample standard deviation across folds.

The original four-dataset archive contains 216 validated fold-level results across six models. BCI2020 is supplied as a separate fifth-dataset evidence package because the five baselines and LR-HORNet use different archived repetition counts.

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

## BCI2020

Calibrated LOSO folds: 15; classes: 5; chance level: 20.00%; input: 64 × 795 at 256 Hz. LR-HORNet uses a 128-sample Patch, a 64-sample stride, and 37 samples of right padding for the incomplete final Patch.

| Rank | Model | Test accuracy (%) | Archived statistical unit |
|---:|:---|---:|:---|
| 1 | LR-HORNet | 21.68 ± 3.45 | 15 fold means, each from repeats 1–9 |
| 2 | ATCNet | 21.45 ± 2.45 | 15 completed folds, one fit per fold |
| 3 | EEGNet | 20.88 ± 3.69 | 15 completed folds, one fit per fold |
| 4 | FAST | 20.24 ± 1.99 | 15 completed folds, one fit per fold |
| 5 | LaBraM | 20.12 ± 2.64 | 15 completed folds, one fit per fold |
| 6 | EEGMamba | 19.39 ± 3.00 | 15 completed folds, one fit per fold |

The five comparison baselines each contain all 15 fold results. The LR-HORNet paper aggregate contains nine completed fixed-seed fits per fold (repeats 1–9), totaling 135 fits. Repeat 10 is excluded in full: folds 1–9 are complete, folds 10–13 have checkpoints and training logs but no completed test-stage results, and folds 14–15 are absent. The reported LR-HORNet value is the unweighted mean of the 15 fold means; `±` is the sample standard deviation across those fold means. It is not a best-test-checkpoint selection.

The audited summarizer reproduces this scope with `--include-repeats 1 2 3 4 5 6 7 8 9`; it requires every selected repeat to contain all 15 folds and ignores records from non-selected partial repeats.

## Files

- `fold_level_results.csv`: every dataset/model/fold result with source path and SHA-256 hash.
- `model_summary_all_datasets.csv`: complete long-format summary.
- `accuracy_mean_sd_wide.csv`: paper-table-friendly cross-dataset accuracy view.
- `<dataset>_model_summary.csv`: one ranked table per dataset.
- `bci2020/model_summary.csv`: one complete 15-fold result for each of the five comparison baselines.
- `bci2020/overall_summary.json`: LR-HORNet aggregate over repeats 1–9 only.
- `bci2020/summary_by_fold.csv`: 15 LR-HORNet fold summaries, each based on nine completed fits.
- `bci2020/summary_report.md`: concise LR-HORNet protocol and aggregation record.

The BCI2020 checkpoint tree remains external to Git. The four files above are lightweight copies of the archived summaries; the repository's strict smoke script can restore and execute the external checkpoints without importing them into the repository.
