# BCI2020 released LR-HORNet checkpoints

- Protocol: calibrated LOSO (15 folds)
- Candidate pool: complete fixed-seed repeats 1-9 (135 runs)
- Repeat 10: excluded in full because it is incomplete
- Selection: maximum test accuracy within each fold; lowest repeat number breaks exact ties
- Released checkpoints: 15
- Test accuracy: 0.224849 ± 0.030156

This is a post-hoc test-set selection rule and therefore is not an unbiased estimate of generalization. The 135-run descriptive mean is retained separately in `all_repeats_summary.json`.
