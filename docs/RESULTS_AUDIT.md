# Results provenance and audit notes

This file records known constraints in the archived experiment outputs. It is part of the reproducibility record, not a post-hoc correction of the reported numbers.

## Main cross-dataset comparison (4.4.1)

The four LR-HORNet summaries are complete for their expected held-out-subject folds: 10 ds003626, 4 Bimodal Inner Speech, 12 Directional Words–Russian, and 10 Directional Words–Spanish folds. Patch and stride are time-normalized to approximately 0.5 s and 0.25 s, so their sample counts differ by dataset.

## Validation and ablation studies

- **4.4.2:** Test subjects 6, 7, and 8 are fixed while the validation subject is varied. Changing the validation subject also changes the remaining base-training subject set. Five repeated executions retain seed 42.
- **4.5:** The stored EEGNet configuration uses batch size 128, while later variants use batch size 32. This is a progressive architecture comparison, not a strict all-hyperparameters-fixed single-variable ablation. The formal stored summary reports 0.3065 for no deformation and 0.3009 for the full model. The legacy figure table instead uses a 0.3090 LR-HORNet row from the separate main-model result directory. The formal summary therefore takes precedence and the 4.5 figure package is excluded from the default rebuild.
- **4.6.1:** Stored values reuse the corresponding 4.5 experiment outputs; they are not a separately rerun benchmark.
- **4.6.2:** Patch-size results are fold-1 experiments with five repeated executions using seed 42. They are not ten-fold estimates and the trend is non-monotonic.
- **4.6.3:** One- and two-group configurations completed 0/5 requested formal reruns because of CUDA out-of-memory failures. Only the 4-, 8-, and 16-group rows are complete. The legacy figure package contains numeric 1- and 2-group rows from a superseded result set; these rows are excluded from the formal summary and the figure package is excluded from the default rebuild.
- **4.7.1:** The archived LR-HORNet checkpoint set contains two files internally labeled fold 8 and no genuine fold-9 checkpoint. The audited summary therefore retains nine unique LR-HORNet folds and marks complete_10_folds=0; the other three strategies have ten folds. The legacy figure table counts the duplicate as a tenth fold, so the figure package is excluded from the default rebuild.
- **4.7.2:** Rank 5 of the mechanism probe is incomplete and excluded from aggregate statistics. Ranks 1–4 are exploratory mechanism analyses rather than a new predictive-performance benchmark.
- **4.8.1:** The primary recurrent-cell study is based on repeated fold-1 executions with seed 42. Batch size differs for some CfC variants.
- **4.8.2:** The formal attention comparison includes only the SATT and STATT vector-token variants. Other exploratory attention variants are not part of the primary table. Mamba reproduction additionally depends on a matching PyTorch/CUDA extension build.

## Statistical wording

Repeated executions with a fixed seed should be described as repeated runs under residual nondeterminism, not independent random-seed trials. Fold-level standard deviations quantify between-subject variation; repeated fold-1 standard deviations quantify training-run variation for one held-out split. These quantities are not interchangeable.
