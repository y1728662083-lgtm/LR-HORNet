# Results provenance and audit notes

This file records known constraints in the archived experiment outputs. It is part of the reproducibility record, not a post-hoc correction of the reported numbers.

## Main cross-dataset comparison (4.4.1)

The original four LR-HORNet summaries are complete for their expected held-out-subject folds: 10 ds003626, 4 Bimodal Inner Speech, 12 Directional Words–Russian, and 10 Directional Words–Spanish folds. Patch and stride are time-normalized to approximately 0.5 s and 0.25 s, so their sample counts differ by dataset. BCI2020 is the fifth dataset and is documented separately below because its archive contains repeated fits and a distinct checkpoint-selection rule.

### BCI2020 evidence boundary

BCI2020 is a five-class dataset with 64 × 795 inputs sampled at 256 Hz. The configuration uses a 128-sample Patch and 64-sample stride; the incomplete final Patch is completed by 37 samples of right padding. Evaluation follows the same calibrated LOSO rule over 15 held-out-subject folds.

Each of the five comparison baselines has all 15 folds complete. LR-HORNet has nine completed fixed-seed fits for each of 15 folds, or 135 candidates from repeats 1–9. The released result selects the maximum test accuracy within each fold and uses the lowest repeat number for exact ties, yielding 22.48 ± 3.02% across the 15 selected folds. This is a post-hoc test-set selection rule and must not be described as an unbiased generalization estimate. Repeat 10 is excluded in full: folds 1–9 are complete, folds 10–13 contain checkpoints and training logs but no completed test-stage results, and folds 14–15 are absent.

The local BCI2020 result tree contains 223 checkpoints. The 15 selected LR-HORNet checkpoints are released under `checkpoints/main/bci2020/`; their repeat, epoch, metric hashes, and weight hashes are recorded in `results/4.4.1/bci2020/selected_checkpoints.csv`. The remaining 208 checkpoints are not committed. The main smoke script strictly restores the 15 released weights, while the external-tree smoke script can audit all 223 local files and execute synthetic 64 × 795 forward passes where dependencies are available.

## Validation and ablation studies

- **4.4.2:** Test subjects 6, 7, and 8 are fixed while the validation subject is varied. Because both held-out subjects are excluded from the base-training pool, changing the validation subject also changes the base-training composition. The study therefore measures sensitivity to the validation-subject/training-composition condition, not an isolated validation-subject effect. Five repeated executions retain seed 42.
- **4.5:** The stored EEGNet configuration uses batch size 128, while later variants use batch size 32. This is a progressive architecture comparison, not a strict all-hyperparameters-fixed single-variable ablation. To control the computational cost, the final LR-HORNet stage reuses the Section 4.4.1 ds003626 ten-fold main result (0.3090 mean accuracy) rather than rerunning the complete model. The checked-in builder verifies its hashes and metrics against the Section 4.4.1 fold-level table.
- **4.6.1:** Stored values reuse the corresponding 4.5 experiment outputs; they are not a separately rerun benchmark.
- **4.6.2:** Patch-size results are fold-1 experiments with five repeated executions using seed 42. They are not ten-fold estimates and the trend is non-monotonic.
- **4.6.3:** All five electrode-group conditions contain five completed fold-1 runs. The checked-in 25-run table is regenerated from each run's test metrics and training log; the earlier 0/5 record for the one- and two-group conditions was superseded.
- **4.7.1:** All four offset-generation strategies contain ten LOSO folds. The LSTM condition includes a genuine fold-9 record with hashes distinct from fold 8. The checked-in 40-run table is regenerated from the source metrics and training logs.
- **4.7.2:** The controlled causal-ablation and artificial-shift analyses report ranks 1–4, for which both outputs are complete. These are exploratory mechanism analyses rather than a new predictive-performance benchmark.
- **4.8.1:** The primary recurrent-cell study is based on repeated fold-1 executions with seed 42. Batch size differs for some CfC variants.
- **4.8.2:** The formal attention comparison includes only the SATT and STATT vector-token variants. Other exploratory attention variants are not part of the primary table. Mamba reproduction additionally depends on a matching PyTorch/CUDA extension build.

## Statistical wording

Repeated executions with a fixed seed should be described as repeated runs under residual nondeterminism, not independent random-seed trials. Fold-level standard deviations quantify between-subject variation; repeated fold-1 standard deviations quantify training-run variation for one held-out split. These quantities are not interchangeable.
