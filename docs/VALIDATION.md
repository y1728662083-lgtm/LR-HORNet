# Validation record

This record separates checks that were executed from checks that remain
dependency- or data-blocked. It reflects the repository audit performed on
2026-10-17.

## Five-dataset release

| Dataset | Input | Classes | Folds | Released LR-HORNet checkpoints |
|---|---:|---:|---:|---:|
| ds003626 | 128 × 1153 | 4 | 10 | 10 |
| Bimodal Inner Speech | 64 × 359 | 8 | 4 | 4 |
| Directional Words–Russian | 34 × 250 | 6 | 12 | 12 |
| Directional Words–Spanish | 38 × 750 | 6 | 10 | 10 |
| BCI2020 | 64 × 795 | 5 | 15 | 15 |

All 51 released files match `checkpoints/main/manifest.csv`, load with
`weights_only=True`, contain finite tensors, restore with `strict=True`, and
have been exercised with their dataset's real input geometry. The reusable
smoke command is:

    python scripts/smoke_test_main_checkpoints.py --device cpu

For routine CPU CI, use `--representative-forward --skip-backward`. This still
hash-checks and strictly restores all 51 states, then runs one real-shape
inference for each dataset. The default release mode executes every checkpoint
and a real-shape backward pass for every dataset and is intentionally much
slower.

The five comparison-model packages each contain configurations for all five
datasets (25 configurations total). Relative `data.root` values are normalized
against the repository root by `portable_paths.py`, so behavior no longer
depends on the launcher's working directory.

## External experiment checkpoint audit

The local experiment archive contains 1,019 `.pt` files. All 1,019 load with
the restricted weights-only loader, contain non-empty tensor state dictionaries,
and contain no NaN or infinity.

- 914/914 checkpoints whose model dependencies are installed restore with
  `strict=True`; no strict-load failure was observed.
- This includes all 51 FAST checkpoints across the five datasets and all 10
  Section 4.5 EEGNet checkpoints. A real-shape FAST forward passed once for
  every dataset geometry.
- 105 Mamba-family checkpoints are dependency-blocked because `mamba_ssm` is
  not installed in the audit environment. They passed the safe-content checks
  but are not reported as strict-load or forward successes.
- The nine Section 4.7.2 checkpoint and original-training-configuration hashes
  match the source archive, and all nine restore strictly. The portable config
  hashes are recorded separately in `results/4.7.2/checkpoint_manifest.csv`.
  Its largest real-shape mechanism forward was stopped after sustained
  computation and is not reported as passed.

## Test evidence

- Repository-relative data-path tests: 34 passed.
- Five datasets × five baseline configuration contracts: 25 passed.
- Main dataset, LOSO, missing-path, Bimodal shape, BCI2020, and Patch-boundary
  contracts: previously executed successfully; a later repeat was blocked in
  module collection by an environment-level `mne` import performance issue.
- All Python sources compile. The vendored Deformable-DETR source emits one
  upstream invalid-escape `SyntaxWarning` but no syntax error.
- The formal Section 4.8.2 SATT/STATT weights restore and execute through the
  PyTorch reference backend. The optional CUDA operator was not compiled in
  this Windows audit environment.

## Data and dependency boundary

Raw EEG data are not distributed. A fresh clone can validate configurations,
code, provenance, and released checkpoints, but a complete training run also
requires the public datasets in the documented layout. Bimodal Inner Speech
must first be converted from the source BDF files to the expected derivative
FIF files. EEGMamba and LaBraM additionally require the pretrained files and
hashes documented in their baseline READMEs.

BCI2020 repeats 1–9 contain 135 complete LR-HORNet candidates. The 15 released
states use the disclosed per-fold maximum-test-accuracy selection rule; repeat
10 is excluded in full. This post-hoc selection is optimistic and must not be
presented as an unbiased generalization estimate.
