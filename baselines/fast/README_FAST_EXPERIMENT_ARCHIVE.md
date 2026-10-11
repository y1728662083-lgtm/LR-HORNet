# FAST multi-dataset experiment archive

This directory records the exact FAST training code and configuration used for
Paper4 section 4.4.1. Experimental outputs can be placed alongside this code
after training.

## Experiment scope

Only FAST is scheduled by this archive. The Paper4 model
`group4_patchLstm_deformableLstm` is not referenced by any of the four YAML
files or by the dispatcher.

The dispatcher creates 36 FAST fits:

| Dataset | Effective input rate | FAST window / stride | Classes | Folds |
|---|---:|---:|---:|---:|
| ds003626 | 256 Hz | 256 / 128 samples | 4 | 10 |
| Russian directional words | 250 Hz | 250 / 125 samples | 6 | 12 |
| Spanish directional words | 500 Hz | 500 / 250 samples | 6 | 10 |
| Bimodal inner speech | 512 Hz | 256 / 128 samples | 8 | 4 |

All four configurations use `dim_token: 32` and the calibrated LOSO split.
Russian and Spanish are independent datasets. Their channel intersections are
computed separately by `src/eeg_pipeline/data/directional_words.py`.

## Important files

- `addition/configs/fast_native_*.yaml`: exact four-dataset configurations.
- `addition/official_sources/FAST/`: author-released FAST source, license,
  citation and README.
- `addition/fast_native_adapter.py`: dataset-grid adapter around official FAST.
- `addition/run_fast_native_all_datasets.py`: one-config/fold runner.
- `addition/dispatch_fast_native.py`: FAST-only dispatcher.
- `addition/training.py`: optimizer, schedule, validation and test evaluation.
- `src/eeg_pipeline/data/`: data readers, preprocessing and calibrated LOSO.
- `addition/official_preprocessing/bimodal_official.zip`: archived official
  Bimodal preprocessing code used by the preprocessing bridge.
- `packages_applied/`: the three server replacement packages, retained only
  for provenance. The unpacked files in this archive are the final effective
  versions; do not reapply the older packages over them.

`addition/source_manifest.json` is intentionally preserved byte-for-byte
because it participates in each run fingerprint. It also mentions EEGMamba
and LaBraM from the broader baseline bundle, but those models are not enabled
or executed by this FAST archive.

## Verification and execution

From the Paper4 repository root, these files retain their original relative
paths. Verify inputs first:

```bash
bash check_fast_inputs_only.sh
```

Run all 36 FAST fits sequentially on GPU 0:

```bash
bash run_fast_native_single_gpu.sh 0
```

The actual output locations are controlled by the `training.output_root`
field in each YAML file. Copy the resulting output directories into this
archive when the experiment finishes.

## Snapshot notes

The final checker normalizes equivalent numeric subject identifiers (`1`,
`01`, and `sub-01`) and correctly distinguishes subject count from
subject-session record count. Thus ds003626 is expected to report 10 subjects
and 30 records (10 subjects x 3 sessions).
