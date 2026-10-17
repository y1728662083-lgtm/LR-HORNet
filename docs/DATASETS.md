# Dataset and split protocol

## Common evaluation rule

The main experiments use calibrated leave-one-subject-out evaluation. For fold i, subject i is the test subject and the next subject in cyclic order is the validation subject. A class-stratified 50% calibration subset from each held-out subject is included in training; the remaining halves are retained for validation and final testing. Calibration and evaluation partitions use seeds 42 and 43, respectively.

This is not identical to strict subject-independent LOSO. Any comparison must state whether target-subject calibration data are available.

## Main dataset configurations

| Config | Input length | Sampling rate | Patch | Stride | LR right boundary | Folds | Archived comparison repeats |
|---|---:|---:|---:|---:|---|---:|---:|
| ds003626.yaml | 1153 | 256 Hz | 128 samples (0.5 s) | 64 samples (0.25 s) | Crop 1 sample | 10 | 1 per fold |
| bimodal_inner_speech.yaml | 359 | 512 Hz | 256 samples (0.5 s) | 128 samples (0.25 s) | Zero-pad 25 samples | 4 | 5 per fold |
| directional_words_russian.yaml | 250 | 250 Hz | 125 samples (0.5 s) | 63 samples (approximately 0.25 s) | Zero-pad 1 sample | 12 | 5 per fold |
| directional_words_spanish.yaml | 750 | 500 Hz | 250 samples (0.5 s) | 125 samples (0.25 s) | Exact; no crop or padding | 10 | 5 per fold |
| bci2020.yaml | 795 | 256 Hz | 128 samples (0.5 s) | 64 samples (0.25 s) | Zero-pad 37 samples | 15 | Baselines: 1 per fold; LR-HORNet: 9 per fold |

LR uses only complete patches when `pad_incomplete_patch: false`, so an
uncovered suffix is cropped. With `pad_incomplete_patch: true`, the patch count
is rounded upward and only the incomplete final patch is right-padded with
zeros. This setting reproduces the dataset-specific boundary handling used by
the archived main experiments.

The checked-in main YAML files execute one run for each listed fold. Repeated-run orchestration was performed outside main.py; repeated archives used the same configured seed (42), so their dispersion reflects residual training nondeterminism rather than independent seed sampling.

BCI2020 has five classes and 64 channels. With a 795-sample input, a 128-sample Patch, and a 64-sample stride, the incomplete final Patch is right-padded by 37 samples to an effective covered length of 832 samples. Its calibrated LOSO protocol contains 15 held-out-subject folds. Each of the five comparison baselines has one complete result per fold. LR-HORNet has 15 folds × 9 complete repeats = 135 candidates; the released artifact selects the maximum test accuracy within each fold, with the lowest repeat number breaking ties. The incomplete tenth repeat is excluded in full. Because this is test-set selection, the resulting 15-fold summary is explicitly post-hoc and optimistic.

## Acquisition-organization categories

- **Randomized independent trials:** ds003626 and Bimodal Inner Speech. Different classes are interleaved along the recording timeline.
- **Randomized class order with consecutive same-class trials:** BCI Competition 2020 Track 3 (BCI2020), included as the fifth cross-dataset evaluation through lightweight result evidence.
- **Fixed repeated class blocks:** Directional Words–Russian and Directional Words–Spanish. Class, block position, relative acquisition time, fatigue, and slow drift may be coupled.

## Data distribution

No raw EEG is included. Users are responsible for obtaining each dataset under its original terms. For the original four datasets, edit only `data.root` if the suggested `data/` layout is not used. For BCI2020, preserve the hash-locked archived YAML and pass the local root through `BCI2020_ROOT` or `--dataset-root`; the launcher writes an ignored runtime configuration. Do not commit local paths, participant data, generated derivatives, or unselected training artifacts. The 51 checkpoints under `checkpoints/main/` cover all five datasets: 36 from the original comparison and 15 selected BCI2020 checkpoints. The remaining 208 local BCI2020 checkpoints stay outside Git; their complete tree can be audited with the external checkpoint smoke test.
