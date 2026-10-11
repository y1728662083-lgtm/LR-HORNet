# Dataset and split protocol

## Common evaluation rule

The main experiments use calibrated leave-one-subject-out evaluation. For fold i, subject i is the test subject and the next subject in cyclic order is the validation subject. A class-stratified 50% calibration subset from each held-out subject is included in training; the remaining halves are retained for validation and final testing. Calibration and evaluation partitions use seeds 42 and 43, respectively.

This is not identical to strict subject-independent LOSO. Any comparison must state whether target-subject calibration data are available.

## Main dataset configurations

| Config | Sampling rate | Patch | Stride | Folds | Archived comparison repeats |
|---|---:|---:|---:|---:|---:|
| ds003626.yaml | 256 Hz | 128 samples (0.5 s) | 64 samples (0.25 s) | 10 | 1 per fold |
| bimodal_inner_speech.yaml | 512 Hz | 256 samples (0.5 s) | 128 samples (0.25 s) | 4 | 5 per fold |
| directional_words_russian.yaml | 250 Hz | 125 samples (0.5 s) | 63 samples (approximately 0.25 s) | 12 | 5 per fold |
| directional_words_spanish.yaml | 500 Hz | 250 samples (0.5 s) | 125 samples (0.25 s) | 10 | 5 per fold |

The checked-in main YAML files execute one run for each listed fold. Repeated-run orchestration was performed outside main.py; repeated archives used the same configured seed (42), so their dispersion reflects residual training nondeterminism rather than independent seed sampling.

## Acquisition-organization categories

- **Randomized independent trials:** ds003626 and Bimodal Inner Speech. Different classes are interleaved along the recording timeline.
- **Randomized class order with consecutive same-class trials:** BCI Competition 2020 Track 3. This dataset is discussed by the paper but is not part of the four completed main-result tables in this repository.
- **Fixed repeated class blocks:** Directional Words–Russian and Directional Words–Spanish. Class, block position, relative acquisition time, fatigue, and slow drift may be coupled.

## Data distribution

No raw EEG is included. Users are responsible for obtaining each dataset under its original terms. After download, edit only data.root if the suggested data/ layout is not used. Do not commit local paths, participant data, generated derivatives, or checkpoints.
