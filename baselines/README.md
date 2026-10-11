# Baseline reproduction archives

This directory contains the finalized code-only reproduction archives used for
the five comparison models in Section 4.4.1:

- `eegnet/` and `atcnet/`: pinned Braindecode implementations and experiment
  configurations;
- `eegmamba/`: pinned author source, dataset adapters, exact configurations,
  and the public pretrained-checkpoint URL, revision, and SHA-256;
- `labram/`: pinned author source, dataset adapters, exact configurations, and
  the public checkpoint mirror revision and SHA-256;
- `fast/`: pinned author source, the native input adapter, and exact
  cross-dataset configurations.

Each archive retains its own README and source manifest. Large third-party
pretrained weights are not duplicated here; their immutable download metadata
and expected hashes are recorded inside the EEGMamba and LaBraM archives.
LR-HORNet itself is implemented by the repository-root pipeline.
