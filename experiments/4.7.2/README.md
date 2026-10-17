# Section 4.7.2 protocol

The released controlled mechanism study contains four ranked checkpoints. Each
rank has both causal-ablation and artificial-shift outputs and is included in
the descriptive summaries. No fifth rank is part of the reported experiment.

`run_offset_mechanism_rank01_04.sh` runs ranks 1–4, and
`summarize_4_7_2.sh` rebuilds the corresponding completion and mechanism tables.

`results/4.7.2/checkpoint_manifest.csv` records two configuration hashes. The
`training_config_sha256` value identifies the immutable YAML archived with the
completed training run. The checked-in YAML differs only in `data.root` and
`training.output_root`; its directly verifiable hash is recorded separately as
`portable_config_sha256`. Model and optimization fields are unchanged, and all
nine referenced state dictionaries restore with `strict=True` under the
portable configurations.
