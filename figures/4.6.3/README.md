# Section 4.6.3 figure package

`generate_result_tables.py` reads the 25 checked-in run records under
`../../experiments/4.6.3/reference_results/`: five electrode-group conditions
with five repeated fold-1 runs per condition. It validates every confusion
matrix and records SHA-256 hashes for the source metrics and training logs.

Regenerate the tables and figure from this directory:

```bash
python generate_result_tables.py
python plot_group_ablation_lollipop.py
```

The reported means and sample standard deviations are therefore derived from
all five completed runs for group counts 1, 2, 4, 8, and 16.
