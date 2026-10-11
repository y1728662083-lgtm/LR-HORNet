# Section 4.7.1 results

`run_level_results.csv` contains four offset-generation strategies across all
ten LOSO folds. `model_summary.csv` is regenerated from those 40 rows by
`../../figures/4.7.1/generate_result_tables.py`.

The source `test_metrics.json` and `training_log.csv` files are stored under
`../../experiments/4.7.1/reference_results/`. The LSTM condition includes a
genuine fold 9 whose source hashes differ from fold 8.
