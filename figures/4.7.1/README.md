# Section 4.7.1 figure package

`generate_result_tables.py` reads the 80 checked-in source files under
`../../experiments/4.7.1/reference_results/`: four offset-generation methods,
ten LOSO folds per method, and a test-metrics/training-log pair for each fold.
It validates metric ranges, recomputes accuracy from each confusion matrix,
and records SHA-256 hashes in the run-level table.

Regenerate the tables and figures from this directory:

```bash
python generate_result_tables.py
python plot_offset_strategy_comparison.py
python plot_offset_strategy_validation_comparison.py
```

The LSTM source contains ten distinct fold records, including a genuine fold 9.
