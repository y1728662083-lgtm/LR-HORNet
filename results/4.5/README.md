# Section 4.5 result provenance

The first four rows of the progressive ablation are derived from their own
ten-fold archived outputs. To avoid duplicating the computational cost of the
complete ten-fold experiment, the final LR-HORNet row reuses the ds003626 main
result reported in Section 4.4.1 (mean accuracy 30.90%).

`figures/4.5/generate_result_tables.py` reconstructs all five rows from the
checked-in source metrics and training logs. It additionally checks every
LR-HORNet fold's metric values and source-file hashes against
`results/4.4.1/fold_level_results_with_val_acc.csv` before exporting the table.
