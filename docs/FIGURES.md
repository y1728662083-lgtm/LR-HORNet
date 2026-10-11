# Figure reproduction

Each section directory under figures/ contains the source CSV files, plotting scripts, and both PNG and editable SVG outputs. The plotting code uses Times New Roman when available and falls back to Matplotlib's configured serif font if it is not installed.

Install the plotting environment and regenerate every section:

    python -m pip install -r requirements-figures.txt
    python scripts/reproduce_figures.py

The driver executes each audited section's top-level plotting scripts in its own directory. Existing files are overwritten only with newly generated versions of the same named figures. Sections 4.5, 4.6.3, and 4.7.1 rebuild their CSV tables from checked-in source metrics and training logs before plotting; the remaining sections read their checked-in source tables.

Section 4.5 reuses the Section 4.4.1 LR-HORNet main-experiment folds for the final stage to avoid duplicating their training cost. Its table builder verifies hashes and metrics against the Section 4.4.1 fold-level table before regenerating the figure.

Section 4.7.2 mechanism images are stored under experiments/4.7.2/reference_results/ because they are direct analysis outputs rather than publication plotting-package outputs. The controlled mechanism study reports four completed ranked checkpoints only.
