# Figure reproduction

Each section directory under figures/ contains the source CSV files, plotting scripts, and both PNG and editable SVG outputs. The plotting code uses Times New Roman when available and falls back to Matplotlib's configured serif font if it is not installed.

Install the plotting environment and regenerate every section:

    python -m pip install -r requirements-figures.txt
    python scripts/reproduce_figures.py

The driver executes each audited section's top-level plotting scripts in its own directory and reads the checked-in CSV source tables. Existing files are overwritten only with newly generated versions of the same named figures. The generate_result_tables.py helpers are retained for users who have complete raw experiment outputs and want to rebuild the CSV tables explicitly.

Sections 4.5, 4.6.3, and 4.7.1 are skipped by the driver because their legacy plot-level CSV files conflict with the formal summaries described in RESULTS_AUDIT.md. Their files remain available for provenance inspection, but they should not be regenerated or cited as canonical results until the conflicts are resolved.

Section 4.7.2 mechanism images are stored under experiments/4.7.2/reference_results/ because they are direct analysis outputs rather than publication plotting-package outputs.
