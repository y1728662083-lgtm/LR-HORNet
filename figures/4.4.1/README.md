# Section 4.4.1 figure package

Run both commands from this directory:

```bash
python build_figure_data.py
python plot_4_4_1_figures.py
```

Accuracy and parameter counts are loaded from `../../results/4.4.1/`. The
builder validates the complete four-dataset by six-model design before writing
the fold-level and cross-dataset plotting tables.

FLOPs use batch size 1 and `1 MAC = 2 FLOPs`. The LR-HORNet count follows each
dataset configuration's right-boundary rule: ds003626 crops its one-sample
suffix, Bimodal and Directional Words–Russian include the zero-padded final
Patch, and Directional Words–Spanish divides exactly. The corresponding
LR-HORNet counts for Bimodal and Russian are 23.547904 M and 10.156800 M FLOPs.

Each figure is exported as a 600-dpi PNG and editable SVG.
