# Data Visualization for Machine Learning

This project is a hands-on introduction to `matplotlib`, the core plotting library of the Python data science stack. Each script builds one chart from raw `numpy` arrays, covering the full range of visual encodings an ML practitioner reaches for day to day: line plots, scatter plots, log-scaled axes, histograms, stacked bar charts, multi-panel figures, colormap-encoded scatter plots, and dimensionality-reduction plots. Visualization is not cosmetic in an ML workflow — it is how you inspect raw data for skew and outliers during exploratory data analysis, diagnose model behavior (loss curves, residuals, decision boundaries), and communicate results to stakeholders who will never read a confusion matrix printed to stdout.

## Overview

Every chart type here maps directly onto a recurring ML task:

- **Line plots** (`0-line.py`, `3-two.py`) — tracking a quantity over an ordered axis, the same shape as a training/validation loss curve plotted over epochs.
- **Log-scaled axes** (`2-change_scale.py`) — visualizing exponential decay or growth processes (radioactive decay here, but the same transform applies to learning-rate schedules or loss on a log scale) without the curve flattening into an unreadable line.
- **Scatter plots** (`1-scatter.py`) — inspecting the relationship between two continuous variables, the first step in checking for correlation or linear separability between features.
- **Histograms** (`4-frequency.py`) — reading the distribution (center, spread, skew) of a single variable, essential before choosing a normalization strategy or spotting class imbalance.
- **Multi-subplot figures** (`5-all_in_one.py`) — assembling several diagnostic views (e.g., loss curve, prediction scatter, error histogram) into a single dashboard-style figure.
- **Stacked bar charts** (`6-bars.py`) — comparing composition across categories, useful for visualizing class distributions across groups or feature contributions.
- **Colormap-encoded scatter plots** (`100-gradient.py`) — adding a third dimension to a 2D scatter via color, the same technique used to visualize a cost surface or a spatial feature colored by target value.
- **PCA projection** (`101-pca.py`) — reducing high-dimensional data to a handful of principal components for a 3D scatter, a standard technique for visualizing class separability before/after feature engineering.

## Contents

| File | Description |
| --- | --- |
| `0-line.py` | Plots `y = x^3` for `x` in `[0, 10]` as a solid red line (`plt.plot(y, 'r-')`), with the x-axis clamped to `[0, 10]`. |
| `1-scatter.py` | Draws a magenta scatter plot of 2000 samples from a bivariate normal distribution (correlated height/weight), with axis labels and a title. |
| `2-change_scale.py` | Plots the exponential decay of Carbon-14 (`y = exp((ln(0.5)/5730) * x)`) over 28,650 years with the y-axis set to a logarithmic scale (`plt.yscale('log')`). |
| `3-two.py` | Overlays two exponential decay curves (C-14 as a red dashed line, Ra-226 as a green solid line) on shared linear axes, with a legend and fixed x/y limits. |
| `4-frequency.py` | Plots a histogram of 50 normally distributed (mean 68, std 15) student grades, binned every 10 units with black bin edges. |
| `5-all_in_one.py` | Reproduces the previous five plots inside a single figure, arranged on a 3x2 grid using `plt.subplot` and `plt.subplot2grid`, with a shared `suptitle`. |
| `6-bars.py` | Plots a stacked bar chart of fruit quantities (apples, bananas, oranges, peaches) owned by three people, using `plt.bar` with cumulative `bottom` offsets. |
| `100-gradient.py` | Scatter plot of 2000 sampled (x, y) coordinates on a synthetic mountain, colored by elevation `z` via the `viridis` colormap with an attached colorbar. |
| `101-pca.py` | Loads the Iris dataset (`pca.npz`), mean-centers it, and computes a 3-component PCA projection via SVD (`np.linalg.svd`) in preparation for a 3D scatter of the reduced features, colored by class label. |

## How It Works

**`5-all_in_one.py` — composite dashboard layout.** This script recreates the five prior charts as subplots of one `Figure`, mixing two layout APIs: `plt.subplot(3, 2, i)` for the first four panels (a regular 3x2 grid) and `plt.subplot2grid((3, 2), (2, 0), colspan=2)` for the histogram, which spans both columns of the bottom row. Per-subplot text (`xlabel`, `ylabel`, `title`) is set to `fontsize='x-small'` so six titled panels stay legible at a compact figure size, and `plt.tight_layout()` is called before `plt.suptitle("All in One")` to prevent the panels' labels from overlapping. This is the same pattern used to build a multi-metric training dashboard (loss, accuracy, gradient norm, prediction scatter) in one image.

**`100-gradient.py` — colormap-encoded scatter.** Instead of a plain 2D scatter, `plt.scatter(x, y, c=z, cmap='viridis')` maps a third numeric variable (`z`, the simulated elevation) onto point color via the `viridis` colormap, and `plt.colorbar(label='elevation (m)')` attaches a labeled scale bar so the color mapping is interpretable. This is the standard way to visualize a scalar field (cost, density, target value) over a 2D coordinate space without resorting to a full 3D plot.

**`101-pca.py` — SVD-based PCA for visualization.** Rather than calling `sklearn.decomposition.PCA`, this script derives the principal components manually with `np.linalg.svd`: it mean-centers the data (`data - data_means`), runs a full SVD on the centered matrix, and projects onto the top three right-singular vectors (`Vh[:3].T`) to get a `(n_samples, 3)` array suitable for a 3D scatter (`Axes3D` is imported for that purpose). This mirrors how PCA is implemented under the hood and is a common way to visualize whether classes are linearly separable after dimensionality reduction.

**`6-bars.py` — stacked bars via cumulative offsets.** Each fruit's bar is drawn with `plt.bar(x, values, bottom=...)`, where `bottom` is the running sum of the categories already plotted (`fruit[0]`, then `fruit[0] + fruit[1]`, etc.). This is the standard matplotlib idiom for stacked bars — there is no single "stacked" flag, so each layer's baseline must be computed explicitly.

**`2-change_scale.py` — logarithmic axis for exponential decay.** Setting `plt.yscale('log')` turns the exponential decay curve into a straight line, making the decay rate visually comparable across time ranges that would otherwise be dominated by the steep early drop-off — the same reason loss curves or histogram counts are often plotted log-scale.

## Requirements

- Python 3.12
- numpy 2.4.5
- matplotlib 3.10.9 (including `mpl_toolkits.mplot3d.Axes3D` for 3D scatter support)

All scripts rely solely on `numpy` and `matplotlib`; none of them import `scikit-learn`. `101-pca.py` expects a `pca.npz` archive (with `data` and `labels` arrays, e.g. the Iris dataset) in the working directory to run end to end.

## Usage

Each deliverable defines a single function with no arguments that builds and displays one figure via `plt.show()`. Run it through its companion `-main.py` driver:

```bash
./0-main.py    # red cubic line, x in [0, 10]
./1-main.py    # magenta scatter of correlated height vs. weight
./2-main.py    # C-14 decay curve on a log-scaled y-axis
./3-main.py    # C-14 (dashed red) vs. Ra-226 (solid green) decay, with legend
./4-main.py    # histogram of 50 student grades, bins of width 10
./5-main.py    # all five charts above combined into one 3x2 dashboard
./6-main.py    # stacked bar chart of fruit per person
./100-main.py  # elevation scatter colored by the viridis colormap
```

For example, `./1-main.py` calls `scatter()` from `1-scatter.py`, which draws 2000 points sampled from a correlated bivariate normal distribution and opens a window showing "Men's Height vs Weight" with labeled axes.

## Design Notes

- Every figure explicitly sets `figsize=(6.4, 4.8)` (matplotlib's default) before plotting, keeping output size consistent across scripts and predictable when embedded elsewhere.
- Axis limits and tick positions are set explicitly (`xlim`, `ylim`, `xticks`, `yticks`) rather than left to matplotlib's auto-scaling, so charts like the stacked bar graph and histogram render with round, human-readable bounds (e.g., grades from 0-100 in steps of 10).
- `2-change_scale.py` uses a logarithmic y-axis specifically because the underlying process is exponential decay; a linear axis would compress the entire multi-thousand-year tail into a near-flat line near zero.
- `5-all_in_one.py` deliberately mixes `plt.subplot` and `plt.subplot2grid` in the same figure to combine a uniform grid with one spanning panel, rather than forcing every chart into equal-sized cells.

## Author

Fjolla Qerimi
