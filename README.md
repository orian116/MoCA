# MoCA (Morphological Condition Alignment)

MoCA (Morphological Condition Alignment) is a lightweight framework for comparing new experimental conditions to gold-standard morphological states using shared feature spaces derived from CellProfiler or similar pipelines.
Tools for:
- condition→state alignment (PCA + centroid distances + soft scores)
- OLD vs NEW feature histogram overlays and range clipping
- UMAP alignment visualization (fit on OLD, transform NEW)
- Combined embedding space construction (joint PCA + UMAP with optional ComBat batch correction)

MoCA answers the question:

“Which known morphological states does each new experimental condition most closely resemble?”

# Install

```bash
pip install git+https://github.com/orian116/MoCA.git
```
# Re-install latest version
```bash
pip install --force-reinstall --no-deps git+https://github.com/orian116/MoCA.git
```

For ComBat batch correction support install with the optional extra:
```bash
pip install “moca[batch] @ git+https://github.com/orian116/MoCA.git”
```

# Recommended workflow

## 1. Inspect Feature Compatibility

```python
from moca import plot_old_new_hist_overlays
plot_old_new_hist_overlays(old_X, new_X, bins=60)
```
- Detects batch shifts
- Identifies outlier features
- Prevents misleading distance metrics

## 1b. Clip new_X to old_X range (optional)

If the histogram overlays reveal that new_X has values outside the range of
old_X for one or more features, remove those out-of-range rows before
continuing:

```python
from moca import clip_new_overlays

new_X_clipped = clip_new_overlays(old_X, new_X)
# optionally restrict to a specific subset of features
new_X_clipped = clip_new_overlays(old_X, new_X, features=[“Area”, “Eccentricity”])
```
- Removes rows in new_X whose value for any shared feature falls outside
  `[old_min, old_max]`
- Reports how many rows were removed
- Returns a filtered copy of new_X; old_X is never modified

## 2. Align conditions to gold states

```python
from moca import align_conditions_to_states
res = align_conditions_to_states(
    old_X=old_X,
    old_states=old_states,
    new_X=new_X,
    new_conditions=new_conditions,
    n_components=0.95,
    weighted=True,
    plot=True,
    save_dir=”figs”,
    save_prefix=”moca_alignment”
)

distance_matrix = res[“distance_matrix”]
```
- Fits StandardScaler + PCA on OLD data only
- Computes gold-state centroids in PCA space
- Measures (weighted) Euclidean distance from NEW cells to centroids
- Aggregates distances to Condition × State matrix

### Interpretation
- Lower distance = closer morphological similarity
- weighted=True emphasizes PCs that explain more variance in the gold atlas
- weighted=False treats all PCs equally

## 3. Visualize Embedding Space
```python
from moca import plot_umap_alignment

plot_umap_alignment(
    old_X, old_states,
    new_X, new_conditions,
    n_components_pca=0.95,
    save_dir=”figs”,
    save_prefix=”moca_umap”
)
```

## 3b. Build a combined embedding space

`create_combined_space` pools old and new cells into a single PCA + UMAP
embedding.  Unlike `plot_umap_alignment` (which fits on OLD only), this
function fits the embedding jointly and is suitable for exploratory
visualisation, quality control, and downstream single-cell-style analyses.

### Basic usage

```python
from moca import create_combined_space

results_df = create_combined_space(
    old_X=old_X,
    new_X=new_X,
    old_id=”control”,
    new_id=”treatment”,
)
```

`results_df` is a DataFrame whose index is `cell_id` (e.g. `”control_0”`,
`”treatment_42”`).  It contains:

| Column group | Columns |
|---|---|
| Original morphological features | all shared columns from old_X / new_X |
| Dataset identity | `batch`, `cell_id` |
| Dimensionality reduction | `PC1`, `PC2`, `UMAP1`, `UMAP2` |
| Extra metadata | any columns from `old_metadata` / `new_metadata` |

### With ComBat batch correction

```python
results_df = create_combined_space(
    old_X=old_X,
    new_X=new_X,
    old_id=”batch_A”,
    new_id=”batch_B”,
    batch_correction=True,   # requires: pip install combat
    log=True,                # log1p-transform before scaling
)
```

### With additional per-cell metadata

```python
results_df = create_combined_space(
    old_X=old_X,
    new_X=new_X,
    old_id=”ctrl”,
    new_id=”treated”,
    old_metadata=old_meta_df,   # same row order as old_X
    new_metadata=new_meta_df,   # same row order as new_X
)
```

Columns shared between `old_metadata` and `new_metadata` are concatenated
directly.  Columns present in only one dataset are included in the output:
numeric columns are filled with `NaN` for the absent dataset, non-numeric
columns are filled with the corresponding `old_id` / `new_id` string.

### Downstream plotting example

```python
import matplotlib.pyplot as plt

for batch, grp in results_df.groupby(“batch”):
    plt.scatter(grp[“UMAP1”], grp[“UMAP2”], s=4, alpha=0.5, label=batch)

plt.xlabel(“UMAP1”); plt.ylabel(“UMAP2”)
plt.legend(); plt.tight_layout(); plt.show()
```

## 4. Generate shared feature heatmaps
```python
from moca import generate_shared_heatmap

generate_shared_heatmap(
    shared_feature_df=shared_X,
    shared_meta_df=shared_meta,
    meta_col=”Condition”,
    cluster_rows=True,
    cluster_columns=True,
    save_dir=”figs”,
    save_prefix=”moca_shared_features”
)
```

- Compare feature programs across conditions
- Identify morphology drivers of alignment
- Produce interpretable feature-level figures







