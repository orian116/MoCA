# MoCA (Morphological Condition Alignment)

MoCA (Morphological Condition Alignment) is a lightweight framework for comparing new experimental conditions to gold-standard morphological states using shared feature spaces derived from CellProfiler or similar pipelines.
Tools for:
- condition→state alignment (PCA + centroid distances + soft scores)
- OLD vs NEW feature histogram overlays
- UMAP alignment visualization (fit on OLD, transform NEW)

MoCA answers the question:

“Which known morphological states does each new experimental condition most closely resemble?”

# Install

```bash
pip install git+https://github.com/orian116/MoCA.git
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
    save_dir="figs",
    save_prefix="moca_alignment"
)

distance_matrix = res["distance_matrix"]
```
- Fits StandardScaler + PCA on OLD data only
- Computes gold-state centroids in PCA space
- Measures (weighted) Euclidean distance from NEW cells to centroids
- Aggregates distances to Condition × State matrix

### Interepretation
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
    save_dir="figs",
    save_prefix="moca_umap"
)
```

## 4. Generate shared feature heatmaps
```python
from moca import generate_shared_heatmap

generate_shared_heatmap(
    shared_feature_df=shared_X,
    shared_meta_df=shared_meta,
    meta_col="Condition",
    cluster_rows=True,
    cluster_columns=True,
    save_dir="figs",
    save_prefix="moca_shared_features"
)
```

- Compare feature programs across conditions
- Identify morphology drivers of alignment
- Produce interpretable feature-level figures







