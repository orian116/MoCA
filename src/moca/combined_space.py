
import numpy as np
import pandas as pd

from typing import Optional, List
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

try:
    import umap as umap_module
except Exception:
    umap_module = None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _merge_metadata_dfs(
    old_meta: pd.DataFrame,
    new_meta: pd.DataFrame,
    old_id: str,
    new_id: str,
) -> pd.DataFrame:
    """
    Row-bind old_meta and new_meta with proper handling of non-shared columns.

    - Columns present in both datasets are concatenated directly.
    - Columns present only in one dataset are added to the other with:
        * NaN  for numeric columns
        * the dataset id string for non-numeric columns
    """
    old_cols = set(old_meta.columns)
    new_cols = set(new_meta.columns)
    only_old = old_cols - new_cols
    only_new = new_cols - old_cols

    old_ext = old_meta.copy()
    new_ext = new_meta.copy()

    # Columns that exist only in new_meta -> add to old_ext
    for col in only_new:
        if pd.api.types.is_numeric_dtype(new_meta[col]):
            old_ext[col] = np.nan
        else:
            old_ext[col] = old_id

    # Columns that exist only in old_meta -> add to new_ext
    for col in only_old:
        if pd.api.types.is_numeric_dtype(old_meta[col]):
            new_ext[col] = np.nan
        else:
            new_ext[col] = new_id

    return pd.concat([old_ext, new_ext], axis=0, ignore_index=True)


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------

def create_combined_space(
    old_X: pd.DataFrame,
    new_X: pd.DataFrame,
    old_id: str,
    new_id: str,
    old_metadata: Optional[pd.DataFrame] = None,
    new_metadata: Optional[pd.DataFrame] = None,
    batch_correction: bool = False,
    batch_key: str = "batch",
    log: bool = False,
    pcs_by_variance: bool = False,
    variance: float = 0.95,
    umap_n_neighbors: int = 30,
    umap_min_dist: float = 0.3,
    umap_metric: str = "euclidean",
    random_state: int = 0,
) -> pd.DataFrame:
    """
    Combine old_X and new_X into a shared embedding space (PCA + UMAP),
    optionally applying ComBat batch correction.

    Parameters
    ----------
    old_X : DataFrame
        Rows = cells, columns = morphological / numeric features (old dataset).
    new_X : DataFrame
        Rows = cells, columns = morphological / numeric features (new dataset).
    old_id : str
        Label prefix for the old dataset (used in ``batch`` and ``cell_id`` columns).
    new_id : str
        Label prefix for the new dataset.
    old_metadata : DataFrame or None
        Extra per-cell metadata for old_X (same row order as old_X).
    new_metadata : DataFrame or None
        Extra per-cell metadata for new_X (same row order as new_X).
    batch_correction : bool, default False
        If True, apply ComBat (pycombat) batch correction before PCA.
        The morphological values are log1p-transformed, passed to pycombat,
        and the corrected matrix is fed directly into PCA (no StandardScaler).
        Requires the ``combat`` package (``pip install combat``).
    batch_key : str, default "batch"
        Column in the combined DataFrame to use as batch labels for pycombat.
        Defaults to the ``batch`` column created automatically from
        ``old_id`` / ``new_id``.  Any other column present in the combined
        DataFrame at correction time can be used (e.g. a plate or run ID
        added via ``old_metadata`` / ``new_metadata``).
        Only used when ``batch_correction=True``.
    log : bool, default False
        If True, apply log1p to the morphological features before scaling.
        Only used when ``batch_correction=False`` (when batch correction is
        enabled, log1p is applied internally as part of the pycombat workflow).
    pcs_by_variance : bool, default False
        If True, the number of PCA components is chosen automatically to
        explain ``variance`` of the total variance.  All retained components
        are passed to UMAP; only PC1 and PC2 are stored in the output.
        If False, exactly 2 components are used.
    variance : float, default 0.95
        Fraction of variance to retain when ``pcs_by_variance=True``.
        Ignored when ``pcs_by_variance=False``.
    umap_n_neighbors : int, default 30
    umap_min_dist : float, default 0.3
    umap_metric : str, default "euclidean"
    random_state : int, default 0

    Returns
    -------
    pd.DataFrame
        ``results_df`` — original morphological features plus all metadata
        columns (``batch``, ``cell_id``, ``PC1``, ``PC2``, ``UMAP1``,
        ``UMAP2``, and any columns from old_metadata / new_metadata).
        Row index is set to ``cell_id``.
    """
    if umap_module is None:
        raise ImportError("umap-learn is required. Install with: pip install umap-learn")

    # ------------------------------------------------------------------
    # Step 1: Align to common morphological features
    # ------------------------------------------------------------------
    common_morph_features: List[str] = [c for c in old_X.columns if c in new_X.columns]
    if not common_morph_features:
        raise ValueError("No shared features found between old_X and new_X.")

    n_old = len(old_X)
    n_new = len(new_X)

    old_morph = old_X[common_morph_features].reset_index(drop=True).copy()
    new_morph = new_X[common_morph_features].reset_index(drop=True).copy()

    # ------------------------------------------------------------------
    # Step 2: Build batch and cell_id metadata columns
    # ------------------------------------------------------------------
    old_cell_ids = [f"{old_id}_{i}" for i in range(n_old)]
    new_cell_ids = [f"{new_id}_{i}" for i in range(n_new)]

    old_morph["batch"] = old_id
    old_morph["cell_id"] = old_cell_ids

    new_morph["batch"] = new_id
    new_morph["cell_id"] = new_cell_ids

    df_combined = pd.concat([old_morph, new_morph], axis=0, ignore_index=True)
    df_combined.index = df_combined["cell_id"].values

    # ------------------------------------------------------------------
    # Step 2b: Merge additional metadata now so all columns are available
    # for batch_key lookup and are present in the final results_df
    # ------------------------------------------------------------------
    if old_metadata is not None or new_metadata is not None:
        if old_metadata is None:
            old_metadata = pd.DataFrame(index=range(n_old))
        if new_metadata is None:
            new_metadata = pd.DataFrame(index=range(n_new))

        old_meta_reset = old_metadata.reset_index(drop=True)
        new_meta_reset = new_metadata.reset_index(drop=True)

        if len(old_meta_reset) != n_old:
            raise ValueError(
                f"old_metadata has {len(old_meta_reset)} rows but old_X has {n_old} rows."
            )
        if len(new_meta_reset) != n_new:
            raise ValueError(
                f"new_metadata has {len(new_meta_reset)} rows but new_X has {n_new} rows."
            )

        merged_meta = _merge_metadata_dfs(old_meta_reset, new_meta_reset, old_id, new_id)
        merged_meta.index = df_combined.index

        extra_cols = [c for c in merged_meta.columns if c not in df_combined.columns]
        if extra_cols:
            df_combined = pd.concat([df_combined, merged_meta[extra_cols]], axis=1)

    # ------------------------------------------------------------------
    # Step 3: Prepare data for PCA
    #
    # Batch-correction path:
    #   log1p(raw) -> pycombat -> data_corrected (fed to PCA)
    #
    # Standard path:
    #   optional log1p -> StandardScaler -> data_corrected (fed to PCA)
    # ------------------------------------------------------------------
    if batch_correction:
        try:
            from combat.pycombat import pycombat
        except ImportError as exc:
            raise ImportError(
                "batch_correction=True requires the 'combat' package. "
                "Install with: pip install combat"
            ) from exc

        if batch_key not in df_combined.columns:
            raise ValueError(
                f"batch_key '{batch_key}' not found in the combined DataFrame. "
                f"Available columns: {list(df_combined.columns)}"
            )

        # ComBat expects rows = features, columns = samples
        data_matrix = pd.DataFrame(
            np.log1p(df_combined[common_morph_features]).T,
            index=common_morph_features,
        )
        batch_labels = df_combined[batch_key].values

        data_corrected = pycombat(data_matrix, batch_labels)
        # Transpose back: rows = samples, columns = features
        data_corrected = data_corrected.T
    else:
        morph_values = df_combined[common_morph_features].values.astype(float)
        if log:
            morph_values = np.log1p(morph_values)
        scaler = StandardScaler()
        data_corrected = scaler.fit_transform(morph_values)

    # ------------------------------------------------------------------
    # Step 6: PCA
    # ------------------------------------------------------------------
    n_components = variance if pcs_by_variance else 2
    pca_corrected = PCA(n_components=n_components).fit_transform(data_corrected)

    # ------------------------------------------------------------------
    # Step 7: Add PC1, PC2 to metadata
    # ------------------------------------------------------------------
    df_combined["PC1"] = pca_corrected[:, 0]
    df_combined["PC2"] = pca_corrected[:, 1]

    # ------------------------------------------------------------------
    # Step 8: UMAP on PCA-reduced data
    # ------------------------------------------------------------------
    reducer = umap_module.UMAP(
        n_neighbors=umap_n_neighbors,
        min_dist=umap_min_dist,
        metric=umap_metric,
        random_state=random_state,
    )
    umap_coords = reducer.fit_transform(pca_corrected)

    # ------------------------------------------------------------------
    # Step 9: Add UMAP1, UMAP2 to metadata
    # ------------------------------------------------------------------
    df_combined["UMAP1"] = umap_coords[:, 0]
    df_combined["UMAP2"] = umap_coords[:, 1]

    return df_combined
