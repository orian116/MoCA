import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


def generate_shared_heatmap(
    shared_feature_df: pd.DataFrame,
    shared_meta_df: pd.DataFrame,
    meta_col: str,
    cluster_rows: bool = True,
    cluster_columns: bool = True,
    agg: str = "mean",
    zscore_rows: bool = True,
    pdist_metric: str = "correlation",
    linkage_method: str = "average",
    figsize=None,
    save_dir: str | None = None,
    save_prefix: str | None = None,
    dpi: int = 300,
):
    """
    Heatmap with metadata groups on x-axis and features on y-axis.

    Parameters
    ----------
    shared_feature_df : pd.DataFrame
        Rows = observations (cells), columns = shared features.
    shared_meta_df : pd.DataFrame
        Rows = observations, contains metadata columns.
    meta_col : str
        Column in shared_meta_df defining x-axis groups.
    cluster_rows : bool
        Hierarchically cluster/reorder features (rows).
    cluster_columns : bool
        Hierarchically cluster/reorder metadata groups (columns).
    agg : {"mean","median"}
        Aggregation within metadata groups.
    zscore_rows : bool
        Z-score each feature across metadata groups.
    pdist_metric : str
        Distance metric for pdist (default: "correlation").
    linkage_method : str
        Linkage method for hierarchical clustering (default: "average").
    figsize : tuple or None
        Figure size. If None, inferred automatically.
    save_dir : str or None
        Directory to save plots. If None, plots are not saved.
    save_prefix : str or None
        File prefix for saved plots (without extension).
    dpi : int
        DPI for TIFF output.

    Returns
    -------
    pd.DataFrame
        Feature × metadata-group matrix used for plotting.
    """
    if meta_col not in shared_meta_df.columns:
        raise ValueError(f"meta_col '{meta_col}' not found in shared_meta_df.")

    if shared_feature_df.shape[0] != shared_meta_df.shape[0]:
        raise ValueError("Feature and metadata dataframes must have the same number of rows.")

    if not shared_feature_df.index.equals(shared_meta_df.index):
        shared_meta_df = shared_meta_df.copy()
        shared_meta_df.index = shared_feature_df.index

    # Aggregate: Feature × Group
    groups = shared_meta_df[meta_col].astype(str)

    if agg == "mean":
        mat = shared_feature_df.groupby(groups).mean(numeric_only=True).T
    elif agg == "median":
        mat = shared_feature_df.groupby(groups).median(numeric_only=True).T
    else:
        raise ValueError("agg must be 'mean' or 'median'.")

    # Z-score per feature (row)
    if zscore_rows:
        X = mat.values.astype(float)
        mu = np.nanmean(X, axis=1, keepdims=True)
        sd = np.nanstd(X, axis=1, keepdims=True)
        sd[sd == 0] = 1.0
        mat = pd.DataFrame((X - mu) / sd, index=mat.index, columns=mat.columns)

    # Clustering helper
    def _cluster_order(X: np.ndarray):
        try:
            from scipy.cluster.hierarchy import linkage, leaves_list
            from scipy.spatial.distance import pdist
            d = pdist(X, metric=pdist_metric)
            Z = linkage(d, method=linkage_method)
            return leaves_list(Z)
        except Exception:
            return np.arange(X.shape[0])

    if cluster_rows and mat.shape[0] > 2:
        mat = mat.iloc[_cluster_order(mat.values), :]

    if cluster_columns and mat.shape[1] > 2:
        mat = mat.iloc[:, _cluster_order(mat.values.T)]

    # Plot
    if figsize is None:
        figsize = (
            max(6, 0.45 * mat.shape[1] + 2),
            max(6, 0.18 * mat.shape[0] + 2),
        )

    fig = plt.figure(figsize=figsize)
    ax = fig.gca()
    im = ax.imshow(mat.values, aspect="auto")
    ax.set_xticks(range(mat.shape[1]))
    ax.set_xticklabels(mat.columns, rotation=45, ha="right")
    ax.set_yticks(range(mat.shape[0]))
    ax.set_yticklabels(mat.index)
    ax.set_title(f"Shared heatmap: features × {meta_col}")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()

    # Save outputs
    if save_dir is not None and save_prefix is not None:
        os.makedirs(save_dir, exist_ok=True)
        fig.savefig(
            os.path.join(save_dir, f"{save_prefix}.svg"),
            bbox_inches="tight"
        )
        fig.savefig(
            os.path.join(save_dir, f"{save_prefix}.tiff"),
            dpi=dpi,
            format="tiff",
            bbox_inches="tight"
        )

    plt.show()
    return mat
