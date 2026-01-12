import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from matplotlib import gridspec
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist


def generate_shared_heatmap(
    shared_feature_df: pd.DataFrame,
    shared_meta_df: pd.DataFrame,
    meta_col: str,
    cluster_rows: bool = False,
    cluster_columns: bool = True,
    show_row_dendrogram: bool = False,
    show_column_dendrogram: bool = True,
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
    Generate a heatmap of shared features (rows) × metadata groups (columns),
    with optional clustering and optional dendrogram display for rows/columns.
    """

    if meta_col not in shared_meta_df.columns:
        raise ValueError(f"meta_col '{meta_col}' not found in shared_meta_df.")

    if not shared_feature_df.index.equals(shared_meta_df.index):
        shared_meta_df = shared_meta_df.loc[shared_feature_df.index]

    # ---- Aggregate: Feature × Group ----
    groups = shared_meta_df[meta_col].astype(str)

    if agg == "mean":
        mat = shared_feature_df.groupby(groups).mean(numeric_only=True).T
    elif agg == "median":
        mat = shared_feature_df.groupby(groups).median(numeric_only=True).T
    else:
        raise ValueError("agg must be 'mean' or 'median'.")

    # ---- Z-score per feature ----
    if zscore_rows:
        X = mat.values.astype(float)
        mu = np.nanmean(X, axis=1, keepdims=True)
        sd = np.nanstd(X, axis=1, keepdims=True)
        sd[sd == 0] = 1.0
        mat = pd.DataFrame((X - mu) / sd, index=mat.index, columns=mat.columns)

    # ---- Clustering ----
    Z_row = Z_col = None

    if cluster_columns and mat.shape[1] > 2:
        Z_col = linkage(pdist(mat.values.T, metric=pdist_metric), method=linkage_method)
        col_order = dendrogram(Z_col, no_plot=True)["leaves"]
        mat = mat.iloc[:, col_order]

    if cluster_rows and mat.shape[0] > 2:
        Z_row = linkage(pdist(mat.values, metric=pdist_metric), method=linkage_method)
        row_order = dendrogram(Z_row, no_plot=True)["leaves"]
        mat = mat.iloc[row_order, :]

    # ---- Figure layout ----
    if figsize is None:
        figsize = (
            max(7, 0.5 * mat.shape[1] + 3),
            max(6, 0.18 * mat.shape[0] + 2),
        )

    fig = plt.figure(figsize=figsize)

    gs = gridspec.GridSpec(
        2 if show_column_dendrogram else 1,
        2 if show_row_dendrogram else 1,
        height_ratios=[1.2, 6] if show_column_dendrogram else [6],
        width_ratios=[1.2, 6] if show_row_dendrogram else [6],
        hspace=0.02,
        wspace=0.02,
    )

    # ---- Column dendrogram ----
    if show_column_dendrogram and Z_col is not None:
        ax_col_dendro = fig.add_subplot(gs[0, -1])
        dendrogram(Z_col, ax=ax_col_dendro, orientation="top", no_labels=True)
        ax_col_dendro.axis("off")

    # ---- Row dendrogram ----
    if show_row_dendrogram and Z_row is not None:
        ax_row_dendro = fig.add_subplot(gs[-1, 0])
        dendrogram(Z_row, ax=ax_row_dendro, orientation="right", no_labels=True)
        ax_row_dendro.axis("off")

    # ---- Heatmap ----
    ax_heat = fig.add_subplot(gs[-1, -1])
    im = ax_heat.imshow(mat.values, aspect="auto")

    ax_heat.set_xticks(range(mat.shape[1]))
    ax_heat.set_xticklabels(mat.columns, rotation=45, ha="right")
    ax_heat.set_yticks(range(mat.shape[0]))
    ax_heat.set_yticklabels(mat.index)
    ax_heat.set_title(f"Shared heatmap: features × {meta_col}")

    fig.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.02)
    fig.tight_layout()

    # ---- Save ----
    if save_dir and save_prefix:
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
