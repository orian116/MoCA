
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from typing import Optional, List, Tuple, Union
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier

try:
    import umap as umap_module
except Exception:
    umap_module = None

from .combined_space import _merge_metadata_dfs


_DEFAULT_COLORS = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52",
    "#8172B3", "#937860", "#DA8BC3", "#8C8C8C",
    "#CCB974", "#64B5CD",
]


# ---------------------------------------------------------------------------
# Internal KDE helper
# ---------------------------------------------------------------------------

def _plot_kde_on_ax(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    color_by: str,
    ax,
    colors: Optional[List] = None,
    hue_order: Optional[List] = None,
    alpha: float = 0.7,
    fill: bool = True,
    thresh: float = 0.05,
    n_contours: int = 5,
    linewidth: float = 1.5,
    show_legend: bool = True,
):
    groups = hue_order if hue_order is not None else list(df[color_by].dropna().unique())
    palette = (
        colors if colors is not None
        else [_DEFAULT_COLORS[i % len(_DEFAULT_COLORS)] for i in range(len(groups))]
    )

    sns.kdeplot(
        data=df,
        x=x_col,
        y=y_col,
        hue=color_by,
        hue_order=groups,
        palette=palette,
        fill=fill,
        alpha=alpha,
        thresh=thresh,
        levels=n_contours + 1,
        linewidth=linewidth,
        common_norm=False,
        ax=ax,
    )

    legend = ax.get_legend()
    if legend:
        if show_legend:
            legend.set_title(color_by, prop={"weight": "bold", "size": 9})
            legend.get_frame().set_linewidth(0)
        else:
            legend.remove()


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------

def plot_and_save_projections(
    old_X: pd.DataFrame,
    new_X: pd.DataFrame,
    old_id: str,
    new_id: str,
    new_column_to_project: str,
    new_metadata: pd.DataFrame,
    old_metadata: Optional[pd.DataFrame] = None,
    old_column_for_knn: Optional[str] = None,
    override_old_coordinates: bool = False,
    old_coordinates: Optional[List] = None,
    kde: bool = False,
    n_components_pca: Union[int, float] = 0.95,
    umap_n_neighbors: int = 30,
    umap_min_dist: float = 0.3,
    umap_metric: str = "euclidean",
    random_state: int = 0,
    knn_n_neighbors: int = 15,
    knn_metric: str = "euclidean",
    knn_weights: str = "uniform",
    kde_alpha: float = 0.7,
    kde_fill: bool = True,
    kde_thresh: float = 0.05,
    kde_n_contours: int = 5,
    kde_linewidth: float = 1.5,
    kde_hue_order: Optional[List] = None,
    kde_colors: Optional[List] = None,
    kde_show_legend: bool = True,
    scatter_s: float = 4,
    scatter_alpha: float = 0.6,
    figsize: Tuple[int, int] = (6, 5),
    save_dir: Optional[str] = None,
    save_prefix: Optional[str] = None,
    dpi: int = 300,
    show: bool = True,
) -> pd.DataFrame:
    """
    Project new_X cells onto the OLD UMAP space and colour them by a new
    metadata column.  Optionally transfer an old metadata label to new cells
    via KNN.

    Two independent goals
    ---------------------
    1. **Visualisation** — new cells are shown in the old UMAP space, coloured
       by ``new_column_to_project`` (a column from ``new_metadata``).
       Old cells form a grey background.

    2. **KNN label transfer** — if ``old_column_for_knn`` is provided, a KNN
       classifier is fitted on old cells (using their UMAP1_old / UMAP2_old
       coordinates and ``old_column_for_knn`` labels), then predicts that label
       for every new cell.  The result is stored as
       ``{old_column_for_knn}_kNN`` in the returned DataFrame.

    Parameters
    ----------
    old_X : DataFrame
        Pre-scaled / batch-corrected feature matrix for the reference dataset
        (right before PCA — no StandardScaler is applied internally).
    new_X : DataFrame
        Same for the query dataset.
    old_id : str
        Display label for old cells (grey in the plot, value in ``batch`` column).
    new_id : str
        Display label for new cells.
    new_column_to_project : str
        Column from ``new_metadata`` used to colour new cells in the plot.
    new_metadata : DataFrame
        Per-cell metadata for new_X (same row order).
    old_metadata : DataFrame or None
        Per-cell metadata for old_X (same row order).
    old_column_for_knn : str or None
        Column from ``old_metadata`` used to fit the KNN classifier.
        Old cells are the reference; new cells are the query.
        The predicted column ``{old_column_for_knn}_kNN`` is added to the
        returned DataFrame for new cells.  Skipped when None.
    override_old_coordinates : bool, default False
        If True, use ``old_coordinates`` to display old cells instead of the
        internally computed UMAP positions.  PCA and UMAP are still fitted on
        old_X to project new cells; the override is for visualisation and
        backwards-compatibility only.
    old_coordinates : list of two array-likes or None
        ``[umap1_array, umap2_array]`` of length ``n_old`` used when
        ``override_old_coordinates=True``.  Stored as ``UMAP1_preRun`` /
        ``UMAP2_preRun`` in the returned DataFrame.
    kde : bool, default False
        If True, show KDE density contours for new cells instead of points.
    n_components_pca : int or float, default 0.95
    umap_n_neighbors : int, default 30
    umap_min_dist : float, default 0.3
    umap_metric : str, default "euclidean"
    random_state : int, default 0
    knn_n_neighbors : int, default 15
    knn_metric : str, default "euclidean"
    knn_weights : str, default "uniform"
    kde_alpha, kde_fill, kde_thresh, kde_n_contours, kde_linewidth : KDE params
    kde_hue_order : list or None
    kde_colors : list or None
    kde_show_legend : bool, default True
    scatter_s : float, default 4
    scatter_alpha : float, default 0.6
    figsize : tuple, default (6, 5)
    save_dir, save_prefix : str or None
        If both given, saves a TIFF to
        ``{save_dir}/{save_prefix}_{new_column_to_project}_projection.tiff``.
    dpi : int, default 300
    show : bool, default True

    Returns
    -------
    pd.DataFrame
        Combined DataFrame (old cells first, then new) with all columns from
        ``old_metadata`` and ``new_metadata``, plus:

        * ``batch``                          — ``old_id`` or ``new_id``
        * ``UMAP1_old``, ``UMAP2_old``       — computed UMAP for old cells (NaN for new)
        * ``UMAP1_new``, ``UMAP2_new``       — projected UMAP for new cells (NaN for old)
        * ``UMAP1_preRun``, ``UMAP2_preRun`` — from ``old_coordinates``
          (only when ``override_old_coordinates=True``; NaN for new cells)
        * ``{old_column_for_knn}_kNN``       — KNN predictions for new cells
          (only when ``old_column_for_knn`` is provided; NaN for old cells)
    """
    if umap_module is None:
        raise ImportError("umap-learn is required. Install with: pip install umap-learn")

    # ------------------------------------------------------------------
    # Step 1: Feature alignment
    # ------------------------------------------------------------------
    shared = [c for c in old_X.columns if c in new_X.columns]
    if not shared:
        raise ValueError("No shared features between old_X and new_X.")

    n_old = len(old_X)
    n_new = len(new_X)

    old_arr = old_X[shared].values.astype(float)
    new_arr = new_X[shared].values.astype(float)

    # ------------------------------------------------------------------
    # Step 2: PCA — fit on old, transform both
    # ------------------------------------------------------------------
    pca = PCA(n_components=n_components_pca, random_state=random_state)
    Xp_old = pca.fit_transform(old_arr)
    Xp_new = pca.transform(new_arr)

    # ------------------------------------------------------------------
    # Step 3: UMAP — fit on old PCA, project new
    # ------------------------------------------------------------------
    reducer = umap_module.UMAP(
        n_neighbors=umap_n_neighbors,
        min_dist=umap_min_dist,
        metric=umap_metric,
        random_state=random_state,
        transform_seed=random_state,
    ).fit(Xp_old)

    U_old = reducer.transform(Xp_old)   # (n_old, 2)
    U_new = reducer.transform(Xp_new)   # (n_new, 2)

    # ------------------------------------------------------------------
    # Step 4: Build combined metadata DataFrame
    # ------------------------------------------------------------------
    old_meta = (
        old_metadata.reset_index(drop=True).copy()
        if old_metadata is not None
        else pd.DataFrame(index=range(n_old))
    )
    new_meta = new_metadata.reset_index(drop=True).copy()

    if len(old_meta) != n_old:
        raise ValueError(f"old_metadata has {len(old_meta)} rows but old_X has {n_old}.")
    if len(new_meta) != n_new:
        raise ValueError(f"new_metadata has {len(new_meta)} rows but new_X has {n_new}.")

    old_meta = old_meta.copy()
    new_meta = new_meta.copy()
    old_meta["batch"] = old_id
    new_meta["batch"] = new_id

    merged = _merge_metadata_dfs(old_meta, new_meta, old_id, new_id)

    # ------------------------------------------------------------------
    # Step 5: UMAP coordinate columns
    # ------------------------------------------------------------------
    merged["UMAP1_old"] = np.concatenate([U_old[:, 0], np.full(n_new, np.nan)])
    merged["UMAP2_old"] = np.concatenate([U_old[:, 1], np.full(n_new, np.nan)])
    merged["UMAP1_new"] = np.concatenate([np.full(n_old, np.nan), U_new[:, 0]])
    merged["UMAP2_new"] = np.concatenate([np.full(n_old, np.nan), U_new[:, 1]])

    if override_old_coordinates and old_coordinates is not None:
        pre_u1, pre_u2 = old_coordinates
        merged["UMAP1_preRun"] = np.concatenate(
            [np.asarray(pre_u1), np.full(n_new, np.nan)]
        )
        merged["UMAP2_preRun"] = np.concatenate(
            [np.asarray(pre_u2), np.full(n_new, np.nan)]
        )

    # ------------------------------------------------------------------
    # Step 6: KNN label transfer (old column → new cells)
    # ref  = old cells (UMAP1_old / UMAP2_old + old_column_for_knn), drop NaN
    # query = new cells (UMAP1_new / UMAP2_new)
    # ------------------------------------------------------------------
    if old_column_for_knn is not None:
        if old_metadata is None or old_column_for_knn not in old_metadata.columns:
            raise ValueError(
                f"old_column_for_knn='{old_column_for_knn}' not found in old_metadata."
            )

        knn_col = f"{old_column_for_knn}_kNN"
        merged[knn_col] = np.nan

        ref_df = (
            merged[merged["batch"] == old_id][
                ["UMAP1_old", "UMAP2_old", old_column_for_knn]
            ]
            .dropna()
        )
        query_coords = (
            merged[merged["batch"] == new_id][["UMAP1_new", "UMAP2_new"]].values
        )

        if len(ref_df) > 0 and len(query_coords) > 0:
            knn = KNeighborsClassifier(
                n_neighbors=knn_n_neighbors,
                metric=knn_metric,
                weights=knn_weights,
            )
            knn.fit(
                ref_df[["UMAP1_old", "UMAP2_old"]].values,
                ref_df[old_column_for_knn].values,
            )
            merged.loc[merged["batch"] == new_id, knn_col] = knn.predict(query_coords)

            print(f"\n{knn_col} value counts (new cells):")
            print(merged.loc[merged["batch"] == new_id, knn_col].value_counts())

    # ------------------------------------------------------------------
    # Step 7: Visualisation
    # ------------------------------------------------------------------
    if override_old_coordinates and old_coordinates is not None:
        disp_old_u1 = np.asarray(old_coordinates[0])
        disp_old_u2 = np.asarray(old_coordinates[1])
    else:
        disp_old_u1 = U_old[:, 0]
        disp_old_u2 = U_old[:, 1]

    fig, ax = plt.subplots(figsize=figsize)

    # Grey background — old cells
    ax.scatter(
        disp_old_u1, disp_old_u2,
        s=scatter_s, alpha=0.3, color="grey", label=old_id, rasterized=True,
    )

    # New cells coloured by new_column_to_project
    new_rows = merged[merged["batch"] == new_id].copy()

    if kde and new_rows[new_column_to_project].notna().any():
        kde_df = new_rows.rename(
            columns={"UMAP1_new": "UMAP1", "UMAP2_new": "UMAP2"}
        ).dropna(subset=[new_column_to_project, "UMAP1", "UMAP2"])

        _plot_kde_on_ax(
            df=kde_df,
            x_col="UMAP1",
            y_col="UMAP2",
            color_by=new_column_to_project,
            ax=ax,
            colors=kde_colors,
            hue_order=kde_hue_order,
            alpha=kde_alpha,
            fill=kde_fill,
            thresh=kde_thresh,
            n_contours=kde_n_contours,
            linewidth=kde_linewidth,
            show_legend=kde_show_legend,
        )
    else:
        groups = (
            kde_hue_order if kde_hue_order is not None
            else list(new_rows[new_column_to_project].dropna().unique())
        )
        colors_to_use = kde_colors if kde_colors is not None else _DEFAULT_COLORS

        for i, grp in enumerate(groups):
            mask = new_rows[new_column_to_project] == grp
            ax.scatter(
                new_rows.loc[mask, "UMAP1_new"],
                new_rows.loc[mask, "UMAP2_new"],
                s=scatter_s,
                alpha=scatter_alpha,
                color=colors_to_use[i % len(colors_to_use)],
                label=str(grp),
                rasterized=True,
            )

    ax.set_xlabel("UMAP1", fontsize=12, fontweight="bold", color="0.2")
    ax.set_ylabel("UMAP2", fontsize=12, fontweight="bold", color="0.2")
    ax.set_title(
        f"{new_id}: {new_column_to_project}", fontsize=13, fontweight="bold", pad=8
    )
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("0.2")
    ax.tick_params(color="0.2", width=0.8)
    if kde_show_legend:
        ax.legend(markerscale=3, fontsize=8, frameon=False)
    plt.tight_layout()

    if save_dir and save_prefix:
        import os
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(
            os.path.join(
                save_dir,
                f"{save_prefix}_{new_column_to_project}_projection.tiff",
            ),
            dpi=dpi,
            format="tiff",
            bbox_inches="tight",
        )

    if show:
        plt.show()
    else:
        plt.close()

    return merged
