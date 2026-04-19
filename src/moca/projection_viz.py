
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.patches as mpatches
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
):
    """Plot KDE on an existing axes; always suppresses seaborn's own legend."""
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

    # Always remove seaborn's legend — caller builds the combined legend
    legend = ax.get_legend()
    if legend:
        legend.remove()


def _build_legend(ax, old_id: str, groups: List, colors: List, show: bool):
    """Attach a combined legend: grey patch for old_id + colour patches for groups."""
    if not show:
        return
    handles = [mpatches.Patch(color="grey", alpha=0.5, label=old_id)]
    handles += [
        mpatches.Patch(color=colors[i % len(colors)], label=str(g))
        for i, g in enumerate(groups)
    ]
    ax.legend(handles=handles, fontsize=8, frameon=False, markerscale=1)


# ---------------------------------------------------------------------------
# fit_old_embedding — fit PCA + UMAP on old data, save .pkl files
# ---------------------------------------------------------------------------

def fit_old_embedding(
    old_X: pd.DataFrame,
    old_id: str,
    old_metadata: Optional[pd.DataFrame] = None,
    n_components_pca: Union[int, float] = 0.95,
    umap_n_neighbors: int = 30,
    umap_min_dist: float = 0.3,
    umap_metric: str = "euclidean",
    random_state: int = 0,
    save_dir: str = ".",
    save_prefix: str = "moca_old",
) -> pd.DataFrame:
    """
    Fit PCA + UMAP on ``old_X``, save both models as ``.pkl`` files, and
    return a metadata DataFrame with ``UMAP1_old`` / ``UMAP2_old`` columns.

    The saved files are used by :func:`plot_and_save_projections` (via
    ``pca_path`` / ``umap_path``) to ensure that multiple new datasets are
    projected into exactly the same embedding space.

    Parameters
    ----------
    old_X : DataFrame
        Pre-scaled / batch-corrected feature matrix (right before PCA).
    old_id : str
        Label for the old dataset (stored in the ``batch`` column).
    old_metadata : DataFrame or None
        Per-cell metadata (same row order as old_X).
    n_components_pca : int or float, default 0.95
    umap_n_neighbors : int, default 30
    umap_min_dist : float, default 0.3
    umap_metric : str, default "euclidean"
    random_state : int, default 0
    save_dir : str, default "."
        Directory where ``.pkl`` files are written.
    save_prefix : str, default "moca_old"
        Filename stem.  Produces ``{save_prefix}_pca.pkl`` and
        ``{save_prefix}_umap.pkl``.

    Returns
    -------
    pd.DataFrame
        ``old_metadata`` columns  + ``batch`` + ``UMAP1_old`` + ``UMAP2_old``.
        Pass this directly as ``old_metadata`` in
        :func:`plot_and_save_projections`.
    """
    if umap_module is None:
        raise ImportError("umap-learn is required. Install with: pip install umap-learn")

    n_old = len(old_X)
    old_arr = old_X.values.astype(float)

    # Fit PCA
    pca = PCA(n_components=n_components_pca, random_state=random_state)
    Xp_old = pca.fit_transform(old_arr)

    # Fit UMAP
    reducer = umap_module.UMAP(
        n_neighbors=umap_n_neighbors,
        min_dist=umap_min_dist,
        metric=umap_metric,
        random_state=random_state,
        transform_seed=random_state,
    ).fit(Xp_old)
    U_old = reducer.transform(Xp_old)

    # Save models
    os.makedirs(save_dir, exist_ok=True)
    pca_path = os.path.join(save_dir, f"{save_prefix}_pca.pkl")
    umap_path = os.path.join(save_dir, f"{save_prefix}_umap.pkl")

    with open(pca_path, "wb") as f:
        pickle.dump({"pca": pca, "feature_names": list(old_X.columns)}, f)
    with open(umap_path, "wb") as f:
        pickle.dump(reducer, f)

    print(f"Saved PCA  → {pca_path}")
    print(f"Saved UMAP → {umap_path}")

    # Build result DataFrame
    old_meta = (
        old_metadata.reset_index(drop=True).copy()
        if old_metadata is not None
        else pd.DataFrame(index=range(n_old))
    )
    if len(old_meta) != n_old:
        raise ValueError(
            f"old_metadata has {len(old_meta)} rows but old_X has {n_old}."
        )

    old_meta = old_meta.copy()
    old_meta["batch"] = old_id
    old_meta["UMAP1_old"] = U_old[:, 0]
    old_meta["UMAP2_old"] = U_old[:, 1]

    return old_meta


# ---------------------------------------------------------------------------
# plot_and_save_projections
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
    pca_path: Optional[str] = None,
    umap_path: Optional[str] = None,
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

    Preferred usage
    ---------------
    Run :func:`fit_old_embedding` once on the reference data to fit the PCA
    and UMAP and save them as ``.pkl`` files.  Pass the returned DataFrame as
    ``old_metadata`` and the paths as ``pca_path`` / ``umap_path`` to
    guarantee every call uses exactly the same embedding space.

    Two independent goals
    ---------------------
    1. **Visualisation** — new cells shown in the old UMAP space, coloured by
       ``new_column_to_project`` (from ``new_metadata``).  Old cells form a
       grey background.  The legend always includes ``old_id`` (grey) and the
       group labels for ``new_column_to_project``.
    2. **KNN label transfer** — if ``old_column_for_knn`` is given, a KNN
       classifier is fitted on old cells (UMAP1_old / UMAP2_old +
       ``old_column_for_knn``), then predicts that label for every new cell.
       Result stored as ``{old_column_for_knn}_kNN`` for new cells.

    Parameters
    ----------
    old_X : DataFrame
        Pre-scaled / batch-corrected feature matrix for the reference dataset.
    new_X : DataFrame
        Same for the query dataset.
    old_id : str
        Label for old cells (grey in plot, value in ``batch`` column).
    new_id : str
        Label for new cells.
    new_column_to_project : str
        Column from ``new_metadata`` used to colour new cells.
    new_metadata : DataFrame
        Per-cell metadata for new_X (same row order).
    old_metadata : DataFrame or None
        Per-cell metadata for old_X (same row order).  If this DataFrame
        contains ``UMAP1_old`` / ``UMAP2_old`` (e.g. from
        :func:`fit_old_embedding`), those coordinates are used directly for
        display and KNN — no re-transformation of old_X is needed.
    old_column_for_knn : str or None
        Column from ``old_metadata`` to fit KNN with.  Predicts
        ``{old_column_for_knn}_kNN`` for new cells.
    pca_path : str or None
        Path to a ``.pkl`` saved by :func:`fit_old_embedding``.
        When provided together with ``umap_path``, the pre-fitted models are
        loaded instead of fitting new ones.
    umap_path : str or None
        Path to the UMAP ``.pkl`` saved by :func:`fit_old_embedding`.
    override_old_coordinates : bool, default False
        Legacy parameter — if True, use ``old_coordinates`` for display.
        Superseded by passing ``old_metadata`` with ``UMAP1_old`` /
        ``UMAP2_old`` columns.
    old_coordinates : list of two array-likes or None
        ``[umap1, umap2]`` for old cells; used when
        ``override_old_coordinates=True``.
    kde : bool, default False
        Show KDE density contours for new cells instead of scatter points.
    n_components_pca : int or float, default 0.95
        Only used when ``pca_path`` is not provided.
    umap_n_neighbors, umap_min_dist, umap_metric, random_state :
        Only used when ``umap_path`` is not provided.
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
    dpi : int, default 300
    show : bool, default True

    Returns
    -------
    pd.DataFrame
        Combined DataFrame with all columns from ``old_metadata`` /
        ``new_metadata``, plus ``batch``, ``UMAP1_old``, ``UMAP2_old``,
        ``UMAP1_new``, ``UMAP2_new``, ``UMAP1_preRun`` / ``UMAP2_preRun``
        (when ``override_old_coordinates=True``), and
        ``{old_column_for_knn}_kNN`` for new cells.
    """
    if umap_module is None:
        raise ImportError("umap-learn is required. Install with: pip install umap-learn")

    n_old = len(old_X)
    n_new = len(new_X)

    # ------------------------------------------------------------------
    # Step 1: Determine feature alignment and load / fit PCA + UMAP
    # ------------------------------------------------------------------
    if pca_path is not None and umap_path is not None:
        with open(pca_path, "rb") as f:
            pca_pkg = pickle.load(f)
        pca = pca_pkg["pca"]
        feature_names = pca_pkg["feature_names"]

        with open(umap_path, "rb") as f:
            reducer = pickle.load(f)

        # Align new_X to the features the PCA was trained on
        missing = [c for c in feature_names if c not in new_X.columns]
        if missing:
            raise ValueError(
                f"new_X is missing {len(missing)} feature(s) required by the "
                f"loaded PCA: {missing[:5]}{'...' if len(missing) > 5 else ''}"
            )
        new_arr = new_X[feature_names].values.astype(float)
        Xp_new = pca.transform(new_arr)
        U_new = reducer.transform(Xp_new)

        # Old UMAP coords: use old_metadata if it already has them
        old_has_umap = (
            old_metadata is not None
            and "UMAP1_old" in old_metadata.columns
            and "UMAP2_old" in old_metadata.columns
        )
        if old_has_umap:
            U_old = old_metadata[["UMAP1_old", "UMAP2_old"]].reset_index(drop=True).values
        else:
            old_arr = old_X[feature_names].values.astype(float)
            Xp_old = pca.transform(old_arr)
            U_old = reducer.transform(Xp_old)

    else:
        # Fit PCA and UMAP on old_X
        shared = [c for c in old_X.columns if c in new_X.columns]
        if not shared:
            raise ValueError("No shared features between old_X and new_X.")

        old_arr = old_X[shared].values.astype(float)
        new_arr = new_X[shared].values.astype(float)

        pca = PCA(n_components=n_components_pca, random_state=random_state)
        Xp_old = pca.fit_transform(old_arr)
        Xp_new = pca.transform(new_arr)

        reducer = umap_module.UMAP(
            n_neighbors=umap_n_neighbors,
            min_dist=umap_min_dist,
            metric=umap_metric,
            random_state=random_state,
            transform_seed=random_state,
        ).fit(Xp_old)

        U_old = reducer.transform(Xp_old)
        U_new = reducer.transform(Xp_new)

    # ------------------------------------------------------------------
    # Step 2: Build combined metadata DataFrame
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
    # Step 3: UMAP coordinate columns (always set explicitly)
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
    # Step 4: KNN label transfer (old_column_for_knn → new cells)
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
            ].dropna()
        )
        query_coords = merged[merged["batch"] == new_id][
            ["UMAP1_new", "UMAP2_new"]
        ].values

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
    # Step 5: Determine display coordinates for old cells
    # ------------------------------------------------------------------
    if override_old_coordinates and old_coordinates is not None:
        disp_old_u1 = np.asarray(old_coordinates[0])
        disp_old_u2 = np.asarray(old_coordinates[1])
    else:
        disp_old_u1 = U_old[:, 0]
        disp_old_u2 = U_old[:, 1]

    # ------------------------------------------------------------------
    # Step 6: Visualisation
    # ------------------------------------------------------------------
    groups = (
        kde_hue_order if kde_hue_order is not None
        else list(
            merged.loc[merged["batch"] == new_id, new_column_to_project]
            .dropna()
            .unique()
        )
    )
    colors_to_use = kde_colors if kde_colors is not None else _DEFAULT_COLORS

    fig, ax = plt.subplots(figsize=figsize)

    # Grey background — old cells
    ax.scatter(
        disp_old_u1, disp_old_u2,
        s=scatter_s, alpha=0.3, color="grey", rasterized=True,
    )

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
            colors=colors_to_use,
            hue_order=groups,
            alpha=kde_alpha,
            fill=kde_fill,
            thresh=kde_thresh,
            n_contours=kde_n_contours,
            linewidth=kde_linewidth,
        )
    else:
        for i, grp in enumerate(groups):
            mask = new_rows[new_column_to_project] == grp
            ax.scatter(
                new_rows.loc[mask, "UMAP1_new"],
                new_rows.loc[mask, "UMAP2_new"],
                s=scatter_s,
                alpha=scatter_alpha,
                color=colors_to_use[i % len(colors_to_use)],
                rasterized=True,
            )

    # Combined legend: old_id (grey) + new groups
    _build_legend(ax, old_id, groups, colors_to_use, show=kde_show_legend)

    ax.set_xlabel("UMAP1", fontsize=12, fontweight="bold", color="0.2")
    ax.set_ylabel("UMAP2", fontsize=12, fontweight="bold", color="0.2")
    ax.set_title(
        f"{new_id}: {new_column_to_project}", fontsize=13, fontweight="bold", pad=8
    )
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("0.2")
    ax.tick_params(color="0.2", width=0.8)
    plt.tight_layout()

    if save_dir and save_prefix:
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
