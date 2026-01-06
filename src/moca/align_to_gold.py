
# align_to_gold.py
# Simple, generalizable alignment of NEW conditions to OLD gold-standard states
# using Euclidean distance in PCA space (PCA & scaler fitted on OLD only).
#
# Dependencies: numpy, pandas, scikit-learn, matplotlib
#
# Usage (sketch):
# from align_to_gold import align_conditions_to_states
# results = align_conditions_to_states(old_X, old_states, new_X, new_conditions, n_components=0.95, plot=True,
#                                      save_dir="figs", save_prefix="myalign") 
# results["distance_matrix"]  # condition x state (mean Euclidean distance)
# results["softscore_matrix"] # condition x state (mean softmax(-0.5 * dist^2))
#
# Where:
# - old_X, new_X are DataFrames (rows=cells, columns=shared morphological features)
# - old_states is a Series of gold-state labels (length == len(old_X))
# - new_conditions is a Series of condition labels (length == len(new_X))

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


@dataclass
class AlignmentModel:
    feature_names: Tuple[str, ...]
    scaler: StandardScaler
    pca: PCA
    centroids_: Dict[str, np.ndarray]
    states_: Tuple[str, ...]
    n_components_: Union[int, float]


def _intersect_and_align(
    old_X: pd.DataFrame, new_X: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, Tuple[str, ...]]:
    shared = tuple(col for col in old_X.columns if col in new_X.columns)
    if len(shared) == 0:
        raise ValueError("No shared feature columns between old and new datasets.")
    old_Xa = old_X.loc[:, shared].copy()
    new_Xa = new_X.loc[:, shared].copy()
    return old_Xa, new_Xa, shared


def _winsorize_to_old_bounds(
    old_X: pd.DataFrame, new_X: pd.DataFrame, clip_quantiles: Optional[Tuple[float, float]]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if clip_quantiles is None:
        return old_X, new_X
    ql, qh = clip_quantiles
    if not (0 <= ql < qh <= 1):
        raise ValueError("clip_quantiles must be a tuple within [0,1] with ql < qh.")
    # Compute bounds from OLD only to prevent data leakage
    low = old_X.quantile(ql, axis=0, numeric_only=True)
    high = old_X.quantile( (qh), axis=0, numeric_only=True)
    old_Xc = old_X.clip(lower=low, upper=high, axis=1)
    new_Xc = new_X.clip(lower=low, upper=high, axis=1)
    return old_Xc, new_Xc


def _fit_transformers_on_old(
    old_X: pd.DataFrame, n_components: Union[int, float], random_state: int
) -> Tuple[StandardScaler, PCA, np.ndarray]:
    scaler = StandardScaler(with_mean=True, with_std=True)
    Xs = scaler.fit_transform(old_X.values)
    pca = PCA(n_components=n_components, random_state=random_state)
    Xp = pca.fit_transform(Xs)
    return scaler, pca, Xp


def _transform_new(
    new_X: pd.DataFrame, scaler: StandardScaler, pca: PCA
) -> np.ndarray:
    Xs_new = scaler.transform(new_X.values)
    Xp_new = pca.transform(Xs_new)
    return Xp_new


def _compute_centroids(Xp_old: np.ndarray, states: pd.Series) -> Dict[str, np.ndarray]:
    centroids = {}
    for s in states.unique():
        mask = (states.values == s)
        if not np.any(mask):
            continue
        centroids[str(s)] = Xp_old[mask].mean(axis=0)
    return centroids


def _pairwise_distances_to_centroids(
    Xp: np.ndarray, centroids: Dict[str, np.ndarray]
) -> np.ndarray:
    # Returns matrix shape (n_samples, n_states) of Euclidean distances
    state_names = list(centroids.keys())
    C = np.vstack([centroids[s] for s in state_names])  # (n_states, p)
    # Efficient squared Euclidean: ||x-c||^2 = ||x||^2 + ||c||^2 - 2 x·c
    x2 = np.sum(Xp**2, axis=1, keepdims=True)           # (n,1)
    c2 = np.sum(C**2, axis=1, keepdims=True).T          # (1,k)
    xc = Xp @ C.T                                       # (n,k)
    dist2 = np.maximum(x2 + c2 - 2.0 * xc, 0.0)
    d = np.sqrt(dist2, out=dist2)  # reuse buffer
    return d  # (n, k)


def _softmax_from_distances(d: np.ndarray) -> np.ndarray:
    # Convert Euclidean distances to soft scores via softmax(-0.5 * d^2)
    # Use a numerically stable trick by subtracting row-wise max of the exponent argument
    z = -0.5 * (d ** 2)
    z_max = np.max(z, axis=1, keepdims=True)
    e = np.exp(z - z_max)
    p = e / np.sum(e, axis=1, keepdims=True)
    return p


def _aggregate_by_condition(
    values: np.ndarray, conditions: pd.Series, col_names: Tuple[str, ...], agg: str = "mean"
) -> pd.DataFrame:
    # values shape (n_samples, k); group by conditions and aggregate columns independently
    df_vals = pd.DataFrame(values, columns=col_names)
    df_vals["_cond_"] = conditions.values
    if agg == "mean":
        out = df_vals.groupby("_cond_").mean(numeric_only=True)
    elif agg == "median":
        out = df_vals.groupby("_cond_").median(numeric_only=True)
    else:
        raise ValueError("agg must be 'mean' or 'median'.")
    out.index.name = "Condition"
    return out


def _plot_heatmap(matrix: pd.DataFrame, title: str, save_path: Optional[str] = None, dpi: int = 300):
    fig = plt.figure(figsize=(max(6, 0.6 * matrix.shape[1] + 2), max(4, 0.4 * matrix.shape[0] + 2)))
    ax = fig.gca()
    im = ax.imshow(matrix.values, aspect="auto")
    ax.set_xticks(range(matrix.shape[1]))
    ax.set_xticklabels(matrix.columns, rotation=45, ha="right")
    ax.set_yticks(range(matrix.shape[0]))
    ax.set_yticklabels(matrix.index)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    if save_path is not None:
        # Save as TIFF at specified DPI
        fig.savefig(save_path, dpi=dpi, format="tiff", bbox_inches="tight")
    plt.show()


def align_conditions_to_states(
    old_X: pd.DataFrame,
    old_states: pd.Series,
    new_X: pd.DataFrame,
    new_conditions: pd.Series,
    n_components: Union[int, float] = 0.95,
    clip_quantiles: Optional[Tuple[float, float]] = (0.005, 0.995),
    aggregate: str = "mean",
    random_state: int = 0,
    plot: bool = True,
    save_dir: Optional[str] = None,
    save_prefix: Optional[str] = None,
    dpi: int = 300
) -> Dict[str, Union[AlignmentModel, pd.DataFrame]]:
    """
    Align NEW conditions to OLD gold states using Euclidean distance in PCA space.

    Parameters
    ----------
    old_X : DataFrame
        OLD dataset features (rows=cells, cols=features) with gold states.
    old_states : Series
        Gold state labels for OLD rows.
    new_X : DataFrame
        NEW dataset features (rows=cells, cols=features).
    new_conditions : Series
        Condition labels for NEW rows.
    n_components : int or float, default 0.95
        PCA dimensionality. If float in (0,1], keeps this proportion of variance.
        If int >= 1, keeps that many components.
    clip_quantiles : tuple(float, float) or None, default (0.005, 0.995)
        If provided, winsorize both OLD and NEW to OLD-derived per-feature bounds at
        the given quantiles to reduce outlier impact. Set to None to disable.
    aggregate : {'mean','median'}, default 'mean'
        Aggregation for condition-level summaries.
    random_state : int, default 0
        Random state for PCA.
    plot : bool, default True
        If True, show heatmaps.
    save_dir : str or None, default None
        Directory to save 300 dpi .tiff plots. If None, no files are written.
    save_prefix : str or None, default None
        File name prefix to use when saving plots. Requires save_dir to be set.
        Files will be named: f"{save_prefix}_distance.tiff" and f"{save_prefix}_softscore.tiff"
    dpi : int, default 300
        Resolution (dots per inch) for saved TIFFs.

    Returns
    -------
    dict with:
        - 'model'             : AlignmentModel with scaler, pca, centroids, etc.
        - 'distance_matrix'   : DataFrame [conditions x states] of mean Euclidean distances.
        - 'softscore_matrix'  : DataFrame [conditions x states] of mean soft scores.
    """
    # 1) Column intersection & alignment
    old_Xa, new_Xa, shared = _intersect_and_align(old_X, new_X)

    # 2) Winsorize to OLD bounds (optional)
    old_Xc, new_Xc = _winsorize_to_old_bounds(old_Xa, new_Xa, clip_quantiles)

    # 3) Fit scaler & PCA on OLD only; transform OLD & NEW
    scaler, pca, Xp_old = _fit_transformers_on_old(old_Xc, n_components, random_state)
    Xp_new = _transform_new(new_Xc, scaler, pca)

    # 4) Compute OLD state centroids in PCA space
    states = pd.Series(old_states).astype(str).reset_index(drop=True)
    centroids = _compute_centroids(Xp_old, states)
    state_names = tuple(centroids.keys())
    if len(state_names) == 0:
        raise ValueError("No centroids could be computed from the provided old_states.")

    # 5) Distances from each NEW cell to each centroid
    D_new = _pairwise_distances_to_centroids(Xp_new, centroids)  # (n_new, k)

    # 6) Soft scores via softmax(-0.5 * d^2)
    P_new = _softmax_from_distances(D_new)  # (n_new, k)

    # 7) Aggregate to condition x state matrices
    new_conds = pd.Series(new_conditions).astype(str).reset_index(drop=True)
    dist_mat = _aggregate_by_condition(D_new, new_conds, state_names, agg=aggregate)
    soft_mat = _aggregate_by_condition(P_new, new_conds, state_names, agg=aggregate)

    # 8) Package model
    model = AlignmentModel(
        feature_names=tuple(shared),
        scaler=scaler,
        pca=pca,
        centroids_=centroids,
        states_=state_names,
        n_components_=n_components,
    )

    # 9) Plot (and optionally save)
    dist_path = soft_path = None
    if save_dir is not None and save_prefix is not None:
        os.makedirs(save_dir, exist_ok=True)
        dist_path = os.path.join(save_dir, f"{save_prefix}_distance.tiff")
        soft_path = os.path.join(save_dir, f"{save_prefix}_softscore.tiff")

    if plot:
        _plot_heatmap(dist_mat, "Mean Euclidean distance: Condition × Gold State", save_path=dist_path, dpi=dpi)
        _plot_heatmap(soft_mat, "Mean Soft Score (softmax(-0.5·d²)): Condition × Gold State", save_path=soft_path, dpi=dpi)

    return {
        "model": model,
        "distance_matrix": dist_mat,
        "softscore_matrix": soft_mat,
    }
