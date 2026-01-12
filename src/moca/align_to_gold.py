
# align_to_gold.py
# Simple, generalizable alignment of NEW conditions to OLD gold-standard states
# using (optionally weighted) Euclidean distance in PCA space
# (PCA & scaler fitted on OLD only).
#
# Dependencies: numpy, pandas, scikit-learn, matplotlib
#
# Usage (sketch):
# from align_to_gold import align_conditions_to_states
# results = align_conditions_to_states(
#     old_X, old_states, new_X, new_conditions,
#     n_components=0.95, weighted=True, plot=True,
#     save_dir="figs", save_prefix="myalign"
# )
# results["distance_matrix"]  # condition x state (mean distance; lower = closer)

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
    weighted_: bool
    pc_weights_: Optional[np.ndarray]  # None if weighted_ is False


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
    high = old_X.quantile(qh, axis=0, numeric_only=True)
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
    Xp: np.ndarray,
    centroids: Dict[str, np.ndarray],
    weighted: bool = False,
    pc_weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Compute distances from samples (Xp) to each centroid.

    If weighted=True, computes a weighted Euclidean distance:
        d(x,c) = sqrt( sum_j w_j * (x_j - c_j)^2 )

    In MoCA, w_j defaults to PCA explained_variance_ratio_ (from OLD PCA fit).
    """
    state_names = list(centroids.keys())
    C = np.vstack([centroids[s] for s in state_names])  # (k, p)

    if weighted:
        if pc_weights is None:
            raise ValueError("pc_weights must be provided when weighted=True.")
        w = np.asarray(pc_weights, dtype=float).reshape(1, -1)  # (1, p)
        if w.shape[1] != Xp.shape[1]:
            raise ValueError("pc_weights length must match the number of PCA components.")
        # Weighted squared Euclidean using scaled coordinates: x' = sqrt(w) * x
        sw = np.sqrt(w)  # (1, p)
        Xw = Xp * sw
        Cw = C * sw
        x2 = np.sum(Xw**2, axis=1, keepdims=True)          # (n,1)
        c2 = np.sum(Cw**2, axis=1, keepdims=True).T        # (1,k)
        xc = Xw @ Cw.T                                     # (n,k)
        dist2 = np.maximum(x2 + c2 - 2.0 * xc, 0.0)
        return np.sqrt(dist2)
    else:
        # Standard Euclidean in PCA space
        x2 = np.sum(Xp**2, axis=1, keepdims=True)          # (n,1)
        c2 = np.sum(C**2, axis=1, keepdims=True).T         # (1,k)
        xc = Xp @ C.T                                      # (n,k)
        dist2 = np.maximum(x2 + c2 - 2.0 * xc, 0.0)
        return np.sqrt(dist2)


def _aggregate_by_condition(
    values: np.ndarray, conditions: pd.Series, col_names: Tuple[str, ...], agg: str = "mean"
) -> pd.DataFrame:
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
        fig.savefig(save_path, dpi=dpi, format="tiff", bbox_inches="tight")
    plt.show()


def align_conditions_to_states(
    old_X: pd.DataFrame,
    old_states: pd.Series,
    new_X: pd.DataFrame,
    new_conditions: pd.Series,
    n_components: Union[int, float] = 0.95,
    weighted: bool = True,
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

    If weighted=True, distances are weighted by PCA component importance
    (explained_variance_ratio_ from the OLD PCA fit).

    Returns
    -------
    dict with:
        - 'model'           : AlignmentModel with scaler, pca, centroids, etc.
        - 'distance_matrix' : DataFrame [conditions x states] of mean distances.
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
    pc_w = pca.explained_variance_ratio_.copy() if weighted else None
    D_new = _pairwise_distances_to_centroids(Xp_new, centroids, weighted=weighted, pc_weights=pc_w)

    # 6) Aggregate to condition x state distance matrix
    new_conds = pd.Series(new_conditions).astype(str).reset_index(drop=True)
    dist_mat = _aggregate_by_condition(D_new, new_conds, state_names, agg=aggregate)

    # 7) Package model
    model = AlignmentModel(
        feature_names=tuple(shared),
        scaler=scaler,
        pca=pca,
        centroids_=centroids,
        states_=state_names,
        n_components_=n_components,
        weighted_=weighted,
        pc_weights_=pc_w,
    )

    # 8) Plot (and optionally save)
    dist_path = None
    if save_dir is not None and save_prefix is not None:
        os.makedirs(save_dir, exist_ok=True)
        dist_path = os.path.join(save_dir, f"{save_prefix}_distance.tiff")

    if plot:
        title = "Mean weighted Euclidean distance: Condition × Gold State" if weighted else "Mean Euclidean distance: Condition × Gold State"
        _plot_heatmap(dist_mat, title, save_path=dist_path, dpi=dpi)

    return {
        "model": model,
        "distance_matrix": dist_mat,
    }
