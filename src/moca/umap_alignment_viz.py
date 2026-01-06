
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from typing import Optional, Tuple, Union, Dict

try:
    import umap
except Exception as e:
    umap = None  # Will raise a clear error on use


from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def _intersect_df(old_X: pd.DataFrame, new_X: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, Tuple[str, ...]]:
    shared = tuple(c for c in old_X.columns if c in new_X.columns)
    if not shared:
        raise ValueError("No shared features between old_X and new_X.")
    return old_X.loc[:, shared].copy(), new_X.loc[:, shared].copy(), shared


def _fit_pca_on_old(old_X: pd.DataFrame, n_components: Union[int, float], random_state: int):
    scaler = StandardScaler(with_mean=True, with_std=True)
    Xs_old = scaler.fit_transform(old_X.values)
    pca = PCA(n_components=n_components, random_state=random_state)
    Xp_old = pca.fit_transform(Xs_old)
    return scaler, pca, Xp_old


def _transform_new(new_X: pd.DataFrame, scaler: StandardScaler, pca: PCA):
    Xs_new = scaler.transform(new_X.values)
    return pca.transform(Xs_new)


def _centroids(X: np.ndarray, labels: pd.Series) -> Dict[str, np.ndarray]:
    out = {}
    u = labels.astype(str).values
    for lab in np.unique(u):
        m = (u == lab)
        if np.any(m):
            out[lab] = X[m].mean(axis=0)
    return out


def _pairwise_dist_to_centroids(X: np.ndarray, centroids: Dict[str, np.ndarray]) -> Tuple[np.ndarray, Tuple[str, ...]]:
    names = tuple(centroids.keys())
    C = np.vstack([centroids[n] for n in names])  # (k, p)
    x2 = np.sum(X**2, axis=1, keepdims=True)
    c2 = np.sum(C**2, axis=1, keepdims=True).T
    xc = X @ C.T
    d2 = np.maximum(x2 + c2 - 2.0 * xc, 0.0)
    return np.sqrt(d2), names


def _softmax_from_dist(d: np.ndarray) -> np.ndarray:
    z = -0.5 * (d ** 2)
    zmax = np.max(z, axis=1, keepdims=True)
    e = np.exp(z - zmax)
    return e / np.sum(e, axis=1, keepdims=True)


def plot_umap_alignment(
    old_X: pd.DataFrame,
    old_states: pd.Series,
    new_X: pd.DataFrame,
    new_conditions: pd.Series,
    n_components_pca: Union[int, float] = 0.95,
    umap_n_neighbors: int = 30,
    umap_min_dist: float = 0.3,
    umap_metric: str = "euclidean",
    random_state: int = 0,
    save_dir: Optional[str] = None,
    save_prefix: Optional[str] = None,
    dpi: int = 300
):
    """
    Visualize how NEW conditions align with OLD states in a shared UMAP space.
    - Fit scaler + PCA on OLD only.
    - Fit UMAP on OLD (in PCA space), transform both OLD and NEW.
    - Show:
        Fig 1: OLD UMAP colored by gold states.
        Fig 2: NEW UMAP colored by condition + OLD state centroids.
        Fig 3: Arrows from each NEW condition centroid to its soft-score-weighted OLD-state centroid.

    Parameters
    ----------
    old_X, new_X : DataFrame (rows=cells, cols=features)
    old_states   : Series of gold state labels for old_X
    new_conditions : Series of condition labels for new_X
    n_components_pca : float in (0,1] to keep variance, or int for fixed components
    umap_n_neighbors, umap_min_dist, umap_metric : standard UMAP params
    random_state : int
    save_dir, save_prefix : if provided, saves 300-dpi TIFFs to f"{save_prefix}_*.tiff"
    dpi : int
    """
    if umap is None:
        raise ImportError("umap-learn is required. Install with: pip install umap-learn")

    # 1) Align features
    oldA, newA, shared = _intersect_df(old_X, new_X)
    old_states = pd.Series(old_states).astype(str).reset_index(drop=True)
    new_conditions = pd.Series(new_conditions).astype(str).reset_index(drop=True)

    # 2) PCA on OLD, transform NEW
    scaler, pca, Xp_old = _fit_pca_on_old(oldA, n_components_pca, random_state)
    Xp_new = _transform_new(newA, scaler, pca)

    # 3) UMAP on OLD, transform both
    reducer = umap.UMAP(
        n_neighbors=umap_n_neighbors,
        min_dist=umap_min_dist,
        metric=umap_metric,
        random_state=random_state,
        transform_seed=random_state
    ).fit(Xp_old)
    U_old = reducer.transform(Xp_old)  # (n_old, 2)
    U_new = reducer.transform(Xp_new)  # (n_new, 2)

    # 4) OLD state centroids in PCA -> distances/softscores for NEW
    cent_pca = _centroids(Xp_old, old_states)
    D_new, state_names = _pairwise_dist_to_centroids(Xp_new, cent_pca)
    P_new = _softmax_from_dist(D_new)  # soft alignment
    top_idx = np.argmax(P_new, axis=1)
    top_state = np.array(state_names)[top_idx]
    top_conf = P_new[np.arange(P_new.shape[0]), top_idx]

    # 5) OLD state centroids in UMAP (for plotting/targets)
    cent_umap = _centroids(U_old, old_states)

    # 6) Figures

    # Fig 1: OLD UMAP colored by state
    plt.figure(figsize=(6, 5))
    for s in np.unique(old_states.values):
        m = (old_states.values == s)
        plt.scatter(U_old[m, 0], U_old[m, 1], s=6, alpha=0.7, label=str(s))
    plt.title("OLD: UMAP colored by gold states")
    plt.xlabel("UMAP-1"); plt.ylabel("UMAP-2")
    plt.legend(markerscale=2, fontsize=8, frameon=False, ncol=2)
    plt.tight_layout()
    if save_dir and save_prefix:
        import os
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f"{save_prefix}_OLD_states.tiff"), dpi=dpi, format="tiff", bbox_inches="tight")
    plt.show()

    # Fig 2: NEW UMAP colored by condition; overlay OLD centroids
    plt.figure(figsize=(6, 5))
    for c in np.unique(new_conditions.values):
        m = (new_conditions.values == c)
        plt.scatter(U_new[m, 0], U_new[m, 1], s=6, alpha=0.6, label=str(c))
    # centroids
    for s, xy in cent_umap.items():
        plt.scatter([xy[0]], [xy[1]], s=60, marker="X")
        plt.text(xy[0], xy[1], f" {s}", va="center", ha="left")
    plt.title("NEW: UMAP colored by condition (+ OLD state centroids)")
    plt.xlabel("UMAP-1"); plt.ylabel("UMAP-2")
    plt.legend(markerscale=2, fontsize=8, frameon=False, ncol=2)
    plt.tight_layout()
    if save_dir and save_prefix:
        plt.savefig(os.path.join(save_dir, f"{save_prefix}_NEW_conditions.tiff"), dpi=dpi, format="tiff", bbox_inches="tight")
    plt.show()

    # Fig 3: Arrows from NEW condition centroids -> weighted OLD-state centroid (by soft scores)
    # Compute per-condition UMAP centroid and weighted OLD target
    conds = np.unique(new_conditions.values)
    cond_cent = {}
    cond_target = {}
    for c in conds:
        m = (new_conditions.values == c)
        if not np.any(m):
            continue
        # centroid in UMAP
        src = U_new[m].mean(axis=0)
        # average soft scores across the condition -> weights
        w = P_new[m].mean(axis=0)  # order aligned with state_names
        # weighted OLD centroid in UMAP
        tgt = np.zeros(2, dtype=float)
        for wi, s in zip(w, state_names):
            tgt += wi * cent_umap[s]
        cond_cent[c] = src
        cond_target[c] = tgt

    plt.figure(figsize=(6, 5))
    # background: faint OLD states for context
    for s in np.unique(old_states.values):
        m = (old_states.values == s)
        plt.scatter(U_old[m, 0], U_old[m, 1], s=4, alpha=0.15)
    # arrows
    for c in conds:
        src = cond_cent[c]; tgt = cond_target[c]
        plt.scatter([src[0]], [src[1]], s=40, label=str(c))
        plt.annotate("", xy=(tgt[0], tgt[1]), xytext=(src[0], src[1]),
                     arrowprops=dict(arrowstyle="->", lw=1.5))
        # label near the target with top state
        plt.text(tgt[0], tgt[1], f" {c} → weighted-old", fontsize=8)
    # overlay state centroids
    for s, xy in cent_umap.items():
        plt.scatter([xy[0]], [xy[1]], s=60, marker="X")
        plt.text(xy[0], xy[1], f" {s}", va="center", ha="left")
    plt.title("Condition drift arrows: NEW centroid → weighted OLD-state centroid")
    plt.xlabel("UMAP-1"); plt.ylabel("UMAP-2")
    plt.legend(markerscale=2, fontsize=8, frameon=False, ncol=2)
    plt.tight_layout()
    if save_dir and save_prefix:
        plt.savefig(os.path.join(save_dir, f"{save_prefix}_arrows.tiff"), dpi=dpi, format="tiff", bbox_inches="tight")
    plt.show()

    # Return useful objects for further analysis
    return {
        "U_old": U_old,
        "U_new": U_new,
        "old_states": old_states,
        "new_conditions": new_conditions,
        "state_centroids_umap": cent_umap,
        "condition_centroids_umap": cond_cent,
        "condition_targets_umap": cond_target,
        "soft_scores_new": P_new,
        "state_names": state_names
    }
