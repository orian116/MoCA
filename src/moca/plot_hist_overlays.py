
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def clip_new_overlays(
    old_X: pd.DataFrame,
    new_X: pd.DataFrame,
    features=None,
    quantiles=None,
) -> pd.DataFrame:
    """
    Remove rows from new_X where any feature falls outside the range of old_X.

    Parameters
    ----------
    old_X : DataFrame
        Old dataset whose per-feature bounds define the valid range.
    new_X : DataFrame
        New dataset to be filtered.
    features : list-like or None
        Subset of feature names to use for range-checking.
        If None, uses the intersection of columns between old_X and new_X.
    quantiles : tuple(float, float) or None
        If provided, use old_X quantile bounds instead of absolute min/max.
        E.g. ``quantiles=(0.01, 0.99)`` keeps new_X rows whose values fall
        within old_X's 1st–99th percentile range for each feature.
        Values must be in (0, 1).  If None, absolute min/max is used.

    Returns
    -------
    pd.DataFrame
        Copy of new_X with any rows outside the per-feature bounds removed.
    """
    if features is None:
        shared = [c for c in old_X.columns if c in new_X.columns]
    else:
        shared = [c for c in features if c in old_X.columns and c in new_X.columns]

    if len(shared) == 0:
        raise ValueError("No shared features found between old_X and new_X.")

    if quantiles is not None:
        lo, hi = quantiles
        if not (0 < lo < hi < 1):
            raise ValueError("quantiles must be a (low, high) tuple with 0 < low < high < 1.")
        bound_label = f"old_X {lo*100:g}th–{hi*100:g}th percentile"
    else:
        bound_label = "old_X min–max"

    mask = pd.Series(True, index=new_X.index)

    for feat in shared:
        x_old = pd.to_numeric(old_X[feat], errors="coerce").dropna()
        x_new = pd.to_numeric(new_X[feat], errors="coerce")

        if x_old.empty:
            continue

        if quantiles is not None:
            old_min = x_old.quantile(lo)
            old_max = x_old.quantile(hi)
        else:
            old_min = x_old.min()
            old_max = x_old.max()

        in_range = (x_new >= old_min) & (x_new <= old_max)
        mask = mask & in_range.reindex(new_X.index, fill_value=False)

    filtered = new_X[mask].copy()
    n_removed = len(new_X) - len(filtered)
    if n_removed > 0:
        print(
            f"clip_new_overlays: removed {n_removed} of {len(new_X)} rows "
            f"({n_removed / len(new_X) * 100:.1f}%) outside {bound_label}."
        )
    else:
        print(f"clip_new_overlays: no rows removed; new_X is fully within {bound_label}.")

    return filtered


def plot_old_new_hist_overlays(old_X: pd.DataFrame,
                               new_X: pd.DataFrame,
                               bins: int = 50,
                               features=None):
    """
    Plot overlaid histograms for OLD vs NEW for each (shared) morphological feature.
    Draw vertical dotted lines at OLD's bounds whenever NEW falls outside OLD's range.

    Parameters
    ----------
    old_X : DataFrame
        Old dataset with gold-standard features (columns = features).
    new_X : DataFrame
        New dataset with the same features (or a superset).
    bins : int, default 50
        Number of histogram bins.
    features : list-like or None
        Subset of feature names to plot. If None, uses the intersection of columns.
    """
    # Determine which features to plot
    if features is None:
        shared = [c for c in old_X.columns if c in new_X.columns]
    else:
        shared = [c for c in features if c in old_X.columns and c in new_X.columns]
    if len(shared) == 0:
        raise ValueError("No shared features found between old_X and new_X.")

    for feat in shared:
        # Coerce to numeric and drop NaNs
        x_old = pd.to_numeric(old_X[feat], errors='coerce').dropna().to_numpy()
        x_new = pd.to_numeric(new_X[feat], errors='coerce').dropna().to_numpy()
        if x_old.size == 0 or x_new.size == 0:
            continue

        old_min, old_max = np.min(x_old), np.max(x_old)
        new_min, new_max = np.min(x_new), np.max(x_new)

        plt.figure(figsize=(6, 4))
        # Overlaid histograms (no explicit colors; uses matplotlib defaults)
        plt.hist(x_old, bins=bins, density=True, alpha=0.5, label="OLD")
        plt.hist(x_new, bins=bins, density=True, alpha=0.5, label="NEW")

        # Vertical dotted lines at OLD bounds if NEW exceeds them
        if new_min < old_min:
            plt.axvline(old_min, linestyle=":", linewidth=1.5)
        if new_max > old_max:
            plt.axvline(old_max, linestyle=":", linewidth=1.5)

        plt.xlabel(feat)
        plt.ylabel("Density")
        plt.title(f"{feat}: OLD vs NEW")
        plt.legend()
        plt.tight_layout()
        plt.show()
