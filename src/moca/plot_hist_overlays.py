
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
