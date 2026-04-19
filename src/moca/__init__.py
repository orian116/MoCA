# moca/__init__.py
"""
MoCA — Morphological Condition Alignment toolkit
Tools for aligning new experimental conditions to gold-standard morphological states,
visualizing feature distributions, and comparing embeddings.
"""

from .align_to_gold import align_conditions_to_states
from .plot_hist_overlays import plot_old_new_hist_overlays, clip_new_overlays
from .umap_alignment_viz import plot_umap_alignment
from .shared_heatmap import generate_shared_heatmap
from .combined_space import create_combined_space
from .projection_viz import plot_and_save_projections, fit_old_embedding

__all__ = [
    "align_conditions_to_states",
    "plot_old_new_hist_overlays",
    "clip_new_overlays",
    "plot_umap_alignment",
    "generate_shared_heatmap",
    "create_combined_space",
    "plot_and_save_projections",
    "fit_old_embedding",
]
