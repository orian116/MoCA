# MoCA (Morphological Condition Alignment)

MoCA (Morphological Condition Alignment) is a lightweight framework for comparing new experimental conditions to gold-standard morphological states using shared feature spaces derived from CellProfiler or similar pipelines.
Tools for:
- condition→state alignment (PCA + centroid distances + soft scores)
- OLD vs NEW feature histogram overlays
- UMAP alignment visualization (fit on OLD, transform NEW)

## Install
```bash
pip install git+https://github.com/orian116/MoCA.git
