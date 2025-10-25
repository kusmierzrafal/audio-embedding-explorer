from enum import StrEnum


class ReduceDimensionsMethod(StrEnum):
    PCA = "PCA"
    TSNE = "t-SNE"
    UMAP = "UMAP"
