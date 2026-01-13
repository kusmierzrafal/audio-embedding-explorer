import numpy as np
import torch
import umap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from src.models.enums.reduce_dimensions_method import ReduceDimensionsMethod


def cosine_similarity(a, b) -> float:
    # Konwersja na tensor (działa dla numpy, list i tensorów)
    if not isinstance(a, torch.Tensor):
        a = torch.as_tensor(a)
    if not isinstance(b, torch.Tensor):
        b = torch.as_tensor(b)

    # Wymuszenie typu float (ważne, jeśli numpy array był np. float64 lub object)
    a = a.float()
    b = b.float()

    # Obliczenie podobieństwa
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()


def reduce_embeddings_dimensionality(
    vectors: np.ndarray, method: ReduceDimensionsMethod = ReduceDimensionsMethod.PCA
) -> np.ndarray:
    n_samples = vectors.shape[0]

    if method == ReduceDimensionsMethod.PCA:
        reducer = PCA(n_components=2, random_state=42)
    elif method == ReduceDimensionsMethod.TSNE:
        if n_samples < 3:
            raise ValueError("t-SNE requires at least 3 samples.")
        reducer = TSNE(
            n_components=2,
            perplexity=min(
                30, max(2, n_samples // 2)
            ),  # TODO this can be managed by user input
            random_state=42,
        )
    elif method == ReduceDimensionsMethod.UMAP:
        if n_samples < 3:
            raise ValueError("UMAP requires at least 3 samples.")
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=min(
                15, n_samples - 1
            ),  # TODO this can be managed by user input
            random_state=42,
        )
    else:
        raise ValueError(f"Unknown reduction method: {method}")
    return reducer.fit_transform(vectors)
