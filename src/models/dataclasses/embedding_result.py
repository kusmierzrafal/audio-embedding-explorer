from dataclasses import dataclass

import torch


@dataclass
class EmbeddingResult:
    vector: torch.Tensor
    normalized_vector: torch.Tensor
    source: str
    model_name: str
