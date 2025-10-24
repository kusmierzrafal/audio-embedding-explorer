from dataclasses import dataclass

import torch


@dataclass
class EmbeddingResult:
    vectors: list[torch.Tensor]
    source: str
    model_name: str
