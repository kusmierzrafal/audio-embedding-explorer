from dataclasses import dataclass
import torch


@dataclass
class EmbeddingResult:
    text_embedding: torch.Tensor
    audio_embedding: torch.Tensor
    similarity: float
