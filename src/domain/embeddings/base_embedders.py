from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from src.models.dataclasses.embedding_result import EmbeddingResult


class AudioEmbedder(ABC):
    @abstractmethod
    def embed_audio(self, audio_path: Path) -> EmbeddingResult:
        raise NotImplementedError
    
class TextEmbedder(ABC):
    @abstractmethod
    def embed_text(self, text: str) -> EmbeddingResult:
        raise NotImplementedError
