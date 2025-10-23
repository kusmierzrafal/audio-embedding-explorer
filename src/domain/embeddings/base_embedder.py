from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional

from src.models.dataclasses.embedding_result import EmbeddingResult


class BaseEmbedder(ABC):
    def __init__(self, model_id: str, model_dir: Path) -> None:
        self.model_id = model_id
        self.model_dir = model_dir
        self.model: Optional[Any] = None
        self.processor: Optional[Any] = None
        self.tokenizer: Optional[Any] = None

    @abstractmethod
    def ensure_exists(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def load_model(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def embed_text(self, text: str) -> EmbeddingResult:
        raise NotImplementedError

    @abstractmethod
    def embed_audio(self, audio_path: Path) -> EmbeddingResult:
        raise NotImplementedError
