from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Union

import numpy as np

from src.models.dataclasses.embedding_result import EmbeddingResult
from src.models.enums.modalities import Modality


class BaseEmbedder(ABC):
    @abstractmethod
    def load(self):
        raise NotImplementedError

    def unload(self):
        pass

    def get_modalities(self) -> list[Modality]:
        raise NotImplementedError


class AudioEmbedder(BaseEmbedder):
    @abstractmethod
    def embed_audio(self, audio: Union[str, np.ndarray], sr: int) -> EmbeddingResult:
        raise NotImplementedError

    @abstractmethod
    def get_sr(self) -> int:
        raise NotImplementedError


class TextEmbedder(BaseEmbedder):
    @abstractmethod
    def embed_text(self, text: str) -> EmbeddingResult:
        raise NotImplementedError
