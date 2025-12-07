from pathlib import Path

import openl3
import streamlit as st
import torch
import torch.nn.functional as F
from openl3 import models

from src.config.error_messages import ERROR_MSG
from src.domain.embeddings.base_embedders import AudioEmbedder
from src.models.dataclasses.embedding_result import EmbeddingResult
from src.utils.audio_utils import AudioHelper


class OpenL3Embedder(AudioEmbedder):
    def __init__(self, input_repr="mel256", embedding_size=512):
        self._hop_size = 0.1  # seconds
        self._input_repr = input_repr
        self._embedding_size = embedding_size

    def load(self):
        self._model = load_weights_cached(
            input_repr=self._input_repr,
            embedding_size=self._embedding_size
        )

    def get_modalities(self) -> list[str]:
        return ["audio"]

    def embed_audio(self, audio_path: Path) -> EmbeddingResult:
        if not audio_path.exists():
            raise FileNotFoundError(ERROR_MSG["AUDIO_FILE_NOT_FOUND"])

        waveform, sr = AudioHelper.load_audio(audio_path, target_sr=48000)
        embeddings, timestamps = openl3.get_audio_embedding(
            waveform, sr, model=self._model, hop_size=self._hop_size, center=True
        )
        embeddings = torch.from_numpy(embeddings)
        global_embedding = embeddings.mean(dim=0, keepdim=True)

        normalized_embeddings = F.normalize(embeddings, p=2, dim=-1)
        # Compute the global embedding by averaging frame-level embeddings
        normalized_global_embedding = normalized_embeddings.mean(dim=0, keepdim=True)

        return EmbeddingResult(
            vector=global_embedding,
            normalized_vector=normalized_global_embedding,
        )

@st.cache_resource(show_spinner=False)
def load_weights_cached(input_repr: str, embedding_size: int):
    model = models.load_audio_embedding_model(
            input_repr=input_repr, content_type="music", embedding_size=embedding_size
        )
    return model