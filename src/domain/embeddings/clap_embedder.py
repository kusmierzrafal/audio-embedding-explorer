from __future__ import annotations

import numpy as np
import streamlit as st
import torch
from transformers import ClapModel, ClapProcessor

from src.config.error_messages import ERROR_MSG
from src.domain.embeddings.base_embedders import AudioEmbedder, TextEmbedder
from src.models.dataclasses.embedding_result import EmbeddingResult
from src.models.enums.modalities import Modality


class ClapEmbedder(AudioEmbedder, TextEmbedder):
    def __init__(self, model_name: str, device: str) -> None:
        self._model_name = model_name
        self._device = device

    def load(self):
        self._model, self._processor = load_weights_cached(
            self._model_name, self._device
        )

    def get_modalities(self) -> list[Modality]:
        return [Modality.AUDIO, Modality.TEXT]

    def embed_text(self, text: str) -> EmbeddingResult:
        if not self._model or not self._processor:
            raise RuntimeError(ERROR_MSG["MODEL_NOT_LOADED"])

        if not text.strip():
            raise ValueError(ERROR_MSG["EMPTY_TEXT_INPUT"])

        text_input = self._processor(text, return_tensors="pt")

        with torch.no_grad():
            features = self._model.get_text_features(**text_input)
            normalized_features = torch.nn.functional.normalize(features, p=2, dim=-1)

        return EmbeddingResult(vector=features, normalized_vector=normalized_features)

    def embed_audio(self, waveform: np.ndarray, sr: int) -> EmbeddingResult:
        if not self._model or not self._processor:
            raise RuntimeError(ERROR_MSG["MODEL_NOT_LOADED"])

        audio_input = self._processor(
            audio=[waveform], return_tensors="pt", sampling_rate=sr
        )

        with torch.no_grad():
            features = self._model.get_audio_features(**audio_input)
            normalized_features = torch.nn.functional.normalize(features, p=2, dim=-1)

        return EmbeddingResult(vector=features, normalized_vector=normalized_features)

    def get_sr(self) -> int:
        return 48000


@st.cache_resource(show_spinner=False)
def load_weights_cached(model_name: str, device: str):
    processor = ClapProcessor.from_pretrained(model_name)
    model = ClapModel.from_pretrained(model_name).to(device)
    return model, processor
