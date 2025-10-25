from __future__ import annotations

from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from transformers import AutoProcessor, AutoTokenizer, ClapModel

from src.config.error_messages import ERROR_MSG
from src.domain.embeddings.base_embedders import AudioEmbedder, TextEmbedder
from src.models.dataclasses.embedding_result import EmbeddingResult
from src.utils.audio_utils import AudioHelper


class ClapEmbedder(AudioEmbedder, TextEmbedder):
    def __init__(self, model_id: str, model_dir: Path) -> None:
        self.model_id = model_id
        if not model_dir.exists():
            model_dir.mkdir(parents=True, exist_ok=True)
            snapshot_download(repo_id=self.model_id, local_dir=model_dir)

        self.model = ClapModel.from_pretrained(model_dir)
        self.processor = AutoProcessor.from_pretrained(model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)

    def embed_text(self, text: str) -> EmbeddingResult:
        if not self.model or not self.tokenizer:
            raise RuntimeError(ERROR_MSG["MODEL_NOT_LOADED"])

        if not text.strip():
            raise ValueError(ERROR_MSG["EMPTY_TEXT_INPUT"])

        text_input = self.tokenizer(text, return_tensors="pt")

        with torch.no_grad():
            features = self.model.get_text_features(**text_input)
            normalized_features = torch.nn.functional.normalize(features, p=2, dim=-1)

        return EmbeddingResult(
            vector=features,
            normalized_vector=normalized_features,
            source="text",
            model_name=self.model_id,
        )

    def embed_audio(self, audio_path: Path) -> EmbeddingResult:
        if not self.model or not self.processor:
            raise RuntimeError(ERROR_MSG["MODEL_NOT_LOADED"])

        if not audio_path.exists():
            raise FileNotFoundError(f"{ERROR_MSG['MODEL_NOT_LOADED']} {audio_path}")

        waveform, sr = AudioHelper.load_audio(audio_path, target_sr=48000)
        audio_input = self.processor(
            audio=[waveform], return_tensors="pt", sampling_rate=sr
        )

        with torch.no_grad():
            features = self.model.get_audio_features(**audio_input)
            normalized_features = torch.nn.functional.normalize(features, p=2, dim=-1)

        return EmbeddingResult(
            vector=features,
            normalized_vector=normalized_features,
            source="audio",
            model_name=self.model_id,
        )
