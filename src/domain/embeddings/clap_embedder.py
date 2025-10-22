from __future__ import annotations
from pathlib import Path
import torch
import librosa
from huggingface_hub import snapshot_download
from transformers import ClapModel, AutoProcessor, AutoTokenizer

from src.config.error_messages import ERROR_MSG
from src.domain.embeddings.base_embedder import BaseEmbedder
from src.models.dataclasses.embedding_result import EmbeddingResult


class ClapEmbedder(BaseEmbedder):
    def ensure_exists(self) -> None:
        if not self.model_dir.exists():
            self.model_dir.mkdir(parents=True, exist_ok=True)
            snapshot_download(repo_id=self.model_id, local_dir=self.model_dir)

    def load_model(self) -> None:
        self.ensure_exists()
        self.model = ClapModel.from_pretrained(self.model_dir)
        self.processor = AutoProcessor.from_pretrained(self.model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)

    def embed_text(self, text: str) -> EmbeddingResult:
        if not self.model or not self.tokenizer:
            raise RuntimeError(ERROR_MSG['MODEL_NOT_LOADED'])

        if not text.strip():
            raise ValueError(ERROR_MSG['EMPTY_TEXT_INPUT'])

        text_input = self.tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            features = self.model.get_text_features(**text_input)

        features = torch.nn.functional.normalize(features, p=2, dim=-1)
        return EmbeddingResult(vector=features, source="text", model_name=self.model_id)

    def embed_audio(self, audio_path: Path) -> EmbeddingResult:
        if not self.model or not self.processor:
            raise RuntimeError(ERROR_MSG['MODEL_NOT_LOADED'])

        if not audio_path.exists():
            raise FileNotFoundError(f"{ERROR_MSG['MODEL_NOT_LOADED']} {audio_path}")

        waveform, sr = librosa.load(audio_path, sr=48000, mono=True)
        waveform = waveform.astype("float32")

        audio_input = self.processor(audio=[waveform], return_tensors="pt", sampling_rate=sr)
        with torch.no_grad():
            features = self.model.get_audio_features(**audio_input)

        features = torch.nn.functional.normalize(features, p=2, dim=-1)
        return EmbeddingResult(vector=features, source="audio", model_name=self.model_id)
