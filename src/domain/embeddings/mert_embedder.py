import numpy as np
import streamlit as st
import torch
import torch.nn.functional as F
from transformers import AutoModel, Wav2Vec2FeatureExtractor

from src.domain.embeddings.base_embedders import AudioEmbedder
from src.models.dataclasses.embedding_result import EmbeddingResult
from src.models.enums.modalities import Modality


class MERTEmbedder(AudioEmbedder):
    def __init__(self, model_name: str, device: str) -> None:
        self.model_name = model_name
        self.device = device

    def load(self):
        self._model, self._processor = load_weights_cached(self.model_name, self.device)

    def get_modalities(self) -> list[Modality]:
        return [Modality.AUDIO]

    def embed_audio(self, waveform: np.ndarray, sr: int) -> EmbeddingResult:
        inputs = self._processor(waveform, sampling_rate=sr, return_tensors="pt")
        # Move input tensors to model device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs, output_hidden_states=True)

        # [layers, batch, time, feature_dim] == [13, 1, T, 768]
        all_layer_hidden_states = torch.stack(outputs.hidden_states).squeeze(1)
        # reduce the representation in time => [13, 768]
        time_reduced_hidden_states = all_layer_hidden_states.mean(dim=1)

        per_layer_embeddings = F.normalize(time_reduced_hidden_states, p=2, dim=-1)
        # [768]
        global_embedding = per_layer_embeddings.mean(dim=0)
        normalized_global_embedding = F.normalize(global_embedding, p=2, dim=-1)

        global_vec = global_embedding.unsqueeze(0)  # [1, 768]
        normalized_global_vec = normalized_global_embedding.unsqueeze(0)  # [1, 768]

        return EmbeddingResult(
            vector=global_vec,
            normalized_vector=normalized_global_vec,
        )

    def get_sr(self) -> int:
        return 24000


@st.cache_resource(show_spinner=False)
def load_weights_cached(model_name: str, device: str):
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    processor = Wav2Vec2FeatureExtractor.from_pretrained(
        model_name, trust_remote_code=True
    )

    # Handle meta tensor issue with newer PyTorch versions
    try:
        model = model.to(device)
    except NotImplementedError as e:
        if "meta tensor" in str(e):
            # Use to_empty() for models loaded with meta tensors
            model = model.to_empty(device=device)
        else:
            raise

    return model, processor
