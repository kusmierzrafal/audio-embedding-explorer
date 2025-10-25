from pathlib import Path
import torch
import torch.nn.functional as F
from openl3 import models, get_audio_embedding
from src.models.dataclasses.embedding_result import EmbeddingResult
from src.config.error_messages import ERROR_MSG
from src.utils.audio_utils import AudioHelper
from src.domain.embeddings.base_embedders import AudioEmbedder

class OpenL3Embedder(AudioEmbedder):
    def __init__(self, input_repr="mel256", embedding_size=512):
        self._hop_size = 0.1  # seconds
        self.model = models.load_audio_embedding_model(
            input_repr=input_repr,
            content_type="music",
            embedding_size=embedding_size
        )
        self.input_repr = input_repr
        self.embedding_size = embedding_size

    def embed_audio(self, audio_path: Path) -> EmbeddingResult:
        if not audio_path.exists():
            raise FileNotFoundError(ERROR_MSG["AUDIO_FILE_NOT_FOUND"])

        waveform, sr = AudioHelper.load_audio(audio_path, target_sr=48000)
        embeddings, timestamps = get_audio_embedding(
            waveform, sr,
            model=self.model,
            hop_size=self._hop_size,
            center=True
        )
        embeddings = torch.from_numpy(embeddings)
        global_embedding = embeddings.mean(dim=0, keepdim=True)

        normalized_embeddings = F.normalize(embeddings, p=2, dim=-1)
        # Compute the global embedding by averaging frame-level embeddings
        normalized_global_embedding = normalized_embeddings.mean(dim=0, keepdim=True)

        return EmbeddingResult(vector=global_embedding, normalized_vector=normalized_global_embedding, source="audio", model_name="openl3")