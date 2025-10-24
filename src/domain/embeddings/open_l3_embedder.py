from pathlib import Path
import torch
import torch.nn.functional as F
from openl3 import models, get_audio_embedding
from src.models.dataclasses.embedding_result import EmbeddingResult
from src.config.error_messages import ERROR_MSG
from src.utils.audio_utils import AudioHelper
from src.domain.embeddings.base_embedder import AudioEmbedder

class OpenL3Embedder(AudioEmbedder):
    def __init__(self, model_id: str, input_repr="mel256", embedding_size=512):
        self.model_id = model_id
        self.model = models.load_audio_embedding_model(
            input_repr=input_repr,
            content_type="music",
            embedding_size=embedding_size
        )
        self.input_repr = input_repr
        self.embedding_size = embedding_size

    def embed_audio(self, audio_path: Path, mode="global", hop_size=0.1) -> EmbeddingResult:
        if not audio_path.exists():
            raise FileNotFoundError(ERROR_MSG["AUDIO_FILE_NOT_FOUND"])
        
        waveform, sr = AudioHelper.load_audio(audio_path, target_sr=48000)
        embeddings, timestamps = get_audio_embedding(
            waveform, sr,
            model=self.model,
            hop_size=hop_size,
            center=True
        )
        embeddings = torch.from_numpy(embeddings)
        embeddings = F.normalize(embeddings, p=2, dim=-1)

        if mode == "global":
            embeddings = embeddings.mean(dim=0, keepdim=True)
        elif mode != "frames":
            raise ValueError(ERROR_MSG["INVALID_AUDIO_EMBEDDING_MODE"])

        return EmbeddingResult(vectors=[embeddings], source="audio", model_name=self.model_id)
