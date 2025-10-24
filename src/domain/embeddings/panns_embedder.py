from pathlib import Path
import torch
import torch.nn.functional as F
from panns_inference import AudioTagging, AudioEmbedding
from config.error_messages import ERROR_MSG
from models.dataclasses.embedding_result import EmbeddingResult
from src.domain.embeddings.base_embedder import AudioEmbedder
from src.utils.audio_utils import AudioHelper


class PANNS_Embedder(AudioEmbedder):
    def __init__(self, sample_rate=32000, device="cpu"):
        self.sr = sample_rate
        self.device = device
        self.model = AudioEmbedding(checkpoint_path=None, sample_rate=sample_rate, device=device)

    def embed_audio(self, audio_path: Path, mode="global", segment_duration=None, hop_size=1.0):
        if segment_duration is not None:
            raise NotImplementedError("TODO: Implement audio embedding with segmentation")
        
        if not audio_path.exists():
            raise FileNotFoundError(ERROR_MSG["AUDIO_FILE_NOT_FOUND"])

        waveform, sr = AudioHelper.load_audio(audio_path, target_sr=self.sr)
        
        audio_input = [waveform]
        with torch.no_grad():
            embeddings = self.model.inference(audio_input)  # shape: (batch, frames, embedding_dim)
        
        embeddings = torch.from_numpy(embeddings[0])
        embeddings = F.normalize(embeddings, p=2, dim=-1)

        if mode == "global":
            embeddings = embeddings.mean(dim=0, keepdim=True)
        elif mode != "frames":
            raise ValueError(ERROR_MSG["INVALID_AUDIO_EMBEDDING_MODE"])

        return EmbeddingResult(vectors=[embeddings], source="audio", model_name="panns")    
