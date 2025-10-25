from pathlib import Path

import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from transformers import AutoModel, Wav2Vec2FeatureExtractor

from src.config.error_messages import ERROR_MSG
from src.domain.embeddings.base_embedders import AudioEmbedder
from src.models.dataclasses.embedding_result import EmbeddingResult
from src.utils.audio_utils import AudioHelper


class MERTEmbedder(AudioEmbedder):
    def __init__(self, model_id: str, model_dir: Path) -> None:
        self.model_id = model_id
        if not model_dir.exists():
            model_dir.mkdir(parents=True, exist_ok=True)
            snapshot_download(repo_id=model_id, local_dir=model_dir)

        self.model = AutoModel.from_pretrained(model_id, trust_remote_code=True)
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(
            model_id, trust_remote_code=True
        )

    def embed_audio(self, audio_path: Path) -> EmbeddingResult:
        if not audio_path.exists():
            raise FileNotFoundError(ERROR_MSG["AUDIO_FILE_NOT_FOUND"])

        waveform, sr = AudioHelper.load_audio(audio_path, target_sr=24000)
        inputs = self.processor(waveform, sampling_rate=sr, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)

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
            source="audio",
            model_name=self.model_id,
        )
