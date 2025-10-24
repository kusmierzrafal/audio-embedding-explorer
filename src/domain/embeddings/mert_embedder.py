from pathlib import Path
from huggingface_hub import snapshot_download
import torch
import torch.nn.functional as F
from transformers import AutoModel, Wav2Vec2FeatureExtractor

from src.config.error_messages import ERROR_MSG
from src.models.dataclasses.embedding_result import EmbeddingResult
from src.domain.embeddings.base_embedder import AudioEmbedder
from src.utils.audio_utils import AudioHelper


class MERTEmbedder(AudioEmbedder):
    def __init__(self, model_id: str, work_dir: Path) -> None:
        self.model_id = model_id
        if not work_dir.exists():
            work_dir.mkdir(parents=True, exist_ok=True)
            snapshot_download(repo_id=model_id, local_dir=work_dir)
        
        self.model = AutoModel.from_pretrained(model_id, trust_remote_code=True)
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(model_id, trust_remote_code=True)


    def embed_audio(self, audio_path: Path) -> EmbeddingResult:
        if not audio_path.exists():
            raise FileNotFoundError(ERROR_MSG["AUDIO_FILE_NOT_FOUND"])

        waveform, sr = AudioHelper.load_audio(audio_path, target_sr=16000)
        inputs = self.processor(waveform, sampling_rate=sr, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)

        # take a look at the output shape, there are 13 layers of representation
        # each layer performs differently in different downstream tasks, you should choose empirically
        all_layer_hidden_states = torch.stack(outputs.hidden_states).squeeze()
        print(all_layer_hidden_states.shape) # [13 layer, Time steps, 768 feature_dim]

        # for utterance level classification tasks, you can simply reduce the representation in time
        time_reduced_hidden_states = all_layer_hidden_states.mean(-2)
        print(time_reduced_hidden_states.shape) # [13, 768]

        # you can even use a learnable weighted average representation
        # aggregator = nn.Conv1d(in_channels=13, out_channels=1, kernel_size=1)
        # weighted_avg_hidden_states = aggregator(time_reduced_hidden_states.unsqueeze(0)).squeeze()
        # print(weighted_avg_hidden_states.shape) # [768]

        embeddings = F.normalize(time_reduced_hidden_states, p=2, dim=-1)
        return EmbeddingResult(vectors=[embeddings], source="audio", model_name=self.model_id)
