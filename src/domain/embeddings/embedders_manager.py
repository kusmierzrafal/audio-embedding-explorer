from pathlib import Path
from typing import Any, Dict

from src.domain.embeddings.clap_embedder import ClapEmbedder
from src.domain.embeddings.mert_embedder import MERTEmbedder
from src.domain.embeddings.openl3_embedder import OpenL3Embedder
from src.models.dataclasses.model_env import ModelEnv
from src.models.enums.embedders_models import EmbedderModel


class EmbeddersManager:
    def __init__(self, models_path: Path, env: ModelEnv) -> None:
        self._models: Dict[str, Any] = {}
        self.models_dir = models_path
        self.env = env

    def get_embedders_lookup(self) -> list[str]:
        return [m.value for m in EmbedderModel]

    def get_embedder(self, model_name: EmbedderModel):
        if model_name in self._models:
            return self._models[model_name]

        if model_name == EmbedderModel.CLAP:
            model = ClapEmbedder(
                model_id=self.env.clap_hf_name,
                model_dir=self.models_dir / self.env.clap_dir_name)
        elif model_name == EmbedderModel.OPENL3:
            model = OpenL3Embedder()
        elif model_name == EmbedderModel.MERT:
            model = MERTEmbedder(
                model_id=self.env.mert_hf_name, 
                model_dir=self.models_dir / self.env.mert_dir_name)
        else:
            raise ValueError(f"Invalid model name: {model_name}")

        self._models[model_name] = model
        return model


    @staticmethod
    def get_embedder_modalities(embedder) -> list[str]:
        modalities = []
        if hasattr(embedder, "embed_audio"):
            modalities.append("audio")
        if hasattr(embedder, "embed_text"):
            modalities.append("text")
        return modalities