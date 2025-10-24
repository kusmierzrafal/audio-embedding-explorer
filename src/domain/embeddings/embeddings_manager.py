from pathlib import Path
from typing import Any, Dict
from src.domain.embeddings.mert_embedder import MERTEmbedder
from src.domain.embeddings.open_l3_embedder import OpenL3Embedder
from src.domain.embeddings.clap_embedder import ClapEmbedder
# from src.domain.embeddings.panns_embedder import PANNS_Embedder
from src.models.enums.available_models import AvailableModel


class ModelsManager:
    def __init__(self, work_dir: Path):
        self._models: Dict[str, Any] = {}
        self.work_dir = work_dir

    def get_model(self, model_name: AvailableModel):
        if model_name in self._models:
            return self._models[model_name]

        if model_name == AvailableModel.CLAP:
            model = ClapEmbedder(model_id="laion/clap-htsat-fused", work_dir=self.work_dir / "clap")
        # elif model_name == AvailableModel.PANNS:
            # model = PANNS_Embedder(model_id="qiuqiangkong/panns_cnn14", work_dir=self.work_dir / "panns")
        elif model_name == AvailableModel.OPENL3:
            model = OpenL3Embedder(model_id="laion/openl3")
        elif model_name == AvailableModel.MERT:
            model = MERTEmbedder(model_id="m-a-p/MERT-v1-95M", work_dir=self.work_dir / "mert")
        else:
            raise ValueError(f"Invalid model name: {model_name}")

        self._models[model_name] = model
        return model


    @staticmethod
    def get_embedder_type(embedder) -> str:
        has_audio = hasattr(embedder, "embed_audio")
        has_text = hasattr(embedder, "embed_text")

        if has_audio and has_text:
            return "multimodal"
        elif has_audio:
            return "audio"
        elif has_text:
            return "text"
        else:
            raise RuntimeError("Embedder does not implement embed_audio or embed_text")