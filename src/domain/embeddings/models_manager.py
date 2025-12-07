from typing import Dict, List, Tuple

try:
    from src.domain.embeddings.clap_embedder import ClapEmbedder

    clap_imported = True
except Exception:
    clap_imported = False

try:
    from src.domain.embeddings.mert_embedder import MERTEmbedder

    mert_imported = True
except Exception:
    mert_imported = False

try:
    from src.domain.embeddings.openl3_embedder import OpenL3Embedder

    openl3_imported = True
except Exception:
    openl3_imported = False

from src.domain.embeddings.stored_model import StoredModel


class ModelsManager:
    def __init__(self, device: str) -> None:
        self.device: str = device
        self.available_models: Dict[str, StoredModel] = {}

        self.available_models["laion/clap-htsat-unfused"] = StoredModel(
            name="laion/clap-htsat-unfused",
            embedder=ClapEmbedder("laion/clap-htsat-unfused", device=self.device)
            if clap_imported
            else None,
            description=(
                "Best for music. Treats the audio input as a whole "
                "(or trims to a specified length) and generates a vector. "
                "Does not use a complex mechanism for combining fragments (fusion) "
                "from different time windows."
            ),
            type="audio-text",
            is_imported=clap_imported,
        )

        self.available_models["laion/clap-htsat-fused"] = StoredModel(
            name="laion/clap-htsat-fused",
            embedder=ClapEmbedder("laion/clap-htsat-fused", device=self.device)
            if clap_imported
            else None,
            description=(
                "Best for ambient sounds, effects. It is designed "
                "to better handle variable-length or longer recordings. "
                "It splits the audio into smaller fragments (windows/patches), "
                "analyzes them independently, and then fuses the information from "
                "these fragments into a single final vector. "
            ),
            type="audio-text",
            is_imported=clap_imported,
        )

        self.available_models["m-a-p/MERT-v1-95M"] = StoredModel(
            name="m-a-p/MERT-v1-95M",
            embedder=MERTEmbedder(model_name="m-a-p/MERT-v1-95M", device=self.device)
            if mert_imported
            else None,
            description="MERT is a state-of-the-art model for general "
            "audio embeddings, excelling in various tasks including "
            "acoustic scene classification, music genre recognition, "
            "and sound event detection.",
            type="audio",
            is_imported=mert_imported,
        )

        self.available_models["openl3-mel256-512"] = StoredModel(
            name="openl3-mel256-512",
            embedder=OpenL3Embedder() if openl3_imported else None,
            description="OpenL3 is a deep audio embedding model "
            "that generates fixed-length vector representations of audio "
            "clips. It is designed to capture high-level semantic features "
            "from audio data, making it useful for tasks such as "
            "audio classification, retrieval, and similarity analysis.",
            type="audio",
            is_imported=openl3_imported,
        )

    def _get(self, id: str) -> StoredModel | None:
        return self.available_models.get(id)

    def is_loaded(self, id: str) -> bool:
        model = self._get(id)
        return model.is_loaded if model else False

    def load_model(self, id: str) -> None:
        model = self._get(id)
        if not model or model.is_loaded:
            return

        model.embedder.load()
        model.is_loaded = True

    def unload_model(self, id: str) -> None:
        model = self._get(id)
        if not model or not model.is_loaded:
            return

        model.embedder.unload()
        model.is_loaded = False

        import gc

        gc.collect()

    def get_model(self, id: str) -> StoredModel | None:
        return self._get(id)

    def get_loaded_models(self) -> list[StoredModel]:
        return [m for m in self.available_models.values() if m.is_loaded]
    
    def get_loaded_models_ids_and_names(self) -> List[Tuple[str, str]]:
        results: List[Tuple[str, str]] = []
    
        for model_id, m in self.available_models.items():
            if m.is_loaded:
                results.append((model_id, m.name))
                
        return results