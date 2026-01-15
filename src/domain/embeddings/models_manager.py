import importlib
import logging
from typing import Dict, List, Tuple, Type

import torch

from src.domain.embeddings.stored_model import StoredModel

logger = logging.getLogger(__name__)


def optional_import(module_path: str, class_name: str) -> tuple[Type | None, bool]:
    try:
        module = importlib.import_module(module_path)
        return getattr(module, class_name), True
    except ImportError:
        return None, False
    except Exception:
        logger.exception("Error while importing %s.%s", module_path, class_name)
        return None, False


class ModelsManager:
    def __init__(self, device: str | None) -> None:
        if not device:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.available_models: Dict[str, StoredModel] = {
            "laion/clap-htsat-unfused": StoredModel(
                name="laion/clap-htsat-unfused",
                embedder=None,
                description=(
                    "Best for music. Treats the audio input as a whole "
                    "(or trims to a specified length) and generates a vector. "
                    "Does not use a complex mechanism for combining fragments (fusion) "
                    "from different time windows."
                ),
                type="audio-text",
            ),
            "laion/clap-htsat-fused": StoredModel(
                name="laion/clap-htsat-fused",
                embedder=None,
                description=(
                    "Best for ambient sounds, effects. It is designed "
                    "to better handle variable-length or longer recordings. "
                    "It splits the audio into smaller fragments (windows/patches), "
                    "analyzes them independently, and then fuses the information from "
                    "these fragments into a single final vector."
                ),
                type="audio-text",
            ),
            "m-a-p/MERT-v1-95M": StoredModel(
                name="m-a-p/MERT-v1-95M",
                embedder=None,
                description=(
                    "MERT is a state-of-the-art model for general "
                    "audio embeddings, excelling in various tasks including "
                    "acoustic scene classification, music genre recognition, "
                    "and sound event detection."
                ),
                type="audio",
            ),
            "openl3-mel256-512": StoredModel(
                name="openl3-mel256-512",
                embedder=None,
                description=(
                    "OpenL3 is a deep audio embedding model "
                    "that generates fixed-length vector representations of audio "
                    "clips. It is designed to capture high-level semantic features "
                    "from audio data, making it useful for tasks such as "
                    "audio classification, retrieval, and similarity analysis."
                ),
                type="audio",
            ),
        }

    def _get(self, id: str) -> StoredModel | None:
        return self.available_models.get(id)

    def is_loaded(self, id: str) -> bool:
        model = self._get(id)
        return model.is_loaded if model else False

    def load_model(self, id: str) -> None:
        model = self._get(id)
        if not model or model.is_loaded:
            return

        if id.startswith("laion/clap"):
            embedderCls, ok = optional_import(
                "src.domain.embeddings.clap_embedder", "ClapEmbedder"
            )
            if not ok:
                model.is_available = False
                return
            model.embedder = embedderCls(model_name=id, device=self.device)

        elif id == "m-a-p/MERT-v1-95M":
            embedderCls, ok = optional_import(
                "src.domain.embeddings.mert_embedder", "MERTEmbedder"
            )
            if not ok:
                model.is_available = False
                return
            model.embedder = embedderCls(model_name=id, device=self.device)

        elif id == "openl3-mel256-512":
            embedderCls, ok = optional_import(
                "src.domain.embeddings.openl3_embedder", "OpenL3Embedder"
            )
            if not ok:
                model.is_available = False
                return
            model.embedder = embedderCls(device=self.device)

        model.embedder.load()
        model.is_loaded = True

    def unload_model(self, id: str) -> None:
        model = self._get(id)
        if not model or not model.is_loaded:
            return

        model.embedder.unload()
        model.embedder = None
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

    def has_any_loaded_model(self) -> bool:
        return any(m.is_loaded for m in self.available_models.values())
