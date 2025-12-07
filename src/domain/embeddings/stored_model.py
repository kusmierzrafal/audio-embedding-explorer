from src.domain.embeddings.base_embedders import BaseEmbedder


class StoredModel:
    def __init__(
        self,
        name: str,
        embedder: BaseEmbedder | None,
        description: str,
        type: str,
        is_imported: bool
    ):
        self.name = name
        self.embedder = embedder
        self.description = description
        self.type = type
        self.is_loaded = False
        self.is_imported = is_imported