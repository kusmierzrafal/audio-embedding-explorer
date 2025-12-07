from src.domain.embeddings.base_embedders import BaseEmbedder


class StoredModel:
    def __init__(
        self,
        name: str,
        embedder: BaseEmbedder,
        description: str,
        type: str
    ):
        self.id = id
        self.name = name
        self.embedder = embedder
        self.description = description
        self.type = type
        self.is_loaded = False