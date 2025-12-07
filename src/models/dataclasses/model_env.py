from dataclasses import dataclass


@dataclass(frozen=True)
class ModelEnv:
    database_url: str
    device: str
