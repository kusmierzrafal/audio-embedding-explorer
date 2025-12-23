from dataclasses import dataclass


@dataclass(frozen=True)
class ModelOption:
    ui_name: str
    model_id: str
