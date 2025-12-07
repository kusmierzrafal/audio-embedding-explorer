import os

from dotenv import load_dotenv

from src.models.dataclasses.model_env import ModelEnv


def load_model_env() -> ModelEnv:
    load_dotenv()
    return ModelEnv(
        database_url=os.getenv("DATABASE_URL"),
        device=os.getenv("DEVICE"),
    )
