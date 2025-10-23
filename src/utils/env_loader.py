import os

from dotenv import load_dotenv

from src.config.env_keys import CLAP_DIR_NAME, CLAP_HF_NAME
from src.models.dataclasses.model_env import ModelEnv


def load_model_env() -> ModelEnv:
    load_dotenv()
    return ModelEnv(
        clap_hf_name=os.environ[CLAP_HF_NAME],
        clap_dir_name=os.environ[CLAP_DIR_NAME],
    )
