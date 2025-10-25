import os

from dotenv import load_dotenv

from src.config.env_keys import CLAP_DIR_NAME, CLAP_HF_NAME, MERT_DIR_NAME, MERT_HF_NAME
from src.models.dataclasses.model_env import ModelEnv


def load_model_env() -> ModelEnv:
    load_dotenv()
    return ModelEnv(
        clap_hf_name=os.getenv("CLAP_HF_NAME", "laion/clap-htsat-fused"),
        clap_dir_name=os.getenv("CLAP_DIR_NAME", "clap-htsat-fused"),
        mert_hf_name=os.getenv("MERT_HF_NAME", "m-a-p/MERT-v1-95M"),
        mert_dir_name=os.getenv("MERT_DIR_NAME", "mert"),
    )