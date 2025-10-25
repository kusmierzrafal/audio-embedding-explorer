from dataclasses import dataclass


@dataclass(frozen=True)
class ModelEnv:
    clap_hf_name: str
    clap_dir_name: str
    mert_hf_name: str
    mert_dir_name: str
