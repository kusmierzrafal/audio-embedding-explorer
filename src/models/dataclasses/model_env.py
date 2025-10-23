from dataclasses import dataclass


@dataclass(frozen=True)
class ModelEnv:
    clap_hf_name: str
    clap_dir_name: str
