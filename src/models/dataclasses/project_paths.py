from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    repo_root: Path
    models_dir: Path
    dataset_dir: Path
