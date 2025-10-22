from pathlib import Path
from src.models.dataclasses.project_paths import ProjectPaths


def resolve_paths() -> ProjectPaths:
    repo_root = Path(__file__).resolve().parents[2]
    return ProjectPaths(
        repo_root=repo_root,
        models_dir=repo_root / "models",
        dataset_dir=repo_root / "dataset",
    )