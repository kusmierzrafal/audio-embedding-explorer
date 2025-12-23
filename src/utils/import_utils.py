from typing import Any, Tuple


def optional_import(module_path: str, name: str) -> Tuple[Any | None, bool]:
    try:
        module = __import__(module_path, fromlist=[name])
        return getattr(module, name), True
    except Exception:
        return None, False
