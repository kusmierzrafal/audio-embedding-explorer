from src.domain.embeddings.stored_model import StoredModel


def model_status_label(model: StoredModel) -> str:
    if model.is_loaded:
        return "Loaded ✅"
    if not model.is_available:
        return "Unavailable ⛔"
    return "Available ⬜"
