import io

import numpy as np
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

from src.domain.db_manager import DbManager
from src.domain.embeddings.base_embedders import (
    AudioEmbedder,
    TextEmbedder,
)
from src.domain.embeddings.models_manager import ModelsManager
from src.domain.visualization import plotly_pca_projection
from src.models.enums.modalities import Modality
from src.models.enums.reduce_dimensions_method import ReduceDimensionsMethod
from src.ui.shared.audio_edit_view import AudioEditView
from src.ui.shared.base_view import BaseView

# Models that support text input
TEXT_CAPABLE_MODELS = {
    "laion/clap-htsat-unfused",
    "laion/clap-htsat-fused",
}


class EmbeddingsPlaygroundView(BaseView):
    title = "Embedding Playground"
    description = (
        "Analyze a single audio-text or audio-audio pair using selected models. "
        "Uses PCA for visualization. You can also load multiple embeddings "
        "into a common space for comparison."
    )

    def render(self) -> None:
        self.header()

        models_manager: ModelsManager = st.session_state["models_manager"]

        # Get all loaded models
        all_loaded_models = models_manager.get_loaded_models_ids_and_names()

        # Filter for text-capable models only
        text_capable_loaded = [
            (model_id, name)
            for model_id, name in all_loaded_models
            if model_id in TEXT_CAPABLE_MODELS
        ]

        if not all_loaded_models:
            st.info(
                "There's no any loaded model. Please load a model in the home view."
            )
            return

        if not text_capable_loaded:
            st.warning(
                "No audio-text capable models loaded. "
                "This playground requires CLAP models. "
                "Please load a CLAP model from the Home page."
            )
            return

        model_id, _ = st.selectbox(
            "Choose model",
            options=text_capable_loaded,
            format_func=lambda x: x[1],
        )

        if model_id is None:
            st.info("Please load and select a model to continue.")
            return

        model = models_manager.get_model(model_id)
        sr = (
            model.embedder.get_sr()
            if Modality.AUDIO in model.embedder.get_modalities()
            else 48000
        )

        st.markdown("---")
        if (
            Modality.AUDIO in model.embedder.get_modalities()
            and Modality.TEXT in model.embedder.get_modalities()
        ):
            self.audio_text_comparison(model.embedder, sr=sr)
        elif Modality.AUDIO in model.embedder.get_modalities():
            self.single_audio(model.embedder, sr=sr)

    def audio_text_comparison(
        self, embedder: AudioEmbedder | TextEmbedder, sr: int
    ) -> None:
        text = st.text_area("Text prompt", placeholder="e.g. calm piano")

        db_manager: DbManager = st.session_state["db_manager"]
        audio_source = st.radio(
            "Audio source",
            ["File Upload", "Database"],
            horizontal=True,
            disabled=not db_manager.is_connected,
        )
        audio_bytes = None
        audio_name = None

        if audio_source == "File Upload":
            uploaded_file: io.BytesIO = st.file_uploader(
                "Load audio", type=["wav", "mp3", "flac"]
            )
            if uploaded_file:
                audio_bytes = uploaded_file
                audio_name = uploaded_file.name

                if st.button("Save to database"):
                    data = uploaded_file.getvalue()
                    if db_manager.insert_audio_if_not_exists(audio_name, data):
                        st.success(f"Saved '{audio_name}' to database.")
                    else:
                        st.info(f"'{audio_name}' already exists in database.")
        else:
            db_audio_files = db_manager.get_audio_files()
            if not db_audio_files:
                st.warning("No audio files found in the database.")
            else:
                selected_audio = st.selectbox(
                    "Select audio from database",
                    options=[(name, id) for id, name in db_audio_files],
                    format_func=lambda x: x[0],
                )
                if selected_audio:
                    audio_name, audio_id = selected_audio
                    audio_bytes, audio_name = db_manager.get_audio_data(audio_id)

        if audio_bytes:
            audio_view = AudioEditView(audio_name, audio_bytes=audio_bytes, sr=sr)
            audio_view.render()

            # Add button to save edited audio to database
            cache = st.session_state.get(f"audio_edit_{audio_name}")
            if cache and db_manager.is_connected:
                # Check if audio has been edited
                is_edited = (
                    cache.speed_rate != 1.0
                    or cache.pitch_steps != 0
                    or cache.noise_amount != 0.0
                )
                if st.button(
                    "Save edited audio",
                    key="save_edited_audio_single",
                    disabled=not is_edited,
                ):
                    from src.utils.audio_utils import AudioHelper

                    # Convert edited audio to bytes
                    edited_bytes = AudioHelper.samples_to_bytes(cache.latest_y, sr=sr)
                    # Generate new name for edited audio
                    edited_name = (
                        f"{audio_name.rsplit('.', 1)[0]}_edited_"
                        + f"{cache.speed_rate}_{cache.pitch_steps}_"
                        + f"{cache.noise_amount}.wav"
                    )

                    if db_manager.insert_audio_if_not_exists(edited_name, edited_bytes):
                        st.success(f"Saved '{edited_name}' to database.")
                        is_edited = False
                    else:
                        st.info(f"'{edited_name}' already exists in database.")

        if st.button(
            "Generate embedding",
            disabled=audio_bytes is None
            or audio_view.latest_y is None
            or not text.strip(),
        ):
            with st.spinner("Generating embeddings and computing similarity..."):
                text_emb = embedder.embed_text(text)
                # Use database caching for audio embedding
                db_manager = st.session_state.get("db_manager")
                if db_manager and audio_bytes:
                    # Get model name from embedder
                    model_name = getattr(embedder, "model_name", "unknown")
                    audio_data = (
                        audio_bytes.getvalue()
                        if hasattr(audio_bytes, "getvalue")
                        else audio_bytes
                    )
                    audio_emb_vector = db_manager.get_or_compute_audio_embedding(
                        embedder,
                        audio_data,
                        getattr(audio_view, "file_name", "unknown"),
                        sr,
                        model_name,
                    )

                    # Create EmbeddingResult-like object
                    class EmbeddingWrapper:
                        def __init__(self, vector):
                            self.vector = vector

                    audio_emb = EmbeddingWrapper(audio_emb_vector)
                else:
                    # Fallback to direct computation
                    audio_emb = embedder.embed_audio(audio_view.latest_y, sr=sr)

                # Convert to numpy arrays and reshape for sklearn
                audio_emb_array = np.asarray(audio_emb.vector).reshape(1, -1)
                text_emb_array = np.asarray(text_emb.vector).reshape(1, -1)
                similarity = cosine_similarity(audio_emb_array, text_emb_array)[0][0]

                st.metric("Cosine similarity", f"{similarity:.4f}")

                st.subheader("2D PCA projection")
                plotly_pca_projection(
                    audio_emb.vector,
                    text_emb.vector,
                    method=ReduceDimensionsMethod.PCA,
                )
        st.markdown("---")

    def single_audio(self, embedder: AudioEmbedder, sr: int) -> None:
        db_manager: DbManager = st.session_state["db_manager"]
        audio_source = st.radio(
            "Audio source",
            ["File Upload", "Database"],
            horizontal=True,
            key="single_audio_source",
            disabled=not db_manager.is_connected,
        )
        audio_bytes = None
        audio_name = None

        if audio_source == "File Upload":
            uploaded_file: io.BytesIO = st.file_uploader(
                "Load audio",
                type=["wav", "mp3", "flac"],
                key="single_audio_upload",
            )
            if uploaded_file:
                audio_bytes = uploaded_file
                audio_name = uploaded_file.name

                if st.button("Save to database", key="single_save_to_db"):
                    data = uploaded_file.getvalue()
                    if db_manager.insert_audio_if_not_exists(audio_name, data):
                        st.success(f"Saved '{audio_name}' to database.")
                    else:
                        st.info(f"'{audio_name}' already exists in database.")
        else:
            db_audio_files = db_manager.get_audio_files()
            if not db_audio_files:
                st.warning("No audio files found in the database.")
            else:
                selected_audio = st.selectbox(
                    "Select audio from database",
                    options=[(name, id) for id, name in db_audio_files],
                    format_func=lambda x: x[0],
                    key="single_audio_db_select",
                )
                if selected_audio:
                    audio_name, audio_id = selected_audio
                    audio_bytes, audio_name = db_manager.get_audio_data(audio_id)

        if audio_bytes:
            audio_view = AudioEditView(audio_name, audio_bytes=audio_bytes, sr=sr)
            audio_view.render()

            # Add button to save edited audio to database
            cache = st.session_state.get(f"audio_edit_{audio_name}")
            if cache and db_manager.is_connected:
                # Check if audio has been edited
                is_edited = (
                    cache.speed_rate != 1.0
                    or cache.pitch_steps != 0
                    or cache.noise_amount != 0.0
                )
                if st.button(
                    "Save edited audio",
                    key="save_edited_audio_single",
                    disabled=not is_edited,
                ):
                    from src.utils.audio_utils import AudioHelper

                    # Convert edited audio to bytes
                    edited_bytes = AudioHelper.samples_to_bytes(cache.latest_y, sr=sr)
                    # Generate new name for edited audio
                    edited_name = (
                        f"{audio_name.rsplit('.', 1)[0]}_edited_"
                        + f"{cache.speed_rate}_{cache.pitch_steps}_"
                        + f"{cache.noise_amount}.wav"
                    )

                    if db_manager.insert_audio_if_not_exists(edited_name, edited_bytes):
                        st.success(f"Saved '{edited_name}' to database.")
                        is_edited = False
                    else:
                        st.info(f"'{edited_name}' already exists in database.")

        if st.button(
            "Generate embedding",
            disabled=audio_bytes is None or audio_view.latest_y is None,
        ):
            with st.spinner("Generating embedding..."):
                # Use database caching for audio embedding
                db_manager = st.session_state.get("db_manager")
                if (
                    db_manager
                    and hasattr(audio_view, "latest_file_data")
                    and audio_view.latest_file_data
                ):
                    # Get model name from embedder
                    model_name = getattr(embedder, "model_name", "unknown")
                    audio_emb_vector = db_manager.get_or_compute_audio_embedding(
                        embedder,
                        audio_view.latest_file_data,
                        getattr(audio_view, "latest_file_name", "unknown"),
                        sr,
                        model_name,
                    )

                    # Create EmbeddingResult-like object
                    class EmbeddingWrapper:
                        def __init__(self, vector):
                            self.vector = vector

                    audio_emb = EmbeddingWrapper(audio_emb_vector)
                else:
                    # Fallback to direct computation
                    audio_emb = embedder.embed_audio(audio_view.latest_y, sr=sr)
                st.info("PCA for single embedding not implemented yet.")
                st.text(audio_emb.vector)
        st.markdown("---")
