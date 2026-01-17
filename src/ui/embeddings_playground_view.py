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
from src.utils.audio_utils import AudioHelper, safe_tensor_to_numpy


class EmbeddingsPlaygroundView(BaseView):
    title = "Embeddings Playground"
    description = (
        "Analyze a single audio-text or audio-audio pair using selected models. "
        "Uses PCA for visualization. You can also load multiple embeddings "
        "into a common space for comparison."
    )

    def render(self) -> None:
        self.header()

        models_manager: ModelsManager = st.session_state["models_manager"]
        all_loaded_models = models_manager.get_loaded_models_ids_and_names()

        if not all_loaded_models:
            st.info(
                "There's no any loaded model. Please load a model in the home view."
            )
            return

        model_id, _ = st.selectbox(
            "Choose model",
            options=all_loaded_models,
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

        has_audio = Modality.AUDIO in model.embedder.get_modalities()
        has_text = Modality.TEXT in model.embedder.get_modalities()

        if has_audio and has_text:
            self.audio_text_comparison(model.embedder, sr=sr)
        elif has_audio:
            self.audio_audio_comparison(model.embedder, sr=sr)


    def audio_text_comparison(
        self, embedder: AudioEmbedder | TextEmbedder, sr: int
    ) -> None:
        text = st.text_input("Text prompt", placeholder="e.g. calm piano")
        db_manager: DbManager = st.session_state["db_manager"]

        audio_info = self._render_audio_selector("at_1", db_manager)

        latest_y = None
        if audio_info:
            latest_y = self._render_audio_editor(audio_info, "at_1", db_manager, sr)

        if st.button(
            "Generate embeddings",
            disabled=latest_y is None or not text.strip(),
            key="gen_emb_at",
        ):
            with st.spinner("Generating embeddings and computing similarity..."):
                text_emb = embedder.embed_text(text)
                audio_emb = embedder.embed_audio(latest_y, sr=sr)

                self._render_results(audio_emb, text_emb)

        st.markdown("---")


    def audio_audio_comparison(self, embedder: AudioEmbedder, sr: int) -> None:
        st.subheader("Audio-Audio Comparison")
        db_manager: DbManager = st.session_state["db_manager"]

        col1, col2 = st.columns(2)
        with col1:
            info1 = self._render_audio_selector("aa_1", db_manager)
        with col2:
            info2 = self._render_audio_selector("aa_2", db_manager)


        y1, y2 = None, None
        ed_col1, ed_col2 = st.columns(2)

        if info1:
            with ed_col1:
                y1 = self._render_audio_editor(info1, "aa_1", db_manager, sr)

        if info2:
            with ed_col2:
                y2 = self._render_audio_editor(info2, "aa_2", db_manager, sr)

        st.markdown("---")

        if st.button(
            "Compare Audios", disabled=y1 is None or y2 is None, key="compare_aa_final"
        ):
            with st.spinner("Computing embeddings..."):
                emb1 = embedder.embed_audio(y1, sr=sr)
                emb2 = embedder.embed_audio(y2, sr=sr)

                self._render_results(emb1, emb2)


    def _render_audio_selector(self, suffix: str, db_manager: DbManager):
        st.markdown("**Source Selection**")
        source = st.radio(
            "Source",
            ["File Upload", "Database"],
            key=f"src_{suffix}",
            horizontal=True,
            label_visibility="collapsed",
        )

        if source == "File Upload":
            f = st.file_uploader(
                "Upload", type=["wav", "mp3", "flac"], key=f"up_{suffix}"
            )
            if f:
                if st.button("Save original to DB", key=f"save_orig_{suffix}"):
                    if db_manager.insert_audio_if_not_exists(f.name, f.getvalue()):
                        st.success(f"Saved '{f.name}'.")
                    else:
                        st.info(f"'{f.name}' already exists.")
                return {"name": f.name, "bytes": f}
        else:
            raw_files = db_manager.get_audio_files()
            db_files = list(raw_files) if raw_files else []

            if db_files:
                sel = st.selectbox(
                    "Select DB",
                    options=[(n, i) for i, n in db_files],
                    format_func=lambda x: str(x[0]),
                    key=f"db_{suffix}",
                )
                if sel:
                    data, name = db_manager.get_audio_data(sel[1])
                    if data:
                        return {"name": name, "bytes": data}
            else:
                st.warning("No files in database.")

        return None


    def _render_audio_editor(
        self, info: dict, suffix: str, db_manager: DbManager, sr: int
    ):
        audio_name, audio_bytes = info["name"], info["bytes"]
        state_prefix = f"edit_state_{suffix}"

        last_file_key = f"last_file_{suffix}"
        if (
            last_file_key not in st.session_state
            or st.session_state[last_file_key] != audio_name
        ):
            if state_prefix in st.session_state:
                del st.session_state[state_prefix]
            st.session_state[last_file_key] = audio_name

        view = AudioEditView(audio_name, audio_bytes, sr, state_prefix)
        view.render()

        cache = st.session_state.get(state_prefix)
        if cache and (
            cache.speed_rate != 1.0
            or cache.pitch_steps != 0
            or cache.noise_amount != 0.0
        ):
            st.markdown("#### Save Edited")

            clean_name = str(audio_name).rsplit(".", 1)[0]
            default_name = (
                f"{clean_name}_edited_"
                f"{cache.speed_rate}_{cache.pitch_steps}_{cache.noise_amount}.wav"
            )

            custom_name = st.text_input(
                "Filename", value=default_name, key=f"save_name_{suffix}"
            )

            if st.button("Save Edited", key=f"save_btn_{suffix}"):
                new_bytes = AudioHelper.samples_to_bytes(cache.latest_y, sr=sr)
                if db_manager.insert_audio_if_not_exists(custom_name, new_bytes):
                    st.success(f"Saved '{custom_name}'")
                else:
                    st.info(f"'{custom_name}' already exists.")

        return view.latest_y


    def _render_results(self, emb1, emb2):
        vec1 = safe_tensor_to_numpy(emb1.vector).reshape(1, -1)
        vec2 = safe_tensor_to_numpy(emb2.vector).reshape(1, -1)

        similarity = cosine_similarity(vec1, vec2)[0][0]
        st.metric("Cosine Similarity", f"{similarity:.4f}")

        st.subheader("2D PCA projection")
        plotly_pca_projection(
            emb1.vector, emb2.vector, method=ReduceDimensionsMethod.PCA
        )
