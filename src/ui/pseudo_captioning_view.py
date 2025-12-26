import io
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from src.domain.db_manager import DbManager
from src.domain.embeddings.models_manager import ModelsManager
from src.ui.shared.base_view import BaseView
from src.ui.shared.model_status_label import model_status_label
from src.utils.audio_utils import AudioHelper

MUSIC_EMBEDDINGS_PATH = Path(
    "assets/pseudo_captioning/music_level_captions_embeddings.json"
)

SOUND_EMBEDDINGS_PATH = Path(
    "assets/pseudo_captioning/sound_level_captions_embeddings.json"
)


class PseudoCaptioningView(BaseView):
    title = "Pseudo-captioning"
    description = (
        "Match uploaded audio with predefined text captions "
        "to simulate pseudo-captioning. "
        "At least one of the following models must be loaded for this view to work: "
        "laion/clap-htsat-unfused, laion/clap-htsat-fused (or both)."
    )

    @staticmethod
    def _load_caption_embeddings(
        path: Path,
    ) -> tuple[str | None, list[str], np.ndarray]:
        if not path.exists():
            st.error(f"Embeddings file not found: {path}")
            return None, [], np.empty((0,))

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            st.error(f"Failed to load embeddings file: {e}")
            return None, [], np.empty((0,))

        meta = data.get("meta", {})
        model_id = meta.get("model")

        entries = data.get("embeddings", [])
        if not entries:
            st.warning(f"No embeddings found in {path.name}")
            return model_id, [], np.empty((0,))

        captions = [e["caption"] for e in entries]
        embeddings = np.vstack(
            [np.asarray(e["embedding"], dtype=np.float32) for e in entries]
        )

        return model_id, captions, embeddings

    def _compute_audio_to_precomputed_captions(
        self,
        clap,
        audio_file,
        captions: list[str],
        caption_embeddings: np.ndarray,
        top_k: int = 20,
    ) -> pd.DataFrame:
        try:
            audio_file.seek(0)
            y, sr = AudioHelper.load_audio(audio_file, clap.get_sr())
            audio_emb = clap.embed_audio(y, sr).vector
        except Exception as e:
            st.error(f"Failed to process audio: {e}")
            return pd.DataFrame(columns=["No.", "Caption", "Cosine similarity"])

        audio_vec = np.asarray(audio_emb, dtype=np.float32).reshape(-1)

        if caption_embeddings.size == 0 or len(captions) == 0:
            st.warning("No caption embeddings available.")
            return pd.DataFrame(columns=["No.", "Caption", "Cosine similarity"])

        if caption_embeddings.ndim != 2:
            st.error("Caption embeddings array must be 2D.")
            return pd.DataFrame(columns=["No.", "Caption", "Cosine similarity"])

        if caption_embeddings.shape[0] != len(captions):
            st.error("Captions and embeddings count mismatch.")
            return pd.DataFrame(columns=["No.", "Caption", "Cosine similarity"])

        if caption_embeddings.shape[1] != audio_vec.shape[0]:
            st.error(
                f"Embedding dimension mismatch: audio={audio_vec.shape[0]}, "
                f"captions={caption_embeddings.shape[1]}"
            )
            return pd.DataFrame(columns=["No.", "Caption", "Cosine similarity"])

        eps = 1e-8
        audio_norm = audio_vec / (np.linalg.norm(audio_vec) + eps)
        cap_norms = caption_embeddings / (
            np.linalg.norm(caption_embeddings, axis=1, keepdims=True) + eps
        )
        sims = cap_norms @ audio_norm

        top_k = min(top_k, sims.shape[0])
        top_idx = np.argpartition(-sims, top_k - 1)[:top_k]
        top_idx = top_idx[np.argsort(-sims[top_idx])]

        return pd.DataFrame(
            {
                "No.": range(1, top_k + 1),
                "Caption": [captions[i] for i in top_idx],
                "Cosine similarity": [round(float(sims[i]), 4) for i in top_idx],
            }
        )

    def render(self) -> None:
        self.header()

        db_manager: DbManager = st.session_state["db_manager"]
        models_manager: ModelsManager = st.session_state["models_manager"]

        music_model_id = "laion/clap-htsat-unfused"
        sound_model_id = "laion/clap-htsat-fused"

        music_model = models_manager.get_model(music_model_id)
        sound_model = models_manager.get_model(sound_model_id)

        music_loaded = music_model.is_loaded if music_model else False
        sound_loaded = sound_model.is_loaded if sound_model else False

        st.markdown("### Required models")

        col1, col2 = st.columns([6, 2])
        with col1:
            st.markdown("**Music-level CLAP (`laion/clap-htsat-unfused`)**")
            st.caption(
                "Optimized for music-level analysis. "
                "Processes the audio as a whole and "
                "is best suited for melodic, harmonic, "
                "and structural musical content."
            )
        with col2:
            st.markdown(model_status_label(music_model))

        col1, col2 = st.columns([6, 2])
        with col1:
            st.markdown("**Sound-level CLAP (`laion/clap-htsat-fused`)**")
            st.caption(
                "Optimized for sound and effect analysis. "
                "Uses temporal fusion over "
                "multiple audio segments and works better for textures, ambience, "
                "and non-musical sound events."
            )
        with col2:
            st.markdown(model_status_label(sound_model))

        st.markdown(
            "<hr style='margin: 16px 0; border: 1px solid rgba(255,255,255,0.1);'/>",
            unsafe_allow_html=True,
        )

        audio_source = st.radio(
            "Audio source",
            ["File Upload", "Database"],
            horizontal=True,
            key="pseudo_captioning_audio_source",
            disabled=not db_manager.is_connected,
        )
        uploaded_audio = None
        if audio_source == "File Upload":
            uploaded_audio = st.file_uploader(
                "Upload audio file",
                type=["wav", "mp3"],
                accept_multiple_files=False,
                key="pseudo_captioning_audio_upload",
            )
            if uploaded_audio:
                if st.button("Save to database", key="pseudo_captioning_save_to_db"):
                    data = uploaded_audio.getvalue()
                    if db_manager.insert_audio_if_not_exists(uploaded_audio.name, data):
                        st.success(f"Saved '{uploaded_audio.name}' to database.")
                    else:
                        st.info(f"'{uploaded_audio.name}' already exists in database.")
        else:
            db_audio_files = db_manager.get_audio_files()
            if not db_audio_files:
                st.warning("No audio files found in the database.")
            else:
                selected_audio = st.selectbox(
                    "Select audio from database",
                    options=[(name, id) for id, name in db_audio_files],
                    format_func=lambda x: x[0],
                    key="pseudo_captioning_audio_db_select",
                )
                if selected_audio:
                    audio_name, audio_id = selected_audio
                    audio_data, _ = db_manager.get_audio_data(audio_id)
                    if audio_data:
                        uploaded_audio = io.BytesIO(audio_data.read())
                        uploaded_audio.name = audio_name

        if not uploaded_audio:
            st.info("Please upload an audio file to generate pseudo-captions.")
            return

        if audio_source == "File Upload" and uploaded_audio:
            allowed_types = ("audio/wav", "audio/x-wav", "audio/mpeg")
            if uploaded_audio.type not in allowed_types:
                st.error(f"Unsupported audio format: {uploaded_audio.type}")
                return
        elif audio_source == "Database" and uploaded_audio:
            supported_extensions = (".wav", ".mp3")
            if not uploaded_audio.name.lower().endswith(supported_extensions):
                st.error(f"Unsupported audio format: {uploaded_audio.name}")
                return

        st.audio(uploaded_audio, format="audio/wav")

        if not (music_loaded or sound_loaded):
            st.warning("At least one CLAP model must be loaded.")
            return

        music_json_ok = MUSIC_EMBEDDINGS_PATH.exists()
        sound_json_ok = SOUND_EMBEDDINGS_PATH.exists()

        can_run_music = music_loaded and music_json_ok
        can_run_sound = sound_loaded and sound_json_ok

        if not (can_run_music or can_run_sound):
            if music_loaded and not music_json_ok:
                st.error(f"Embeddings file not found: {MUSIC_EMBEDDINGS_PATH}")
            if sound_loaded and not sound_json_ok:
                st.error(f"Embeddings file not found: {SOUND_EMBEDDINGS_PATH}")
            st.warning(
                "No runnable block: load a model and ensure its embeddings JSON exists."
            )
            return

        if not st.button("Generate pseudo-captions"):
            return

        slot_music = st.empty()
        slot_sound = st.empty()

        results: list[tuple[str, pd.DataFrame]] = []

        if can_run_music:
            with slot_music.container():
                with st.spinner("Computing music-level captions..."):
                    time.sleep(1)
                    emb_model_id, captions, emb = self._load_caption_embeddings(
                        MUSIC_EMBEDDINGS_PATH
                    )
                    if emb_model_id != music_model_id:
                        st.error(
                            "Music caption embeddings were generated "
                            "with a different model."
                        )
                        emb_model_id, captions, emb = None, [], np.empty((0,))

                    if captions and emb.size > 0:
                        df = self._compute_audio_to_precomputed_captions(
                            clap=music_model.embedder,
                            audio_file=uploaded_audio,
                            captions=captions,
                            caption_embeddings=emb,
                        )
                    else:
                        df = pd.DataFrame(
                            columns=["No.", "Caption", "Cosine similarity"]
                        )

                slot_music.empty()
                if not df.empty:
                    results.append(("🎵 Music-level captions", df))
                else:
                    pass

        if can_run_sound:
            with slot_sound.container():
                with st.spinner("Computing sound-level captions..."):
                    time.sleep(1)

                    emb_model_id, captions, emb = self._load_caption_embeddings(
                        SOUND_EMBEDDINGS_PATH
                    )
                    if emb_model_id != sound_model_id:
                        st.error(
                            "Sound caption embeddings were generated "
                            "with a different model."
                        )
                        emb_model_id, captions, emb = None, [], np.empty((0,))

                    if captions and emb.size > 0:
                        df = self._compute_audio_to_precomputed_captions(
                            clap=sound_model.embedder,
                            audio_file=uploaded_audio,
                            captions=captions,
                            caption_embeddings=emb,
                        )
                    else:
                        df = pd.DataFrame(
                            columns=["No.", "Caption", "Cosine similarity"]
                        )

                slot_sound.empty()
                if not df.empty:
                    results.append(("🔊 Sound-level captions", df))
                else:
                    pass

        if not results:
            st.warning(
                "No results to display (check model/json compatibility "
                "and embeddings content)."
            )
            return

        columns = st.columns(len(results))
        for col, (title, df) in zip(columns, results):
            with col:
                st.markdown(f"### {title}")
                st.dataframe(df, use_container_width=True, hide_index=True)
