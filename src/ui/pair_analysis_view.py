import time
from pathlib import Path

import streamlit as st

from src.domain.embeddings.embedders_manager import EmbeddersManager
from src.domain.metrics import cosine_similarity
from src.domain.visualization import plotly_pca_projection
from src.models.enums.reduce_dimensions_method import ReduceDimensionsMethod
from src.ui.shared.base_view import BaseView


class PairAnalysisView(BaseView):
    title = "Pair Analysis"
    description = (
        "Analyze a single audio-text or audio-audio pair using specified models. "
        "This view uses **PCA (Principal Component Analysis)** to visualize embeddings "
        "in 2D space — it is the only method for comparing just two samples."
    )

    def render(self) -> None:
        self.header()

        embedders_manager: EmbeddersManager = st.session_state["embedders_manager"]
        selected_model = st.selectbox(
            "Select embedder model",
            [""] + embedders_manager.get_embedders_lookup()
        )

        if not selected_model:
            st.info("Please select an embedder model to continue.")
            return

        embedder = embedders_manager.get_embedder(selected_model)
        modalities = embedders_manager.get_embedder_modalities(embedder)

        if "audio" in modalities and "text" in modalities:
            self.audio_text_comparison(embedder)
        elif "audio" in modalities:
            self.audio_audio_comparison(embedder)
        elif "text" in modalities:
            self.text_text_comparison(embedder)




    def audio_text_comparison(self, embedder) -> None:
        audio = st.file_uploader("Audio file", type=["wav", "mp3"])
        if audio is not None:
            st.audio(audio, format="audio/wav")
        
        text = st.text_area("Text prompt", placeholder="e.g. calm piano")
        
        button_disabled = not audio or not text.strip()
        if st.button("Generate embeddings", disabled=button_disabled):
            temp_path = Path("temp_audio.wav")
            with open(temp_path, "wb") as f:
                f.write(audio.read())

            with st.spinner("Generating embeddings and computing similarity..."):
                try:
                    text_emb = embedder.embed_text(text)
                    audio_emb = embedder.embed_audio(temp_path)
                    similarity = cosine_similarity(
                        audio_emb.vector, text_emb.vector
                    )
                    time.sleep(0.5)
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    return

            st.metric("Cosine similarity", f"{similarity:.4f}")

            st.subheader("2D PCA projection")
            plotly_pca_projection(
                audio_emb.vector,
                text_emb.vector,
                method=ReduceDimensionsMethod.PCA,
            )

    def audio_audio_comparison(self, embedder) -> None:
        pass

    def text_text_comparison(self, embedder) -> None:
        pass