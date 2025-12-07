import io

import streamlit as st
import torch

from src.domain.embeddings.base_embedders import (
    AudioEmbedder,
    TextEmbedder,
)
from src.domain.embeddings.models_manager import ModelsManager
from src.domain.metrics import cosine_similarity
from src.domain.visualization import plotly_pca_projection
from src.models.enums.reduce_dimensions_method import ReduceDimensionsMethod
from src.ui.shared.audio_edit_view import AudioEditView
from src.ui.shared.base_view import BaseView


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
        model = st.selectbox(
            "Choose model",
            options=models_manager.get_loaded_models(),
            format_func=lambda x: x.name
        )

        if not models_manager.get_loaded_models():
            st.info("There's no any loaded model. Please load a model in the home view.") 
            return
        
        if model is None:
            st.info("Please load and select a model to continue.")
            return
        
        sr = 48000
        st.markdown("---")
        if 'audio' in model.embedder.get_modalities() and 'text' in model.embedder.get_modalities():
            self.audio_text_comparison(model.embedder, sr=sr)
        elif 'audio' in model.embedder.get_modalities():
            self.audio_audio_comparison(model.embedder, sr=sr)

        

    def audio_text_comparison(self, embedder: AudioEmbedder | TextEmbedder, sr: int) -> None:
        text = st.text_area("Text prompt", placeholder="e.g. calm piano")
        
        audio_bytes: io.Bytes = st.file_uploader("Load audio", type=["wav", "mp3", "flac"])
        if audio_bytes:
            audio_view = AudioEditView(audio_bytes.name, audio_bytes=audio_bytes, sr=sr)
            audio_view.render()

        if st.button("Generate embedding", disabled=audio_bytes is None or audio_view.latest_y is None or not text.strip()):
            with st.spinner("Generating embeddings and computing similarity..."):
                text_emb = embedder.embed_text(text)
                audio_emb = embedder.embed_audio(audio_view.latest_y, sr=sr)
                similarity = cosine_similarity(audio_emb.vector, text_emb.vector)

                st.metric("Cosine similarity", f"{similarity:.4f}")

                st.subheader("2D PCA projection")
                plotly_pca_projection(
                    audio_emb.vector,
                    text_emb.vector,
                    method=ReduceDimensionsMethod.PCA,
                )
        st.markdown("---")
