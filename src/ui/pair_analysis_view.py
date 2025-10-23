from pathlib import Path

import streamlit as st

from src.domain.metrics import cosine_similarity
from src.domain.visualization import plot_embedding_heatmap, plot_embedding_pca
from src.ui.shared.base_view import BaseView


class PairAnalysisView(BaseView):
    title = "Pair Analysis"
    description = "Analyze one audio–text pair using CLAP, SLAP, or both models. "

    def render(self) -> None:
        self.header()

        clap = st.session_state["models"]["CLAP"]

        mode = st.radio(
            "Select mode",
            ["CLAP", "SLAP", "CLAP vs SLAP"],
            horizontal=True,
        )  # noqa: F841
        audio = st.file_uploader("Audio file", type=["wav", "mp3"])
        text = st.text_area("Text prompt", placeholder="e.g. calm piano")

        if text and audio and st.button("Generate"):
            temp_path = Path("temp_audio.wav")
            with open(temp_path, "wb") as f:
                f.write(audio.read())

            text_emb = clap.embed_text(text)
            audio_emb = clap.embed_audio(temp_path)

            similarity = cosine_similarity(audio_emb.vector, text_emb.vector)

            st.metric("Cosine similarity", f"{similarity:.4f}")
            st.subheader("Heatmap of embedding differences")
            plot_embedding_heatmap(audio_emb.vector, text_emb.vector)
            st.subheader("2D PCA projection")
            plot_embedding_pca(audio_emb.vector, text_emb.vector)
