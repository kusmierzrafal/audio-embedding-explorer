import time
from pathlib import Path

import streamlit as st

from src.domain.metrics import cosine_similarity
from src.domain.visualization import plotly_pca_projection
from src.models.enums.reduce_dimensions_method import ReduceDimensionsMethod
from src.ui.shared.base_view import BaseView


class PairAnalysisView(BaseView):
    title = "Pair Analysis"
    description = (
        "Analyze a single audio–text pair using CLAP, SLAP, or both models. "
        "This view uses **PCA (Principal Component Analysis)** to visualize embeddings "
        "in 2D space — it is the only method for comparing just two samples."
    )

    def render(self) -> None:
        self.header()

        clap = st.session_state["models"]["CLAP"]

        mode = st.radio(  # noqa: F841
            "Select mode",
            ["CLAP", "SLAP", "CLAP vs SLAP"],
            horizontal=True,
        )

        audio = st.file_uploader("Audio file", type=["wav", "mp3"])

        if audio is not None:
            st.audio(audio, format="audio/wav")

        text = st.text_area("Text prompt", placeholder="e.g. calm piano")

        if text and audio:
            if st.button("Generate embeddings"):
                temp_path = Path("temp_audio.wav")
                with open(temp_path, "wb") as f:
                    f.write(audio.read())

                with st.spinner("Generating embeddings and computing similarity..."):
                    try:
                        text_emb = clap.embed_text(text)
                        audio_emb = clap.embed_audio(temp_path)
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
        else:
            st.info("Please provide both text and audio inputs to continue.")
