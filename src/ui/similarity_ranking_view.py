import streamlit as st
from src.ui.shared.base_view import BaseView


class SimilarityRankingView(BaseView):
    title = "Similarity Ranking"
    description = (
        "Search for the most similar audio or text items."
    )

    def render(self) -> None:
        self.header()

        tab1, tab2 = st.tabs(["Text → Audio", "Audio → Text"])

        with tab1:
            tab_1_texts = st.text_area("Enter descriptions (one per line)", key="t1t")
            tab_1_audio_files = st.file_uploader(
                "Upload audio files", type=["wav", "mp3"], accept_multiple_files=True,
                key="t1a"

            )

        with tab2:
            tab_2_texts = st.text_area("Enter descriptions (one per line)", key="t2t")
            tab_2_audio_files = st.file_uploader(
                "Upload audio files", type=["wav", "mp3"], accept_multiple_files=True,
                key="t2a"
            )
