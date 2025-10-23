import streamlit as st

from src.ui.shared.base_view import BaseView


class SimilarityRankingView(BaseView):
    title = "Similarity Ranking"
    description = "Search for the most similar audio or text items."

    def render(self) -> None:
        self.header()

        tab1, tab2 = st.tabs(["Text → Audio", "Audio → Text"])

        with tab1:
            st.text_area("Enter descriptions (one per line)", key="t1t")
            st.file_uploader(
                "Upload audio files",
                type=["wav", "mp3"],
                accept_multiple_files=True,
                key="t1a",
            )

        with tab2:
            st.text_area("Enter descriptions (one per line)", key="t2t")
            st.file_uploader(
                "Upload audio files",
                type=["wav", "mp3"],
                accept_multiple_files=True,
                key="t2a",
            )
