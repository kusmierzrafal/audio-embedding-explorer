import streamlit as st
from src.config.navbar_config import PAGE_TITLE
from src.ui.shared.base_view import BaseView


class HomeView(BaseView):
    title = PAGE_TITLE
    description = (
         "A tool for comparing and visualizing audio–text embeddings, "
         "that allows users to upload audio files, enter text, "
         "and analyze how embedding models CLAP and SLAP interpret the meaning of the provided data."
    )

    def render(self) -> None:
        self.header()
        st.info(
            "Use the sidebar to navigate between views.\n\n"
            "Current modules:\n"
            "- Pair Analysis (CLAP / SLAP / both)\n"
            "- Model Comparison on demo data\n"
            "- Similarity Ranking (Text→Audio / Audio→Text)\n"
            "- Pseudo-captioning\n"
            "- Local Database search"
        )
