import streamlit as st
from src.ui.shared.base_view import BaseView


class PairAnalysisView(BaseView):
    title = "Pair Analysis"
    description = (
        "Analyze one audio–text pair using CLAP, SLAP, or both models. "
    )

    def render(self) -> None:
        self.header()

        mode = st.radio(
            "Select mode",
            ["CLAP", "SLAP", "CLAP vs SLAP"],
            horizontal=True,
        )
        audio = st.file_uploader("Audio file", type=["wav", "mp3"])
        text = st.text_area("Text prompt", placeholder="e.g. calm piano")

