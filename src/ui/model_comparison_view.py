import streamlit as st

from src.ui.shared.base_view import BaseView


class ModelComparisonView(BaseView):
    title = "Model Comparison"
    description = (
        "Compare CLAP and SLAP on a predefined demo dataset. "
        "Displays how both models group the same audio files."
    )

    def render(self) -> None:
        self.header()

        st.markdown("**Demo dataset:** 10 audio clips")
