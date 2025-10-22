import streamlit as st
from src.ui.shared.base_view import BaseView


class LocalDbView(BaseView):
    title = "Local Database"
    description = (
        "Search within your local collection of audio embeddings "
    )

    def render(self) -> None:
        self.header()
        query = st.text_input("Search query (text description)")
