import streamlit as st


class BaseView:
    title: str = ""
    description: str = ""

    def header(self) -> None:
        if self.title:
            st.markdown(f"### {self.title}")
        if self.description:
            st.markdown(
                f'<p style="color:#9aa4b2;font-size:0.9rem;">{self.description}</p>',
                unsafe_allow_html=True,
            )

    def render(self) -> None:
        raise NotImplementedError("Subclasses must implement the render() method.")
