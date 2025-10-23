import numpy as np
import pandas as pd
import streamlit as st

from src.ui.shared.base_view import BaseView


class PseudoCaptioningView(BaseView):
    title = "Pseudo-captioning"
    description = (
        "Match uploaded audio with predefined text captions "
        "to simulate pseudo-captioning."
    )

    def render(self) -> None:
        self.header()

        st.file_uploader("Upload audio", type=["wav", "mp3"])
        if st.button("Generate mock captions"):
            df = pd.DataFrame(
                {
                    "Caption": [
                        "calm acoustic melody",
                        "energetic drum loop",
                        "soft ambient pad",
                    ],
                    "Score": np.random.uniform(0.4, 0.95, 3).round(3),
                }
            )
            st.dataframe(df)
            st.success("Mock captions generated.")
