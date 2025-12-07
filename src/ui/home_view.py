import streamlit as st

from src.config.navbar_config import PAGE_TITLE
from src.domain.db_manager import DbManager
from src.domain.embeddings.models_manager import ModelsManager
from src.ui.shared.base_view import BaseView


class HomeView(BaseView):
    title = PAGE_TITLE
    description = (
        "A tool for comparing and visualizing audio and audio-text embeddings, "
        "that allows users to upload audio files, enter text, "
        "and analyze how embedding models CLAP and SLAP interpret the meaning of "
        "the provided data."
    )

    def render(self) -> None:
        self.header()

        st.divider()
        models_manager: ModelsManager = st.session_state.models_manager
        db_manager: DbManager = st.session_state.get("db_manager", None)

        loaded_count = len(models_manager.get_loaded_models())
        total_count = len(models_manager.available_models)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Loaded embedding models", f"{loaded_count} / {total_count}")
        c2.metric(
            "System Status",
            "Ready" if loaded_count > 0 else "Waiting",
            delta="Active" if loaded_count > 0 else "No models",
            delta_color="normal" if loaded_count > 0 else "off",
        )
        c3.metric(
            "Database Status",
            "Connected" if db_manager.is_connected else "Not connected",
        )

        c4.metric("Device", models_manager.device.upper())

        st.divider()
        st.subheader("📦 Available Models")
        for id, model_data in models_manager.available_models.items():
            is_loaded = models_manager.is_loaded(id)
            with st.container(border=True):
                col_icon, col_desc, col_action = st.columns([1, 4, 2])

                with col_icon:
                    if is_loaded:
                        st.markdown("# 🟢")
                    else:
                        st.markdown("# ⚪")

                with col_desc:
                    st.markdown(f"**{model_data.name}**")
                    st.caption(f"{model_data.description} `{model_data.type}`")

                with col_action:
                    if is_loaded:
                        if st.button("Unload", key=f"unload_{id}", type="secondary"):
                            models_manager.unload_model(id)
                            st.rerun()
                    else:
                        if st.button(
                            "Load",
                            key=f"load_{id}",
                            type="primary",
                            disabled=not model_data.is_imported,
                            help="Model not available. Please check the installation "
                            "instructions."
                            if not model_data.is_imported
                            else None,
                        ):
                            with st.spinner(f"Loading {model_data.name}..."):
                                models_manager.load_model(id)
                            st.rerun()

        st.markdown("---")
