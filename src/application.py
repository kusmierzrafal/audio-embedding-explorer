from typing import Dict, Type

import streamlit as st
from streamlit_option_menu import option_menu

from src.config.navbar_config import NAVBAR_ICONS, NAVBAR_STYLES, PAGE_TITLE
from src.domain.db_manager import DbManager
from src.domain.embeddings.models_manager import ModelsManager
from src.models.enums.view_names import ViewName
from src.ui.database_management_view import DatabaseManagementView
from src.ui.embeddings_playground_view import EmbeddingsPlaygroundView
from src.ui.home_view import HomeView
from src.ui.local_db_view import LocalDbView
from src.ui.model_comparison_view import ModelComparisonView
from src.ui.pseudo_captioning_view import PseudoCaptioningView
from src.ui.shared.base_view import BaseView
from src.ui.similarity_ranking_view import SimilarityRankingView
from src.utils.env_loader import load_model_env


class Application:
    def __init__(self) -> None:
        st.set_page_config(
            page_title=PAGE_TITLE,
            layout="wide",
        )

        self.views: Dict[ViewName, Type[BaseView]] = {
            ViewName.HOME: HomeView,
            ViewName.EMBEDDINGS_PLAYGROUND: EmbeddingsPlaygroundView,
            ViewName.MODEL_COMPARISON: ModelComparisonView,
            ViewName.SIMILARITY_RANKING: SimilarityRankingView,
            ViewName.PSEUDO_CAPTIONING: PseudoCaptioningView,
            ViewName.LOCAL_DB: LocalDbView,
            ViewName.DATABASE_MANAGEMENT: DatabaseManagementView,
        }

    def prepare_env(self) -> None:
        if "app_ready" not in st.session_state:
            st.session_state.app_ready = False

        if st.session_state.app_ready:
            return

        if "model_env" not in st.session_state:
            st.session_state.model_env = load_model_env()

        if "models_manager" not in st.session_state:
            device = st.session_state.model_env.device
            st.session_state.models_manager = ModelsManager(device)

        if "db_manager" not in st.session_state:
            st.session_state.db_manager = DbManager()

        st.session_state.app_ready = True

    def run(self) -> None:
        if "app_ready" not in st.session_state or not st.session_state.app_ready:
            self.prepare_env()

        # Custom CSS to control sidebar width
        st.markdown("""
        <style>
        section[data-testid="stSidebar"] {
            width: 21rem !important;
            background-color: #161B22 !important;
        }
        </style>
        """, unsafe_allow_html=True)

        models_manager: ModelsManager = st.session_state["models_manager"]

        with st.sidebar:
            st.markdown(f"## {PAGE_TITLE}")

            selected_label = option_menu(
                menu_title=None,
                options=[v.value for v in self.views.keys()],
                icons=[NAVBAR_ICONS[v] for v in self.views.keys()],
                orientation="vertical",
                default_index=0,
                styles=NAVBAR_STYLES,
            )

        selected_view = next(v for v in self.views if v.value == selected_label)

        if (
            selected_view is not ViewName.HOME
            and selected_view is not ViewName.DATABASE_MANAGEMENT
            and not models_manager.has_any_loaded_model()
        ):
            st.info(
                "No embedding models are loaded.\n\n"
                "Please go to Home page and load at least one model to continue."
            )
            return

        view_class = self.views[selected_view]
        view = view_class()
        view.render()
