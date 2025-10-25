from typing import Dict, Type

import streamlit as st
from streamlit_option_menu import option_menu

from src.config.navbar_config import NAVBAR_ICONS, NAVBAR_STYLES, PAGE_TITLE
from src.domain.embeddings.embedders_manager import EmbeddersManager
from src.models.enums.view_names import ViewName
from src.ui.home_view import HomeView
from src.ui.local_db_view import LocalDbView
from src.ui.model_comparison_view import ModelComparisonView
from src.ui.pair_analysis_view import PairAnalysisView
from src.ui.pseudo_captioning_view import PseudoCaptioningView
from src.ui.shared.base_view import BaseView
from src.ui.similarity_ranking_view import SimilarityRankingView
from src.utils.env_loader import load_model_env
from src.utils.path_resolver import resolve_paths


class Application:
    def __init__(self) -> None:
        st.set_page_config(
            page_title=PAGE_TITLE,
            layout="wide",
        )

        self.views: Dict[ViewName, Type[BaseView]] = {
            ViewName.HOME: HomeView,
            ViewName.PAIR_ANALYSIS: PairAnalysisView,
            ViewName.MODEL_COMPARISON: ModelComparisonView,
            ViewName.SIMILARITY_RANKING: SimilarityRankingView,
            ViewName.PSEUDO_CAPTIONING: PseudoCaptioningView,
            ViewName.LOCAL_DB: LocalDbView,
        }

    def prepare_env(self) -> None:
        if "embedders_manager" in st.session_state:
            return

        env = load_model_env()
        paths = resolve_paths()

        st.session_state["embedders_manager"] = EmbeddersManager(paths.models_dir, env)
        st.success("Environment loaded successfully!")

    def run(self) -> None:
        if "embedders_manager" not in st.session_state:
            with st.spinner("Preparing environment..."):
                self.prepare_env()
            st.rerun()
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
        view_class = self.views[selected_view]
        view = view_class()
        view.render()
