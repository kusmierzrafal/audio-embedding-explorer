import logging
from typing import List, Optional, Tuple

import numpy as np
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

from src.domain.db_manager import DbManager
from src.domain.embeddings.models_manager import ModelsManager
from src.ui.shared.base_view import BaseView

logger = logging.getLogger(__name__)

# Models that support text input
TEXT_CAPABLE_MODELS = {
    "laion/clap-htsat-unfused",
    "laion/clap-htsat-fused",
}


class LocalDbView(BaseView):
    title = "Local Database"
    description = (
        "Search within your local collection of audio embeddings "
        "using text descriptions or audio similarity"
    )

    def render(self) -> None:
        self.header()

        db_manager: DbManager = st.session_state.get("db_manager")
        models_manager: ModelsManager = st.session_state.get("models_manager")

        if not db_manager or not db_manager.is_connected:
            st.error("Database not connected. Please check your database connection.")
            return

        if not models_manager:
            st.error("Models manager not initialized.")
            return

        loaded_models = models_manager.get_loaded_models()
        if not loaded_models:
            st.warning(
                "No models loaded. Please load at least one model from the Home page."
            )
            return

        self._render_search_interface(db_manager, models_manager, loaded_models)

    def _render_search_interface(
        self, db_manager: DbManager, models_manager: ModelsManager, loaded_models: list
    ) -> None:
        """Render the main search interface"""

        # Get available audio files from DB
        audio_files = self._get_audio_files_from_db(db_manager)

        if not audio_files:
            st.warning(
                "No audio files found in database. "
                "Please add some files using other views."
            )
            return

        st.markdown("### Search Configuration")

        col1, col2 = st.columns(2)

        with col1:
            # Search mode selection
            search_mode = st.radio(
                "Search Mode",
                ["Text → Audio", "Audio → Audio"],
                help=(
                    "Text→Audio: Find audio using text description (CLAP models only)\n"
                    "Audio→Audio: Find similar audio (all models)"
                ),
            )

        with col2:
            # Model selection
            model_options = [model.name for model in loaded_models]

            if search_mode == "Text → Audio":
                # Filter to only text-capable models
                model_options = [m for m in model_options if m in TEXT_CAPABLE_MODELS]
                if not model_options:
                    st.error(
                        "Text search requires CLAP models. Please load a CLAP model."
                    )
                    return

            selected_model = st.selectbox(
                "Model", model_options, help="Model to use for similarity computation"
            )

        # Top-K selection
        top_k = st.slider(
            "Number of results",
            min_value=1,
            max_value=min(20, len(audio_files)),
            value=5,
        )

        st.markdown("### Query")

        query_vec = None
        query_description = ""

        if search_mode == "Text → Audio":
            # Text input for semantic search
            query_text = st.text_input(
                "Enter text description",
                placeholder=(
                    "e.g. piano melody, drum beat, guitar solo, noisy synthesizer"
                ),
                help="Describe the audio you're looking for",
            )

            if query_text and st.button("Search by Text", type="primary"):
                with st.spinner("Computing text embedding..."):
                    query_vec = self._compute_text_embedding(
                        query_text, selected_model, models_manager
                    )
                    query_description = f'Text: "{query_text}"'

        else:  # Audio → Audio
            # Audio selection from database
            audio_options = [(f["id"], f["name"]) for f in audio_files]

            col1, col2 = st.columns([3, 1])
            with col1:
                selected_audio_id = st.selectbox(
                    "Select query audio",
                    [aid for aid, _ in audio_options],
                    format_func=lambda aid: next(
                        name for id_, name in audio_options if id_ == aid
                    ),
                    help="Choose an audio file from your database to find similar ones",
                )

            with col2:
                # Audio preview
                if selected_audio_id:
                    audio_data, audio_name = db_manager.get_audio_data(
                        selected_audio_id
                    )
                    if audio_data:
                        st.audio(audio_data, format="audio/wav")
                        st.caption(f"Query: {audio_name}")

            if selected_audio_id and st.button("Search by Audio", type="primary"):
                with st.spinner("Loading audio embedding..."):
                    query_vec = self._get_audio_embedding_from_db(
                        selected_audio_id, selected_model, db_manager
                    )
                    query_description = (
                        "Audio: "
                        f"{
                            next(
                                name
                                for id_, name in audio_options
                                if id_ == selected_audio_id
                            )
                        }"
                    )

        # Show results if we have a query
        if query_vec is not None:
            self._show_search_results(
                query_vec,
                selected_model,
                query_description,
                top_k,
                db_manager,
                exclude_audio_id=selected_audio_id
                if search_mode == "Audio → Audio"
                else None,
            )

    def _get_audio_files_from_db(self, db_manager: DbManager) -> List[dict]:
        """Get all audio files from database"""
        try:
            with db_manager.conn:
                cursor = db_manager.conn.cursor()
                cursor.execute(
                    "SELECT id, original_name, sha256 FROM audio ORDER BY original_name"
                )
                rows = cursor.fetchall()
                return [
                    {"id": row[0], "name": row[1], "sha256": row[2]} for row in rows
                ]
        except Exception as e:
            st.error(f"Failed to load audio files: {e}")
            return []

    def _compute_text_embedding(
        self, text: str, model_id: str, models_manager: ModelsManager
    ) -> Optional[np.ndarray]:
        """Compute text embedding for CLAP models"""
        try:
            stored_model = models_manager.get_model(model_id)
            if not stored_model or not stored_model.is_loaded:
                st.error(f"Model {model_id} is not loaded")
                return None
            embedder = stored_model.embedder
            if not hasattr(embedder, "embed_text"):
                st.error(f"Model {model_id} does not support text embedding")
                return None

            embedding_result = embedder.embed_text(text)
            return embedding_result.vector.detach().cpu().numpy()

        except Exception as e:
            st.error(f"Failed to compute text embedding: {e}")
            logger.exception("Text embedding failed")
            return None

    def _get_audio_embedding_from_db(
        self, audio_id: int, model_id: str, db_manager: DbManager
    ) -> Optional[np.ndarray]:
        """Get audio embedding from database"""
        try:
            with db_manager.conn:
                cursor = db_manager.conn.cursor()
                cursor.execute(
                    """
                    SELECT e.vector_f32 
                    FROM embedding e 
                    JOIN model m ON e.model_id = m.id 
                    WHERE e.audio_id = ? AND m.name = ?
                """,
                    (audio_id, model_id),
                )

                row = cursor.fetchone()
                if not row:
                    st.error(
                        f"No embedding found for audio {audio_id} with model {model_id}"
                    )
                    return None

                # Decode float32 blob
                embedding_bytes = row[0]
                embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
                return embedding

        except Exception as e:
            st.error(f"Failed to load audio embedding: {e}")
            logger.exception("Audio embedding loading failed")
            return None

    def _show_search_results(
        self,
        query_vec: np.ndarray,
        model_id: str,
        query_description: str,
        top_k: int,
        db_manager: DbManager,
        exclude_audio_id: Optional[int] = None,
    ) -> None:
        """Show search results ranked by similarity"""

        st.markdown("### Search Results")
        st.markdown(f"**Query:** {query_description}")
        st.markdown(f"**Model:** {model_id}")

        try:
            results = self._rank_against_db(
                query_vec, model_id, top_k, db_manager, exclude_audio_id
            )

            if not results:
                st.info("No results found.")
                return

            # Display results
            for i, (audio_id, audio_name, score) in enumerate(results, 1):
                with st.container():
                    col1, col2, col3 = st.columns([1, 4, 2])

                    with col1:
                        st.markdown(f"**#{i}**")

                    with col2:
                        st.markdown(f"**{audio_name}**")
                        st.markdown(f"Similarity: {score:.4f}")

                    with col3:
                        # Audio player
                        audio_data, _ = db_manager.get_audio_data(audio_id)
                        if audio_data:
                            st.audio(audio_data, format="audio/wav")

                    st.divider()

        except Exception as e:
            st.error(f"Search failed: {e}")
            logger.exception("Search results failed")

    def _rank_against_db(
        self,
        query_vec: np.ndarray,
        model_id: str,
        top_k: int,
        db_manager: DbManager,
        exclude_audio_id: Optional[int] = None,
    ) -> List[Tuple[int, str, float]]:
        """Rank all audio in DB against query vector"""

        try:
            with db_manager.conn:
                cursor = db_manager.conn.cursor()

                # Build query with optional exclusion
                query_sql = """
                    SELECT a.id, a.original_name, e.vector_f32
                    FROM embedding e
                    JOIN audio a ON e.audio_id = a.id
                    JOIN model m ON e.model_id = m.id
                    WHERE m.name = ?
                """
                params = [model_id]

                if exclude_audio_id is not None:
                    query_sql += " AND a.id != ?"
                    params.append(exclude_audio_id)

                cursor.execute(query_sql, params)
                rows = cursor.fetchall()

                if not rows:
                    return []

                # Compute similarities
                results = []
                query_vec = query_vec.reshape(1, -1)

                for audio_id, audio_name, embedding_bytes in rows:
                    embedding = np.frombuffer(
                        embedding_bytes, dtype=np.float32
                    ).reshape(1, -1)
                    similarity = cosine_similarity(query_vec, embedding)[0, 0]
                    results.append((audio_id, audio_name, float(similarity)))

                # Sort by similarity (descending) and take top-k
                results.sort(key=lambda x: x[2], reverse=True)
                return results[:top_k]

        except Exception:
            logger.exception("Ranking failed")
            raise
