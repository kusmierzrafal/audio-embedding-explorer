import pandas as pd
import streamlit as st

from src.domain.db_manager import DbManager
from src.ui.shared.base_view import BaseView


class DatabaseManagementView(BaseView):
    title = "Database Management"
    description = (
        "Manage your audio embedding database. View, insert, and delete audio files, "
        "models, and embeddings stored in the local SQLite database."
    )

    def render(self) -> None:
        self.header()

        db_manager: DbManager = st.session_state["db_manager"]

        if not db_manager.is_connected:
            st.error(
                "Database is not connected. Please check your database configuration."
            )
            return

        # Tabs for different operations
        tab1, tab2, tab3, tab4 = st.tabs(
            [
                "📊 View Data",
                "➕ Insert Audio",
                "🗑️ Delete Data",
                "📈 Statistics",
            ]
        )

        with tab1:
            self._render_view_data(db_manager)

        with tab2:
            self._render_insert_audio(db_manager)

        with tab3:
            self._render_delete_data(db_manager)

        with tab4:
            self._render_statistics(db_manager)

    def _render_view_data(self, db_manager: DbManager) -> None:
        st.subheader("Database Contents")

        data_type = st.radio(
            "Select data to view",
            ["Audio Files", "Models", "Embeddings"],
            horizontal=True,
        )

        if data_type == "Audio Files":
            self._show_audio_files(db_manager)
        elif data_type == "Models":
            self._show_models(db_manager)
        elif data_type == "Embeddings":
            self._show_embeddings(db_manager)

    def _show_audio_files(self, db_manager: DbManager) -> None:
        st.markdown("### Audio Files")

        try:
            with db_manager.conn:
                cursor = db_manager.conn.cursor()
                cursor.execute("""
                    SELECT id, original_name, sha256, length(data) as size_bytes
                    FROM audio 
                    ORDER BY id DESC
                """)
                rows = cursor.fetchall()

            if not rows:
                st.info("No audio files found in the database.")
                return

            df = pd.DataFrame(rows, columns=["ID", "Name", "SHA256", "Size (bytes)"])
            df["Size (MB)"] = (df["Size (bytes)"] / 1024 / 1024).round(2)

            st.dataframe(df, use_container_width=True)

            # Audio playback section
            if len(df) > 0:
                st.markdown("### Preview Audio")
                selected_id = st.selectbox(
                    "Select audio to preview",
                    df["ID"].tolist(),
                    format_func=lambda x: (
                        f"ID {x}: {df[df['ID'] == x]['Name'].iloc[0]}"
                    ),
                )

                if selected_id:
                    audio_data, audio_name = db_manager.get_audio_data(selected_id)
                    if audio_data:
                        st.audio(audio_data, format="audio/wav")
                        st.caption(f"File: {audio_name}")

        except Exception as e:
            st.error(f"Failed to load audio files: {e}")

    def _show_models(self, db_manager: DbManager) -> None:
        st.markdown("### Models")

        try:
            with db_manager.conn:
                cursor = db_manager.conn.cursor()
                cursor.execute("""
                    SELECT m.id, m.name, COUNT(e.audio_id) as embedding_count
                    FROM model m
                    LEFT JOIN embedding e ON m.id = e.model_id
                    GROUP BY m.id, m.name
                    ORDER BY m.id
                """)
                rows = cursor.fetchall()

            if not rows:
                st.info("No models found in the database.")
                return

            df = pd.DataFrame(rows, columns=["ID", "Model Name", "Embeddings Count"])
            st.dataframe(df, use_container_width=True)

        except Exception as e:
            st.error(f"Failed to load models: {e}")

    def _show_embeddings(self, db_manager: DbManager) -> None:
        st.markdown("### Embeddings")

        try:
            with db_manager.conn:
                cursor = db_manager.conn.cursor()
                cursor.execute("""
                    SELECT a.original_name, m.name as model_name,
                        length(e.vector_f32) as vector_size
                    FROM embedding e
                    JOIN audio a ON e.audio_id = a.id
                    JOIN model m ON e.model_id = m.id
                    ORDER BY a.original_name, m.name
                """)
                rows = cursor.fetchall()

            if not rows:
                st.info("No embeddings found in the database.")
                return

            df = pd.DataFrame(
                rows, columns=["Audio File", "Model", "Vector Size (bytes)"]
            )
            df["Vector Dimensions"] = (df["Vector Size (bytes)"] / 4).astype(
                int
            )  # float32 = 4 bytes per dimension

            st.dataframe(df, use_container_width=True)

        except Exception as e:
            st.error(f"Failed to load embeddings: {e}")

    def _render_insert_audio(self, db_manager: DbManager) -> None:
        st.subheader("Insert New Audio Files")

        uploaded_files = st.file_uploader(
            "Upload audio files to database",
            type=["wav", "mp3"],
            accept_multiple_files=True,
            key="db_mgmt_upload",
        )

        if uploaded_files:
            st.markdown(f"**Selected {len(uploaded_files)} file(s)**")
            for file in uploaded_files:
                st.write(f"- {file.name}")

            if st.button("Insert All Files", type="primary"):
                success_count = 0
                duplicate_count = 0

                progress_bar = st.progress(0)
                status_text = st.empty()

                for i, file in enumerate(uploaded_files):
                    status_text.text(f"Processing {file.name}...")
                    progress_bar.progress((i + 1) / len(uploaded_files))

                    try:
                        data = file.getvalue()
                        if db_manager.insert_audio_if_not_exists(file.name, data):
                            success_count += 1
                        else:
                            duplicate_count += 1
                    except Exception as e:
                        st.error(f"Failed to insert {file.name}: {e}")

                status_text.empty()
                progress_bar.empty()

                if success_count > 0:
                    st.success(f"Successfully inserted {success_count} file(s)")
                if duplicate_count > 0:
                    st.info(f"{duplicate_count} file(s) already existed in database")

    def _render_delete_data(self, db_manager: DbManager) -> None:
        st.subheader("Delete Data")

        delete_type = st.radio(
            "Select data type to delete",
            ["Audio Files", "Embeddings"],
            horizontal=True,
        )

        if delete_type == "Audio Files":
            self._delete_audio_files(db_manager)
        elif delete_type == "Embeddings":
            self._delete_embeddings(db_manager)

    def _delete_audio_files(self, db_manager: DbManager) -> None:
        st.markdown("### Delete Audio Files")
        st.warning("⚠️ Deleting audio files will also remove all associated embeddings!")

        try:
            audio_files = db_manager.get_audio_files()
            if not audio_files:
                st.info("No audio files to delete.")
                return

            selected_files = st.multiselect(
                "Select audio files to delete",
                options=[(id, name) for id, name in audio_files],
                format_func=lambda x: f"ID {x[0]}: {x[1]}",
            )

            if selected_files:
                st.markdown("**Files to be deleted:**")
                for file_id, file_name in selected_files:
                    st.write(f"- {file_name} (ID: {file_id})")

                if st.button("🗑️ Delete Selected Files", type="primary"):
                    deleted_count = 0
                    for file_id, file_name in selected_files:
                        try:
                            with db_manager.conn:
                                cursor = db_manager.conn.cursor()
                                cursor.execute(
                                    "DELETE FROM audio WHERE id = ?",
                                    (file_id,),
                                )
                                if cursor.rowcount > 0:
                                    deleted_count += 1
                        except Exception as e:
                            st.error(f"Failed to delete {file_name}: {e}")

                    if deleted_count > 0:
                        st.success(
                            f"Successfully deleted {deleted_count} audio file(s) "
                            "and their embeddings"
                        )
                    st.rerun()

        except Exception as e:
            st.error(f"Failed to load audio files: {e}")

    def _delete_embeddings(self, db_manager: DbManager) -> None:
        st.markdown("### Delete Embeddings")
        st.warning(
            (
                "⚠️ Deleting embeddings will remove cached computations - "
                "they can be regenerated!"
            )
        )

        try:
            with db_manager.conn:
                cursor = db_manager.conn.cursor()
                cursor.execute("""
                    SELECT e.audio_id, e.model_id, 
                           COALESCE(a.original_name, '[DELETED AUDIO]') as audio_name, 
                           COALESCE(m.name, '[DELETED MODEL]') as model_name
                    FROM embedding e
                    LEFT JOIN audio a ON e.audio_id = a.id
                    LEFT JOIN model m ON e.model_id = m.id
                    ORDER BY COALESCE(a.original_name, '[DELETED AUDIO]'), COALESCE(m.name, '[DELETED MODEL]')
                """)
                embeddings = cursor.fetchall()

            if not embeddings:
                st.info("No embeddings to delete.")
                return

            selected_embeddings = st.multiselect(
                "Select embeddings to delete",
                options=[
                    (audio_id, model_id, audio_name, model_name)
                    for audio_id, model_id, audio_name, model_name in embeddings
                ],
                format_func=lambda x: (
                    f"{x[2]} + {x[3]} (Audio ID: {x[0]}, Model ID: {x[1]})"
                ),
            )

            if selected_embeddings:
                st.markdown("**Embeddings to be deleted:**")
                for (
                    audio_id,
                    model_id,
                    audio_name,
                    model_name,
                ) in selected_embeddings:
                    st.write(f"- {audio_name} + {model_name}")

                if st.button("🗑️ Delete Selected Embeddings", type="primary"):
                    deleted_count = 0
                    for (
                        audio_id,
                        model_id,
                        audio_name,
                        model_name,
                    ) in selected_embeddings:
                        try:
                            with db_manager.conn:
                                cursor = db_manager.conn.cursor()
                                cursor.execute(
                                    "DELETE FROM embedding "
                                    "WHERE audio_id = ? AND model_id = ?",
                                    (audio_id, model_id),
                                )
                                if cursor.rowcount > 0:
                                    deleted_count += 1
                        except Exception as e:
                            st.error(
                                f"Failed to delete embedding {audio_name} + "
                                f"{model_name}: {e}"
                            )

                    if deleted_count > 0:
                        st.success(f"Successfully deleted {deleted_count} embedding(s)")
                    st.rerun()

        except Exception as e:
            st.error(f"Failed to load embeddings: {e}")

    def _render_statistics(self, db_manager: DbManager) -> None:
        st.subheader("Database Statistics")

        try:
            with db_manager.conn:
                cursor = db_manager.conn.cursor()

                # Basic counts
                cursor.execute("SELECT COUNT(*) FROM audio")
                audio_count = cursor.fetchone()[0]

                cursor.execute("SELECT COUNT(*) FROM model")
                model_count = cursor.fetchone()[0]

                cursor.execute("SELECT COUNT(*) FROM embedding")
                embedding_count = cursor.fetchone()[0]

                # Storage usage
                cursor.execute("SELECT SUM(length(data)) FROM audio")
                total_audio_size = cursor.fetchone()[0] or 0

                cursor.execute("SELECT SUM(length(vector_f32)) FROM embedding")
                total_embedding_size = cursor.fetchone()[0] or 0

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Audio Files", audio_count)
                st.metric(
                    "Audio Storage",
                    f"{total_audio_size / (1024 * 1024):.1f} MB",
                )

            with col2:
                st.metric("Models", model_count)
                st.metric(
                    "Embedding Storage",
                    f"{total_embedding_size / (1024 * 1024):.1f} MB",
                )

            with col3:
                st.metric("Total Embeddings", embedding_count)
                st.metric(
                    "Total Storage",
                    f"{(total_audio_size + total_embedding_size) / (1024 * 1024):.1f} "
                    "MB",
                )

            # Embeddings per model chart
            if embedding_count > 0:
                with db_manager.conn:
                    cursor = db_manager.conn.cursor()
                    cursor.execute("""
                        SELECT m.name, COUNT(e.audio_id) as count
                        FROM model m
                        LEFT JOIN embedding e ON m.id = e.model_id
                        GROUP BY m.id, m.name
                        ORDER BY count DESC
                    """)
                    model_stats = cursor.fetchall()

                if model_stats:
                    st.markdown("### Embeddings per Model")
                    df = pd.DataFrame(model_stats, columns=["Model", "Embeddings"])
                    st.bar_chart(df.set_index("Model"))

        except Exception as e:
            st.error(f"Failed to load statistics: {e}")
