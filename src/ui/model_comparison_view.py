import io
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import streamlit as st

from src.domain.db_manager import DbManager
from src.domain.embeddings.models_manager import ModelsManager
from src.domain.visualization import compute_pca_fig, compute_umap_fig
from src.models.dataclasses.model_option import ModelOption
from src.ui.shared.base_view import BaseView
from src.ui.shared.model_status_label import model_status_label
from src.utils.audio_utils import AudioHelper

logger = logging.getLogger(__name__)

ALLOWED_AUDIO_TYPES = ("audio/wav", "audio/x-wav", "audio/mpeg", "audio/mp3")


MODEL_OPTIONS: List[ModelOption] = [
    ModelOption(ui_name="CLAP (unfused)", model_id="laion/clap-htsat-unfused"),
    ModelOption(ui_name="CLAP (fused)", model_id="laion/clap-htsat-fused"),
    ModelOption(ui_name="OpenL3", model_id="openl3-mel256-512"),
    ModelOption(ui_name="MERT", model_id="m-a-p/MERT-v1-95M"),
]


@st.dialog("Remove audio file?")
def _confirm_remove_audio():
    item = st.session_state.get("mc_audio_to_delete")

    if not item:
        st.write("No audio selected.")
        return

    st.warning(
        "Removing this audio file will invalidate all existing embeddings "
        "and visualizations. You will need to regenerate them."
    )

    st.markdown(f"**File:** `{item['name']}`")

    col_confirm, col_cancel = st.columns(2)

    with col_confirm:
        if st.button("Confirm removal", type="primary"):
            key = (item.get("name"), len(item.get("bytes") or b""))
            st.session_state["mc_audio_files"] = [
                f
                for f in st.session_state["mc_audio_files"]
                if (f.get("name"), len(f.get("bytes") or b"")) != key
            ]
            st.session_state["mc_model_runs"].clear()
            st.session_state["mc_audio_to_delete"] = None
            st.session_state["mc_show_delete_audio_modal"] = False
            st.session_state["mc_audio_uploader_nonce"] += 1
            st.rerun()

    with col_cancel:
        if st.button("Cancel", type="secondary"):
            st.session_state["mc_audio_to_delete"] = None
            st.session_state["mc_show_delete_audio_modal"] = False
            st.rerun()


def _init_state() -> None:
    st.session_state.setdefault("mc_audio_files", [])
    st.session_state.setdefault("mc_show_audio_uploader", False)
    st.session_state.setdefault("mc_selected_models", [])
    st.session_state.setdefault("mc_show_model_picker", False)
    st.session_state.setdefault("mc_model_picker_value", None)
    st.session_state.setdefault("mc_audio_to_delete", None)
    st.session_state.setdefault("mc_show_delete_audio_modal", False)
    st.session_state.setdefault("mc_audio_uploader_nonce", 0)
    st.session_state.setdefault("mc_model_runs", {})


def _read_uploaded_files(uploaded: List[Any]) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for f in uploaded:
        try:
            f.seek(0)
            data = f.read()
            results.append({"name": f.name, "bytes": data, "mime": f.type})
        except Exception as e:
            st.error(f"Failed to read {getattr(f, 'name', 'file')}: {e}")
    return results


def _dedupe_files(files: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out: List[Dict[str, Any]] = []
    for item in files:
        key = (item.get("name"), len(item.get("bytes") or b""))
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def _get_embedder_sr(embedder: Any) -> Optional[int]:
    try:
        if hasattr(embedder, "get_sr") and callable(embedder.get_sr):
            sr = embedder.get_sr()
            if isinstance(sr, int) and sr > 0:
                return sr
    except Exception:
        pass
    return None


def _load_audio_from_bytes(
    data: bytes, target_sr: Optional[int]
) -> Tuple[np.ndarray, int]:
    bio = io.BytesIO(data)
    try:
        if target_sr is None:
            y, sr = AudioHelper.load_audio(bio, 44100)
        else:
            y, sr = AudioHelper.load_audio(bio, target_sr)
        return y, sr
    except Exception as e:
        raise RuntimeError(f"Audio decoding failed: {e}") from e


def _compute_embeddings_for_files(
    embedder: Any, files: List[Dict[str, Any]], model_name: str
) -> Tuple[List[str], np.ndarray]:
    names: List[str] = []
    vecs: List[np.ndarray] = []

    db_manager = st.session_state.get("db_manager")
    model_id = None
    if db_manager and db_manager.is_connected:
        model_id = db_manager.get_or_insert_model(model_name)

    target_sr = _get_embedder_sr(embedder)

    for item in files:
        name = item["name"]
        data = item["bytes"]

        # Try to get embedding from database first
        cached_vector = None
        if db_manager and model_id:
            audio_id = db_manager.get_audio_id_by_data(data)
            if audio_id:
                cached_vector = db_manager.get_embedding(audio_id, model_id)

        if cached_vector is not None:
            # Use cached embedding
            v = cached_vector.reshape(-1)
        else:
            # Generate new embedding
            y, sr = _load_audio_from_bytes(data, target_sr)
            try:
                emb = embedder.embed_audio(y, sr)
                v = getattr(emb, "vector", emb)
                v = np.asarray(v, dtype=np.float32).reshape(-1)

                # Save to database if possible
                if db_manager and model_id:
                    audio_id = db_manager.get_audio_id_by_data(data)
                    if not audio_id:
                        # Insert audio first
                        db_manager.insert_audio_if_not_exists(name, data)
                        audio_id = db_manager.get_audio_id_by_data(data)
                    if audio_id:
                        db_manager.save_embedding(audio_id, model_id, v)
            except Exception as e:
                raise RuntimeError(f"Embedding failed for '{name}': {e}") from e

        names.append(name)
        vecs.append(v)

    X = np.vstack(vecs) if vecs else np.empty((0, 0), dtype=np.float32)
    return names, X


class ModelComparisonView(BaseView):
    title = "Model Comparison"
    description = (
        "Compare audio embedding models (CLAP fused/unfused, OpenL3, and MERT). "
        "For each selected model, the view visualizes how the same set of audio files "
        "is embedded by displaying PCA and UMAP projections."
    )

    def render(self) -> None:
        self.header()
        _init_state()

        models_manager: ModelsManager = st.session_state["models_manager"]

        top_left, top_right = st.columns([1, 1])

        with top_left:
            if st.button("Add audio files"):
                st.session_state["mc_show_audio_uploader"] = True

        used = set(st.session_state["mc_selected_models"])
        unused = [
            m
            for m in MODEL_OPTIONS
            if m.model_id not in used and models_manager.is_loaded(m.model_id)
        ]

        with top_right:
            if unused:
                if st.button("Add model to comparison"):
                    st.session_state["mc_show_model_picker"] = True
            else:
                st.session_state["mc_show_model_picker"] = False

        if st.session_state["mc_show_audio_uploader"]:
            db_manager: DbManager = st.session_state["db_manager"]
            audio_source = st.radio(
                "Audio source",
                ["File Upload", "Database"],
                horizontal=True,
                key="mc_audio_source",
                disabled=not db_manager.is_connected,
            )
            uploaded = []
            if audio_source == "File Upload":
                uploaded_files = st.file_uploader(
                    "Upload audio files",
                    type=["wav", "mp3"],
                    accept_multiple_files=True,
                    key=f"mc_audio_uploader_{st.session_state['mc_audio_uploader_nonce']}",
                )
                if uploaded_files:
                    uploaded = uploaded_files
                    if st.button("Save all to database", key="mc_save_all"):
                        saved_count = 0
                        for file in uploaded_files:
                            data = file.getvalue()
                            if db_manager.insert_audio_if_not_exists(file.name, data):
                                saved_count += 1
                        if saved_count > 0:
                            st.success(f"Saved {saved_count} file(s) to database.")
                        else:
                            st.info("All files already exist in database.")
            else:
                db_audio_files = db_manager.get_audio_files()
                if not db_audio_files:
                    st.warning("No audio files found in the database.")
                else:
                    selected_audios = st.multiselect(
                        "Select audio files from database",
                        options=[(name, id) for id, name in db_audio_files],
                        format_func=lambda x: x[0],
                        key="mc_audio_db_select",
                    )
                    if selected_audios and st.button(
                        "Add selected files", key="mc_add_db_files"
                    ):
                        uploaded = []
                        for audio_name, audio_id in selected_audios:
                            audio_data, _ = db_manager.get_audio_data(audio_id)
                            if audio_data:
                                audio_file = io.BytesIO(audio_data.read())
                                audio_file.name = audio_name
                                audio_file.type = (
                                    "audio/wav"  # Set type for compatibility
                                )
                                uploaded.append(audio_file)
            if uploaded:
                new_files = _read_uploaded_files(uploaded)
                filtered: List[Dict[str, Any]] = []
                for item in new_files:
                    mime = item.get("mime")
                    if mime and mime not in ALLOWED_AUDIO_TYPES:
                        st.warning(f"Unsupported type for {item.get('name')}: {mime}")
                        continue
                    filtered.append(item)

                all_files = st.session_state["mc_audio_files"] + filtered
                st.session_state["mc_audio_files"] = _dedupe_files(all_files)
                st.session_state["mc_show_audio_uploader"] = False
                st.session_state["mc_audio_uploader_nonce"] += 1
                st.rerun()

        files = st.session_state["mc_audio_files"]

        if not files:
            st.info(
                "No audio files added yet. "
                "Add at least one file "
                "to enable embedding generation."
            )
        else:
            st.markdown("### Audio files")
            # Create a scrollable container with fixed height
            with st.container(height=400):
                for i, item in enumerate(files, start=1):
                    row_left, row_right = st.columns([10, 1])

                    with row_left:
                        st.markdown(f"**{i}. {item['name']}**")
                        st.audio(item["bytes"])

                    with row_right:
                        if st.button("✕", key=f"mc_del_audio_{i}", help="Remove audio"):
                            st.session_state["mc_audio_to_delete"] = item
                            st.session_state["mc_show_delete_audio_modal"] = True
                            st.rerun()

            # Add save to database button below the list
            db_manager: DbManager = st.session_state["db_manager"]
            if db_manager.is_connected and st.button(
                "Save all to database", key="mc_save_all_files"
            ):
                saved_count = 0
                skipped_count = 0
                for item in files:
                    if db_manager.insert_audio_if_not_exists(
                        item["name"], item["bytes"]
                    ):
                        saved_count += 1
                    else:
                        skipped_count += 1

                if saved_count > 0:
                    st.success(f"Saved {saved_count} file(s) to database.")
                if skipped_count > 0:
                    st.info(f"{skipped_count} file(s) already existed in database.")

        if st.session_state.get("mc_show_delete_audio_modal"):
            _confirm_remove_audio()

        st.markdown(
            "<hr style='margin: 16px 0; border: 1px solid rgba(255,255,255,0.1);'/>",
            unsafe_allow_html=True,
        )

        if st.session_state["mc_show_model_picker"] and unused:
            ui_names = [m.ui_name for m in unused]
            choice = st.selectbox(
                "Select a model to add",
                options=ui_names,
                index=None,
                placeholder="Choose...",
                key="mc_model_picker_selectbox",
            )
            add_col1, add_col2 = st.columns([1, 4])
            with add_col1:
                if st.button("Add selected model", disabled=choice is None):
                    selected = next(m for m in unused if m.ui_name == choice)
                    st.session_state["mc_selected_models"].append(selected.model_id)
                    st.session_state["mc_show_model_picker"] = False
                    st.rerun()

        selected_models: List[str] = st.session_state["mc_selected_models"]

        if not selected_models:
            st.markdown(
                "**Demo dataset:** add audio clips to begin, "
                "then add models for comparison."
            )
            return

        cols = st.columns(len(selected_models))
        for col, model_id in zip(cols, selected_models):
            opt = next((m for m in MODEL_OPTIONS if m.model_id == model_id), None)
            ui_name = opt.ui_name if opt else model_id

            stored = models_manager.get_model(model_id)
            run_state: Dict[str, Any] = st.session_state["mc_model_runs"].setdefault(
                model_id,
                {
                    "status": "idle",
                    "error": None,
                    "pca_fig": None,
                    "umap_fig": None,
                },
            )

            with col:
                header_left, header_right = st.columns([6, 1])
                with header_left:
                    st.markdown(f"### {ui_name}")
                with header_right:
                    if st.button(
                        "✕",
                        key=f"mc_del_{model_id}",
                        help="Remove from comparison",
                    ):
                        st.session_state["mc_selected_models"] = [
                            mid
                            for mid in st.session_state["mc_selected_models"]
                            if mid != model_id
                        ]
                        st.session_state["mc_model_runs"].pop(model_id, None)
                        st.rerun()

                st.markdown(model_status_label(stored))

                if not files:
                    st.caption(
                        "Add at least one audio file to enable embedding generation."
                    )
                    continue

                if not stored or not stored.is_loaded:
                    st.caption("Load this model to enable embedding generation.")
                    continue

                status = run_state.get("status", "idle")

                if status == "error":
                    err = run_state.get("error") or "Unknown error"
                    st.error(err)

                if status in ("idle", "error"):
                    if st.button("Generate embeddings", key=f"mc_gen_{model_id}"):
                        run_state["status"] = "running"
                        run_state["error"] = None
                        run_state["pca_fig"] = None
                        run_state["umap_fig"] = None
                        st.rerun()

                if status == "running":
                    with st.spinner("Generating embeddings and projections..."):
                        try:
                            names, X = _compute_embeddings_for_files(
                                stored.embedder, files, model_id
                            )

                            if X.size == 0:
                                raise RuntimeError("No embeddings were produced.")

                            if X.ndim != 2 or X.shape[0] != len(names):
                                raise RuntimeError("Invalid embeddings shape.")

                            pca_fig = compute_pca_fig(names, X)
                            umap_fig = compute_umap_fig(names, X)

                            run_state["pca_fig"] = pca_fig
                            run_state["umap_fig"] = umap_fig
                            run_state["status"] = "done"
                        except Exception as e:
                            logger.exception("Model comparison failed for %s", model_id)
                            run_state["status"] = "error"
                            run_state["error"] = str(e)

                    st.rerun()

                if run_state.get("status") == "done":
                    st.markdown("#### PCA visualization")
                    if run_state.get("pca_fig") is not None:
                        st.pyplot(run_state["pca_fig"], clear_figure=False)

                    st.markdown("#### UMAP visualization")
                    if run_state.get("umap_fig") is not None:
                        st.pyplot(run_state["umap_fig"], clear_figure=False)
