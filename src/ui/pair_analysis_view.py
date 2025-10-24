import streamlit as st

<<<<<<< Updated upstream
=======
from src.domain.embeddings.embeddings_manager import ModelsManager
from src.models.enums.available_models import AvailableModel
from src.domain.metrics import cosine_similarity
from src.domain.visualization import plot_embedding_heatmap, plot_embedding_pca
>>>>>>> Stashed changes
from src.ui.shared.base_view import BaseView


class PairAnalysisView(BaseView):
    title = "Pair Analysis"
    description = "Analyze one audio-text pair using specified embedding models."

    def render(self) -> None:
        self.header()

<<<<<<< Updated upstream
        st.radio(
            "Select mode",
            ["CLAP", "SLAP", "CLAP vs SLAP"],
            horizontal=True,
        )
        st.file_uploader("Audio file", type=["wav", "mp3"])
        st.text_area("Text prompt", placeholder="e.g. calm piano")
=======
        selected_model = st.selectbox("Select model:", [m.value for m in AvailableModel])
        models_manager: ModelsManager = st.session_state["models_manager"]

        with st.spinner(f"Loading model... {selected_model}"):
            embedder = models_manager.get_model(selected_model)
            st.success(f"Loaded model: {selected_model}")
        embedder_type = models_manager.get_embedder_type(embedder)

        audio = None
        if embedder_type == "multimodal" or embedder_type == "audio":
            audio = st.file_uploader("Audio file", type=["wav", "mp3"])
            st.audio(audio) if audio else None

        text = ""
        if embedder_type == "multimodal" or embedder_type == "text":
            text = st.text_area("Text prompt", placeholder="e.g. calm piano")
        
        button_disabled = embedder_type == "audio" and audio is None or \
                          embedder_type == "text" and text.strip() == "" or \
                          embedder_type == "multimodal" and (audio is None or text.strip() == "")

        if st.button("Generate", disabled=button_disabled):
            audio_emb = None

            if embedder_type == "audio" or embedder_type == "multimodal":
                temp_path = Path("temp_audio.wav")
                with open(temp_path, "wb") as f:
                    f.write(audio.read())
                audio_emb = embedder.embed_audio(temp_path)
                st.subheader("Audio Embedding")
                st.write(audio_emb.vectors[0].shape)

            text_emb = None
            if embedder_type == "text" or embedder_type == "multimodal":
                text_emb = embedder.embed_text(text)
                st.subheader("Text Embedding")
                st.write(text_emb.vectors[0].shape)

            if embedder_type == "multimodal":
                similarity = cosine_similarity(audio_emb.vectors[0], text_emb.vectors[0])
                st.metric("Cosine similarity", f"{similarity:.4f}")
                st.subheader("Heatmap of embedding differences")
                plot_embedding_heatmap(audio_emb.vectors[0], text_emb.vectors[0])
                st.subheader("2D PCA projection")
                plot_embedding_pca(audio_emb.vectors[0], text_emb.vectors[0])
>>>>>>> Stashed changes
