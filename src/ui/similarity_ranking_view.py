import time

import streamlit as st

from src.domain.embeddings.base_embedders import AudioEmbedder, TextEmbedder
from src.domain.embeddings.models_manager import ModelsManager
from src.domain.metrics import cosine_similarity
from src.ui.shared.base_view import BaseView
from src.utils.audio_utils import AudioHelper


class SimilarityRankingView(BaseView):
    title = "Similarity Ranking"
    description = "Compare text descriptions against uploaded audio, or vice versa."

    def compute_text_to_audio(
        self, clap: AudioEmbedder | TextEmbedder, text, audio_files
    ):
        with st.spinner("Computing similarities..."):
            text_emb = clap.embed_text(text).vector
            results = []

            for file in audio_files:
                y, sr = AudioHelper.load_audio(file, clap.get_sr())
                audio_emb = clap.embed_audio(y, sr).vector
                sim = cosine_similarity(audio_emb, text_emb)

                results.append(
                    {
                        "text": text,
                        "audio": file.name,
                        "similarity": float(sim),
                        "file": file,
                    }
                )

            results.sort(key=lambda x: x["similarity"], reverse=True)
            time.sleep(0.5)
        return results

    def display_text_to_audio_results(self, results, text_input):
        st.markdown(
            f"""
            <h3 style='margin-bottom: 12px;'>Text description: <code>{text_input}</code>
            </h3>
            """,
            unsafe_allow_html=True,
        )

        header_cols = st.columns([0.3, 3, 1, 2])
        with header_cols[0]:
            st.markdown("**No.**")
        with header_cols[1]:
            st.markdown("**Audio file name**")
        with header_cols[2]:
            st.markdown("**Similarity**")
        with header_cols[3]:
            st.markdown("**Preview**")

        st.markdown(
            "<hr style='margin: 4px 0; "
            "margin-bottom: 26px; "
            "border: 1px solid rgba(255,255,255,0.1);'"
            "/>",
            unsafe_allow_html=True,
        )

        for i, r in enumerate(results, start=1):
            col1, col2, col3, col4 = st.columns([0.3, 3, 1, 2])
            with col1:
                st.markdown(f"**{i}.**")
            with col2:
                st.markdown(f"{r['audio']}")
            with col3:
                similarity = r["similarity"]
                if similarity > 0.7:
                    st.markdown(f":green[**{similarity:.4f}**]")
                elif similarity > 0.3:
                    st.markdown(f":orange[**{similarity:.4f}**]")
                else:
                    st.markdown(f":red[**{similarity:.4f}**]")

            with col4:
                st.audio(r["file"], format="audio/wav")

    def compute_audio_to_text(
        self, 
        clap: AudioEmbedder | TextEmbedder, 
        audio_file, 
        texts):
        with st.spinner("Computing similarities..."):
            y, sr = AudioHelper.load_audio(audio_file, clap.get_sr())
            audio_emb = clap.embed_audio(y, sr).vector
            text_embs = [clap.embed_text(t).vector for t in texts]

            results = []
            for i, text_emb in enumerate(text_embs):
                sim = cosine_similarity(audio_emb, text_emb)
                results.append(
                    {
                        "text": texts[i],
                        "audio": audio_file.name,
                        "similarity": float(sim),
                    }
                )
            results.sort(key=lambda x: x["similarity"], reverse=True)
            time.sleep(0.5)
        return results

    def display_audio_to_text_results(self, results, audio_input):
        st.markdown(
            f"""
                 <h3 style='margin-bottom: 12px;'>Audio: <code>{audio_input}</code></h3>
                 """,
            unsafe_allow_html=True,
        )

        header_cols = st.columns([0.3, 1, 1])
        with header_cols[0]:
            st.markdown("**No.**")
        with header_cols[1]:
            st.markdown("**Input text**")
        with header_cols[2]:
            st.markdown("**Similarity**")

        st.markdown(
            "<hr style='margin: 4px 0; "
            "margin-bottom: 26px; "
            "border: 1px solid rgba(255,255,255,0.1);' "
            "/>",
            unsafe_allow_html=True,
        )

        for i, r in enumerate(results, start=1):
            col1, col2, col3 = st.columns([0.3, 1, 1])
            with col1:
                st.markdown(f"**{i}.**")
            with col2:
                st.markdown(f"{r['text']}")
            with col3:
                similarity = r["similarity"]
                if similarity > 0.7:
                    st.markdown(f":green[**{similarity:.4f}**]")
                elif similarity > 0.3:
                    st.markdown(f":orange[**{similarity:.4f}**]")
                else:
                    st.markdown(f":red[**{similarity:.4f}**]")

    def render(self) -> None:
        self.header()

        models_manager: ModelsManager = st.session_state["models_manager"]
        clap = models_manager.get_model("laion/clap-htsat-unfused").embedder
        st.text("Using model: laion/clap-htsat-unfused")

        tab1, tab2 = st.tabs(["Text → Audio", "Audio → Text"])

        with tab1:
            text_input = st.text_input(
                "Enter one text descriptions:",
                placeholder="calm piano",
                key="t2a_texts",
            )
            uploaded_audios = st.file_uploader(
                "Upload more than one audio files:",
                type=["wav", "mp3"],
                accept_multiple_files=True,
                key="t2a_audios",
            )
            if text_input and uploaded_audios:
                if st.button("Compute similarities", key="btn_t2a"):
                    results = self.compute_text_to_audio(
                        clap, text_input, uploaded_audios
                    )
                    self.display_text_to_audio_results(results, text_input)
            else:
                st.info("Please provide both text descriptions and audio files.")

        with tab2:
            uploaded_audio = st.file_uploader(
                "Upload one audio file:",
                type=["wav", "mp3"],
                accept_multiple_files=False,
                key="a2t_audio",
            )

            if uploaded_audio is not None:
                st.audio(uploaded_audio, format="audio/wav")

            text_input = st.text_area(
                "Enter text descriptions (one per line):",
                placeholder="calm piano\nenergetic rock\ndark ambient",
                key="a2t_texts",
            )
            if uploaded_audio and text_input:
                if st.button("Compute similarities", key="btn_a2t"):
                    texts = [t.strip() for t in text_input.split("\n") if t.strip()]
                    results = self.compute_audio_to_text(clap, uploaded_audio, texts)
                    self.display_audio_to_text_results(results, uploaded_audio.name)
            else:
                st.info(
                    "Please provide both a single audio file and text descriptions."
                )
