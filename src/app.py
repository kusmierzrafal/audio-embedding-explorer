import os
from pathlib import Path
from huggingface_hub import snapshot_download
import librosa
import streamlit as st
import torch
from transformers import AutoProcessor, AutoTokenizer, ClapModel
from dotenv import load_dotenv
import torch.nn.functional as F


def resolve_paths():
    global CURRENT_FILE, REPO_ROOT, MODELS_DIR, DATASET_DIR
    CURRENT_FILE = Path(__file__).resolve()
    SRC_ROOT = CURRENT_FILE.parents[0]
    REPO_ROOT = CURRENT_FILE.parents[1]
    MODELS_DIR = REPO_ROOT / "models"
    DATASET_DIR = REPO_ROOT / "dataset"


def load_env_variables():
    load_dotenv()
    global CLAP_HF_NAME, CLAP_DIR_NAME
    CLAP_HF_NAME = os.environ["CLAP_HF_NAME"]
    CLAP_DIR_NAME = os.environ["CLAP_DIR_NAME"]


def ensure_clap_exists(clap_model_dir):
    if not os.path.exists(clap_model_dir):
        st.info(f"Model {CLAP_HF_NAME} not found in {clap_model_dir}")
        
        os.makedirs(clap_model_dir, exist_ok=True)
        with st.spinner(f"Downloading model {CLAP_HF_NAME} to directory {clap_model_dir}..."):
            snapshot_download(
                repo_id=CLAP_HF_NAME,
                local_dir=clap_model_dir
            )
        st.success(f"Model {CLAP_HF_NAME} downloaded to {clap_model_dir}")
    else:
        st.info(f"Model {CLAP_HF_NAME} already exists in {clap_model_dir}")


def load_clap_model(clap_model_dir):
    ensure_clap_exists(clap_model_dir)
    model = ClapModel.from_pretrained(clap_model_dir)
    processor = AutoProcessor.from_pretrained(clap_model_dir)
    tokenizer = AutoTokenizer.from_pretrained(clap_model_dir)
    return model, processor, tokenizer


def calculate_text_embedding(model, tokenizer, user_text):
    with st.spinner("Processing text..."):
        text_input = tokenizer(user_text, return_tensors="pt")
        with torch.no_grad():
            text_output = model.get_text_features(**text_input)
    st.success("Text processed.")
    return text_output


def calculate_audio_embedding(model, processor, audio_file):
    with st.spinner("Processing audio..."):
        waveform, sr = librosa.load(audio_file, sr=48000, mono=True)
        waveform = waveform.astype("float32")
        audio_input = processor(audios=[waveform], return_tensors="pt", sampling_rate=sr)
        with torch.no_grad():
            audio_output = model.get_audio_features(**audio_input)
    st.success("Audio processed.")
    return audio_output


def main():
    st.title("Audio Embedding Explorer")

    with st.spinner("Loading environment..."):
        resolve_paths()
        load_env_variables()
        clap_model_dir = MODELS_DIR / CLAP_DIR_NAME
    st.success("Environment variables loaded.")
    
    with st.spinner(f"Loading model {CLAP_HF_NAME} from directory {clap_model_dir}..."):
        load_clap_model(clap_model_dir)
        model, processor, tokenizer = load_clap_model(clap_model_dir)
    st.success(f"Model {CLAP_HF_NAME} loaded.")

    audio_file = st.file_uploader("Upload audio file", type=["wav"])
    user_text = st.text_input("Enter text")

    text_embedding = None
    if user_text is not None and user_text.strip() != "":
        text_embedding = calculate_text_embedding(model, tokenizer, user_text)
        st.write(f"Text Embedding {tuple(text_embedding.shape)}:")
        # st.write(text_embedding)

    audio_embedding = None
    if audio_file is not None:
        audio_embedding = calculate_audio_embedding(model, processor, audio_file)
        st.write(f"Audio Embedding {tuple(audio_embedding.shape)}:")
        # st.write(audio_embedding)

    if audio_embedding is not None and text_embedding is not None:
        similarity = F.cosine_similarity(audio_embedding, text_embedding)
        st.write(f"Similarity score: {similarity.item():.4f}")
    else:
        st.info("Upload both audio and text to compute similarity.")

    
if __name__ == "__main__":
    main()