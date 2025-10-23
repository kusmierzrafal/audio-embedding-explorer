import io

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch
from sklearn.decomposition import PCA


def plot_embedding_heatmap(audio_emb: torch.Tensor, text_emb: torch.Tensor) -> None:
    a = audio_emb.detach().cpu().numpy()
    t = text_emb.detach().cpu().numpy()

    a = a.reshape(1, -1)
    t = t.reshape(1, -1)

    diff = np.abs(a - t)

    fig, ax = plt.subplots()
    im = ax.imshow(diff, cmap="coolwarm", aspect="auto")
    ax.set_title("Heatmapa różnic między embeddingami")
    ax.set_xlabel("Wymiar wektora embeddingu")
    ax.set_ylabel("Audio ↔ Text")
    fig.colorbar(im, ax=ax)
    # st.pyplot(fig)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    st.image(buf, width=650)


def plot_embedding_pca(audio_emb: torch.Tensor, text_emb: torch.Tensor) -> None:
    a = audio_emb.detach().cpu().numpy().reshape(1, -1)
    t = text_emb.detach().cpu().numpy().reshape(1, -1)
    combined = np.vstack([a, t])

    pca = PCA(n_components=2)
    points_2d = pca.fit_transform(combined)
    fig, ax = plt.subplots()
    ax.scatter(points_2d[0, 0], points_2d[0, 1], color="blue", label="Audio")
    ax.scatter(points_2d[1, 0], points_2d[1, 1], color="red", label="Text")
    ax.set_title("PCA embeddingów (2D)")
    ax.legend()
    # st.pyplot(fig, width='content')
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    st.image(buf, width=650)
