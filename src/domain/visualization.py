import io
from typing import List, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import torch
from sklearn.decomposition import PCA

from src.domain.metrics import reduce_embeddings_dimensionality
from src.models.enums.reduce_dimensions_method import ReduceDimensionsMethod
from src.utils.import_utils import optional_import


def plot_pca_projection(
    audio_emb: torch.Tensor,
    text_emb: torch.Tensor,
    method: Union[ReduceDimensionsMethod, str],
) -> None:
    if method != ReduceDimensionsMethod.PCA:
        st.warning("This visualization is available only for PCA projection.")
        return

    a = audio_emb.detach().cpu().numpy().reshape(1, -1)
    t = text_emb.detach().cpu().numpy().reshape(1, -1)
    combined = np.vstack([a, t])

    points_2d = reduce_embeddings_dimensionality(combined, method)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(points_2d[0, 0], points_2d[0, 1], color="blue", label="Audio", s=80)
    ax.scatter(points_2d[1, 0], points_2d[1, 1], color="red", label="Text", s=80)
    ax.legend()
    ax.set_title(f"{method} projection of embeddings (2D)")
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    st.image(buf, width=650)


def plotly_pca_projection(
    audio_emb: torch.Tensor,
    text_emb: torch.Tensor,
    method: Union[ReduceDimensionsMethod, str],
) -> None:
    if method != ReduceDimensionsMethod.PCA:
        st.warning("This visualization is available only for PCA projection.")
        return

    a = audio_emb.detach().cpu().numpy().reshape(1, -1)
    t = text_emb.detach().cpu().numpy().reshape(1, -1)
    combined = np.vstack([a, t])
    labels = ["Audio", "Text"]

    points_2d = reduce_embeddings_dimensionality(combined, method)

    df = pd.DataFrame({"x": points_2d[:, 0], "y": points_2d[:, 1], "label": labels})

    fig = px.scatter(
        df,
        x="x",
        y="y",
        color="label",
        color_discrete_map={"Audio": "#3B82F6", "Text": "#EF4444"},
        size=[18, 18],
        hover_name="label",
        title=f"2D {method.name} Projection of Embeddings",
    )

    fig.update_layout(
        template="plotly_dark",
        font=dict(size=14, family="Arial"),
        xaxis_title="Component 1",
        yaxis_title="Component 2",
        xaxis=dict(
            showgrid=True,
            zeroline=True,
            showline=True,
            mirror=True,
            tickformat=".2e",
            exponentformat="e",
        ),
        yaxis=dict(
            showgrid=True,
            zeroline=True,
            showline=True,
            mirror=True,
            tickformat=".2e",
            exponentformat="e",
        ),
        legend_title_text=None,
        width=700,
        height=500,
        margin=dict(l=40, r=40, t=60, b=40),
    )

    fig.update_traces(marker=dict(line=dict(width=1, color="white")))

    st.plotly_chart(fig, use_container_width=True)


def plot_2d(points: np.ndarray, labels: List[str], title: str):
    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")

    if points.size == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.set_axis_off()
        return fig

    ax.scatter(points[:, 0], points[:, 1])
    for (x, y), lab in zip(points, labels):
        ax.annotate(lab, (x, y), fontsize=8)

    ax.grid(True, linestyle="--", linewidth=0.5)
    return fig


def compute_pca_fig(names: List[str], X: np.ndarray):
    if X.ndim != 2 or X.shape[0] < 2:
        return plot_2d(
            np.zeros((len(names), 2), dtype=np.float32),
            names,
            "PCA (needs ≥ 2 samples)",
        )
    pca = PCA(n_components=2)
    pts = pca.fit_transform(X)
    return plot_2d(pts, names, "PCA")


def compute_umap_fig(names: List[str], X: np.ndarray):
    UMAP, ok = optional_import("umap", "UMAP")
    if not ok or UMAP is None:
        fig, ax = plt.subplots()
        ax.set_title("UMAP (not available)")
        ax.text(
            0.5,
            0.5,
            "Install 'umap-learn' to enable UMAP.",
            ha="center",
            va="center",
        )
        ax.set_axis_off()
        return fig

    if X.ndim != 2 or X.shape[0] < 2:
        return plot_2d(
            np.zeros((len(names), 2), dtype=np.float32),
            names,
            "UMAP (needs ≥ 2 samples)",
        )

    reducer = UMAP(n_components=2, random_state=42)
    pts = reducer.fit_transform(X)
    return plot_2d(pts, names, "UMAP")
