import io
from typing import List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
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

    # Convert to numpy arrays (handle both torch tensors and numpy arrays)
    if isinstance(audio_emb, torch.Tensor):
        a = audio_emb.detach().cpu().numpy().reshape(1, -1)
    else:
        a = np.asarray(audio_emb).reshape(1, -1)

    if isinstance(text_emb, torch.Tensor):
        t = text_emb.detach().cpu().numpy().reshape(1, -1)
    else:
        t = np.asarray(text_emb).reshape(1, -1)

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

    if X.ndim != 2 or X.shape[0] < 5:
        return plot_2d(
            np.zeros((len(names), 2), dtype=np.float32),
            names,
            "UMAP (needs ≥ 5 samples)",
        )

    reducer = UMAP(n_components=2, random_state=42)
    pts = reducer.fit_transform(X)
    return plot_2d(pts, names, "UMAP")


def compute_interactive_pca_fig(
    names: List[str], X: np.ndarray, audio_ids: Optional[List[int]] = None
):
    """Create interactive PCA plot using plotly"""
    if X.ndim != 2 or X.shape[0] < 2:
        # Create empty plotly figure
        fig = go.Figure()
        fig.update_layout(
            title="PCA (needs ≥ 2 samples)",
            xaxis_title="Component 1",
            yaxis_title="Component 2",
        )
        fig.add_annotation(
            text="Need at least 2 samples for PCA",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
        )
        return fig

    pca = PCA(n_components=2)
    pts = pca.fit_transform(X)

    # Create DataFrame for plotly
    df = pd.DataFrame(
        {
            "PC1": pts[:, 0],
            "PC2": pts[:, 1],
            "name": names,
            "audio_id": audio_ids if audio_ids else list(range(len(names))),
        }
    )

    fig = px.scatter(
        df,
        x="PC1",
        y="PC2",
        hover_data=["name", "audio_id"],
        title="PCA - Interactive (click points for details)",
        labels={"PC1": "Component 1", "PC2": "Component 2"},
    )

    fig.update_traces(
        marker=dict(size=8, line=dict(width=1, color="white")),
        hovertemplate="<b>%{customdata[0]}</b><br>"
        + "PC1: %{x:.3f}<br>"
        + "PC2: %{y:.3f}<br>"
        + "Audio ID: %{customdata[1]}<extra></extra>",
    )

    fig.update_layout(width=700, height=500, hovermode="closest")

    return fig


def compute_interactive_umap_fig(
    names: List[str], X: np.ndarray, audio_ids: Optional[List[int]] = None
):
    """Create interactive UMAP plot using plotly"""
    UMAP, ok = optional_import("umap", "UMAP")
    if not ok or UMAP is None:
        fig = go.Figure()
        fig.update_layout(
            title="UMAP (not available)",
            xaxis_title="UMAP 1",
            yaxis_title="UMAP 2",
        )
        fig.add_annotation(
            text="Install 'umap-learn' to enable UMAP",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
        )
        return fig

    if X.ndim != 2 or X.shape[0] < 5:
        fig = go.Figure()
        fig.update_layout(
            title="UMAP (needs ≥ 5 samples)",
            xaxis_title="UMAP 1",
            yaxis_title="UMAP 2",
        )
        fig.add_annotation(
            text="Need at least 5 samples for UMAP",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
        )
        return fig

    reducer = UMAP(n_components=2, random_state=42)
    pts = reducer.fit_transform(X)

    # Create DataFrame for plotly
    df = pd.DataFrame(
        {
            "UMAP1": pts[:, 0],
            "UMAP2": pts[:, 1],
            "name": names,
            "audio_id": audio_ids if audio_ids else list(range(len(names))),
        }
    )

    fig = px.scatter(
        df,
        x="UMAP1",
        y="UMAP2",
        hover_data=["name", "audio_id"],
        title="UMAP - Interactive (click points for details)",
        labels={"UMAP1": "UMAP 1", "UMAP2": "UMAP 2"},
    )

    fig.update_traces(
        marker=dict(size=8, line=dict(width=1, color="white")),
        hovertemplate="<b>%{customdata[0]}</b><br>"
        + "UMAP1: %{x:.3f}<br>"
        + "UMAP2: %{y:.3f}<br>"
        + "Audio ID: %{customdata[1]}<extra></extra>",
    )

    fig.update_layout(width=700, height=500, hovermode="closest")

    return fig
