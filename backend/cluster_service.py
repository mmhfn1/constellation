"""
cluster_service.py
-------------------
Turns latent vectors into:
  1. A 2D (x, y) projection via UMAP — the visible coordinates on the map.
  2. cluster_id per movie, via HDBSCAN *on the 2D UMAP output*.

CRITICAL ORDERING: We run UMAP first, then HDBSCAN in that 2D space.
Running HDBSCAN on the high-D latent space independently from UMAP produces
clusters that are semantically coherent but spatially scattered on the map —
a movie shows inside a cluster blob that doesn't contain it, and selection
rings float in empty space. Clustering on the 2D UMAP output guarantees that
every movie in the same archetype is physically adjacent on the visible map.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from config import (
    HDBSCAN_MIN_CLUSTER_SIZE,
    HDBSCAN_MIN_SAMPLES,
    REASSIGN_NOISE_POINTS,
    UMAP_MIN_DIST,
    UMAP_N_COMPONENTS,
    UMAP_N_NEIGHBORS,
    UMAP_SPREAD,
)

logger = logging.getLogger(__name__)

NOISE_LABEL = -1


def project_to_2d(latent_vectors: np.ndarray) -> np.ndarray:
    """Project latent vectors to 2D UMAP coordinates.

    Run this BEFORE run_hdbscan so clustering happens in the visible 2D space.

    n_neighbors=20: wider neighbourhood than the default 15 captures more
        global structure, so related-but-distant movies still pull together.
    min_dist=0.05: much tighter than the default 0.1 — creates distinct visual
        cluster blobs with clearer gaps between archetypes.
    """
    try:
        import umap as umap_module
    except ImportError as exc:
        raise ImportError("umap-learn not installed: pip install umap-learn") from exc

    reducer = umap_module.UMAP(
        n_components=UMAP_N_COMPONENTS,
        n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        spread=UMAP_SPREAD,
        metric="euclidean",
        random_state=42,
        low_memory=False,
    )
    coords = reducer.fit_transform(latent_vectors)
    logger.info(
        "UMAP: %d vectors %dD→2D (n_neighbors=%d, min_dist=%.2f)",
        len(latent_vectors), latent_vectors.shape[1], UMAP_N_NEIGHBORS, UMAP_MIN_DIST,
    )
    return coords.astype(np.float32)


def run_hdbscan(coords_2d: np.ndarray, reassign_noise: bool = REASSIGN_NOISE_POINTS) -> np.ndarray:
    """Run HDBSCAN on 2D coordinates (not the high-D latent space).

    This guarantees spatial coherence: a cluster's members are literally
    adjacent on the map. With reassign_noise=True (default), points marked
    noise are soft-assigned to their best-fitting archetype so no movie
    appears as an isolated island.
    """
    try:
        import hdbscan
    except ImportError as exc:
        raise ImportError("hdbscan not installed: pip install hdbscan") from exc

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
        min_samples=HDBSCAN_MIN_SAMPLES,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=reassign_noise,
    )
    labels = clusterer.fit_predict(coords_2d)
    n_clusters = len(set(labels)) - (1 if NOISE_LABEL in labels else 0)
    n_noise = int((labels == NOISE_LABEL).sum())
    logger.info("HDBSCAN (2D): %d archetypes, %d noise points", n_clusters, n_noise)

    if reassign_noise and n_noise > 0 and n_clusters > 0:
        membership = hdbscan.all_points_membership_vectors(clusterer)
        noise_mask = labels == NOISE_LABEL
        labels = labels.copy()
        labels[noise_mask] = membership[noise_mask].argmax(axis=1)
        logger.info("Reassigned %d noise points to nearest archetype", n_noise)

    return labels


def summarize_clusters(metadata: pd.DataFrame, labels: np.ndarray, coords: np.ndarray, genre_columns: list) -> list:
    """Build centroid summaries — galaxy bubble positions, sizes and genre labels."""
    df = metadata.copy()
    df["cluster_id"] = labels
    df["x"], df["y"] = coords[:, 0], coords[:, 1]

    summaries = []
    for cluster_id, group in df.groupby("cluster_id"):
        if cluster_id == NOISE_LABEL:
            continue
        genre_counts = group[genre_columns].sum().sort_values(ascending=False)
        top_genres = [c.replace("genre_", "") for c in genre_counts.head(3).index if genre_counts[c] > 0]
        summaries.append({
            "cluster_id": int(cluster_id),
            "x": float(group["x"].mean()),
            "y": float(group["y"].mean()),
            "size": int(len(group)),
            "avg_rating": round(float(group["imdb_rating"].mean()), 2),
            "top_genres": top_genres,
            "label": " / ".join(top_genres) if top_genres else f"Archetype {cluster_id}",
            "representative_titles": group.nlargest(3, "num_votes")["title"].tolist(),
        })

    summaries.sort(key=lambda s: -s["size"])
    logger.info("Summarized %d clusters", len(summaries))
    return summaries


def build_cluster_artifacts(metadata: pd.DataFrame, latent_vectors: np.ndarray, genre_columns: list):
    """Main entry point. ORDER MATTERS: 2D first, cluster in 2D."""
    coords = project_to_2d(latent_vectors)          # step 1: 2D positions
    labels = run_hdbscan(coords)                     # step 2: cluster IN 2D
    summaries = summarize_clusters(metadata, labels, coords, genre_columns)
    return labels, coords, summaries


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    dummy = np.random.randn(500, 16).astype(np.float32)
    coords = project_to_2d(dummy)
    labels = run_hdbscan(coords)
    print(labels[:10], coords[:5])
