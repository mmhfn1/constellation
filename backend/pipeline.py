"""
pipeline.py
------------
Orchestrates the full *offline* pipeline:

    raw CSV -> data_loader -> [metadata features] -+
                                                     +-> concat -> model.py (train) -> latent vectors
    titles  -> embeddings.py -> [semantic features] -+                                    |
                                                                                            v
                                                                            cluster_service.py (HDBSCAN + UMAP)
                                                                                            |
                                                                                            v
                                                                json artifacts in backend/data/
                                                                (graph_tier1, movie_details, cluster_summary)

This is meant to be run *once* (or periodically, e.g. nightly, as new
movies are added) — NOT on every API request. app.py just reads the JSON
artifacts this script produces, which is what keeps the API fast (see the
two-tier design in app.py's docstring).

Usage:
    python pipeline.py                  # uses backend/data/imdb_top_movies_1980_2026.csv
    python generate_sample_data.py      # ...or generate a synthetic CSV first
"""
from __future__ import annotations

import json
import logging

import numpy as np

import cluster_service
import data_loader
import embeddings
from config import (
    CLUSTER_SUMMARY_PATH,
    GRAPH_TIER1_PATH,
    LATENT_VECTORS_PATH,
    MOVIE_DETAILS_PATH,
    SENTENCE_EMBEDDING_DIM,
)

logger = logging.getLogger(__name__)


def run_pipeline() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    # 1. Data prep ----------------------------------------------------------
    dataset = data_loader.prepare_dataset()
    metadata = dataset.metadata

    # 2. Semantic title embeddings, concatenated with metadata features -----
    try:
        title_embeddings = embeddings.embed_titles(metadata["title"].tolist())
    except ImportError as exc:
        logger.warning(
            "%s — falling back to zero vectors for title embeddings so the rest "
            "of the pipeline can still be exercised. Install torch + "
            "sentence-transformers for real semantic embeddings.",
            exc,
        )
        title_embeddings = np.zeros((len(metadata), SENTENCE_EMBEDDING_DIM), dtype=np.float32)

    combined_features = np.concatenate([dataset.feature_matrix, title_embeddings], axis=1)
    logger.info(
        "Combined feature matrix: %d metadata dims + %d embedding dims = %d total",
        dataset.feature_matrix.shape[1], title_embeddings.shape[1], combined_features.shape[1],
    )

    # 3. Train the autoencoder, get latent vectors for every movie ----------
    try:
        import model  # deferred: this import itself requires torch to be installed
        trained_model, device, history = model.train_autoencoder(combined_features)
        latent_vectors = model.encode_all(trained_model, combined_features, device)
        logger.info("Final test reconstruction loss: %.5f", history["test_loss"][-1])
    except ImportError as exc:
        logger.warning("%s — falling back to PCA so clustering/projection can still run.", exc)
        from sklearn.decomposition import PCA

        n_components = 16
        latent_vectors = PCA(n_components=n_components, random_state=42).fit_transform(combined_features)

    np.save(LATENT_VECTORS_PATH, latent_vectors)

    # 4. Cluster into archetypes + project to 2D for the map ----------------
    try:
        labels, coords, cluster_summaries = cluster_service.build_cluster_artifacts(
            metadata, latent_vectors, dataset.genre_columns
        )
    except ImportError as exc:
        logger.warning(
            "%s — falling back to KMeans + PCA(2D) so you can see real output "
            "end-to-end. Install hdbscan + umap-learn for true density-based "
            "archetypes and a neighborhood-preserving 2D layout.",
            exc,
        )
        from sklearn.cluster import KMeans
        from sklearn.decomposition import PCA

        k = max(2, min(20, len(metadata) // 200))
        labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(latent_vectors)
        coords = PCA(n_components=2, random_state=42).fit_transform(latent_vectors).astype(np.float32)
        cluster_summaries = cluster_service.summarize_clusters(metadata, labels, coords, dataset.genre_columns)

    # 5. Write the two-tier artifacts app.py will serve ----------------------
    write_tier1_graph(metadata, labels, coords)
    write_tier2_details(metadata)
    write_cluster_summary(cluster_summaries)

    logger.info("Pipeline complete. Artifacts written to backend/data/.")


def write_tier1_graph(metadata, labels: np.ndarray, coords: np.ndarray) -> None:
    """Tier 1: the *only* thing needed to draw the map. Deliberately just
    four numbers per movie so this payload stays small even at 16k+ rows."""
    records = [
        {
            "movie_id": str(mid),
            "x": float(x),
            "y": float(y),
            "cluster_id": int(cid),
        }
        for mid, x, y, cid in zip(metadata["movie_id"], coords[:, 0], coords[:, 1], labels)
    ]
    with open(GRAPH_TIER1_PATH, "w") as f:
        json.dump(records, f)
    logger.info("Wrote tier-1 graph data (%d nodes) to %s", len(records), GRAPH_TIER1_PATH)


def write_tier2_details(metadata) -> None:
    """Tier 2: rich metadata, looked up by movie_id on demand (hover/click)
    instead of shipped up front."""
    details = {}
    for _, row in metadata.iterrows():
        details[str(row["movie_id"])] = {
            "movie_id": str(row["movie_id"]),
            "title": row["title"],
            "year": int(row["year"]),
            "runtime_minutes": int(row["runtime_minutes"]),
            "imdb_rating": float(row["imdb_rating"]),
            "num_votes": int(row["num_votes"]),
            "genres": row["genres"],
            "poster_url": row.get("poster_url", None),
            "imdb_url": row.get("imdb_url", None),
        }
    with open(MOVIE_DETAILS_PATH, "w") as f:
        json.dump(details, f)
    logger.info("Wrote tier-2 movie details (%d movies) to %s", len(details), MOVIE_DETAILS_PATH)


def write_cluster_summary(cluster_summaries: list) -> None:
    with open(CLUSTER_SUMMARY_PATH, "w") as f:
        json.dump(cluster_summaries, f)
    logger.info("Wrote %d cluster summaries to %s", len(cluster_summaries), CLUSTER_SUMMARY_PATH)


if __name__ == "__main__":
    run_pipeline()
