"""
embeddings.py
-------------
Generates semantic embeddings of movie titles with a pre-trained
Sentence-Transformer (all-MiniLM-L6-v2, 384-dim). Kept separate from
model.py because this embedding step is "frozen" — we don't fine-tune the
transformer, we just use it as a feature extractor — whereas model.py
contains the autoencoder we *do* train.

--------------------------------------------------------------------------
HOW TO SCALE THIS STEP (read before pointing it at >100k titles)
--------------------------------------------------------------------------
1. Batch size: SENTENCE_BATCH_SIZE in config.py controls GPU memory vs.
   throughput. On a single modern GPU, 256-512 is usually safe for short
   titles; raise it until you see diminishing returns or OOM, then back off.

2. Caching: embeddings are deterministic for a frozen model, so cache them
   keyed by a hash of the title text and only embed *new or changed*
   titles on each pipeline run. `_load_cache` / `_save_cache` below show
   the minimal version of this; swap the JSON file for Redis/SQLite if the
   catalog grows past a few hundred thousand rows.

3. GPU batching across a larger corpus: SentenceTransformer.encode already
   batches internally, but for multi-million-row corpora prefer:
       - `model.encode(..., batch_size=512, show_progress_bar=True)`
       - splitting the corpus across multiple processes/GPUs with
         `SentenceTransformer.start_multi_process_pool` and
         `encode_multi_process`, which shards batches across all visible
         CUDA devices.

4. Throughput at serving time (real-time queries, not the offline
   pipeline): swap in a quantized/ONNX export of MiniLM
   (`optimum[onnxruntime]`) or move encoding to a small dedicated
   embedding microservice so the API process never loads the full model.

5. If the catalog grows beyond ~1M titles, also consider an approximate
   nearest-neighbour index (FAISS, ScaNN, or pgvector) on the embeddings
   themselves, so "find similar titles" queries don't require a full
   linear scan even before they reach HDBSCAN.
--------------------------------------------------------------------------
"""
from __future__ import annotations

import hashlib
import json
import logging

import numpy as np

from config import DATA_DIR, SENTENCE_BATCH_SIZE, SENTENCE_EMBEDDING_DIM, SENTENCE_MODEL_NAME

logger = logging.getLogger(__name__)

_CACHE_PATH = DATA_DIR / "title_embedding_cache.json"


def _hash_title(title: str) -> str:
    return hashlib.sha1(title.strip().lower().encode("utf-8")).hexdigest()


def _load_cache() -> dict:
    if _CACHE_PATH.exists():
        with open(_CACHE_PATH, "r") as f:
            return json.load(f)
    return {}


def _save_cache(cache: dict) -> None:
    with open(_CACHE_PATH, "w") as f:
        json.dump(cache, f)


def embed_titles(titles: list, device: str = None, use_cache: bool = True) -> np.ndarray:
    """
    Returns an (N, SENTENCE_EMBEDDING_DIM) float32 array of title embeddings.

    `device` is auto-detected (CUDA if available) when None — see
    model.py's `get_device()` for the same pattern used by the autoencoder.
    """
    try:
        import torch
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:  # pragma: no cover - guidance for local setup
        raise ImportError(
            "sentence-transformers/torch not installed. Run "
            "`pip install -r requirements.txt` (see that file for the correct "
            "torch build for your CUDA version)."
        ) from exc

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Loading %s on %s", SENTENCE_MODEL_NAME, device)
    model = SentenceTransformer(SENTENCE_MODEL_NAME, device=device)

    cache = _load_cache() if use_cache else {}
    hashes = [_hash_title(t) for t in titles]

    to_encode_idx = [i for i, h in enumerate(hashes) if h not in cache]
    if to_encode_idx:
        logger.info("Embedding %d/%d uncached titles", len(to_encode_idx), len(titles))
        new_vectors = model.encode(
            [titles[i] for i in to_encode_idx],
            batch_size=SENTENCE_BATCH_SIZE,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,  # cosine-ready embeddings
        )
        for idx, vec in zip(to_encode_idx, new_vectors):
            cache[hashes[idx]] = vec.tolist()
        if use_cache:
            _save_cache(cache)
    else:
        logger.info("All %d titles found in embedding cache", len(titles))

    embeddings = np.array([cache[h] for h in hashes], dtype=np.float32)
    assert embeddings.shape == (len(titles), SENTENCE_EMBEDDING_DIM)
    return embeddings


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sample = ["The Dark Knight", "Eternal Sunshine of the Spotless Mind", "Mad Max: Fury Road"]
    vecs = embed_titles(sample)
    print(vecs.shape)
