"""
app.py
------
FastAPI surface for the Movie Similarity Map. Deliberately thin: all the
heavy computation (autoencoder training, HDBSCAN, UMAP) happens offline in
pipeline.py. This file just serves precomputed JSON and logs interactions.

Two-tier API (the whole point being: don't make 16k+ nodes carry rich
metadata they don't need until a user actually looks at them):

  Tier 1 — GET /api/graph
      -> [{movie_id, x, y, cluster_id}, ...] for every movie.
      This is the only payload needed to draw the full scatter/force graph.
      Small (4 fields/row) so it stays cheap even at 16k+ rows.

  Tier 1.5 — GET /api/clusters
      -> centroid summaries (position, size, label, top genres).
      Powers the initial "zoomed out" hierarchical view.

  Tier 2 — GET /api/movies/{movie_id}
      -> rich metadata (title, rating, poster, IMDb url). Fetched on
      demand by the frontend's hover/click handlers, not bundled into
      Tier 1.

  POST /api/interactions
      -> logs a click/like/etc. to SQLite for the feedback loop.

Run with:
    uvicorn app:app --reload --port 8000
"""
from __future__ import annotations

import json
import logging
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import database
from config import CLUSTER_SUMMARY_PATH, GRAPH_TIER1_PATH, MOVIE_DETAILS_PATH, DEFAULT_CORS_ORIGINS, EXTRA_CORS_ORIGINS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Movie Similarity Map API", version="1.0.0")

# Local dev origins + anything in ALLOWED_ORIGINS are allowed explicitly;
# allow_origin_regex additionally allows *any* GitHub Pages site (covers
# both project pages and <user>.github.io) without needing to know the
# exact username/repo ahead of time. Only relevant if you're running this
# backend live and pointing a separately-hosted frontend at it — see the
# README's "Optional: a live backend" section.
app.add_middleware(
    CORSMiddleware,
    allow_origins=DEFAULT_CORS_ORIGINS + EXTRA_CORS_ORIGINS,
    allow_origin_regex=r"https://[\w-]+\.github\.io",
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory cache of the precomputed artifacts. They're written once by
# pipeline.py and only re-read on process restart — there's no reason to
# touch disk on every request for data this size.
_graph_cache: list = []
_details_cache: dict = {}
_cluster_cache: list = []
_position_by_id: dict = {}  # movie_id -> {x, y, cluster_id}, built once at startup for O(1) lookups


@app.on_event("startup")
def load_artifacts() -> None:
    database.init_db()

    global _graph_cache, _details_cache, _cluster_cache, _position_by_id
    try:
        with open(GRAPH_TIER1_PATH) as f:
            _graph_cache = json.load(f)
        with open(MOVIE_DETAILS_PATH) as f:
            _details_cache = json.load(f)
        with open(CLUSTER_SUMMARY_PATH) as f:
            _cluster_cache = json.load(f)
        _position_by_id = {n["movie_id"]: n for n in _graph_cache}
        logger.info(
            "Loaded artifacts: %d graph nodes, %d movie details, %d clusters",
            len(_graph_cache), len(_details_cache), len(_cluster_cache),
        )
    except FileNotFoundError:
        logger.warning(
            "Precomputed artifacts not found in backend/data/. Run "
            "`python generate_sample_data.py && python pipeline.py` first."
        )


# ---------------------------------------------------------------------------
# Tier 1 — visualization payload
# ---------------------------------------------------------------------------
@app.get("/api/graph")
def get_graph():
    """The full node list for the map: x, y, cluster_id, movie_id only."""
    if not _graph_cache:
        raise HTTPException(status_code=503, detail="Graph data not loaded. Run the pipeline first.")
    return {"nodes": _graph_cache, "count": len(_graph_cache)}


@app.get("/api/clusters")
def get_clusters():
    """Centroid summaries for the initial hierarchical view."""
    return {"clusters": _cluster_cache}


@app.get("/api/clusters/{cluster_id}/nodes")
def get_cluster_nodes(cluster_id: int):
    """Convenience endpoint for the 'explode' interaction: just this
    cluster's nodes, so the frontend doesn't have to filter 16k rows
    client-side every time a cluster is expanded."""
    nodes = [n for n in _graph_cache if n["cluster_id"] == cluster_id]
    if not nodes:
        raise HTTPException(status_code=404, detail=f"No nodes found for cluster_id={cluster_id}")
    return {"cluster_id": cluster_id, "nodes": nodes}


# ---------------------------------------------------------------------------
# Tier 2 — on-demand detail payload
# ---------------------------------------------------------------------------
@app.get("/api/movies/{movie_id}")
def get_movie_detail(movie_id: str):
    detail = _details_cache.get(movie_id)
    if detail is None:
        raise HTTPException(status_code=404, detail=f"Unknown movie_id: {movie_id}")
    return detail


class MovieIdsRequest(BaseModel):
    movie_ids: list[str]


@app.post("/api/movies/batch")
def get_movies_batch(payload: MovieIdsRequest):
    """Lets the detail panel pre-fetch a small batch (e.g. all nodes in a
    just-exploded cluster) in one round trip instead of N requests."""
    return {mid: _details_cache[mid] for mid in payload.movie_ids if mid in _details_cache}


# ---------------------------------------------------------------------------
# Search — looks up titles server-side so Tier 1 never has to carry them
# ---------------------------------------------------------------------------
@app.get("/api/search")
def search_movies(q: str, limit: int = 25):
    q_lower = q.strip().lower()
    if len(q_lower) < 2:
        return {"results": []}

    results = []
    for movie_id, detail in _details_cache.items():
        if q_lower in detail["title"].lower():
            pos = _position_by_id.get(movie_id, {})
            results.append(
                {
                    "movie_id": movie_id,
                    "title": detail["title"],
                    "year": detail["year"],
                    "x": pos.get("x"),
                    "y": pos.get("y"),
                    "cluster_id": pos.get("cluster_id"),
                }
            )
            if len(results) >= limit:
                break
    return {"results": results}


# ---------------------------------------------------------------------------
# Feedback loop
# ---------------------------------------------------------------------------
class InteractionRequest(BaseModel):
    movie_id: str
    action: Literal["view", "click", "like", "unlike", "explode_cluster"]
    cluster_id: int | None = None
    session_id: str | None = None


@app.post("/api/interactions")
def post_interaction(payload: InteractionRequest):
    interaction_id = database.log_interaction(
        movie_id=payload.movie_id,
        action=payload.action,
        cluster_id=payload.cluster_id,
        session_id=payload.session_id,
    )
    return {"id": interaction_id, "status": "logged"}


@app.get("/api/interactions/likes")
def get_like_counts():
    """Aggregated like counts per movie — the simplest possible signal a
    future fine-tuning job could read (e.g. to up-weight liked movies'
    rows, or as positive pairs for a future recommendation objective)."""
    return database.get_like_counts()


@app.get("/health")
def health():
    return {"status": "ok", "nodes_loaded": len(_graph_cache)}
