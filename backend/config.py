"""
config.py
---------
Single source of truth for paths and hyperparameters. Every other module
imports from here instead of hard-coding values, so the whole pipeline can
be retuned (or pointed at a bigger dataset) from one place.
"""
from pathlib import Path
import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BACKEND_DIR = Path(__file__).resolve().parent
DATA_DIR = BACKEND_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

RAW_CSV_PATH = DATA_DIR / "imdb_top_movies_1980_2026.csv"

# Precomputed artifacts produced by pipeline.py and served by app.py.
# Splitting these into two files is what makes the "two-tier" API cheap:
# Tier 1 loads a small array of floats/ints, Tier 2 is looked up by id.
GRAPH_TIER1_PATH = DATA_DIR / "graph_tier1.json"   # x, y, cluster_id, movie_id
MOVIE_DETAILS_PATH = DATA_DIR / "movie_details.json"  # rich metadata, keyed by movie_id
CLUSTER_SUMMARY_PATH = DATA_DIR / "cluster_summary.json"  # centroids + archetype labels
LATENT_VECTORS_PATH = DATA_DIR / "latent_vectors.npy"     # cached, for re-clustering w/o retraining

SQLITE_PATH = DATA_DIR / "feedback.db"
AUTOENCODER_WEIGHTS_PATH = DATA_DIR / "autoencoder.pt"

# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
NUMERIC_COLUMNS = ["runtime_minutes", "imdb_rating", "log_votes", "year"]
GENRE_COLUMN = "genres"            # pipe- or comma-separated string in the raw CSV
TITLE_COLUMN = "title"
ID_COLUMN = "movie_id"

# ---------------------------------------------------------------------------
# Sentence-transformer (semantic title embeddings)
# ---------------------------------------------------------------------------
SENTENCE_MODEL_NAME = "all-MiniLM-L6-v2"   # 384-dim, fast, good default for short text
SENTENCE_EMBEDDING_DIM = 384
SENTENCE_BATCH_SIZE = 256                  # see embeddings.py for scaling notes

# ---------------------------------------------------------------------------
# Autoencoder
# ---------------------------------------------------------------------------
AUTOENCODER_HIDDEN_DIMS = [256, 64]
LATENT_DIM = 16
TRAIN_TEST_SPLIT = 0.70   # 70% train / 30% test, per spec
RANDOM_SEED = 42
EPOCHS = 60
BATCH_SIZE = 128
LEARNING_RATE = 1e-3

# ---------------------------------------------------------------------------
# Clustering / projection
# ---------------------------------------------------------------------------
HDBSCAN_MIN_CLUSTER_SIZE = 25
HDBSCAN_MIN_SAMPLES = 3   # reduced from 5 — 2D space is less forgiving, fewer samples needed

# HDBSCAN's hard cluster labels mark anything that didn't fit cleanly into
# a dense region as "noise" (-1) -- that's by design, not a flaw, but on a
# movie map it just reads as islands with nothing nearby, not because a
# movie is unique, just because it didn't clear the density bar. With this
# on, every noise point is reassigned to its best-fitting archetype via
# HDBSCAN's own soft-clustering membership (see cluster_service.run_hdbscan)
# instead of being left unassigned -- every movie ends up in *some*
# archetype, even a weak-fit one, rather than none.
REASSIGN_NOISE_POINTS = True

# ---------------------------------------------------------------------------
# UMAP — 2D projection of the latent space
#
# CRITICAL PIPELINE ORDER: We run UMAP *first* to get 2D positions, then
# run HDBSCAN on those 2D positions. This is the only way to guarantee that
# a cluster's members are spatially adjacent on the visible map. Running
# HDBSCAN on the high-D latent space and UMAP separately (the naive order)
# produces clusters that are semantically coherent in latent space but
# visually incoherent on the map — a movie shows up "inside" a cluster blob
# that doesn't actually contain it, and the selection ring floats in empty
# space because the star is rendered at a position unrelated to its cluster.
# ---------------------------------------------------------------------------
UMAP_N_COMPONENTS = 2
UMAP_N_NEIGHBORS = 20     # was 15 — larger neighbourhood captures more global structure
UMAP_MIN_DIST = 0.05      # was 0.1 — tighter local clusters with clearer visual gaps
UMAP_SPREAD = 1.0         # default; controls how spread-apart clusters are globally

# ---------------------------------------------------------------------------
# CORS — only relevant if you deploy this backend somewhere and point a
# separately-hosted frontend at it (see app.py). The frontend itself doesn't
# need this backend at all for a GitHub Pages deploy — it reads bundled
# static JSON instead (see frontend/src/staticData.js) — this only matters
# if you've chosen to run a live backend too.
# ---------------------------------------------------------------------------
# Comma-separated list of extra origins allowed to call this API, e.g.
#   ALLOWED_ORIGINS="https://example.com,https://staging.example.com"
_extra_origins = os.environ.get("ALLOWED_ORIGINS", "")
EXTRA_CORS_ORIGINS = [o.strip() for o in _extra_origins.split(",") if o.strip()]

# Local dev origins are always allowed; *.github.io is allowed via regex
# below (see app.py) so any GitHub Pages deploy works without extra config.
DEFAULT_CORS_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:4173",  # vite preview
    "http://127.0.0.1:4173",
]
