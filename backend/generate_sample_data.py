"""
generate_sample_data.py
------------------------
Utility script (not part of the core pipeline) that fabricates a
realistic-looking imdb_top_movies_1980_2026.csv so you can run the full
pipeline end-to-end before plugging in the real dataset. Replace the file
at backend/data/imdb_top_movies_1980_2026.csv with your real export and
everything downstream (data_loader.py onward) works unchanged.

Usage:
    python generate_sample_data.py --n 16000
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from config import RAW_CSV_PATH

GENRE_POOL = [
    "Action", "Adventure", "Animation", "Comedy", "Crime", "Drama",
    "Fantasy", "Horror", "Mystery", "Romance", "Sci-Fi", "Thriller",
    "War", "Western", "Biography", "Documentary", "Family", "Musical",
]

_ADJ = ["Last", "Silent", "Broken", "Hidden", "Crimson", "Golden", "Final", "Lost",
        "Eternal", "Midnight", "Velvet", "Quiet", "Burning", "Forgotten", "Wild"]
_NOUN = ["Horizon", "Shadow", "City", "Garden", "River", "Empire", "Signal", "Mirror",
         "Storm", "Harbor", "Echo", "Kingdom", "Engine", "Letter", "Coast"]


def _fake_title(rng: np.random.Generator) -> str:
    pattern = rng.integers(0, 3)
    a, n = rng.choice(_ADJ), rng.choice(_NOUN)
    if pattern == 0:
        return f"The {a} {n}"
    if pattern == 1:
        return f"{a} {n}"
    return f"{n} of {a}ness".replace("ofness", "of Loss")


def generate(n: int, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    years = rng.integers(1980, 2027, size=n)
    runtimes = np.clip(rng.normal(110, 22, size=n), 70, 220).round().astype(int)
    ratings = np.clip(rng.normal(6.8, 1.1, size=n), 1.0, 9.9).round(1)
    # heavy-tailed vote counts -> this is exactly what log_transform_votes fixes downstream
    votes = np.round(np.exp(rng.normal(9, 2.2, size=n))).astype(int)
    votes = np.clip(votes, 50, 3_000_000)

    titles = [_fake_title(rng) for _ in range(n)]
    seen = set()
    deduped_titles = []
    for t in titles:
        candidate = t
        i = 2
        while candidate in seen:
            candidate = f"{t} {i}"
            i += 1
        seen.add(candidate)
        deduped_titles.append(candidate)

    genres_col = []
    for _ in range(n):
        k = rng.integers(1, 4)
        genres_col.append(",".join(rng.choice(GENRE_POOL, size=k, replace=False)))

    # introduce some realistic nulls for handle_nulls() to exercise
    runtime_f = runtimes.astype(float)
    rating_f = ratings.astype(float)
    null_idx = rng.choice(n, size=max(1, n // 200), replace=False)
    runtime_f[null_idx] = np.nan
    null_idx2 = rng.choice(n, size=max(1, n // 250), replace=False)
    rating_f[null_idx2] = np.nan

    df = pd.DataFrame(
        {
            "movie_id": [f"tt{rng.integers(1_000_000, 9_999_999)}" for _ in range(n)],
            "title": deduped_titles,
            "year": years,
            "runtime_minutes": runtime_f,
            "imdb_rating": rating_f,
            "num_votes": votes,
            "genres": genres_col,
            "poster_url": [f"https://picsum.photos/seed/{i}/240/360" for i in range(n)],
            "imdb_url": [f"https://www.imdb.com/title/tt{rng.integers(1_000_000, 9_999_999)}/" for _ in range(n)],
        }
    )
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=16000, help="number of synthetic rows to generate")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = generate(args.n, args.seed)
    df.to_csv(RAW_CSV_PATH, index=False)
    print(f"Wrote {len(df)} synthetic rows to {RAW_CSV_PATH}")
