"""
data_loader.py
---------------
All raw-data concerns live here: reading the CSV, cleaning nulls, the
log-transform on vote counts, one-hot encoding genres, and standardizing
numeric columns. Nothing in this module knows about PyTorch, HDBSCAN, or
FastAPI — it just turns a messy CSV into a clean, model-ready DataFrame
plus a numpy feature matrix.

Expected raw columns in imdb_top_movies_1980_2026.csv (renamed if needed
in `_normalize_column_names`):
    movie_id, title, year, runtime_minutes, imdb_rating, num_votes,
    genres ("Action,Drama" or "Action|Drama"), poster_url, imdb_url
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from config import (
    GENRE_COLUMN,
    ID_COLUMN,
    NUMERIC_COLUMNS,
    RAW_CSV_PATH,
    TITLE_COLUMN,
)

logger = logging.getLogger(__name__)

# Maps common alternate header spellings -> the canonical names this module expects.
# Extend this if your CSV export uses different headers.
_COLUMN_ALIASES = {
    "id": ID_COLUMN,
    "tconst": ID_COLUMN,
    "primaryTitle": TITLE_COLUMN,
    "runtimeMinutes": "runtime_minutes",
    "runtime": "runtime_minutes",
    "averageRating": "imdb_rating",
    "rating": "imdb_rating",
    "numVotes": "num_votes",
    "votes": "num_votes",
    "startYear": "year",
    "poster": "poster_url",
    "url": "imdb_url",
}


@dataclass
class PreparedDataset:
    """Everything downstream modules (model.py, cluster_service.py) need."""

    metadata: pd.DataFrame          # one row per movie, human-readable columns kept for the API
    feature_matrix: np.ndarray      # numeric+one-hot features, standardized, ready for the encoder
    genre_columns: list[str]        # names of the one-hot genre columns, in feature_matrix order
    numeric_columns: list[str]
    scaler: StandardScaler          # fitted scaler, kept so new rows could be transformed identically


def _normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={k: v for k, v in _COLUMN_ALIASES.items() if k in df.columns})
    return df


def _split_genres(raw: str) -> list[str]:
    if not isinstance(raw, str) or not raw.strip():
        return []
    sep = "|" if "|" in raw else ","
    return [g.strip() for g in raw.split(sep) if g.strip()]


def load_raw_csv(path=RAW_CSV_PATH) -> pd.DataFrame:
    logger.info("Loading raw CSV from %s", path)
    df = pd.read_csv(path)
    df = _normalize_column_names(df)

    required = {ID_COLUMN, TITLE_COLUMN, "year", "runtime_minutes", "imdb_rating", "num_votes", GENRE_COLUMN}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"CSV is missing required columns {missing}. "
            f"Found columns: {list(df.columns)}. Update _COLUMN_ALIASES in data_loader.py "
            f"if your export uses different headers."
        )
    return df


def handle_nulls(df: pd.DataFrame) -> pd.DataFrame:
    """Impute rather than drop, so a few missing fields don't shrink the dataset."""
    df = df.copy()

    # Numeric: median-impute (robust to outlier blockbusters skewing the mean).
    for col in ["runtime_minutes", "imdb_rating", "num_votes", "year"]:
        if df[col].isna().any():
            median = df[col].median()
            logger.info("Imputing %d nulls in %s with median=%.2f", df[col].isna().sum(), col, median)
            df[col] = df[col].fillna(median)

    # Genres: missing -> "Unknown" so the one-hot encoder gets a real category
    # instead of silently dropping the movie from every genre dimension.
    df[GENRE_COLUMN] = df[GENRE_COLUMN].fillna("Unknown")

    # Title: a movie with no title can't get a semantic embedding; drop those rows only.
    before = len(df)
    df = df.dropna(subset=[TITLE_COLUMN, ID_COLUMN])
    if len(df) < before:
        logger.warning("Dropped %d rows with missing title/id (cannot be embedded or keyed)", before - len(df))

    return df.reset_index(drop=True)


def log_transform_votes(df: pd.DataFrame) -> pd.DataFrame:
    """log1p normalizes the heavy right-skew of vote counts (a handful of
    blockbusters otherwise dominate the Euclidean distance used downstream)."""
    df = df.copy()
    df["log_votes"] = np.log1p(df["num_votes"].clip(lower=0))
    return df


def one_hot_genres(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    df = df.copy()
    genre_lists = df[GENRE_COLUMN].apply(_split_genres)
    all_genres = sorted({g for genres in genre_lists for g in genres})

    onehot = pd.DataFrame(
        {f"genre_{g}": genre_lists.apply(lambda gl, g=g: float(g in gl)) for g in all_genres},
        index=df.index,
    )
    df = pd.concat([df, onehot], axis=1)
    genre_columns = list(onehot.columns)
    logger.info("One-hot encoded %d distinct genres", len(genre_columns))
    return df, genre_columns


def standardize(df: pd.DataFrame, columns: list[str]) -> tuple[pd.DataFrame, StandardScaler]:
    scaler = StandardScaler()
    df = df.copy()
    df[[f"{c}_z" for c in columns]] = scaler.fit_transform(df[columns])
    return df, scaler


def prepare_dataset(path=RAW_CSV_PATH) -> PreparedDataset:
    """Full pipeline: raw CSV -> clean metadata + standardized feature matrix."""
    df = load_raw_csv(path)
    df = handle_nulls(df)
    df = log_transform_votes(df)
    df, genre_columns = one_hot_genres(df)
    df, scaler = standardize(df, NUMERIC_COLUMNS)

    standardized_numeric_cols = [f"{c}_z" for c in NUMERIC_COLUMNS]
    feature_columns = standardized_numeric_cols + genre_columns
    feature_matrix = df[feature_columns].to_numpy(dtype=np.float32)

    logger.info(
        "Prepared dataset: %d movies, %d features (%d numeric + %d genre)",
        len(df), feature_matrix.shape[1], len(standardized_numeric_cols), len(genre_columns),
    )

    return PreparedDataset(
        metadata=df,
        feature_matrix=feature_matrix,
        genre_columns=genre_columns,
        numeric_columns=standardized_numeric_cols,
        scaler=scaler,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    dataset = prepare_dataset()
    print(dataset.metadata.head())
    print("Feature matrix shape:", dataset.feature_matrix.shape)
