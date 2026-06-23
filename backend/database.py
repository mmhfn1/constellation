"""
database.py
------------
A small SQLite-backed store for user interactions (hovers/clicks/likes).
This is the "Feedback Loop" piece: every interaction is logged with enough
context (movie_id, cluster_id, the action, a timestamp) that a future
training run could use it as implicit signal — e.g. weighting the
autoencoder's loss, or as labels for a "movies you might like" model.

Kept intentionally simple (raw sqlite3, no ORM) since the schema is small
and unlikely to need migrations frequently. Swap in SQLAlchemy if it grows.
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone

from config import SQLITE_PATH

_SCHEMA = """
CREATE TABLE IF NOT EXISTS interactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    movie_id TEXT NOT NULL,
    cluster_id INTEGER,
    action TEXT NOT NULL CHECK (action IN ('view', 'click', 'like', 'unlike', 'explode_cluster')),
    session_id TEXT,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_interactions_movie_id ON interactions(movie_id);
CREATE INDEX IF NOT EXISTS idx_interactions_action ON interactions(action);
"""


@contextmanager
def get_connection():
    conn = sqlite3.connect(SQLITE_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with get_connection() as conn:
        conn.executescript(_SCHEMA)


def log_interaction(movie_id: str, action: str, cluster_id: int = None, session_id: str = None) -> int:
    """Records one interaction event. Called from the FastAPI endpoint on
    every hover/click/like the frontend sends."""
    with get_connection() as conn:
        cursor = conn.execute(
            """
            INSERT INTO interactions (movie_id, cluster_id, action, session_id, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (movie_id, cluster_id, action, session_id, datetime.now(timezone.utc).isoformat()),
        )
        return cursor.lastrowid


def get_like_counts() -> dict:
    """movie_id -> like count. Useful for a future 'popular with users'
    signal, or for re-weighting samples in the next autoencoder training run."""
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT movie_id, COUNT(*) as likes FROM interactions WHERE action = 'like' GROUP BY movie_id"
        ).fetchall()
        return {row["movie_id"]: row["likes"] for row in rows}


def get_recent_interactions(limit: int = 100) -> list:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM interactions ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(row) for row in rows]


if __name__ == "__main__":
    init_db()
    log_interaction(movie_id="tt0468569", action="like", cluster_id=3, session_id="demo")
    print(get_recent_interactions(5))
