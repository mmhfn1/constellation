// DetailPanel.jsx
//
// This is the "Tier 2" half of the two-tier API: nothing here is fetched
// until a movie is actually selected. The map itself never carries this
// data for all 16k nodes — only the one the user clicked.
import { useEffect, useState } from "react";
import { fetchMovieDetail, logInteraction, isLikedLocally } from "../api.js";

export default function DetailPanel({ movieId, clusterId, clusterLabel, sessionId, onClose }) {
  const [detail, setDetail] = useState(null);
  const [liked, setLiked] = useState(false);

  useEffect(() => {
    setDetail(null);
    // In static (GitHub Pages) mode, likes persist per-visitor in
    // localStorage — restore whether this movie was already liked instead
    // of always resetting to false on open.
    setLiked(movieId ? isLikedLocally(movieId) : false);
    if (!movieId) return;
    let cancelled = false;
    fetchMovieDetail(movieId).then((d) => {
      if (!cancelled) setDetail(d);
    });
    logInteraction({ movieId, action: "click", clusterId, sessionId });
    return () => {
      cancelled = true;
    };
  }, [movieId, clusterId, sessionId]);

  const isOpen = Boolean(movieId);

  function toggleLike() {
    const next = !liked;
    setLiked(next);
    logInteraction({ movieId, action: next ? "like" : "unlike", clusterId, sessionId });
  }

  return (
    <div className={`detail-panel glass-panel ${isOpen ? "" : "is-hidden"}`}>
      {isOpen && (
        <button className="detail-close" onClick={onClose} aria-label="Close detail panel">
          ✕
        </button>
      )}
      {!detail ? (
        <div className="detail-skeleton">loading…</div>
      ) : (
        <>
          {detail.poster_url && <img className="detail-poster" src={detail.poster_url} alt="" />}
          <div className="detail-body">
            <h2 className="detail-title">{detail.title}</h2>
            <div className="detail-meta-row">
              <span>{detail.year}</span>
              <span>·</span>
              <span>{detail.runtime_minutes} min</span>
              <span>·</span>
              <span className="detail-rating">★ {detail.imdb_rating}</span>
            </div>
            <div className="detail-genres">
              {(detail.genres || "")
                .split(/[,|]/)
                .filter(Boolean)
                .map((g) => (
                  <span className="genre-chip" key={g}>
                    {g.trim()}
                  </span>
                ))}
            </div>
            {clusterLabel && (
              <p className="detail-archetype">
                Archetype: <strong>{clusterLabel}</strong>
              </p>
            )}
            <div className="detail-actions">
              <button className={`like-button ${liked ? "is-liked" : ""}`} onClick={toggleLike}>
                {liked ? "♥ Liked" : "♡ Like"}
              </button>
              {detail.imdb_url && (
                <a className="imdb-link" href={detail.imdb_url} target="_blank" rel="noreferrer">
                  IMDb ↗
                </a>
              )}
            </div>
          </div>
        </>
      )}
    </div>
  );
}
