// Sidebar.jsx
import { useState, useRef, useCallback } from "react";
import { colorForCluster } from "../palette.js";
import { searchMovies } from "../api.js";

export default function Sidebar({
  clusters,
  explodedIds,
  hiddenIds,
  fullSky,
  onToggleExplode,
  onToggleHidden,
  onSetFullSky,
  onResetView,
  onSearchSelect,
  searchAnchor,
  onRecallAnchor,
  onDismissAnchor,
  onOpenDocs,
}) {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState([]);
  const debounceRef = useRef(null);

  const runSearch = useCallback((value) => {
    setQuery(value);
    clearTimeout(debounceRef.current);
    if (value.trim().length < 2) {
      setResults([]);
      return;
    }
    debounceRef.current = setTimeout(async () => {
      try {
        const res = await searchMovies(value.trim());
        setResults(res.results);
      } catch {
        setResults([]);
      }
    }, 220);
  }, []);

  return (
    <div className="sidebar glass-panel">
      <div className="sidebar-header">
        <div>
          <h1 className="wordmark">
            Constellation<span className="dot">.</span>
          </h1>
          <p className="tagline">An atlas of cinematic similarity</p>
        </div>
        <button className="docs-button" onClick={onOpenDocs} title="About this map" aria-label="About this map">
          ?
        </button>
      </div>

      <div className="search-box">
        <input
          className="search-input"
          type="text"
          placeholder="Search a title…"
          value={query}
          onChange={(e) => runSearch(e.target.value)}
          aria-label="Search movie titles"
        />
        {results.length > 0 && (
          <div className="search-results">
            {results.map((r) => (
              <button
                key={r.movie_id}
                className="search-result-row"
                onClick={() => {
                  onSearchSelect(r);
                  setQuery(r.title);
                  setResults([]);
                }}
              >
                <span>{r.title}</span>
                <span className="search-result-year">{r.year}</span>
              </button>
            ))}
          </div>
        )}
      </div>

      {searchAnchor && (
        <div className="anchor-chip">
          <span className="anchor-chip-dot" />
          <button className="anchor-chip-label" onClick={onRecallAnchor} title="Fly back to this search result">
            {searchAnchor.title}
          </button>
          <button className="anchor-chip-dismiss" onClick={onDismissAnchor} aria-label="Stop tracking this result">
            ×
          </button>
        </div>
      )}

      <div className="divider" />

      <div style={{ display: "flex", gap: 8 }}>
        <button
          className={`ghost-button ${fullSky ? "is-active" : ""}`}
          style={{ flex: 1 }}
          onClick={() => onSetFullSky(!fullSky)}
        >
          {fullSky ? "★ Full sky" : "Reveal full sky"}
        </button>
        <button className="ghost-button" onClick={onResetView}>
          Reset
        </button>
      </div>

      <div>
        <p className="section-label" style={{ marginBottom: 8 }}>
          Archetypes · {clusters.length}
        </p>
        <div className="legend-list">
          {clusters.map((c) => {
            const isExploded = explodedIds.has(c.cluster_id);
            const isHidden = hiddenIds.has(c.cluster_id);
            return (
              <button
                key={c.cluster_id}
                className={`legend-row ${isExploded ? "is-exploded" : ""} ${isHidden ? "is-dimmed" : ""}`}
                onClick={() => onToggleExplode(c.cluster_id)}
                onContextMenu={(e) => {
                  e.preventDefault();
                  onToggleHidden(c.cluster_id);
                }}
                title="Click to explode/collapse · right-click to hide"
              >
                <span
                  className="legend-swatch"
                  style={{ background: colorForCluster(c.cluster_id), color: colorForCluster(c.cluster_id) }}
                />
                <span className="legend-text">
                  <span className="legend-label">{c.label}</span>
                </span>
                <span className="legend-count">{c.size}</span>
              </button>
            );
          })}
        </div>
      </div>
    </div>
  );
}
