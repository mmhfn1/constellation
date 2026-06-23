// api.js
//
// Every function here keeps the exact shape the rest of the app already
// expects (App.jsx, Sidebar.jsx, DetailPanel.jsx, useGraphData.js don't
// know or care which mode is active). Two modes:
//
//   "live"   — VITE_API_BASE_URL is set at build time: talk to the real
//              FastAPI backend (backend/app.py) over REST, exactly as
//              before. Use this for local dev, or a production deploy
//              where you've hosted the backend yourself (Render, Fly,
//              Railway, a VPS, etc.) — see the README for CORS setup.
//
//   "static" — no VITE_API_BASE_URL: read the same pipeline-produced JSON
//              straight out of the bundle (frontend/public/data/, synced
//              from backend/data/ — see scripts/sync-data.mjs) and do
//              search/feedback client-side. This is what makes the site
//              fully self-contained on GitHub Pages, where there's no
//              server to call. See staticData.js for the implementation.
//
// The only real functional difference between the two: in "static" mode,
// likes/clicks persist to the visitor's own browser (localStorage) instead
// of a shared SQLite database, since there's no server to aggregate them
// across visitors. Everything else — the map, search, detail panel,
// explode/collapse, full sky — works identically either way.
import {
  fetchGraphStatic,
  fetchClustersStatic,
  fetchMovieDetailStatic,
  fetchMoviesBatchStatic,
  searchMoviesStatic,
  logInteractionStatic,
  getLikeCountsStatic,
  isLikedStatic,
} from "./staticData.js";

const BASE_URL = import.meta.env.VITE_API_BASE_URL || null;
export const API_MODE = BASE_URL ? "live" : "static";

async function getJSON(path) {
  const res = await fetch(`${BASE_URL}${path}`);
  if (!res.ok) {
    throw new Error(`GET ${path} failed: ${res.status}`);
  }
  return res.json();
}

async function postJSON(path, body) {
  const res = await fetch(`${BASE_URL}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    throw new Error(`POST ${path} failed: ${res.status}`);
  }
  return res.json();
}

// --- Tier 1: visualization payload (x, y, cluster_id, movie_id for every movie) ---
export function fetchGraph() {
  return API_MODE === "live" ? getJSON("/api/graph") : fetchGraphStatic();
}

export function fetchClusters() {
  return API_MODE === "live" ? getJSON("/api/clusters") : fetchClustersStatic();
}

export function fetchClusterNodes(clusterId) {
  if (API_MODE === "live") return getJSON(`/api/clusters/${clusterId}/nodes`);
  return fetchGraphStatic().then(({ nodes }) => ({
    cluster_id: clusterId,
    nodes: nodes.filter((n) => n.cluster_id === clusterId),
  }));
}

// --- Tier 2: on-demand rich detail, fetched on hover/click only ---
export function fetchMovieDetail(movieId) {
  return API_MODE === "live" ? getJSON(`/api/movies/${encodeURIComponent(movieId)}`) : fetchMovieDetailStatic(movieId);
}

export function fetchMoviesBatch(movieIds) {
  return API_MODE === "live" ? postJSON("/api/movies/batch", { movie_ids: movieIds }) : fetchMoviesBatchStatic(movieIds);
}

// --- Search (server-side live, in-memory static — title text never ships
// in the bulk Tier-1 payload either way) ---
export function searchMovies(query) {
  return API_MODE === "live" ? getJSON(`/api/search?q=${encodeURIComponent(query)}`) : searchMoviesStatic(query);
}

// --- Feedback loop ---
export function logInteraction({ movieId, action, clusterId = null, sessionId = null }) {
  const payload = { movieId, action, clusterId, sessionId };
  if (API_MODE === "static") return logInteractionStatic(payload);
  return postJSON("/api/interactions", {
    movie_id: movieId,
    action,
    cluster_id: clusterId,
    session_id: sessionId,
  }).catch((err) => {
    // Interaction logging is best-effort — never let it break the UI.
    console.warn("Failed to log interaction", err);
  });
}

export function getLikeCounts() {
  return API_MODE === "live" ? getJSON("/api/interactions/likes") : getLikeCountsStatic();
}

// Only meaningful in static mode (live mode has no per-visitor concept —
// DetailPanel just tracks "liked this session" either way, this just lets
// it restore that state from a previous visit).
export function isLikedLocally(movieId) {
  return API_MODE === "static" ? isLikedStatic(movieId) : false;
}
