// staticData.js
//
// A client-side stand-in for backend/app.py, used when no live API is
// configured (the default for a GitHub Pages deploy, since Pages can't run
// the FastAPI backend). It reads the exact same JSON artifacts pipeline.py
// produces — just bundled as static files instead of served over HTTP —
// and reimplements the same two-tier contract: cheap graph/cluster data
// eager-loaded, rich movie detail lazy-loaded on first hover/click, search
// done in-memory, and interactions logged to localStorage instead of
// SQLite.
//
// Why localStorage and not "nothing": GitHub Pages has no server, so there
// is no way to aggregate likes/clicks *across visitors* without standing up
// a backend somewhere (see api.js / the README for that path). localStorage
// is the closest static equivalent — each visitor's own likes persist for
// them across visits, the DetailPanel's like button still works exactly as
// designed, and the data shape matches what database.py would have stored
// closely enough that swapping in a real backend later is a drop-in change.
const DATA_BASE = `${import.meta.env.BASE_URL}data`;
const LIKES_KEY = "constellation_likes_v1";
const INTERACTIONS_KEY = "constellation_interactions_v1";
const MAX_LOGGED_INTERACTIONS = 500; // keep localStorage from growing unbounded

let graphPromise = null;
let clustersPromise = null;
let detailsPromise = null;
let positionById = null; // movie_id -> {x, y, cluster_id}, built once details+graph are both in

async function getJSON(path) {
  const res = await fetch(`${DATA_BASE}/${path}`);
  if (!res.ok) {
    throw new Error(`Failed to load bundled data file: ${path} (${res.status})`);
  }
  return res.json();
}

function loadGraph() {
  if (!graphPromise) graphPromise = getJSON("graph_tier1.json");
  return graphPromise;
}

function loadClusters() {
  if (!clustersPromise) clustersPromise = getJSON("cluster_summary.json");
  return clustersPromise;
}

// Deliberately NOT loaded until something actually needs movie titles
// (first hover/click, or the first search keystroke) — this is the static
// equivalent of "Tier 2 is only fetched on demand," just fetching one
// bundled file instead of one REST call per movie. It's cached by the
// browser (and this module) after the first request either way.
function loadDetails() {
  if (!detailsPromise) {
    detailsPromise = getJSON("movie_details.json").then((details) => {
      return details;
    });
  }
  return detailsPromise;
}

async function ensurePositionIndex() {
  if (positionById) return positionById;
  const graph = await loadGraph();
  positionById = new Map(graph.map((n) => [n.movie_id, n]));
  return positionById;
}

export async function fetchGraphStatic() {
  const nodes = await loadGraph();
  return { nodes, count: nodes.length };
}

export async function fetchClustersStatic() {
  const clusters = await loadClusters();
  return { clusters };
}

export async function fetchMovieDetailStatic(movieId) {
  const details = await loadDetails();
  const detail = details[movieId];
  if (!detail) throw new Error(`Unknown movie_id: ${movieId}`);
  return detail;
}

export async function fetchMoviesBatchStatic(movieIds) {
  const details = await loadDetails();
  const out = {};
  for (const id of movieIds) {
    if (details[id]) out[id] = details[id];
  }
  return out;
}

export async function searchMoviesStatic(query, limit = 25) {
  const q = query.trim().toLowerCase();
  if (q.length < 2) return { results: [] };
  const [details, positions] = await Promise.all([loadDetails(), ensurePositionIndex()]);
  const results = [];
  for (const movieId in details) {
    const detail = details[movieId];
    if (!detail.title.toLowerCase().includes(q)) continue;
    const pos = positions.get(movieId) || {};
    results.push({
      movie_id: movieId,
      title: detail.title,
      year: detail.year,
      x: pos.x,
      y: pos.y,
      cluster_id: pos.cluster_id,
    });
    if (results.length >= limit) break;
  }
  return { results };
}

// --- localStorage-backed feedback loop, mirroring database.py's shape ---
function readLocalStore(key, fallback) {
  try {
    const raw = localStorage.getItem(key);
    return raw ? JSON.parse(raw) : fallback;
  } catch {
    return fallback;
  }
}

function writeLocalStore(key, value) {
  try {
    localStorage.setItem(key, JSON.stringify(value));
  } catch {
    // Storage can be full or disabled (private browsing); feedback logging
    // is best-effort everywhere in this app, static mode included.
  }
}

export function logInteractionStatic({ movieId, action, clusterId = null, sessionId = null }) {
  if (action === "like" || action === "unlike") {
    const likes = readLocalStore(LIKES_KEY, {});
    if (action === "like") likes[movieId] = (likes[movieId] || 0) + 1;
    else delete likes[movieId];
    writeLocalStore(LIKES_KEY, likes);
  }
  const log = readLocalStore(INTERACTIONS_KEY, []);
  log.push({ movie_id: movieId, action, cluster_id: clusterId, session_id: sessionId, created_at: new Date().toISOString() });
  if (log.length > MAX_LOGGED_INTERACTIONS) log.splice(0, log.length - MAX_LOGGED_INTERACTIONS);
  writeLocalStore(INTERACTIONS_KEY, log);
  return Promise.resolve({ status: "logged-locally" });
}

export function getLikeCountsStatic() {
  return Promise.resolve(readLocalStore(LIKES_KEY, {}));
}

export function isLikedStatic(movieId) {
  const likes = readLocalStore(LIKES_KEY, {});
  return Boolean(likes[movieId]);
}
