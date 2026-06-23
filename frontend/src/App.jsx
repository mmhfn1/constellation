// App.jsx
import { useCallback, useMemo, useRef, useState } from "react";
import GalaxyCanvas from "./components/GalaxyCanvas.jsx";
import Sidebar from "./components/Sidebar.jsx";
import DetailPanel from "./components/DetailPanel.jsx";
import HoverTooltip from "./components/HoverTooltip.jsx";
import { Hud, Hint } from "./components/Hud.jsx";
import DocsPanel from "./components/DocsPanel.jsx";
import { useGraphData } from "./hooks/useGraphData.js";
import { fetchMovieDetail, logInteraction } from "./api.js";

// One id per browser session, sent with every logged interaction so a
// future fine-tuning pass could group "this person's" clicks/likes
// together without any login system.
const SESSION_ID = (() => {
  const key = "constellation_session_id";
  let id = sessionStorage.getItem(key);
  if (!id) {
    id = Math.random().toString(36).slice(2);
    sessionStorage.setItem(key, id);
  }
  return id;
})();

export default function App() {
  const { nodes, clusters, error, isLoading } = useGraphData();

  const [explodedIds, setExplodedIds] = useState(() => new Set());
  const [hiddenIds, setHiddenIds] = useState(() => new Set());
  const [fullSky, setFullSky] = useState(false);
  const [selectedMovie, setSelectedMovie] = useState(null); // {id, clusterId}
  const [hover, setHover] = useState(null); // {movieId, clientX, clientY, title, clusterLabel}
  const [highlightTarget, setHighlightTarget] = useState(null);
  // Separate from selectedMovie on purpose: selectedMovie changes every
  // time you click any star (that's what drives the detail panel), but a
  // search result should stay marked on the map as a fixed point of
  // reference while you click around nearby stars to compare them —
  // otherwise there's no way to find your way back to what you searched
  // for in the first place. Cleared only by a new search or Reset, never
  // by clicking a different star.
  const [searchAnchor, setSearchAnchor] = useState(null); // {movieId, title}
  const [zoom, setZoom] = useState(1);
  const [hintVisible, setHintVisible] = useState(true);
  const [docsOpen, setDocsOpen] = useState(false);

  const detailCacheRef = useRef(new Map());
  const hoverDebounceRef = useRef(null);

  const clusterLabelById = useMemo(() => {
    const m = new Map();
    (clusters || []).forEach((c) => m.set(c.cluster_id, c.label));
    return m;
  }, [clusters]);

  const dismissHint = useCallback(() => setHintVisible(false), []);

  const handleToggleExplode = useCallback(
    (clusterId) => {
      dismissHint();
      setExplodedIds((prev) => {
        const next = new Set(prev);
        if (next.has(clusterId)) next.delete(clusterId);
        else {
          next.add(clusterId);
          logInteraction({ movieId: `cluster:${clusterId}`, action: "explode_cluster", clusterId, sessionId: SESSION_ID });
        }
        return next;
      });
    },
    [dismissHint]
  );

  const handleToggleHidden = useCallback((clusterId) => {
    setHiddenIds((prev) => {
      const next = new Set(prev);
      if (next.has(clusterId)) next.delete(clusterId);
      else next.add(clusterId);
      return next;
    });
  }, []);

  const handleHoverNode = useCallback((movieId, clientX, clientY) => {
    clearTimeout(hoverDebounceRef.current);
    if (!movieId) {
      setHover(null);
      return;
    }
    const cached = detailCacheRef.current.get(movieId);
    setHover((prev) => ({
      movieId,
      clientX,
      clientY,
      title: cached ? cached.title : prev && prev.movieId === movieId ? prev.title : null,
      clusterLabel: null,
    }));
    if (!cached) {
      // Debounced Tier-2 fetch — only hit the API once the cursor has
      // actually settled on a node, not on every pixel of mouse travel.
      hoverDebounceRef.current = setTimeout(() => {
        fetchMovieDetail(movieId).then((detail) => {
          detailCacheRef.current.set(movieId, detail);
          setHover((prev) => (prev && prev.movieId === movieId ? { ...prev, title: detail.title } : prev));
        });
        logInteraction({ movieId, action: "view", sessionId: SESSION_ID });
      }, 140);
    }
  }, []);

  const handleSelectNode = useCallback(
    (movieId, clusterId) => {
      dismissHint();
      setSelectedMovie({ id: movieId, clusterId });
    },
    [dismissHint]
  );

  const handleBackgroundClick = useCallback(() => {
    setSelectedMovie(null);
  }, []);

  const handleSearchSelect = useCallback(
    (result) => {
      dismissHint();
      setHighlightTarget({ movieId: result.movie_id, x: result.x, y: result.y, nonce: Math.random() });
      setSelectedMovie({ id: result.movie_id, clusterId: result.cluster_id });
      setSearchAnchor({ movieId: result.movie_id, title: result.title });
      // Auto-explode the cluster so the searched movie is immediately visible
      // among its archetype peers, rather than hidden inside a bubble.
      // Guarded to not collapse a cluster that's already open.
      if (result.cluster_id !== -1) {
        setExplodedIds((prev) => {
          if (prev.has(result.cluster_id)) return prev; // already exploded — don't collapse
          const next = new Set(prev);
          next.add(result.cluster_id);
          return next;
        });
      }
    },
    [dismissHint]
  );

  const handleResetView = useCallback(() => {
    setExplodedIds(new Set());
    setHiddenIds(new Set());
    setFullSky(false);
    setSelectedMovie(null);
    setSearchAnchor(null);
  }, []);

  const handleRecallAnchor = useCallback(() => {
    if (!searchAnchor) return;
    const wn = nodes.find((n) => n.movie_id === searchAnchor.movieId);
    if (!wn) return;
    setHighlightTarget({ movieId: wn.movie_id, x: wn.x, y: wn.y, nonce: Math.random() });
    setSelectedMovie({ id: wn.movie_id, clusterId: wn.cluster_id });
    // Re-explode if the user collapsed the cluster after the initial search.
    if (wn.cluster_id !== -1) {
      setExplodedIds((prev) => {
        if (prev.has(wn.cluster_id)) return prev;
        const next = new Set(prev);
        next.add(wn.cluster_id);
        return next;
      });
    }
  }, [searchAnchor, nodes]);

  const handleDismissAnchor = useCallback(() => setSearchAnchor(null), []);

  if (error) {
    return (
      <div className="error-screen">
        <h1 className="wordmark">Constellation</h1>
        <p>
          Couldn't reach the API. Make sure the backend is running (
          <code>uvicorn app:app --reload --port 8000</code>) and that the pipeline has produced its
          data artifacts (<code>python pipeline.py</code>).
        </p>
      </div>
    );
  }

  if (isLoading) {
    return (
      <div className="loading-screen">
        <h1 className="wordmark">Constellation</h1>
        <div className="loading-pulse" />
      </div>
    );
  }

  const visibleCount = fullSky
    ? nodes.filter((n) => !hiddenIds.has(n.cluster_id)).length
    : explodedIds.size > 0
    ? nodes.filter((n) => explodedIds.has(n.cluster_id) && !hiddenIds.has(n.cluster_id)).length
    : clusters.filter((c) => !hiddenIds.has(c.cluster_id)).length;

  return (
    <div className="app-shell">
      <GalaxyCanvas
        nodes={nodes}
        clusters={clusters}
        explodedIds={explodedIds}
        hiddenIds={hiddenIds}
        fullSky={fullSky}
        selectedId={selectedMovie?.id || null}
        anchorId={searchAnchor?.movieId || null}
        highlightTarget={highlightTarget}
        onHoverNode={handleHoverNode}
        onSelectNode={handleSelectNode}
        onToggleExplode={handleToggleExplode}
        onBackgroundClick={handleBackgroundClick}
        onZoomChange={setZoom}
      />

      <Sidebar
        clusters={clusters}
        explodedIds={explodedIds}
        hiddenIds={hiddenIds}
        fullSky={fullSky}
        onToggleExplode={handleToggleExplode}
        onToggleHidden={handleToggleHidden}
        onSetFullSky={(v) => {
          dismissHint();
          setFullSky(v);
        }}
        onResetView={handleResetView}
        onSearchSelect={handleSearchSelect}
        searchAnchor={searchAnchor}
        onRecallAnchor={handleRecallAnchor}
        onDismissAnchor={handleDismissAnchor}
        onOpenDocs={() => setDocsOpen(true)}
      />

      <DetailPanel
        movieId={selectedMovie?.id || null}
        clusterId={selectedMovie?.clusterId}
        clusterLabel={selectedMovie ? clusterLabelById.get(selectedMovie.clusterId) : null}
        sessionId={SESSION_ID}
        onClose={() => setSelectedMovie(null)}
      />

      <HoverTooltip hover={hover} />
      <Hud
        nodeCount={visibleCount}
        zoom={zoom}
        visibleLabel={fullSky ? "full sky" : explodedIds.size > 0 ? `${explodedIds.size} exploded` : "overview"}
      />
      <Hint visible={hintVisible} />
      <DocsPanel open={docsOpen} onClose={() => setDocsOpen(false)} />
    </div>
  );
}
