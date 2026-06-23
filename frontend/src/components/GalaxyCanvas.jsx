// GalaxyCanvas.jsx
//
// Renders the whole map on a single <canvas> (not SVG/per-node DOM nodes —
// that's what keeps 16k+ nodes fluid; see the "SCALING NOTES" comment near
// the bottom of this file). D3 supplies the math (force simulation, zoom
// transform, quadtree hit-testing) while this component owns the canvas
// painting and the requestAnimationFrame loop by hand.
//
// Hierarchical view, per the brief:
//   - Overview: one bubble per HDBSCAN cluster ("archetype"), rendered as a
//     small galaxy (gradient core, soft atmosphere, scattered sub-stars,
//     slow spiral arms on the largest few) and positioned by a real
//     d3-force simulation (collision + a gentle anchor back to the
//     cluster's true position in the latent-space projection).
//   - Explode: clicking a cluster bubble spawns a small force simulation
//     for just that cluster's movies, seeded at the centroid so they bloom
//     outward into their real positions in a brief supernova-style flash —
//     physics does the "explosion" animation for free, no manual tweening.
//   - Full sky: every movie at once, using its precomputed (x, y) directly
//     with no live physics, which is what stays smooth at 16k+ nodes.
//
// Three things this revision adds on top of that original architecture:
//   1. A layered deep-space backdrop (nebula wash + three parallax star
//      layers + a couple of seeded background curiosities) so the canvas
//      reads as "looking out at space," not a dark rectangle with dots.
//   2. A fixed look for the ~44% of movies HDBSCAN couldn't assign to an
//      archetype ("noise"/cluster_id -1) — previously one flat dull color,
//      which read as muddy dark patches ("dark spots") right next to the
//      bright cluster stars. They're now a deterministic mix of dim/warm/
//      bright field-star tiers, same as a real, unevenly-lit starfield.
//   3. A Google Maps-style locked camera (translateExtent + a scaleExtent
//      floor computed from the data's own bounds) plus a render-loop camera
//      that eases toward the zoom/pan target instead of snapping to it, so
//      every zoom — wheel, drag, or a search fly-to — feels smooth.
import { useEffect, useRef, useCallback } from "react";
import * as d3 from "d3";
import { colorForCluster, hexToRgba, fieldStarTierFor } from "../palette.js";

const WORLD_SCALE = 70; // data-space units -> world (canvas) units
const STAR_RADIUS = 3.1;
const STAR_RADIUS_HOVER = 5.5;
const HIT_TOLERANCE = 10; // world units, generous enough for fat-finger taps

const MAX_ZOOM = 9;
const VIEW_PADDING_RATIO = 0.22; // breathing room beyond the data's own bounds
const GALAXY_COUNT = 3; // largest N clusters get the spiral-arm "galaxy" treatment

const PARALLAX_MARGIN = 70; // px of overscan baked into the backdrop layers
const PARALLAX_CLAMP = 280; // px, defensive cap on how far we read the pan delta
const CAMERA_SMOOTH_TAU = 95; // ms — how quickly the rendered camera catches the target
const MAX_FRAME_DT = 100; // ms, clamp so a backgrounded tab doesn't jump on return

// A star's *world*-space radius is fixed, so its on-screen size grows
// linearly with zoom — fine at low zoom, but by the time you're zoomed
// into a dense cluster at MAX_ZOOM, a fixed-world-size halo can cover
// several neighboring stars entirely, which is exactly backwards for
// "zoom in to tell movies apart." screenCappedRadius keeps a radius's
// *world* size as-is at low zoom (where it'd render small anyway) but
// stops its *screen* size from growing past maxPx once zoom would push it
// over that — so neighboring stars stay visually distinct at any zoom
// level instead of blooming into one solid blob.
const MAX_CORE_SCREEN_PX = 9;
const MAX_HALO_SCREEN_PX = 13;
function screenCappedRadius(worldR, k, maxPx) {
  const screenPx = worldR * k;
  return screenPx > maxPx ? maxPx / k : worldR;
}

function clamp(v, lo, hi) {
  return Math.max(lo, Math.min(hi, v));
}

// Deterministic, fast, non-cryptographic string hash -> [0, 1). Used to pick
// a stable field-star tier / twinkle phase per movie so the sky doesn't
// reshuffle itself on every re-render, without needing real per-star state.
function hashUnit(id) {
  const s = String(id);
  let h = 2166136261;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return ((h >>> 0) % 100000) / 100000;
}

export default function GalaxyCanvas({
  nodes, // tier-1 array: [{movie_id, x, y, cluster_id}, ...]
  clusters, // [{cluster_id, x, y, size, label, ...}, ...]
  explodedIds, // Set<number>
  hiddenIds, // Set<number> — clusters toggled off in the legend
  fullSky, // boolean — render every node directly, no centroid layer
  selectedId,
  anchorId, // movie_id of the current search result, or null — stays put across clicks (see App.jsx)
  highlightTarget, // {movieId, x, y} from search, or null
  onHoverNode, // (movieId | null, clientX, clientY) => void
  onSelectNode, // (movieId, clusterId) => void
  onToggleExplode, // (clusterId) => void
  onBackgroundClick,
  onZoomChange, // (k) => void, throttled HUD readout
}) {
  const containerRef = useRef(null);
  const canvasRef = useRef(null);
  const starfieldRef = useRef(null); // offscreen decorative backdrop layers

  // Mutable, render-loop-only state lives in refs so updating it never
  // triggers a React re-render — only requestAnimationFrame redraws.
  // `transformRef` is what's actually drawn each frame; `targetTransformRef`
  // is where the camera is headed (set instantly by d3-zoom on every wheel
  // tick / drag / fly-to). The draw loop eases the former toward the latter
  // every frame — see "CAMERA SMOOTHING" in the scaling notes below.
  const transformRef = useRef(d3.zoomIdentity);
  const targetTransformRef = useRef(d3.zoomIdentity);
  const fitTransformRef = useRef(null);
  const worldBoundsRef = useRef(null);
  const zoomBehaviorRef = useRef(null);
  const worldNodesRef = useRef(new Map()); // movie_id -> {id, clusterId, trueX, trueY, fieldTier}
  const clusterMembersRef = useRef(new Map()); // cluster_id -> movie_id[]
  const centroidSimRef = useRef(null);
  const centroidNodesRef = useRef([]);
  const explodedSimsRef = useRef(new Map()); // cluster_id -> {sim, simNodes}
  const burstsRef = useRef([]); // explosion shockwave rings: {x, y, start}
  const cometsRef = useRef([]); // ambient shooting stars, screen-space
  const nextCometAtRef = useRef(0);
  const hoverEaseRef = useRef(new Map()); // id -> eased 0..1 hover amount
  const hoveredRef = useRef(null);
  const lastFrameTimeRef = useRef(0);
  const lastZoomReportRef = useRef(0);
  const rafRef = useRef(null);
  const dprRef = useRef(Math.min(window.devicePixelRatio || 1, 2));

  // Latest values of props that change often (clicks/toggles), read inside
  // the main effect's closures via .current instead of being effect
  // dependencies — that's what keeps the canvas/zoom/listeners setup below
  // from being torn down and recreated (and the camera re-fit!) on every
  // interaction. A separate cheap effect just keeps these in sync.
  const fullSkyRef = useRef(fullSky);
  const hiddenIdsRef = useRef(hiddenIds);
  const explodedIdsRef = useRef(explodedIds);
  const selectedIdRef = useRef(selectedId);
  const anchorIdRef = useRef(anchorId);
  useEffect(() => {
    fullSkyRef.current = fullSky;
    hiddenIdsRef.current = hiddenIds;
    explodedIdsRef.current = explodedIds;
    selectedIdRef.current = selectedId;
    anchorIdRef.current = anchorId;
  }, [fullSky, hiddenIds, explodedIds, selectedId, anchorId]);

  // -- one-time index build whenever the tier-1 node list changes --------
  useEffect(() => {
    const worldNodes = new Map();
    const byCluster = new Map();
    for (const n of nodes) {
      const wn = {
        id: n.movie_id,
        clusterId: n.cluster_id,
        trueX: n.x * WORLD_SCALE,
        trueY: n.y * WORLD_SCALE,
        // Stable per-movie unit value, reused for both the field-star tier
        // (noise nodes) and a per-star twinkle phase offset.
        unit: hashUnit(n.movie_id),
      };
      worldNodes.set(n.movie_id, wn);
      if (!byCluster.has(n.cluster_id)) byCluster.set(n.cluster_id, []);
      byCluster.get(n.cluster_id).push(n.movie_id);
    }
    worldNodesRef.current = worldNodes;
    clusterMembersRef.current = byCluster;
  }, [nodes]);

  // -- centroid layer: a real d3-force simulation for the overview bubbles
  useEffect(() => {
    if (!clusters.length) return;
    const sizeScale = d3
      .scaleSqrt()
      .domain([1, d3.max(clusters, (c) => c.size) || 1])
      .range([13, 42]);

    // The largest few clusters get the spiral-arm "galaxy" treatment; the
    // rest read as tighter star clusters. A continuous size scale already
    // drives bubble radius/glow, so this is just a small extra flourish on
    // the handful of clusters substantial enough to read as galaxies.
    const galaxyIds = new Set(
      [...clusters]
        .sort((a, b) => b.size - a.size)
        .slice(0, Math.min(GALAXY_COUNT, clusters.length))
        .map((c) => c.cluster_id)
    );

    const simNodes = clusters.map((c) => {
      // Seeded per-cluster so the sub-star scatter inside each bubble is
      // stable across renders instead of reshuffling every mount.
      const rng = d3.randomLcg(((c.cluster_id + 1) * 0.0131) % 1 || 0.5);
      const scatterCount = 5 + Math.floor(rng() * 4);
      const scatter = Array.from({ length: scatterCount }, () => {
        const angle = rng() * Math.PI * 2;
        const radius = 0.22 + rng() * 0.68;
        return {
          dx: Math.cos(angle) * radius,
          dy: Math.sin(angle) * radius,
          r: 0.6 + rng() * 1.15,
          alpha: 0.32 + rng() * 0.48,
        };
      });
      return {
        clusterId: c.cluster_id,
        trueX: c.x * WORLD_SCALE,
        trueY: c.y * WORLD_SCALE,
        x: c.x * WORLD_SCALE,
        y: c.y * WORLD_SCALE,
        r: sizeScale(c.size),
        size: c.size,
        label: c.label,
        isGalaxy: galaxyIds.has(c.cluster_id),
        spiralSeed: rng() * Math.PI * 2,
        scatter,
      };
    });
    centroidNodesRef.current = simNodes;

    const sim = d3
      .forceSimulation(simNodes)
      .force("collide", d3.forceCollide((d) => d.r + 6).strength(0.85))
      .force("anchorX", d3.forceX((d) => d.trueX).strength(0.12))
      .force("anchorY", d3.forceY((d) => d.trueY).strength(0.12))
      .alphaDecay(0.02)
      .on("tick", () => {}); // drawing happens in the rAF loop, not here

    centroidSimRef.current = sim;
    return () => sim.stop();
  }, [clusters]);

  // -- explode / collapse: spin up or tear down a per-cluster simulation -
  useEffect(() => {
    const members = clusterMembersRef.current;
    const worldNodes = worldNodesRef.current;
    const centroids = centroidNodesRef.current;
    const liveSims = explodedSimsRef.current;

    // Start a bloom simulation for newly-exploded clusters.
    for (const clusterId of explodedIds) {
      if (liveSims.has(clusterId)) continue;
      const ids = members.get(clusterId) || [];
      const centroid = centroids.find((c) => c.clusterId === clusterId);
      if (!centroid) continue;

      burstsRef.current.push({ x: centroid.x, y: centroid.y, start: performance.now(), color: colorForCluster(clusterId) });

      const simNodes = ids.map((id) => {
        const wn = worldNodes.get(id);
        // Seed every member AT the centroid (with tiny jitter so the
        // force layout has something to push apart) — this is what makes
        // the cluster visually "bloom" outward into its real positions.
        return {
          id,
          clusterId,
          trueX: wn.trueX,
          trueY: wn.trueY,
          x: centroid.x + (Math.random() - 0.5) * 4,
          y: centroid.y + (Math.random() - 0.5) * 4,
        };
      });

      const sim = d3
        .forceSimulation(simNodes)
        .force("collide", d3.forceCollide(STAR_RADIUS + 1.4).strength(1))
        .force("anchorX", d3.forceX((d) => d.trueX).strength(0.4))
        .force("anchorY", d3.forceY((d) => d.trueY).strength(0.4))
        .alphaDecay(0.028)
        .on("tick", () => {});

      liveSims.set(clusterId, { sim, simNodes });
    }

    // Tear down simulations for clusters that were just collapsed.
    for (const clusterId of Array.from(liveSims.keys())) {
      if (!explodedIds.has(clusterId)) {
        liveSims.get(clusterId).sim.stop();
        liveSims.delete(clusterId);
      }
    }
  }, [explodedIds]);

  // -- deep-space backdrop: nebula wash + parallax star layers -----------
  // Built once per resize into offscreen canvases (never per-frame — that
  // would defeat the whole point of a backdrop). Each layer is overscanned
  // by PARALLAX_MARGIN px so the per-frame parallax shift never exposes an
  // edge. Twinkle is faked at the *layer* level (a slow shared sine driving
  // globalAlpha for the whole layer) instead of per star, which is what
  // keeps this free even with thousands of background stars.
  const buildStarfield = useCallback((w, h) => {
    const ow = w + PARALLAX_MARGIN * 2;
    const oh = h + PARALLAX_MARGIN * 2;

    function makeLayer() {
      const c = document.createElement("canvas");
      c.width = ow;
      c.height = oh;
      return c;
    }

    // --- nebula wash: a handful of huge, very soft tinted gradients ----
    const nebula = makeLayer();
    const nctx = nebula.getContext("2d");
    const nebRng = d3.randomLcg(0.4271);
    const nebulaHues = ["#241a4a", "#142d33", "#3a1d33", "#16213d", "#2c1a22"];
    for (let i = 0; i < 5; i++) {
      const cx = nebRng() * ow;
      const cy = nebRng() * oh;
      const r = (0.35 + nebRng() * 0.4) * Math.max(ow, oh);
      const g = nctx.createRadialGradient(cx, cy, 0, cx, cy, r);
      const color = nebulaHues[i % nebulaHues.length];
      g.addColorStop(0, hexToRgba(color, 0.5));
      g.addColorStop(0.6, hexToRgba(color, 0.16));
      g.addColorStop(1, hexToRgba(color, 0));
      nctx.fillStyle = g;
      nctx.fillRect(0, 0, ow, oh);
    }
    // A couple of seeded deep-space curiosities — purely decorative, fixed
    // in the backdrop, never interactive. They're what makes the sky feel
    // observed rather than generated: a quiet black hole with a thin
    // accretion glow, and a faded supernova remnant shell.
    const curioRng = d3.randomLcg(0.8123);
    const bhx = curioRng() * ow;
    const bhy = curioRng() * oh * 0.7;
    const bhR = 9 + curioRng() * 5;
    const disk = nctx.createRadialGradient(bhx, bhy, bhR * 0.4, bhx, bhy, bhR * 2.6);
    disk.addColorStop(0, "rgba(0,0,0,0)");
    disk.addColorStop(0.38, "rgba(0,0,0,0)");
    disk.addColorStop(0.55, hexToRgba("#e8b84b", 0.22));
    disk.addColorStop(0.7, hexToRgba("#5fb3d9", 0.1));
    disk.addColorStop(1, "rgba(0,0,0,0)");
    nctx.fillStyle = disk;
    nctx.beginPath();
    nctx.arc(bhx, bhy, bhR * 2.6, 0, Math.PI * 2);
    nctx.fill();
    nctx.fillStyle = "#05060a";
    nctx.beginPath();
    nctx.arc(bhx, bhy, bhR * 0.85, 0, Math.PI * 2);
    nctx.fill();

    const remRng = d3.randomLcg(0.5566);
    const rx = remRng() * ow;
    const ry = remRng() * oh;
    for (let ring = 0; ring < 3; ring++) {
      nctx.beginPath();
      nctx.arc(rx, ry, 14 + ring * 11, 0, Math.PI * 2);
      nctx.strokeStyle = hexToRgba(ring % 2 === 0 ? "#e2667d" : "#5fd9c8", 0.07 - ring * 0.012);
      nctx.lineWidth = 2.5;
      nctx.stroke();
    }

    // --- star layers: far (dim/dense) -> near (bright/sparse) -----------
    function paintStars(ctx, density, rRange, alphaRange, colorMix) {
      const rng = d3.randomLcg((density * 7.13) % 1 || 0.31);
      const n = Math.floor((ow * oh) / density);
      for (let i = 0; i < n; i++) {
        const x = rng() * ow;
        const y = rng() * oh;
        const r = rRange[0] + rng() * (rRange[1] - rRange[0]);
        const a = alphaRange[0] + rng() * (alphaRange[1] - alphaRange[0]);
        const tint = rng();
        ctx.fillStyle = tint < colorMix[0] ? "#cfd6ff" : tint < colorMix[1] ? "#fff3d6" : "#ffffff";
        ctx.globalAlpha = a;
        ctx.beginPath();
        ctx.arc(x, y, r, 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.globalAlpha = 1;
    }

    const far = makeLayer();
    paintStars(far.getContext("2d"), 1900, [0.35, 0.9], [0.18, 0.45], [0.5, 0.85]);

    const mid = makeLayer();
    paintStars(mid.getContext("2d"), 5200, [0.6, 1.4], [0.3, 0.6], [0.45, 0.8]);

    const near = makeLayer();
    paintStars(near.getContext("2d"), 16000, [0.9, 2.1], [0.45, 0.85], [0.4, 0.75]);

    starfieldRef.current = { nebula, far, mid, near, ow, oh };
  }, []);

  // -- bounds + camera lock -------------------------------------------------
  // Computes the world-space bounding box of every star and cluster bubble,
  // padded for breathing room. translateExtent pins the camera to that box
  // (you can pan until the edge of the data meets the edge of the
  // viewport, then it stops — the Google Maps "can't pan into the void"
  // behavior) and the scaleExtent floor is the exact zoom level where that
  // box fits the viewport (you can't zoom out past "the whole sky, framed").
  const computeWorldBounds = useCallback(() => {
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    for (const [, wn] of worldNodesRef.current) {
      if (wn.trueX < minX) minX = wn.trueX;
      if (wn.trueX > maxX) maxX = wn.trueX;
      if (wn.trueY < minY) minY = wn.trueY;
      if (wn.trueY > maxY) maxY = wn.trueY;
    }
    for (const c of centroidNodesRef.current) {
      if (c.trueX < minX) minX = c.trueX;
      if (c.trueX > maxX) maxX = c.trueX;
      if (c.trueY < minY) minY = c.trueY;
      if (c.trueY > maxY) maxY = c.trueY;
    }
    if (!isFinite(minX)) return null;
    const spanX = Math.max(maxX - minX, 200);
    const spanY = Math.max(maxY - minY, 200);
    const padX = spanX * VIEW_PADDING_RATIO;
    const padY = spanY * VIEW_PADDING_RATIO;
    const bounds = { minX: minX - padX, maxX: maxX + padX, minY: minY - padY, maxY: maxY + padY };
    worldBoundsRef.current = bounds;
    return bounds;
  }, []);

  const applyViewLock = useCallback(
    (zoomBehavior, w, h) => {
      const bounds = computeWorldBounds();
      if (!bounds || !w || !h) return null;
      const spanX = bounds.maxX - bounds.minX;
      const spanY = bounds.maxY - bounds.minY;
      const fitK = Math.min(w / spanX, h / spanY, 2.5);
      zoomBehavior.extent([[0, 0], [w, h]]);
      zoomBehavior.scaleExtent([fitK, MAX_ZOOM]);
      zoomBehavior.translateExtent([[bounds.minX, bounds.minY], [bounds.maxX, bounds.maxY]]);
      return { bounds, fitK };
    },
    [computeWorldBounds]
  );

  // -- resize handling + initial fit-to-view ------------------------------
  const fitToView = useCallback(
    (zoomBehavior, selection) => {
      if (!canvasRef.current) return;
      const w = canvasRef.current.clientWidth;
      const h = canvasRef.current.clientHeight;
      const lock = applyViewLock(zoomBehavior, w, h);
      if (!lock) return;
      const { bounds, fitK } = lock;
      const cx = (bounds.minX + bounds.maxX) / 2;
      const cy = (bounds.minY + bounds.maxY) / 2;
      const t = d3.zoomIdentity.translate(w / 2 - cx * fitK, h / 2 - cy * fitK).scale(fitK);
      zoomBehavior.transform(selection, t);
      fitTransformRef.current = t;
      transformRef.current = t;
      targetTransformRef.current = t;
    },
    [applyViewLock]
  );

  // -- main effect: canvas setup, zoom/pan/click/hover wiring, rAF loop --
  useEffect(() => {
    const container = containerRef.current;
    const canvas = canvasRef.current;
    if (!container || !canvas) return;
    const ctx = canvas.getContext("2d");

    function resize() {
      const w = container.clientWidth;
      const h = container.clientHeight;
      const dpr = dprRef.current;
      canvas.width = w * dpr;
      canvas.height = h * dpr;
      canvas.style.width = `${w}px`;
      canvas.style.height = `${h}px`;
      buildStarfield(w, h);
      if (zoomBehaviorRef.current) applyViewLock(zoomBehaviorRef.current, w, h);
    }
    resize();
    const resizeObserver = new ResizeObserver(resize);
    resizeObserver.observe(container);

    const zoomBehavior = d3
      .zoom()
      .scaleExtent([0.05, MAX_ZOOM]) // real floor gets set by applyViewLock once data is ready
      .filter((event) => !event.button)
      .on("zoom", (event) => {
        targetTransformRef.current = event.transform;
      });

    const selection = d3.select(canvas);
    selection.call(zoomBehavior);
    selection.on("dblclick.zoom", null);
    zoomBehaviorRef.current = zoomBehavior;

    // Wait one tick so centroid sim nodes exist before fitting/locking.
    const fitTimer = setTimeout(() => fitToView(zoomBehavior, selection), 60);

    // -- hit testing -------------------------------------------------------
    // Stars are tiny, uniform points, so a quadtree + a flat screen-space
    // tolerance (below) is the right tool. Cluster bubbles are not uniform
    // — they range from r=13 to r=42 world units — so they're tested
    // directly against their own rendered radius instead of being forced
    // through that same flat tolerance, which would only register clicks
    // very close to a bubble's center and miss most of a large bubble's
    // actual visible area.
    function starCandidatesForHitTest() {
      const fullSky = fullSkyRef.current;
      const hiddenIds = hiddenIdsRef.current;
      if (fullSky) {
        const out = [];
        for (const [id, wn] of worldNodesRef.current) {
          if (!hiddenIds.has(wn.clusterId)) out.push({ id, clusterId: wn.clusterId, x: wn.trueX, y: wn.trueY, kind: "star" });
        }
        return out;
      }
      const out = [];
      // Field stars (cluster_id -1) are drawn in overview mode now too (see
      // drawFieldStars), so they need to be hit-testable there as well —
      // otherwise they'd be visible but inert, which would be its own
      // confusing half-fix.
      for (const [id, wn] of worldNodesRef.current) {
        if (wn.clusterId === -1 && !hiddenIds.has(wn.clusterId)) out.push({ id, clusterId: wn.clusterId, x: wn.trueX, y: wn.trueY, kind: "star" });
      }
      for (const [clusterId, { simNodes }] of explodedSimsRef.current) {
        if (hiddenIds.has(clusterId)) continue;
        for (const sn of simNodes) out.push({ id: sn.id, clusterId, x: sn.x, y: sn.y, kind: "star" });
      }
      return out;
    }

    function hitCentroid(x, y) {
      const hiddenIds = hiddenIdsRef.current;
      const explodedIds = explodedIdsRef.current;
      let best = null;
      let bestDist = Infinity;
      for (const c of centroidNodesRef.current) {
        if (hiddenIds.has(c.clusterId) || explodedIds.has(c.clusterId)) continue;
        const d = Math.hypot(c.x - x, c.y - y);
        // +6 world units of edge fudge, same spirit as HIT_TOLERANCE's own
        // "+8" — a little forgiveness right at the rim of a bubble.
        if (d <= c.r + 6 && d < bestDist) {
          bestDist = d;
          best = { id: `cluster:${c.clusterId}`, clusterId: c.clusterId, x: c.x, y: c.y, kind: "centroid" };
        }
      }
      return best;
    }

    function pointerToWorld(clientX, clientY) {
      const rect = canvas.getBoundingClientRect();
      const t = transformRef.current;
      const px = clientX - rect.left;
      const py = clientY - rect.top;
      return { x: (px - t.x) / t.k, y: (py - t.y) / t.k };
    }

    function findNearest(clientX, clientY) {
      const { x, y } = pointerToWorld(clientX, clientY);
      if (!fullSkyRef.current) {
        const centroidHit = hitCentroid(x, y);
        if (centroidHit) return centroidHit;
      }
      const candidates = starCandidatesForHitTest();
      if (!candidates.length) return null;
      const tree = d3.quadtree(candidates, (d) => d.x, (d) => d.y);
      const tolerance = HIT_TOLERANCE / transformRef.current.k + 8;
      return tree.find(x, y, tolerance);
    }

    let lastMoveAt = 0;
    function handleMove(event) {
      const now = performance.now();
      if (now - lastMoveAt < 28) return; // ~35fps hover sampling is plenty
      lastMoveAt = now;
      const hit = findNearest(event.clientX, event.clientY);
      hoveredRef.current = hit ? hit.id : null;
      canvas.style.cursor = hit ? "pointer" : "grab";
      if (onHoverNode) onHoverNode(hit && hit.kind === "star" ? hit.id : null, event.clientX, event.clientY);
    }

    function handleClick(event) {
      const hit = findNearest(event.clientX, event.clientY);
      if (!hit) {
        if (onBackgroundClick) onBackgroundClick();
        return;
      }
      if (hit.kind === "centroid") {
        if (onToggleExplode) onToggleExplode(hit.clusterId);
      } else {
        if (onSelectNode) onSelectNode(hit.id, hit.clusterId);
      }
    }

    canvas.addEventListener("mousemove", handleMove);
    canvas.addEventListener("click", handleClick);
    canvas.addEventListener("mouseleave", () => {
      hoveredRef.current = null;
      if (onHoverNode) onHoverNode(null, 0, 0);
    });

    // -- per-frame eased hover amount (0..1) for whatever's hovered -------
    // Smooths the galaxy glow-on-hover instead of snapping it, same spirit
    // as the camera easing below.
    function updateHoverEase(dt) {
      const map = hoverEaseRef.current;
      const activeKey = hoveredRef.current;
      const factor = 1 - Math.exp(-dt / 90);
      for (const key of Array.from(map.keys())) {
        const val = map.get(key);
        const target = key === activeKey ? 1 : 0;
        const next = val + (target - val) * factor;
        if (next < 0.002 && target === 0) map.delete(key);
        else map.set(key, next);
      }
      if (activeKey && !map.has(activeKey)) map.set(activeKey, factor);
    }

    // -- screen-space deep-space backdrop: nebula + parallax stars --------
    function drawBackground(w, h, now) {
      const sf = starfieldRef.current;
      if (!sf) return;
      const fit = fitTransformRef.current;
      const t = transformRef.current;
      const dx = fit ? clamp(t.x - fit.x, -PARALLAX_CLAMP, PARALLAX_CLAMP) : 0;
      const dy = fit ? clamp(t.y - fit.y, -PARALLAX_CLAMP, PARALLAX_CLAMP) : 0;

      function layer(canvasLayer, factor, alpha) {
        const ox = clamp(dx * factor, -PARALLAX_MARGIN + 4, PARALLAX_MARGIN - 4);
        const oy = clamp(dy * factor, -PARALLAX_MARGIN + 4, PARALLAX_MARGIN - 4);
        ctx.globalAlpha = alpha;
        ctx.drawImage(canvasLayer, -PARALLAX_MARGIN - ox, -PARALLAX_MARGIN - oy, sf.ow, sf.oh);
        ctx.globalAlpha = 1;
      }

      layer(sf.nebula, 0.012, 1);
      layer(sf.far, 0.022, 0.55 + 0.45 * Math.sin(now * 0.00017));
      layer(sf.mid, 0.038, 0.6 + 0.4 * Math.sin(now * 0.00025 + 2.1));
      layer(sf.near, 0.055, 0.7 + 0.3 * Math.sin(now * 0.00032 + 4.2));
      void w;
      void h;
    }

    // -- ambient shooting stars, purely decorative, screen-space ----------
    function updateAndDrawComets(w, h, now) {
      if (now > nextCometAtRef.current && cometsRef.current.length < 2) {
        const fromLeft = Math.random() < 0.5;
        const len = 130 + Math.random() * 170;
        const angle = Math.PI / 7 + Math.random() * 0.3;
        const dir = fromLeft ? 1 : -1;
        cometsRef.current.push({
          x0: fromLeft ? -30 : w + 30,
          y0: Math.random() * h * 0.55,
          dx: Math.cos(angle) * len * dir,
          dy: Math.sin(angle) * len,
          start: now,
          duration: 850 + Math.random() * 500,
        });
        nextCometAtRef.current = now + 5500 + Math.random() * 9000;
      }
      cometsRef.current = cometsRef.current.filter((c) => now - c.start < c.duration);
      for (const c of cometsRef.current) {
        const p = (now - c.start) / c.duration;
        const easeIn = p < 0.15 ? p / 0.15 : 1;
        const easeOut = p > 0.7 ? Math.max(0, (1 - p) / 0.3) : 1;
        const amt = easeIn * easeOut;
        const head = Math.min(p * 1.2, 1);
        const tail = Math.max(p * 1.2 - 0.45, 0);
        const hx = c.x0 + c.dx * head;
        const hy = c.y0 + c.dy * head;
        const tx = c.x0 + c.dx * tail;
        const ty = c.y0 + c.dy * tail;
        const grad = ctx.createLinearGradient(tx, ty, hx, hy);
        grad.addColorStop(0, "rgba(230,238,255,0)");
        grad.addColorStop(1, `rgba(230,238,255,${0.85 * amt})`);
        ctx.strokeStyle = grad;
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.moveTo(tx, ty);
        ctx.lineTo(hx, hy);
        ctx.stroke();
        ctx.beginPath();
        ctx.arc(hx, hy, 1.5, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(255,255,255,${0.9 * amt})`;
        ctx.fill();
      }
    }

    // ---- the draw loop ----------------------------------------------
    function draw(now) {
      if (!lastFrameTimeRef.current) lastFrameTimeRef.current = now;
      const dt = Math.min(now - lastFrameTimeRef.current, MAX_FRAME_DT);
      lastFrameTimeRef.current = now;

      // Ease the rendered camera toward the zoom target. This is what
      // makes wheel zoom, drag-pan, and the search fly-to all feel
      // continuous instead of stepping with the input device — see
      // "CAMERA SMOOTHING" in the scaling notes below.
      const cur = transformRef.current;
      const target = targetTransformRef.current;
      const smoothing = 1 - Math.exp(-dt / CAMERA_SMOOTH_TAU);
      if (Math.abs(target.k - cur.k) < 0.0003 && Math.abs(target.x - cur.x) < 0.05 && Math.abs(target.y - cur.y) < 0.05) {
        transformRef.current = target;
      } else {
        transformRef.current = d3.zoomIdentity
          .translate(cur.x + (target.x - cur.x) * smoothing, cur.y + (target.y - cur.y) * smoothing)
          .scale(cur.k + (target.k - cur.k) * smoothing);
      }

      updateHoverEase(dt);

      const dpr = dprRef.current;
      const w = canvas.clientWidth;
      const h = canvas.clientHeight;
      ctx.save();
      ctx.scale(dpr, dpr);
      ctx.clearRect(0, 0, w, h);

      drawBackground(w, h, now);
      updateAndDrawComets(w, h, now);

      const t = transformRef.current;
      ctx.translate(t.x, t.y);
      ctx.scale(t.k, t.k);

      // Supernova shockwaves from recently-exploded clusters --------------
      burstsRef.current = burstsRef.current.filter((b) => now - b.start < 900);
      for (const b of burstsRef.current) {
        const p = (now - b.start) / 750;
        const flash = Math.max(0, 1 - p * 3);
        if (flash > 0) {
          ctx.beginPath();
          ctx.arc(b.x, b.y, 4 + p * 16, 0, Math.PI * 2);
          ctx.fillStyle = `rgba(255,250,235,${flash * 0.75})`;
          ctx.fill();
        }
        for (let ring = 0; ring < 2; ring++) {
          const rp = p - ring * 0.14;
          if (rp <= 0 || rp >= 1) continue;
          ctx.beginPath();
          ctx.arc(b.x, b.y, 8 + rp * 110, 0, Math.PI * 2);
          ctx.strokeStyle = hexToRgba(b.color || "#E8B84B", 0.5 * (1 - rp));
          ctx.lineWidth = (ring === 0 ? 2.1 : 1.1) / t.k;
          ctx.stroke();
        }
      }

      if (fullSkyRef.current) {
        drawFullSky(ctx, t.k);
      } else {
        // Unclustered ("noise") movies have no centroid bubble to live in —
        // without this, anything HDBSCAN couldn't assign to an archetype
        // (about 44% of the dataset) was simply never drawn outside full
        // sky mode, including whatever a search result might resolve to:
        // the camera would fly to it and show a selection ring with no
        // star under it. Drawing them as the same ambient field-star layer
        // full sky uses — underneath the galaxy bubbles — fixes that and
        // means the background starfield is always present, the way real
        // sky doesn't disappear just because you're looking at a galaxy.
        drawFieldStars(ctx, t.k);
        drawCentroids(ctx, t.k, now);
        drawExplodedClusters(ctx, t.k);
      }

      // Highlight ring for the selected / search-targeted star -----------
      drawSelectionRing(ctx, t.k, now);
      drawAnchorMarker(ctx, t.k, now);

      ctx.restore();

      // Throttled HUD readout, driven by the rendered (eased) zoom level
      // rather than raw input events.
      if (now - lastZoomReportRef.current > 80) {
        lastZoomReportRef.current = now;
        if (onZoomChange) onZoomChange(t.k);
      }

      rafRef.current = requestAnimationFrame(draw);
    }

    // Batched-by-style draw of every visible node, directly at its true
    // projected position — no physics, which is what keeps this path
    // smooth at 16k+ nodes (see SCALING NOTES below). Noise/unclustered
    // movies are split into three deterministic field-star tiers instead
    // of one flat color — see palette.js for why.
    // Renders only the unclustered ("noise", cluster_id -1) movies — the
    // ~44% of the dataset HDBSCAN couldn't fit into an archetype. Full sky
    // already includes these via drawFullSky; this is the same field-star
    // styling/batching, pulled out so the overview can show them too (see
    // the call site below for why that matters).
    function drawFieldStars(ctx, k) {
      const hiddenIds = hiddenIdsRef.current;
      const coreGroups = new Map();
      const haloGroups = new Map();
      const styleMeta = new Map();

      for (const [, wn] of worldNodesRef.current) {
        if (wn.clusterId !== -1 || hiddenIds.has(wn.clusterId)) continue;
        const tier = fieldStarTierFor(wn.unit);
        const r = screenCappedRadius(tier.r, k, MAX_CORE_SCREEN_PX);
        const haloR = screenCappedRadius(tier.r * 2.5, k, MAX_HALO_SCREEN_PX);
        const style = { key: tier.key, color: tier.color, r, coreAlpha: tier.coreAlpha, haloAlpha: tier.haloAlpha, haloR };
        if (!coreGroups.has(style.key)) {
          coreGroups.set(style.key, new Path2D());
          haloGroups.set(style.key, new Path2D());
          styleMeta.set(style.key, style);
        }
        const core = coreGroups.get(style.key);
        core.moveTo(wn.trueX + style.r, wn.trueY);
        core.arc(wn.trueX, wn.trueY, style.r, 0, Math.PI * 2);
        if (style.haloAlpha > 0) {
          const halo = haloGroups.get(style.key);
          halo.moveTo(wn.trueX + style.haloR, wn.trueY);
          halo.arc(wn.trueX, wn.trueY, style.haloR, 0, Math.PI * 2);
        }
      }
      for (const [key, style] of styleMeta) {
        if (style.haloAlpha <= 0) continue;
        ctx.fillStyle = style.color;
        ctx.globalAlpha = style.haloAlpha;
        ctx.fill(haloGroups.get(key));
      }
      for (const [key, style] of styleMeta) {
        ctx.fillStyle = style.color;
        ctx.globalAlpha = style.coreAlpha;
        ctx.fill(coreGroups.get(key));
      }
      ctx.globalAlpha = 1;
    }

    function drawFullSky(ctx, k) {
      const hiddenIds = hiddenIdsRef.current;
      const coreGroups = new Map(); // styleKey -> Path2D
      const haloGroups = new Map();
      const styleMeta = new Map();

      for (const [, wn] of worldNodesRef.current) {
        if (hiddenIds.has(wn.clusterId)) continue;
        let style;
        if (wn.clusterId === -1) {
          const tier = fieldStarTierFor(wn.unit);
          const r = screenCappedRadius(tier.r, k, MAX_CORE_SCREEN_PX);
          const haloR = screenCappedRadius(tier.r * 2.5, k, MAX_HALO_SCREEN_PX);
          style = { key: tier.key, color: tier.color, r, coreAlpha: tier.coreAlpha, haloAlpha: tier.haloAlpha, haloR };
        } else {
          const color = colorForCluster(wn.clusterId);
          const r = screenCappedRadius(STAR_RADIUS, k, MAX_CORE_SCREEN_PX);
          const haloR = screenCappedRadius(STAR_RADIUS * 2.2, k, MAX_HALO_SCREEN_PX);
          style = { key: `c:${wn.clusterId}`, color, r, coreAlpha: 0.88, haloAlpha: 0.15, haloR };
        }
        if (!coreGroups.has(style.key)) {
          coreGroups.set(style.key, new Path2D());
          haloGroups.set(style.key, new Path2D());
          styleMeta.set(style.key, style);
        }
        const core = coreGroups.get(style.key);
        core.moveTo(wn.trueX + style.r, wn.trueY);
        core.arc(wn.trueX, wn.trueY, style.r, 0, Math.PI * 2);
        if (style.haloAlpha > 0) {
          const halo = haloGroups.get(style.key);
          halo.moveTo(wn.trueX + style.haloR, wn.trueY);
          halo.arc(wn.trueX, wn.trueY, style.haloR, 0, Math.PI * 2);
        }
      }

      // Halos under, crisp cores on top — still just a handful of fill()
      // calls total, not one per star.
      for (const [key, style] of styleMeta) {
        if (style.haloAlpha <= 0) continue;
        ctx.fillStyle = style.color;
        ctx.globalAlpha = style.haloAlpha;
        ctx.fill(haloGroups.get(key));
      }
      for (const [key, style] of styleMeta) {
        ctx.fillStyle = style.color;
        ctx.globalAlpha = style.coreAlpha;
        ctx.fill(coreGroups.get(key));
      }
      ctx.globalAlpha = 1;
    }

    function drawCentroids(ctx, k, now) {
      const hiddenIds = hiddenIdsRef.current;
      const explodedIds = explodedIdsRef.current;
      for (const c of centroidNodesRef.current) {
        if (hiddenIds.has(c.clusterId) || explodedIds.has(c.clusterId)) continue;
        const color = colorForCluster(c.clusterId);
        const hoverT = hoverEaseRef.current.get(`cluster:${c.clusterId}`) || 0;
        const r = c.r;

        // Soft atmospheric glow — what makes a bubble read as a galaxy
        // glimpsed from a distance rather than a flat translucent disc.
        const glowR = r * (1.75 + hoverT * 0.3);
        const glow = ctx.createRadialGradient(c.x, c.y, r * 0.15, c.x, c.y, glowR);
        glow.addColorStop(0, hexToRgba(color, 0.32 + hoverT * 0.14));
        glow.addColorStop(0.55, hexToRgba(color, 0.11));
        glow.addColorStop(1, hexToRgba(color, 0));
        ctx.fillStyle = glow;
        ctx.beginPath();
        ctx.arc(c.x, c.y, glowR, 0, Math.PI * 2);
        ctx.fill();

        if (c.isGalaxy) {
          const rot = c.spiralSeed + now * 0.000012;
          ctx.save();
          ctx.translate(c.x, c.y);
          ctx.rotate(rot);
          ctx.strokeStyle = hexToRgba(color, 0.16 + hoverT * 0.08);
          ctx.lineWidth = Math.max(0.8, r * 0.05);
          for (let arm = 0; arm < 2; arm++) {
            ctx.beginPath();
            const armRot = arm * Math.PI;
            for (let a = 0; a <= Math.PI * 1.35; a += 0.14) {
              const rad = (a / (Math.PI * 1.35)) * r * 1.55;
              const x = Math.cos(a + armRot) * rad;
              const y = Math.sin(a + armRot) * rad * 0.68;
              if (a === 0) ctx.moveTo(x, y);
              else ctx.lineTo(x, y);
            }
            ctx.stroke();
          }
          ctx.restore();
        }

        // Bright core with a hot white center fading into the cluster hue.
        const core = ctx.createRadialGradient(c.x, c.y, 0, c.x, c.y, r);
        core.addColorStop(0, "rgba(255,255,255,0.88)");
        core.addColorStop(0.2, hexToRgba(color, 0.82));
        core.addColorStop(1, hexToRgba(color, 0.04));
        ctx.fillStyle = core;
        ctx.beginPath();
        ctx.arc(c.x, c.y, r, 0, Math.PI * 2);
        ctx.fill();

        ctx.beginPath();
        ctx.arc(c.x, c.y, r, 0, Math.PI * 2);
        ctx.lineWidth = (1 + hoverT * 0.9) / k;
        ctx.strokeStyle = hexToRgba(color, 0.45 + hoverT * 0.35);
        ctx.stroke();

        // Scattered sub-stars so the bubble reads as "a cluster of stars,"
        // not a flat circle — seeded once per cluster, not regenerated.
        for (const s of c.scatter) {
          ctx.beginPath();
          ctx.arc(c.x + s.dx * r, c.y + s.dy * r, s.r, 0, Math.PI * 2);
          ctx.fillStyle = "#ffffff";
          ctx.globalAlpha = s.alpha;
          ctx.fill();
        }
        ctx.globalAlpha = 1;
      }
    }

    function drawExplodedClusters(ctx, k) {
      const hiddenIds = hiddenIdsRef.current;
      for (const [clusterId, { simNodes }] of explodedSimsRef.current) {
        if (hiddenIds.has(clusterId)) continue;
        const color = colorForCluster(clusterId);
        const centroid = centroidNodesRef.current.find((c) => c.clusterId === clusterId);

        // Faint constellation lines back to the cluster's center.
        if (centroid) {
          ctx.beginPath();
          for (const sn of simNodes) {
            ctx.moveTo(centroid.x, centroid.y);
            ctx.lineTo(sn.x, sn.y);
          }
          ctx.strokeStyle = color;
          ctx.globalAlpha = 0.08;
          ctx.lineWidth = 1 / k;
          ctx.stroke();
          ctx.globalAlpha = 1;
        }

        const halo = new Path2D();
        const core = new Path2D();
        const haloR = screenCappedRadius(STAR_RADIUS * 2.2, k, MAX_HALO_SCREEN_PX);
        const r = screenCappedRadius(STAR_RADIUS, k, MAX_CORE_SCREEN_PX);
        for (const sn of simNodes) {
          halo.moveTo(sn.x + haloR, sn.y);
          halo.arc(sn.x, sn.y, haloR, 0, Math.PI * 2);
          core.moveTo(sn.x + r, sn.y);
          core.arc(sn.x, sn.y, r, 0, Math.PI * 2);
        }
        ctx.fillStyle = color;
        ctx.globalAlpha = 0.16;
        ctx.fill(halo);
        ctx.globalAlpha = 1;
        ctx.fill(core);
      }
    }

    // Resolves a movie_id to its current on-screen position — where it is
    // actually rendered right now — so the selection ring and anchor marker
    // can follow the star rather than floating over empty space.
    //
    // Priority order:
    //   1. An exploded cluster's live simulation (has the up-to-date position
    //      as the bloom animation plays out — updates every frame)
    //   2. Full-sky mode true world position (star IS drawn at trueX/trueY)
    //   3. Overview, cluster NOT yet exploded: the cluster's centroid bubble
    //      (the galaxy bubble IS drawn there; floating the ring at the world
    //      position would show it hovering in empty space with nothing under it)
    function findStarPos(targetId) {
      // Check exploded sims first — these have live-updating positions.
      for (const [, { simNodes }] of explodedSimsRef.current) {
        const found = simNodes.find((sn) => sn.id === targetId);
        if (found) return found;
      }
      const wn = worldNodesRef.current.get(targetId);
      if (!wn) return null;
      // Full sky: star IS rendered at true UMAP position.
      if (fullSkyRef.current) return { x: wn.trueX, y: wn.trueY };
      // Overview, cluster not exploded: ring should appear on the galaxy bubble
      // (the star itself is not separately drawn in overview mode).
      const centroid = centroidNodesRef.current.find((c) => c.clusterId === wn.clusterId);
      if (centroid && !explodedIdsRef.current.has(wn.clusterId)) {
        return { x: centroid.x, y: centroid.y };
      }
      return { x: wn.trueX, y: wn.trueY };
    }

    function drawSelectionRing(ctx, k, now) {
      const targetId = selectedIdRef.current || hoveredRef.current;
      if (!targetId || (typeof targetId === "string" && targetId.startsWith("cluster:"))) return;
      const pos = findStarPos(targetId);
      if (!pos) return;
      // A slow pulsing double ring — an "event horizon" glow for whatever
      // star is currently selected or hovered.
      const pulse = 0.5 + 0.5 * Math.sin(now * 0.0026);
      const baseR = STAR_RADIUS_HOVER + 2;
      ctx.beginPath();
      ctx.arc(pos.x, pos.y, baseR + pulse * 1.6, 0, Math.PI * 2);
      ctx.strokeStyle = "rgba(232, 184, 75, 0.85)";
      ctx.lineWidth = 1.5 / k;
      ctx.stroke();
      ctx.beginPath();
      ctx.arc(pos.x, pos.y, baseR + 5 + pulse * 3.5, 0, Math.PI * 2);
      ctx.strokeStyle = `rgba(232, 184, 75, ${0.16 + pulse * 0.12})`;
      ctx.lineWidth = 1 / k;
      ctx.stroke();
    }

    // A persistent marker for the current search result — deliberately
    // styled nothing like the gold selection ring above (steady, not
    // pulsing; cyan, not gold; a reticle, not a glow) so the two can never
    // be confused, and deliberately NOT cleared by clicking a different
    // star (selectedIdRef changes on every click; anchorIdRef only
    // changes on a new search or Reset — see App.jsx). This is the literal
    // "how do I find my way back to what I searched for" answer: it just
    // stays on the map, a fixed-screen-size reticle so it stays equally
    // findable whether you're zoomed in past it or all the way out.
    function drawAnchorMarker(ctx, k, now) {
      const anchorId = anchorIdRef.current;
      if (!anchorId) return;
      const pos = findStarPos(anchorId);
      if (!pos) return; // not currently rendered (e.g. inside a different, still-collapsed cluster)
      const shimmer = 0.55 + 0.25 * Math.sin(now * 0.0009);
      const r = 9 / k; // constant screen size at any zoom level
      ctx.save();
      ctx.strokeStyle = `rgba(95, 179, 217, ${shimmer})`;
      ctx.lineWidth = 1.4 / k;
      ctx.beginPath();
      ctx.arc(pos.x, pos.y, r, 0, Math.PI * 2);
      ctx.stroke();
      // four short outward ticks — a target reticle, reading as "marked
      // reference point" rather than "currently active."
      ctx.beginPath();
      for (const [dx, dy] of [
        [0, -1],
        [0, 1],
        [1, 0],
        [-1, 0],
      ]) {
        const x1 = pos.x + dx * (r + 1.5 / k);
        const y1 = pos.y + dy * (r + 1.5 / k);
        const x2 = pos.x + dx * (r + 5 / k);
        const y2 = pos.y + dy * (r + 5 / k);
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
      }
      ctx.stroke();
      ctx.restore();
    }

    rafRef.current = requestAnimationFrame(draw);

    return () => {
      cancelAnimationFrame(rafRef.current);
      clearTimeout(fitTimer);
      resizeObserver.disconnect();
      canvas.removeEventListener("mousemove", handleMove);
      canvas.removeEventListener("click", handleClick);
      selection.on(".zoom", null);
    };
    // Setup (canvas sizing, zoom/pan, hover/click listeners, the rAF loop)
    // runs once on mount. Interactive props (fullSky, hiddenIds,
    // explodedIds, selectedId) are read via the refs kept in sync above,
    // not as dependencies here — otherwise every click/toggle would tear
    // down and recreate the zoom behavior and re-fit the camera, which is
    // exactly the jarring "view jumps on every interaction" bug this
    // structure avoids. The callback props (onHoverNode, etc.) are stable
    // across renders because App.jsx wraps them in useCallback.
  }, [onHoverNode, onSelectNode, onToggleExplode, onBackgroundClick, onZoomChange, buildStarfield, fitToView, applyViewLock]);

  // -- fly the camera to a search result -----------------------------------
  useEffect(() => {
    if (!highlightTarget || !canvasRef.current || !zoomBehaviorRef.current) return;
    const canvas = canvasRef.current;
    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    const k = Math.max(transformRef.current.k, 2.2);
    const tx = highlightTarget.x * WORLD_SCALE;
    const ty = highlightTarget.y * WORLD_SCALE;
    const target = d3.zoomIdentity.translate(w / 2 - tx * k, h / 2 - ty * k).scale(k);
    // Reuse the exact zoom behavior already bound to the canvas (not a new
    // instance) so its existing on("zoom", ...) handler fires on every tick
    // of this transition and keeps targetTransformRef in sync. The render
    // loop's own camera easing then smooths this transition further, which
    // is what keeps a search fly-to feeling like the same kind of motion
    // as a manual zoom instead of a separate, snappier animation.
    d3.select(canvas).transition().duration(900).ease(d3.easeCubicInOut).call(zoomBehaviorRef.current.transform, target);
  }, [highlightTarget]);

  return (
    <div ref={containerRef} className="canvas-layer">
      <canvas ref={canvasRef} />
    </div>
  );
}

/* ===========================================================================
SCALING NOTES — fluid interaction at 16k+ nodes
===========================================================================
1. Canvas, not SVG/DOM-per-node: each star is a couple of `ctx.arc` calls
   into a shared Path2D, not a DOM/SVG element. SVG starts to stutter well
   below 16k live nodes because every element is a DOM node the browser
   has to lay out and paint individually; canvas just rasterizes pixels.

2. Batched fills: `drawFullSky` groups every node by *style* (cluster color,
   or one of three field-star tiers for unclustered movies) into one Path2D
   per style and calls `ctx.fill()` once or twice per style (a few dozen
   calls total) instead of once per node (16k+ calls). State changes
   (fillStyle, beginPath) are the expensive part of canvas drawing, not the
   arcs themselves — batching collapses thousands of state changes into a
   handful. The halo/glow pass doubles the call count, not the order of
   magnitude — still O(distinct styles), not O(nodes).

3. No live physics for the full-sky view: x/y come straight from the
   backend's precomputed UMAP/PCA projection. Live force simulations only
   run for the small number of currently-exploded clusters (tens to a few
   hundred nodes each), where physics is cheap and adds real value (the
   "bloom" animation). Running a force simulation over all 16k nodes every
   frame is the single biggest thing that *would* make this laggy — so we
   deliberately never do it.

4. Hit-testing is decoupled from the render loop: a fresh `d3.quadtree` is
   only built on mousemove/click (throttled to ~35fps), not every animation
   frame, and only over the currently *visible* nodes (a handful of
   centroids, or one exploded cluster's members) rather than all 16k.

5. devicePixelRatio is capped at 2 (`dprRef`) — on a 3x/4x retina display,
   rendering at native pixel density buys invisible sharpness at a real
   frame-time cost; capping is a standard canvas-perf tradeoff.

6. The decorative backdrop (nebula + 3 star layers) is rasterized once per
   resize into offscreen canvases, not redrawn per frame — the render loop
   only ever does a handful of `drawImage`/gradient calls for it, with
   "twinkle" faked as a shared sine-driven alpha per layer instead of
   per-star animation.

7. Beyond ~50-100k nodes: swap the canvas 2D batched-Path2D approach for a
   real WebGL point-sprite renderer (regl, PixiJS particle container, or a
   custom GL program rendering one POINTS draw call for all nodes with
   per-instance color/position attributes). Canvas2D batching scales well
   into the tens of thousands; WebGL is the right tool once you're
   comfortably past that, because a single GPU draw call replaces what
   would otherwise be many thousands of canvas path operations per frame.

===========================================================================
VIEW LOCK — bounded camera, Google Maps style
===========================================================================
`applyViewLock` computes the world-space bounding box of every star and
cluster bubble (padded by VIEW_PADDING_RATIO) and feeds it to d3-zoom as
`translateExtent`. Combined with a `scaleExtent` floor set to the exact k
where that box fits the current viewport, this reproduces the Google Maps
"can't pan past the edge of the data, can't zoom out past the whole map"
behavior natively — d3-zoom's own constrain logic does the clamping, we
just keep feeding it the right box. It's recomputed on every resize (since
the fit zoom level depends on viewport size) and once data finishes
loading (since the bounds depend on the data itself).

===========================================================================
CAMERA SMOOTHING — decoupling input from the rendered transform
===========================================================================
d3-zoom's "zoom" event fires once per wheel tick / pointer-move with the
*new* transform already computed — applying that directly to the canvas is
what makes notchy mice or quick trackpad gestures feel stepped. Instead,
the zoom handler only updates `targetTransformRef` (where the camera should
end up); every animation frame, `transformRef` (what's actually drawn) eases
toward that target with a simple exponential smoothing factor based on the
frame's real elapsed time (`1 - exp(-dt / CAMERA_SMOOTH_TAU)`), not a fixed
per-frame constant — so it settles at the same *speed* regardless of frame
rate. The same mechanism smooths the search fly-to transition further (it's
already a d3 transition; the camera eases toward each of its intermediate
frames too), which is what keeps every kind of zoom — wheel, drag, or
search — feeling like one continuous motion instead of several different
animation systems.
=========================================================================== */
