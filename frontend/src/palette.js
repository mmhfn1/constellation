// palette.js
// Categorical color palette for HDBSCAN cluster ids ("archetypes"). Picked
// as jewel tones that read clearly against the near-black map background
// and stay distinguishable for the most common forms of color vision
// deficiency (no pure red/green pair sits adjacent in the cycle).
export const CLUSTER_COLORS = [
  "#E8B84B", // marquee gold
  "#5FB3D9", // cyan
  "#E2667D", // rose
  "#7FD9A6", // mint
  "#B07FE0", // violet
  "#E0935F", // amber
  "#5FD9C8", // teal
  "#D95F9E", // magenta
  "#9FD95F", // chartreuse
  "#5F8AE0", // periwinkle
  "#D9C25F", // ochre
  "#8E7FE0", // indigo
];

export const NOISE_COLOR = "#4a4d63"; // legacy export, kept for anything still importing it

export function colorForCluster(clusterId) {
  if (clusterId === -1 || clusterId === null || clusterId === undefined) {
    return NOISE_COLOR;
  }
  return CLUSTER_COLORS[clusterId % CLUSTER_COLORS.length];
}

// ---------------------------------------------------------------------------
// Field stars — the ~44% of movies HDBSCAN couldn't assign to an archetype
// ("noise", cluster_id -1). Painting all of these as one flat dull color is
// what previously read as "dark spots": a muddy, hole-punched look against
// the night sky, especially where they sit beside bright cluster stars.
// Real starfields aren't one brightness — they're mostly dim, scattered
// pinpoints with the occasional standout. These three tiers (assigned
// deterministically per movie, see hashUnit() in GalaxyCanvas) recreate that
// variation cheaply: still just a few batched canvas draws, not one extra
// call per star.
// ---------------------------------------------------------------------------
export const FIELD_STAR_TIERS = [
  { key: "field-dim", color: "#7782ad", r: 1.5, coreAlpha: 0.42, haloAlpha: 0, threshold: 0.56 },
  { key: "field-warm", color: "#d2c08c", r: 1.95, coreAlpha: 0.64, haloAlpha: 0.1, threshold: 0.87 },
  { key: "field-bright", color: "#f1f4ff", r: 2.5, coreAlpha: 0.92, haloAlpha: 0.22, threshold: 1.01 },
];

export function fieldStarTierFor(unit) {
  for (const tier of FIELD_STAR_TIERS) {
    if (unit < tier.threshold) return tier;
  }
  return FIELD_STAR_TIERS[FIELD_STAR_TIERS.length - 1];
}

// Convert a palette hex string to rgba() with a given alpha — every glow,
// gradient, and halo in the canvas renderer needs this since canvas
// gradients/fills want explicit alpha channels, not CSS color-mix.
export function hexToRgba(hex, alpha) {
  const h = hex.replace("#", "");
  const full = h.length === 3 ? h.split("").map((c) => c + c).join("") : h;
  const int = parseInt(full, 16);
  const r = (int >> 16) & 255;
  const g = (int >> 8) & 255;
  const b = int & 255;
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}
