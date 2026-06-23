// HoverTooltip.jsx
// A DOM element that follows the cursor, rather than canvas text — keeps
// label rendering crisp at any zoom level without fighting canvas text
// scaling, and costs nothing extra since it only exists while hovering.
export default function HoverTooltip({ hover }) {
  if (!hover) return null;
  return (
    <div className="hover-tooltip" style={{ left: hover.clientX, top: hover.clientY }}>
      {hover.title || "…"}
      {hover.clusterLabel && <span className="ht-cluster">{hover.clusterLabel}</span>}
    </div>
  );
}
