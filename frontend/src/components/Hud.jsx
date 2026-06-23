// Hud.jsx
export function Hud({ nodeCount, zoom, visibleLabel }) {
  return (
    <div className="hud glass-panel">
      <span>
        <strong>{nodeCount.toLocaleString()}</strong> movies
      </span>
      <span>
        zoom <strong>{zoom.toFixed(2)}×</strong>
      </span>
      <span>{visibleLabel}</span>
    </div>
  );
}

export function Hint({ visible }) {
  return (
    <div className={`hint glass-panel ${visible ? "" : "is-gone"}`}>
      scroll to zoom · drag to pan · click a cluster to explode it
    </div>
  );
}
