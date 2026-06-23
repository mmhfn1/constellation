// DocsPanel.jsx
//
// In-app documentation for the person looking at the map — not a README
// (that's for someone setting the project up), this is "what am I looking
// at and how do I use it," written for a visitor who just wants to
// understand the thing in front of them. Opened from the "?" button in
// the sidebar header.
import { useEffect } from "react";

export default function DocsPanel({ open, onClose }) {
  useEffect(() => {
    if (!open) return;
    function onKeyDown(e) {
      if (e.key === "Escape") onClose();
    }
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [open, onClose]);

  if (!open) return null;

  return (
    <div className="docs-overlay" onMouseDown={(e) => e.target === e.currentTarget && onClose()}>
      <div className="docs-panel glass-panel">
        <div className="docs-panel-header">
          <h1 className="wordmark">
            About Constellation<span className="dot">.</span>
          </h1>
          <button className="docs-close" onClick={onClose} aria-label="Close">
            ×
          </button>
        </div>

        <section className="docs-section">
          <p className="docs-lede">
            Constellation is a map of roughly 16,000 movies, arranged so that movies with a similar feel sit near
            each other — the same way a real star map groups stars into constellations, not because anyone decided
            they belonged together, but because of where they actually are.
          </p>
        </section>

        <section className="docs-section">
          <h2 className="docs-heading">Reading the map</h2>
          <dl className="docs-glossary">
            <dt>Galaxies</dt>
            <dd>
              Each glowing bubble is an <em>archetype</em> — a group of movies that turned out to be unusually
              similar to each other. Its size reflects how many movies are in it; its label is the genres most
              common among its members.
            </dd>
            <dt>Stars</dt>
            <dd>
              Click a galaxy and it blooms outward into its individual movies — a small supernova, then each one
              settles at its real position. Click any star to see its details.
            </dd>
            <dt>Loosely-fitting movies</dt>
            <dd>
              Not every movie fits neatly into one archetype — some are a genuine blend of several. Rather than
              leaving those unsorted, each one is placed in whichever archetype it resembles most closely, even
              when that resemblance is loose. Every movie belongs to a galaxy; some just sit nearer its edge.
            </dd>
            <dt>The locked camera</dt>
            <dd>
              You can't scroll or drag past the edge of the actual data, the same way a map application stops you
              at the edge of the world — there's nothing out there to find, so the camera doesn't pretend otherwise.
            </dd>
          </dl>
        </section>

        <section className="docs-section">
          <h2 className="docs-heading">Using it</h2>
          <ul className="docs-list">
            <li>
              <strong>Search</strong> a title to fly straight to it. A small cyan marker stays on that result even
              after you click around nearby stars, so you can always find your way back to it.
            </li>
            <li>
              <strong>Click a galaxy</strong> to explode it into its movies; click again (or right-click its row in
              the list) to collapse or hide it.
            </li>
            <li>
              <strong>"Reveal full sky"</strong> shows every movie at once, at its true position, instead of grouped
              into galaxies.
            </li>
            <li>
              <strong>Scroll to zoom, drag to pan.</strong> "Reset" returns to the very first view.
            </li>
          </ul>
        </section>

        <section className="docs-section">
          <h2 className="docs-heading">How the map was actually made</h2>
          <p>
            Each movie's metadata — genres, rating, runtime, vote count — and a semantic reading of its title are
            compressed by a small neural network into a compact numeric "fingerprint." Movies that are described
            similarly end up with similar fingerprints, the same way two songs in a similar key sound alike even
            with different lyrics.
          </p>
          <p>
            An unsupervised algorithm then looks at all ~16,000 fingerprints and finds natural groupings —
            archetypes nobody hand-labeled in advance, discovered purely from which movies' fingerprints cluster
            together densely. A separate step flattens that high-dimensional fingerprint space down to the two
            coordinates you actually see, trying to preserve one rule above all: things that were similar up there
            should still be near each other down here.
          </p>
          <p>
            None of this reads a plot summary or understands a movie the way a person does — it's working entirely
            from structured metadata and title text. Archetypes are statistical groupings, not editorial genres, and
            occasionally a film will land in a neighborhood that's a reasonable nearest-fit rather than an obvious
            one.
          </p>
        </section>

        <section className="docs-section docs-footer">
          <p>Built with a PyTorch autoencoder, HDBSCAN clustering, and a UMAP projection.</p>
        </section>
      </div>
    </div>
  );
}
