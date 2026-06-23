# Constellation — A Movie Similarity Map

An interactive atlas of ~16k films: a PyTorch autoencoder + sentence-transformer
embeddings compress each movie into a latent vector, HDBSCAN groups latent
vectors into "archetypes," and a D3 + Canvas frontend renders the whole thing
as a navigable night sky — cluster bubbles ("galaxies") you can click to
explode into their individual movies ("stars"), with a locked, Google Maps–
style camera and a layered deep-space backdrop.

```
constellation/
├── .github/workflows/deploy.yml   # builds frontend/ and publishes it to GitHub Pages
├── backend/
│   ├── config.py              # every path/hyperparameter/CORS setting, in one place
│   ├── data_loader.py         # CSV -> clean, standardized feature matrix
│   ├── embeddings.py          # all-MiniLM-L6-v2 title embeddings (+ scaling notes)
│   ├── model.py                # PyTorch Autoencoder, GPU training, 70/30 split
│   ├── cluster_service.py     # HDBSCAN archetypes + UMAP 2D projection
│   ├── database.py            # SQLite feedback loop (clicks/likes)
│   ├── pipeline.py            # orchestrates the steps above, writes JSON artifacts
│   ├── generate_sample_data.py# fabricates a CSV matching the expected schema
│   ├── app.py                  # FastAPI: the two-tier API (optional — see below)
│   └── requirements.txt
└── frontend/
    ├── src/
    │   ├── components/GalaxyCanvas.jsx   # the D3 + Canvas star map
    │   ├── components/Sidebar.jsx        # search + cluster legend/filter
    │   ├── components/DetailPanel.jsx    # Tier-2 on-demand movie detail
    │   ├── api.js          # switches between live backend and bundled static data
    │   ├── staticData.js   # the static-mode data engine (no backend required)
    │   ├── palette.js, App.jsx, styles.css
    ├── public/
    │   ├── data/            # bundled pipeline output — what makes GitHub Pages work
    │   └── sample.html      # standalone, no-build-step preview of the map
    ├── scripts/sync-data.mjs   # re-syncs public/data/ from backend/data/
    └── package.json
```

## What this revision changed

Starting from the original two-tier-API design, this pass focused on four things:

1. **Fixed the "dark spots."** In "reveal full sky" mode, the ~44% of movies
   HDBSCAN couldn't assign to an archetype ("noise") were all painted one flat
   dull color — against the near-black background, that read as muddy holes
   right next to the bright cluster stars. They're now split into three
   deterministic field-star tiers (dim/warm/bright, picked per movie by a
   stable hash so the sky doesn't reshuffle on reload), the same way a real,
   unevenly-lit starfield looks. See `fieldStarTierFor` in `palette.js`.
2. **Locked the camera.** `GalaxyCanvas.jsx` now computes the world-space
   bounding box of the whole dataset and feeds it to d3-zoom as
   `translateExtent`, with a `scaleExtent` floor at the exact zoom level
   where that box fits the viewport — the Google Maps "can't pan into the
   void, can't zoom out past the whole map" behavior, recomputed on resize.
3. **Made it feel like space.** A layered backdrop (a soft nebula wash, three
   parallax star layers with shared-sine twinkle, a couple of seeded black-
   hole/supernova-remnant curiosities), galaxy-style cluster bubbles
   (gradient core, atmospheric glow, scattered sub-stars, slow spiral arms on
   the largest few), enhanced supernova-style explode bursts, a pulsing
   "event horizon" selection ring, and ambient shooting stars.
4. **Smoothed every zoom.** The camera no longer snaps directly to each
   wheel/drag event; it eases toward a target transform every frame with
   frame-rate-independent exponential smoothing, so wheel zoom, drag-pan, and
   the search fly-to all feel like one continuous motion. See "CAMERA
   SMOOTHING" in the comment block at the bottom of `GalaxyCanvas.jsx`.

Also fixed along the way:
- Cluster-bubble hit-testing previously used one flat click-tolerance
  regardless of a bubble's actual rendered size, so clicking anywhere but
  very close to a large galaxy's center could miss it — bubbles are now
  tested against their own radius.
- The ~44% of movies in the "noise" bucket (`cluster_id: -1`, no archetype)
  had no rendering path at all outside full-sky mode — no centroid bubble
  (they're not a real cluster) and no exploded-cluster stars (they're not
  in one). Searching for or otherwise selecting one of these movies in
  overview mode would fly the camera to it and show a selection ring over
  empty space, with nothing actually drawn there. They're now rendered as
  the same ambient field-star layer full sky uses, underneath the galaxy
  bubbles, in overview mode too — both fixing that and making the
  background starfield permanent rather than something that only appears
  in one view mode.

A second pass addressed three more things:

1. **No more unclustered "island" movies.** HDBSCAN's hard labeling marks
   anything that doesn't clear its density bar as noise (`cluster_id: -1`)
   — by design, not a bug, but on a map that reads as islands floating with
   nothing nearby, not because a movie is unique, just because it didn't
   fit cleanly. `cluster_service.run_hdbscan` now reassigns every noise
   point to its best-fitting archetype via HDBSCAN's own soft-clustering
   membership vectors (`REASSIGN_NOISE_POINTS = True` in `config.py`) —
   every movie ends up in *some* archetype, even a loose-fit one, rather
   than none. Verified against the real dataset: 7,015 noise points out of
   16,252 reassigned, 0 left over, all 68 archetypes summing back to the
   full count. Set `REASSIGN_NOISE_POINTS = False` to restore the old
   explicit-noise-bucket behavior. This only takes effect on a
   `pipeline.py` re-run — see "Running it locally" if you're working from
   already-generated `backend/data/*.json`.
2. **Dense clusters stay legible at high zoom.** A star's on-screen size
   used to scale linearly with zoom forever, so zooming into a
   several-hundred-movie cluster could blow individual stars up into
   overlapping blobs that hid their neighbors entirely — the opposite of
   what zooming in is for. `screenCappedRadius` in `GalaxyCanvas.jsx` (and
   the equivalent in `sample.html`) now caps how large a star's halo/core
   can render on screen regardless of zoom level, so neighboring movies
   stay visually distinct even zoomed all the way in.
3. **A persistent marker for search results.** Searching for a movie, then
   clicking a different nearby star to compare it, used to lose track of
   the original result entirely — `selectedId` (whatever's currently
   clicked) and the search target were the same piece of state. They're
   now separate: a steady cyan reticle marks the current search result and
   stays put regardless of what else you click, paired with a small chip
   in the sidebar (click to fly back to it, × to dismiss) — see
   `searchAnchor` in `App.jsx`.

Also new: an in-app "About" page (the **?** button next to the wordmark) —
written for whoever's looking at the map, not whoever's setting it up. The
existing developer-facing docs stay here in this README.

## Running it locally

### Backend (optional — see "Deploying" below for why)

```bash
cd backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # see the file for a CPU-only torch note

python generate_sample_data.py --n 16000   # or drop in your own CSV
python pipeline.py                          # writes backend/data/*.json
uvicorn app:app --reload --port 8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Open the printed localhost URL. By default the frontend runs in **static
mode** — no backend required, it reads `frontend/public/data/*.json` directly
(see "Two ways to run this" below). To use a live backend instead, copy
`.env.example` to `.env` and set `VITE_API_BASE_URL=http://localhost:8000`.

If you re-run the pipeline and want the static build to reflect it:

```bash
cd frontend && npm run sync-data
```

## Deploying to GitHub Pages

GitHub Pages only serves static files — it can't run the FastAPI backend.
This repo handles that by giving the frontend two modes:

- **Static mode (default, what GitHub Pages uses):** the app reads the exact
  same JSON `pipeline.py` produces, just bundled as static files
  (`frontend/public/data/`) instead of served over HTTP. Search runs
  in-memory; likes/clicks persist to the visitor's own browser
  (`localStorage`) instead of a shared database, since there's no server to
  aggregate them across visitors. Everything else — the map, explode/
  collapse, full sky, the detail panel — works identically. See
  `frontend/src/staticData.js`.
- **Live mode:** if `VITE_API_BASE_URL` is set at build time, the app talks
  to a real `backend/app.py` over REST instead, exactly as the original
  design — useful for local dev, or if you've deployed the backend somewhere
  yourself (Render, Fly, Railway, a VPS) and want shared, cross-visitor
  feedback aggregation.

### Steps

1. Push this repo to GitHub.
2. In the repo's **Settings → Pages**, set **Source** to **GitHub Actions**.
3. Push to `main`. `.github/workflows/deploy.yml` builds `frontend/` and
   publishes it automatically — including detecting whether your repo is a
   normal project repo (served at `https://<user>.github.io/<repo-name>/`)
   or the special `<user>.github.io` repo (served at the domain root), and
   setting Vite's `base` path accordingly. You don't need to hardcode either
   your username or the repo name anywhere.
4. Your site is live at the URL GitHub Pages shows in Settings → Pages once
   the workflow finishes (check the Actions tab for progress/logs).

If you re-run the pipeline later, commit the updated `backend/data/*.json`
and push — the workflow re-syncs `frontend/public/data/` and redeploys
automatically (it also runs `npm run sync-data` itself as a safety net, in
case `public/data/` wasn't updated locally before the push).

### Optional: a live backend

To point the deployed site at a live backend instead of static data, set
`VITE_API_BASE_URL` as a repository variable (Settings → Secrets and
variables → Actions → Variables) and uncomment the corresponding line in
`.github/workflows/deploy.yml`. `backend/app.py`'s CORS config already
allows any `*.github.io` origin out of the box; for a custom domain, set the
`ALLOWED_ORIGINS` environment variable on whatever's running the backend
(comma-separated list of origins).

## sample.html — a standalone preview

`frontend/public/sample.html` is a single self-contained file — vanilla JS,
D3 pulled from a CDN, no React, no build step — that renders the same space
backdrop, fixed field-star treatment, galaxy-style clusters, locked camera,
and explode interaction as the full app, reading the same bundled
`data/*.json`. It deploys automatically alongside the main app (reachable at
`/sample.html` on your Pages URL) and is also useful as a quick way to read
or fork the core rendering logic without a React/Vite toolchain in the way.

It skips search, the detail panel, and likes — those stay exclusive to the
full app. To view it locally, serve the `frontend/public/` folder with any
static file server (`npx serve`, `python3 -m http.server`, etc.) — opening it
directly as a `file://` URL won't work, since browsers block local `fetch()`
calls for security reasons.

## Using the map

- **Scroll** to zoom, **drag** to pan — the camera is locked to the data's
  own bounds, the same way Google Maps stops you at the edge of the world.
- **Click a galaxy** to explode it into its individual stars (a small live
  force simulation blooms them outward into their real positions, with a
  brief supernova-style flash). Click an exploded galaxy's position again —
  or right-click any legend row — to fold it back / hide it.
- **"Reveal full sky"** switches to rendering every movie at once, directly
  at its precomputed position (no live physics) — the path that stays fluid
  at 16k+ nodes. See the `SCALING NOTES` comment at the bottom of
  `GalaxyCanvas.jsx` for the performance reasoning.
- **Search** flies the camera to a title (resolved server-side in live mode,
  in-memory in static mode — either way, raw titles never ship in the bulk
  Tier-1 payload used to draw the map itself). A cyan marker stays on that
  result even after you click elsewhere — click its chip in the sidebar to
  fly back, or × to dismiss it.
- **Click a star** to open its detail panel and **like** it — in live mode,
  likes/clicks log to `backend/data/feedback.db` (SQLite); in static mode
  (GitHub Pages), they persist to the visitor's own browser.
- The **?** button next to the wordmark opens an in-app, plain-language
  explanation of the map for whoever's looking at it — distinct from this
  README, which is for whoever's setting the project up.

## Notes on what's been verified vs. left for your machine

The original pipeline (`data_loader.py`, `pipeline.py` with its PCA/KMeans
fallbacks when torch/hdbscan/umap-learn aren't installed) was built and run
end-to-end in a sandbox without GPU/network access, producing the real,
inspectable JSON artifacts currently sitting in `backend/data/`. The
noise-reassignment change above was verified against real HDBSCAN + UMAP,
not the fallback — `hdbscan`/`umap-learn` were installed and
`cluster_service.build_cluster_artifacts` was re-run against the actual
cached `latent_vectors.npy` for this dataset (no retraining needed — that's
exactly what that cache is for), confirming 0 noise points and all 68
archetypes summing back to the full 16,252-movie count before the data in
this repo was regenerated. Frontend changes — explode/collapse, search, the
anchor marker, the size-capped zoom, the docs panel, the detail panel, the
like button's localStorage persistence, the view lock, and the GitHub Pages
base-path handling — were all exercised against a live Vite build and a
real Chromium browser, clicked through and screenshotted, not just read
over. Still worth a final smoke test on your own machine and an actual
GitHub Pages deploy before you consider it done. `backend/data/` already
includes a trained `autoencoder.pt` and its `latent_vectors.npy` from
whatever pipeline run originally produced this dataset — if you change the
underlying movie data, you'll need `torch` + `sentence-transformers`
installed to retrain it (`python pipeline.py`); to only re-cluster the
existing latent space (e.g. after changing `HDBSCAN_MIN_CLUSTER_SIZE` or
`REASSIGN_NOISE_POINTS`), you only need `hdbscan` + `umap-learn` and can
reuse the cached `latent_vectors.npy` directly, which is what verifying the
noise-reassignment change above actually did.

## Extending the feedback loop

`database.py`'s `get_like_counts()` is the simplest possible signal a future
training run could use — e.g. up-weighting liked movies' rows when
retraining the autoencoder, or as positive pairs for a future "movies you
might like" model. The `interactions` table also logs `view`/`click`/
`explode_cluster` events with a per-browser-session id, if you want richer
implicit signal than just likes. This only applies in live mode — static
mode's `localStorage` equivalent (`staticData.js`) is shaped the same way,
but is obviously per-visitor rather than aggregated.
