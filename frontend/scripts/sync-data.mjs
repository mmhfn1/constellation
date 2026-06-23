#!/usr/bin/env node
// scripts/sync-data.mjs
//
// Copies the precomputed pipeline artifacts (backend/data/*.json) into
// frontend/public/data/ so the static (no-backend) build can fetch them
// directly. This is what makes a GitHub Pages deploy fully self-contained:
// the same JSON app.py would otherwise serve over HTTP just ships as
// static files instead.
//
// Run this any time you re-run `python pipeline.py` and want the change
// reflected in the static build:
//   cd frontend && npm run sync-data
//
// The GitHub Actions deploy workflow runs it automatically before every
// build, so committing fresh backend/data/*.json is enough to redeploy.
import { copyFileSync, existsSync, mkdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const BACKEND_DATA = join(__dirname, "..", "..", "backend", "data");
const PUBLIC_DATA = join(__dirname, "..", "public", "data");

const FILES = ["graph_tier1.json", "cluster_summary.json", "movie_details.json"];

mkdirSync(PUBLIC_DATA, { recursive: true });

let copied = 0;
for (const file of FILES) {
  const src = join(BACKEND_DATA, file);
  const dest = join(PUBLIC_DATA, file);
  if (!existsSync(src)) {
    console.warn(`⚠ ${file} not found in backend/data/ — skipping. Run the pipeline first:`);
    console.warn("  cd backend && python pipeline.py");
    continue;
  }
  copyFileSync(src, dest);
  copied += 1;
  console.log(`✓ synced ${file}`);
}

if (copied === FILES.length) {
  console.log("All static data files synced into frontend/public/data/.");
} else if (copied === 0) {
  console.error("No data files found — the static build will have nothing to show.");
  process.exitCode = 1;
}
