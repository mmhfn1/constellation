import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// GitHub Pages project sites are served from https://<user>.github.io/<repo-name>/,
// so every asset URL needs that repo name prefixed as a base path — without
// it the deployed page requests /assets/... at the domain root and 404s.
// User/org sites (a repo literally named <user>.github.io) are served from
// the domain root instead. The included GitHub Actions workflow
// (.github/workflows/deploy.yml) detects which case applies and sets
// BASE_PATH automatically; this just falls back to "/" for local dev and
// for anyone building by hand.
export default defineConfig({
  plugins: [react()],
  base: process.env.BASE_PATH || "/",
  server: {
    port: 5173,
  },
  preview: {
    port: 4173,
  },
});
