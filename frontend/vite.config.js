// vite.config.js - configuration for Vite, the dev server and build tool.

// `defineConfig` only adds editor autocompletion; it returns the object unchanged.
import { defineConfig } from "vite";
// The official plugin that compiles JSX and enables React Fast Refresh (instant reloads).
import react from "@vitejs/plugin-react";

// Export the configuration object Vite reads at startup.
export default defineConfig({
  // Enable the React plugin.
  plugins: [react()],
  // Settings for `npm run dev`.
  server: {
    // Serve the UI on http://localhost:5173.
    port: 5173,
    // Forward every request that starts with /api to the FastAPI backend.
    // The browser then talks to ONE origin, so no CORS issues in development.
    proxy: {
      "/api": {
        // Where the backend runs (override with VITE_PROXY_TARGET if needed).
        target: process.env.VITE_PROXY_TARGET ?? "http://localhost:8000",
        // Rewrite the Host header to match the target.
        changeOrigin: true,
      },
    },
  },
});
