import { defineConfig } from "vite";
import preact from "@preact/preset-vite";

// Vite config notes (phase 8):
//
// `base` — the SPA is mounted at /_plugin/ui/, so every emitted
//   asset path needs that prefix. Without this, the built
//   index.html would reference /assets/foo.js and the wrapper
//   would route those requests to bifrost instead of the plugin.
//
// `build.rollupOptions.output.entryFileNames` (and chunk/asset
//   variants) — the plan explicitly disables filename hashing.
//   Long-term browser caching is meaningless for an embedded
//   admin UI behind session auth, and non-hashed filenames keep
//   the Go //go:embed index predictable across redeploys.
//
// `server.proxy` — dev-server proxy for `/_plugin/*` -> the
//   locally-running plugin (started via `make docker-up`). Lets a
//   developer iterate on the UI with Vite HMR while the backend
//   runs in the same container topology as production.
//
// `resolve.alias` — Tanstack Query ships as a React-only package.
//   Aliasing react/react-dom to preact/compat lets us consume it
//   directly without forking. The same alias is mirrored in
//   tsconfig.json's `paths`.
export default defineConfig({
  base: "/_plugin/ui/",
  plugins: [preact()],
  resolve: {
    alias: {
      react: "preact/compat",
      "react-dom": "preact/compat",
    },
  },
  build: {
    outDir: "dist",
    // Keep the placeholder .gitkeep around between local builds so
    // `npm run build` doesn't dirty the working tree. We re-enable
    // content-hashed filenames below so the browser cache busts
    // automatically on every redeploy — the embed handler doesn't
    // care about specific asset names (it serves whatever's in
    // dist/assets/), and `index.html` is the only file with a
    // fixed name, which already has `Cache-Control: no-store`.
    emptyOutDir: false,
    rollupOptions: {
      output: {
        entryFileNames: "assets/[name]-[hash].js",
        chunkFileNames: "assets/[name]-[hash].js",
        assetFileNames: "assets/[name]-[hash].[ext]",
      },
    },
  },
  server: {
    port: 5173,
    proxy: {
      "/_plugin": {
        target: "http://localhost:8181",
        changeOrigin: false,
      },
    },
  },
});
