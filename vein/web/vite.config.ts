import { defineConfig } from "vite";
import preact from "@preact/preset-vite";

export default defineConfig({
  // Relative asset URLs so the built UI can be served from any mount path
  // (root or a sub-path like /lab) without a baked-in base. Pairs with the
  // runtime API-base derivation in src/api.ts. Requires the host to serve
  // the SPA at a trailing-slash path so relative assets resolve correctly.
  base: "./",
  plugins: [preact()],
  server: {
    port: 5173,
    proxy: {
      "/workflows": {
        target: "http://localhost:3000",
        // Disable buffering so SSE events stream through in real time
        configure: (proxy) => {
          proxy.on("proxyRes", (proxyRes) => {
            if (proxyRes.headers["content-type"]?.includes("text/event-stream")) {
              proxyRes.headers["cache-control"] = "no-cache";
              proxyRes.headers["x-accel-buffering"] = "no";
            }
          });
        },
      },
      "/steps": "http://localhost:3000",
      "/chat": "http://localhost:3000",
      "/health": "http://localhost:3000",
    },
  },
  build: {
    outDir: "dist",
  },
});
