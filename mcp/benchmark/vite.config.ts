import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import path from "path";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  base: "/sessions/",
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  server: {
    port: 3366,
    proxy: {
      "/api": "http://localhost:3355",
      // Concept documentation lives behind auth on the main app, not on the
      // (public) sessions router — see api.ts.
      "/gitree": "http://localhost:3355",
    },
  },
});
