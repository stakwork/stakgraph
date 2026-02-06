#!/usr/bin/env node

import { FlagEmbedding, EmbeddingModel } from "fastembed";

console.log("[download-models] Downloading fastembed models...");

FlagEmbedding.init({
  model: EmbeddingModel.BGESmallENV15,
  cacheDir: "local_cache",
  showDownloadProgress: true,
})
  .then(() => {
    console.log("[download-models] ✓ Models downloaded successfully");
    process.exit(0);
  })
  .catch((error) => {
    console.error("[download-models] ✗ Download failed:", error.message);
    process.exit(1);
  });
