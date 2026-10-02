#!/usr/bin/env node

// fastembed-js downloads models from Qdrant's GCS bucket, which now denies
// anonymous reads (403 XML body → TAR_BAD_ARCHIVE). Seed the cache directory
// from the HuggingFace mirror instead; FlagEmbedding.init skips the download
// when `<cacheDir>/<model>` already exists.

import fs from "fs";
import path from "path";
import { FlagEmbedding, EmbeddingModel } from "fastembed";

const MODEL = EmbeddingModel.BGESmallENV15;
const CACHE_DIR = "local_cache";
const HF_REPO = "Qdrant/bge-small-en-v1.5-onnx-Q";
const FILES = [
  "model_optimized.onnx",
  "tokenizer.json",
  "tokenizer_config.json",
  "config.json",
  "special_tokens_map.json",
];

async function download(file, dir) {
  const url = `https://huggingface.co/${HF_REPO}/resolve/main/${file}`;
  const res = await fetch(url);
  if (!res.ok) throw new Error(`GET ${url} → ${res.status}`);
  fs.writeFileSync(path.join(dir, file), Buffer.from(await res.arrayBuffer()));
  console.log(`[download-models]   ${file}`);
}

async function main() {
  console.log("[download-models] Downloading fastembed models...");
  const modelDir = path.join(CACHE_DIR, MODEL);
  if (!fs.existsSync(modelDir)) {
    const tmpDir = `${modelDir}.partial`;
    fs.rmSync(tmpDir, { recursive: true, force: true });
    fs.mkdirSync(tmpDir, { recursive: true });
    for (const file of FILES) await download(file, tmpDir);
    fs.renameSync(tmpDir, modelDir);
  }
  // Load once to verify the files are usable.
  await FlagEmbedding.init({ model: MODEL, cacheDir: CACHE_DIR });
}

// Exit naturally: process.exit() after an onnxruntime session loads can abort
// with "mutex lock failed" on macOS.
main()
  .then(() => {
    console.log("[download-models] ✓ Models downloaded successfully");
  })
  .catch((error) => {
    console.error("[download-models] ✗ Download failed:", error.message);
    process.exitCode = 1;
  });
