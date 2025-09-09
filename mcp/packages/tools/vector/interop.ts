import { createRequire } from "module";
const require = createRequire(import.meta.url);
const fastembed = require("fastembed");

export const EmbeddingModel = fastembed.EmbeddingModel;
export const FlagEmbedding = fastembed.FlagEmbedding;
