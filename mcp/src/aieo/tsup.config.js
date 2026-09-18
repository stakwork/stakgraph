import { defineConfig } from "tsup";

export default defineConfig({
  entry: ["src/index.ts"],
  format: ["esm"], // ESM only — ai v7 dropped CommonJS
  dts: true, // Generate declaration file (.d.ts)
  splitting: false,
  sourcemap: true,
  clean: true,
  noExternal: [
    "@ai-sdk/anthropic",
    "@ai-sdk/google",
    "@ai-sdk/openai",
    "@ai-sdk/provider-utils",
  ], // Bundle AI SDK dependencies to avoid version conflicts
});
