import { defineConfig } from "tsup";

export default defineConfig({
  entry: ["src/index.ts"],
  format: ["cjs", "esm"], // Build for commonJS and ESmodules
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
