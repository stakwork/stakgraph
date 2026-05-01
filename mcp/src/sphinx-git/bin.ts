#!/usr/bin/env node
import { run } from "./cli.js";

run(process.argv.slice(2)).catch((err) => {
  process.stderr.write(`sphinx-git: ${err?.message ?? err}\n`);
  process.exit(1);
});
