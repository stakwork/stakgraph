import { z } from "zod";
import { defineStep } from "../../core.js";

export default defineStep({
  type: "log",
  input: z.object({
    message: z.string(),
    level: z.enum(["info", "warn", "error"]).default("info"),
  }),
  output: z.string(),
  async run(cfg) {
    const prefix = `[${cfg.level.toUpperCase()}]`;
    const line = `${prefix} ${cfg.message}`;

    switch (cfg.level) {
      case "error":
        console.error(line);
        break;
      case "warn":
        console.warn(line);
        break;
      default:
        console.log(line);
    }

    return cfg.message;
  },
});
