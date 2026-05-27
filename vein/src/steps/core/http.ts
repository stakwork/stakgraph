import { z } from "zod";
import { defineStep } from "../../core.js";

export default defineStep({
  type: "http",
  input: z.object({
    url: z.string(),
    method: z.enum(["GET", "POST", "PUT", "DELETE", "PATCH"]).default("GET"),
    body: z.any().optional(),
    headers: z.record(z.string()).optional(),
  }),
  output: z.any(),
  async run(cfg) {
    const res = await fetch(cfg.url, {
      method: cfg.method,
      body: cfg.body !== undefined ? JSON.stringify(cfg.body) : undefined,
      headers: {
        ...(cfg.body !== undefined
          ? { "Content-Type": "application/json" }
          : {}),
        ...cfg.headers,
      },
    });

    let body: unknown;
    const contentType = res.headers.get("content-type") ?? "";
    if (contentType.includes("application/json")) {
      body = await res.json();
    } else {
      body = await res.text();
    }

    if (!res.ok) {
      throw new Error(
        `HTTP ${cfg.method} ${cfg.url} returned ${res.status}: ${typeof body === "string" ? body : JSON.stringify(body)}`,
      );
    }

    return { status: res.status, body };
  },
});
