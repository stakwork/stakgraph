/**
 * Vision service: the agent's "eyes" — judge a full-page screenshot (plus the
 * browser console/network errors and the server log tail) with a vision model.
 * Lifted from `boot-and-exercise.ts`. Stateless, so it's a plain object (no
 * per-run session).
 */
import { readFileSync } from "node:fs";
import { z } from "zod";
import { usageFromResult, computeCost } from "vein";
import type { Observations } from "./browser.js";
import { summarizeObs } from "./browser.js";

export interface VisionVerdict {
  working: boolean;
  reason: string;
  usage: ReturnType<typeof usageFromResult>;
  cost: number;
}

export interface VisionService {
  assess(
    pngPath: string,
    url: string,
    obs: Observations,
    logs: string,
    model?: string,
  ): Promise<VisionVerdict>;
}

export function buildVisionService(): VisionService {
  return {
    async assess(pngPath, url, obs, logs, model) {
      const { generateObject } = await import("ai");
      const { anthropic } = await import("@ai-sdk/anthropic");
      const m = anthropic(model ?? process.env["VEIN_LLM_MODEL"] ?? "claude-sonnet-4-6");
      const schema = z.object({
        working: z
          .boolean()
          .describe(
            "true ONLY if the intended app UI rendered (a real, populated, styled page — a login/landing page counts) AND there is no blank/white page, error overlay, stack trace, or 404/500, AND no fatal console/network/server error is breaking the page.",
          ),
        reason: z.string().describe("one or two short sentences: what the screenshot shows + whether the errors look fatal."),
      });
      const image = readFileSync(pngPath);
      const { object, usage: rawUsage } = await generateObject({
        model: m as any,
        schema: schema as any,
        messages: [
          {
            role: "user",
            content: [
              {
                type: "text",
                text: `A web app is booted locally at ${url}. Below is a full-page screenshot, the browser console/network errors, and the server log tail. Did the app's intended UI render and is it functional (not a blank/error/404/500 page, no fatal errors)?

BROWSER OBSERVATIONS:
${summarizeObs(obs)}

SERVER LOGS (tail):
${logs ? logs.slice(-6000) : "(none)"}`,
              },
              { type: "image", image },
            ],
          },
        ],
      });
      const usage = usageFromResult(rawUsage);
      const o = object as { working: boolean; reason: string };
      return { working: o.working, reason: o.reason, usage, cost: computeCost("anthropic", usage) };
    },
  };
}
