import { Request, Response } from "express";
import fs from "fs";
import { signApiToken } from "./repo/events.js";

/**
 * Serves an index.html file with a short-lived JWT injected into
 * `window.__AUTH_TOKEN__` so the client-side app can authenticate
 * subsequent API calls without exposing the API_TOKEN itself.
 *
 * If API_TOKEN is not set (dev mode), the token is left empty and
 * API calls will pass through unauthenticated.
 */
export function sendIndexWithToken(indexPath: string) {
  return (_req: Request, res: Response) => {
    fs.readFile(indexPath, "utf8", (err, html) => {
      if (err) {
        res.status(500).send("Error loading app");
        return;
      }
      let token = "";
      try {
        token = signApiToken("1h");
      } catch (_) {
        // API_TOKEN not set (dev mode) — leave token empty
      }
      const injected = html.replace(
        "</head>",
        `<script>window.__AUTH_TOKEN__=${JSON.stringify(token)}</script></head>`,
      );
      res.type("html").send(injected);
    });
  };
}
