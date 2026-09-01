import { z, defineStep, type StepContext, type VeinCapabilities } from "vein";
import { createSign } from "node:crypto";

// ── sheets/* preamble — duplicated in every sheets/* step; see _shared.ts ──
const SHEETS_API = "https://sheets.googleapis.com/v4/spreadsheets";
const DRIVE_API = "https://www.googleapis.com/drive/v3/files";
const OAUTH_SCOPES =
  "https://www.googleapis.com/auth/spreadsheets https://www.googleapis.com/auth/drive";
const DEFAULT_TOKEN_URI = "https://oauth2.googleapis.com/token";
/** Refresh the cached access token this many seconds before it expires. */
const TOKEN_EXPIRY_SLACK_S = 60;

interface ServiceAccount {
  client_email: string;
  private_key: string;
  token_uri?: string;
}

/** Accept a service account as an object, a JSON string, or base64-encoded JSON. */
function parseServiceAccount(input: unknown): ServiceAccount {
  let obj: any = input;
  if (typeof input === "string") {
    const trimmed = input.trim();
    const json = trimmed.startsWith("{")
      ? trimmed
      : Buffer.from(trimmed, "base64").toString("utf-8");
    obj = JSON.parse(json);
  }
  if (
    !obj ||
    typeof obj !== "object" ||
    typeof obj.client_email !== "string" ||
    typeof obj.private_key !== "string"
  ) {
    throw new Error(
      "GOOGLE_SERVICE_ACCOUNT_JSON must be service-account JSON with client_email and private_key",
    );
  }
  return {
    client_email: obj.client_email,
    // Keys pasted through env/JSON sometimes carry literal "\n" sequences.
    private_key: obj.private_key.replace(/\\n/g, "\n"),
    token_uri: typeof obj.token_uri === "string" ? obj.token_uri : undefined,
  };
}

function truncate(value: unknown, maxChars: number): string {
  const str = typeof value === "string" ? value : JSON.stringify(value);
  if (str.length <= maxChars) return str;
  return `${str.slice(0, maxChars)}…[truncated, ${str.length} chars total]`;
}

function errorResult(label: string, status: number, body: any): string {
  const message = body?.error?.message ?? body;
  return `${label} failed: HTTP ${status}: ${truncate(message, 500)}`;
}

/** Access token cached per step module (Google tokens last ~1h), keyed by
 *  client_email so different credentials can never mix. */
let cachedToken: { email: string; value: string; exp: number } | null = null;

/** Resolve credentials (cfg wins → secret store → env) and return an
 *  authenticated Sheets/Drive request helper over ctx.services.http. */
async function sheetsCtx(
  ctx: StepContext<VeinCapabilities> | undefined,
  cfg: { serviceAccount?: unknown; driveFolderId?: string },
) {
  const httpMaybe = ctx?.services?.http;
  if (!httpMaybe) {
    throw new Error("sheets: ctx.services.http unavailable — run with a services bag");
  }
  const http = httpMaybe;
  const secrets = ctx?.services?.secrets;
  const rawSa = cfg.serviceAccount ?? (await secrets?.get("GOOGLE_SERVICE_ACCOUNT_JSON"));
  if (!rawSa) {
    throw new Error(
      // Aimed at the AGENT that called this tool, not a human operator: an
      // instructional "paste the key into the env" message was observed live
      // sending an agent credential-hunting (env greps, full-disk finds for
      // service-account JSONs). State it as an environment FACT with a
      // fallback, never as a fixable configuration step.
      "Google Sheets tools are UNAVAILABLE in this environment (no service account is configured). " +
        "This is not something you can fix: do NOT search the environment or filesystem for credentials, " +
        "and do NOT retry other sheets tools — they will all fail the same way. Proceed WITHOUT " +
        "spreadsheets: keep tabular and numeric work in local markdown files in your working directory.",
    );
  }
  const sa = parseServiceAccount(rawSa);
  const driveFolderId =
    cfg.driveFolderId ?? (await secrets?.get("GOOGLE_DRIVE_FOLDER_ID")) ?? undefined;

  async function getToken(): Promise<string> {
    const now = Math.floor(Date.now() / 1000);
    if (
      cachedToken &&
      cachedToken.email === sa.client_email &&
      cachedToken.exp - TOKEN_EXPIRY_SLACK_S > now
    ) {
      return cachedToken.value;
    }
    const tokenUri = sa.token_uri || DEFAULT_TOKEN_URI;
    // RS256 service-account JWT built with node:crypto (no jsonwebtoken dep).
    const enc = (o: unknown) => Buffer.from(JSON.stringify(o)).toString("base64url");
    const signingInput = `${enc({ alg: "RS256", typ: "JWT" })}.${enc({
      iss: sa.client_email,
      scope: OAUTH_SCOPES,
      aud: tokenUri,
      iat: now,
      exp: now + 3600,
    })}`;
    const signature = createSign("RSA-SHA256")
      .update(signingInput)
      .sign(sa.private_key)
      .toString("base64url");
    const res = await http(tokenUri, {
      method: "POST",
      headers: { "Content-Type": "application/x-www-form-urlencoded" },
      body: new URLSearchParams({
        grant_type: "urn:ietf:params:oauth:grant-type:jwt-bearer",
        assertion: `${signingInput}.${signature}`,
      }).toString(),
      timeout: 30_000,
    });
    const body = res.body as any;
    if (!res.ok || !body?.access_token) {
      throw new Error(
        `Google OAuth token exchange failed: HTTP ${res.status}: ${truncate(res.body, 300)}`,
      );
    }
    cachedToken = {
      email: sa.client_email,
      value: body.access_token,
      exp: now + (Number(body.expires_in) || 3600),
    };
    return cachedToken.value;
  }

  async function api(
    method: "GET" | "POST" | "PUT" | "DELETE",
    url: string,
    payload?: unknown,
  ): Promise<{ ok: boolean; status: number; body: any }> {
    const token = await getToken();
    const res = await http(url, {
      method,
      headers: { Authorization: `Bearer ${token}` },
      ...(payload !== undefined ? { body: payload } : {}),
      timeout: 60_000,
    });
    return { ok: res.ok, status: res.status, body: res.body as any };
  }

  return { api, driveFolderId, clientEmail: sa.client_email };
}
// ── end preamble ───────────────────────────────────────────────────────────

const spreadsheetUrl = (id: string) => `https://docs.google.com/spreadsheets/d/${id}/edit`;

export default defineStep({
  type: "sheets/create-spreadsheet",
  description:
    "Create a new Google Spreadsheet and return its spreadsheet_id and url. " +
    "When a shared Drive folder is configured (GOOGLE_DRIVE_FOLDER_ID), it is created inside " +
    "that folder, so the user can open it immediately. " +
    "Start here for any calculation task, then write inputs and formulas with sheets_update_values. " +
    "Every spreadsheet starts with one tab named 'Sheet1'; pass extra_sheet_titles to add more tabs up front.",
  input: z.object({
    title: z.string().describe("Spreadsheet title shown in Drive."),
    extra_sheet_titles: z
      .array(z.string())
      .optional()
      .describe("Additional tabs to create beyond the default 'Sheet1'."),
    serviceAccount: z
      .any()
      .optional()
      .describe(
        "Service-account credentials override (JSON object, JSON string, or base64 JSON). " +
          "Normally omitted — resolved from the GOOGLE_SERVICE_ACCOUNT_JSON secret.",
      ),
    driveFolderId: z
      .string()
      .optional()
      .describe(
        "Drive folder to create the spreadsheet in (must be shared with the service account's " +
          "client_email). Normally omitted — resolved from the GOOGLE_DRIVE_FOLDER_ID secret.",
      ),
  }),
  output: z.any(),
  async run(cfg, ctx) {
    const { api, driveFolderId, clientEmail } = await sheetsCtx(
      ctx as StepContext<VeinCapabilities>,
      cfg,
    );
    // A 403/404 here almost always means the folder isn't shared with the SA.
    const shareHint = driveFolderId
      ? ` (if this is a permission error, share the Drive folder ${driveFolderId} with the service account's client_email ${clientEmail})`
      : "";
    try {
      let spreadsheetId: string;
      if (driveFolderId) {
        // Creating through Drive places the file directly in the shared
        // folder (a Sheets-API create would land in the service account's
        // own root Drive, invisible to the user).
        const resp = await api("POST", `${DRIVE_API}?supportsAllDrives=true`, {
          name: cfg.title,
          mimeType: "application/vnd.google-apps.spreadsheet",
          parents: [driveFolderId],
        });
        if (!resp.ok) {
          return errorResult("sheets_create_spreadsheet", resp.status, resp.body) + shareHint;
        }
        spreadsheetId = resp.body.id;
      } else {
        const resp = await api("POST", SHEETS_API, { properties: { title: cfg.title } });
        if (!resp.ok) return errorResult("sheets_create_spreadsheet", resp.status, resp.body);
        spreadsheetId = resp.body.spreadsheetId;
      }

      if (cfg.extra_sheet_titles && cfg.extra_sheet_titles.length > 0) {
        const resp = await api(
          "POST",
          `${SHEETS_API}/${encodeURIComponent(spreadsheetId)}:batchUpdate`,
          {
            requests: cfg.extra_sheet_titles.map((t: string) => ({
              addSheet: { properties: { title: t } },
            })),
          },
        );
        if (!resp.ok) {
          return {
            spreadsheet_id: spreadsheetId,
            url: spreadsheetUrl(spreadsheetId),
            warning: errorResult("adding extra sheets", resp.status, resp.body),
          };
        }
      }

      return {
        spreadsheet_id: spreadsheetId,
        url: spreadsheetUrl(spreadsheetId),
        sheets: ["Sheet1", ...(cfg.extra_sheet_titles ?? [])],
      };
    } catch (err: any) {
      return `sheets_create_spreadsheet failed: ${err?.message ?? String(err)}`;
    }
  },
});
