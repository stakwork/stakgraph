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
      "GOOGLE_SERVICE_ACCOUNT_JSON not configured — paste the service-account JSON into the vein secrets UI or mcp env",
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

export default defineStep({
  type: "sheets/add-sheet",
  description:
    "Add a new tab to an existing spreadsheet (e.g. a 'Scenarios' tab next to 'Model'). " +
    "Reference its cells from other tabs as 'TabName!A1'.",
  input: z.object({
    spreadsheet_id: z.string().describe("Spreadsheet id."),
    title: z.string().describe("Title of the new tab."),
    serviceAccount: z
      .any()
      .optional()
      .describe(
        "Service-account credentials override (JSON object, JSON string, or base64 JSON). " +
          "Normally omitted — resolved from the GOOGLE_SERVICE_ACCOUNT_JSON secret.",
      ),
  }),
  output: z.any(),
  async run(cfg, ctx) {
    const { api } = await sheetsCtx(ctx as StepContext<VeinCapabilities>, cfg);
    try {
      const resp = await api(
        "POST",
        `${SHEETS_API}/${encodeURIComponent(cfg.spreadsheet_id)}:batchUpdate`,
        { requests: [{ addSheet: { properties: { title: cfg.title } } }] },
      );
      if (!resp.ok) return errorResult("sheets_add_sheet", resp.status, resp.body);
      const props = resp.body.replies?.[0]?.addSheet?.properties;
      return { sheet_id: props?.sheetId, title: props?.title ?? cfg.title };
    } catch (err: any) {
      return `sheets_add_sheet failed: ${err?.message ?? String(err)}`;
    }
  },
});
