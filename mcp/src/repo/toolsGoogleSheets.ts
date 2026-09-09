import { tool, Tool } from "ai";
import { z } from "zod";
import axios from "axios";
import jwt from "jsonwebtoken";

/**
 * Google Sheets tools — create spreadsheets and read/write cell values and
 * formulas via the Google Sheets + Drive REST APIs, authenticated with a
 * caller-supplied service account.
 *
 * Registered only when the caller supplies service-account credentials on the
 * request body (`googleSheets.serviceAccount`) — like `stakworkApiKey`, the
 * credentials are plumbed server-to-server and are never an LLM-visible
 * parameter. Spreadsheets are created inside `googleSheets.driveFolderId`
 * when set; sharing that folder with the service account's client_email is
 * what makes agent-created sheets visible to humans.
 *
 * Auth is a plain service-account JWT flow (RS256 assertion signed with the
 * key from the credentials JSON, exchanged at token_uri) — no googleapis
 * dependency. The access token is cached per registration (one agent run).
 */

const SHEETS_API = "https://sheets.googleapis.com/v4/spreadsheets";
const DRIVE_API = "https://www.googleapis.com/drive/v3/files";
const OAUTH_SCOPES =
  "https://www.googleapis.com/auth/spreadsheets https://www.googleapis.com/auth/drive";
const DEFAULT_TOKEN_URI = "https://oauth2.googleapis.com/token";
/** Refresh the cached access token this many seconds before it expires. */
const TOKEN_EXPIRY_SLACK_S = 60;

export type GoogleSheetsToolName =
  | "sheets_create_spreadsheet"
  | "sheets_update_values"
  | "sheets_batch_update_values"
  | "sheets_get_values"
  | "sheets_add_sheet"
  | "sheets_import_spreadsheet";

export const GOOGLE_SHEETS_TOOL_NAMES: GoogleSheetsToolName[] = [
  "sheets_create_spreadsheet",
  "sheets_update_values",
  "sheets_batch_update_values",
  "sheets_get_values",
  "sheets_add_sheet",
  "sheets_import_spreadsheet",
];

export interface GoogleSheetsToolsOptions {
  /** Service-account credentials: parsed JSON object, JSON string, or base64-encoded JSON. */
  serviceAccount: unknown;
  /** Drive folder to create spreadsheets in (share it with the service account's client_email). */
  driveFolderId?: string;
}

/** Per-tool tweaks resolved by the caller from toolsConfig. */
export type GoogleSheetsToolOverrides = Partial<
  Record<GoogleSheetsToolName, { disabled?: boolean; description?: string }>
>;

interface ServiceAccount {
  client_email: string;
  private_key: string;
  token_uri?: string;
}

/** Accept a service account as an object, a JSON string, or base64-encoded JSON. */
export function parseServiceAccount(input: unknown): ServiceAccount {
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
      "googleSheets.serviceAccount must be service-account JSON with client_email and private_key"
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

async function apiRequest(
  method: "get" | "post" | "put" | "delete",
  url: string,
  token: string,
  payload?: unknown
): Promise<{ ok: boolean; status: number; body: any }> {
  const resp = await axios.request({
    method,
    url,
    data: payload,
    headers: {
      Authorization: `Bearer ${token}`,
      ...(payload !== undefined ? { "Content-Type": "application/json" } : {}),
    },
    validateStatus: () => true,
    timeout: 60_000,
  });
  return { ok: resp.status >= 200 && resp.status < 300, status: resp.status, body: resp.data };
}

const spreadsheetUrl = (id: string) => `https://docs.google.com/spreadsheets/d/${id}/edit`;

/** 2D array of cell values; strings starting with '=' become live formulas under USER_ENTERED. */
const valuesSchema = z
  .array(z.array(z.union([z.string(), z.number(), z.boolean(), z.null()])))
  .describe(
    "2D array of rows. Strings starting with '=' are entered as live formulas (e.g. '=SUM(B2:B10)')."
  );

export function registerGoogleSheetsTools(
  allTools: Record<string, Tool<any, any>>,
  options: GoogleSheetsToolsOptions,
  overrides?: GoogleSheetsToolOverrides
): void {
  let sa: ServiceAccount;
  try {
    sa = parseServiceAccount(options.serviceAccount);
  } catch (err: any) {
    console.error(`[google-sheets] invalid credentials, skipping tools: ${err?.message ?? err}`);
    return;
  }
  const driveFolderId = options.driveFolderId;

  // One cached access token per registration (i.e. per agent run) — never
  // shared across requests, so different callers' credentials can't mix.
  let cachedToken: { value: string; exp: number } | null = null;
  async function getToken(): Promise<string> {
    const now = Math.floor(Date.now() / 1000);
    if (cachedToken && cachedToken.exp - TOKEN_EXPIRY_SLACK_S > now) return cachedToken.value;
    const tokenUri = sa.token_uri || DEFAULT_TOKEN_URI;
    const assertion = jwt.sign(
      { iss: sa.client_email, scope: OAUTH_SCOPES, aud: tokenUri, iat: now, exp: now + 3600 },
      sa.private_key,
      { algorithm: "RS256" }
    );
    const resp = await axios.post(
      tokenUri,
      new URLSearchParams({
        grant_type: "urn:ietf:params:oauth:grant-type:jwt-bearer",
        assertion,
      }).toString(),
      {
        headers: { "Content-Type": "application/x-www-form-urlencoded" },
        validateStatus: () => true,
        timeout: 30_000,
      }
    );
    if (resp.status < 200 || resp.status >= 300 || !resp.data?.access_token) {
      throw new Error(
        `Google OAuth token exchange failed: HTTP ${resp.status}: ${truncate(resp.data, 300)}`
      );
    }
    cachedToken = {
      value: resp.data.access_token,
      exp: now + (Number(resp.data.expires_in) || 3600),
    };
    return cachedToken.value;
  }

  const register = (
    name: GoogleSheetsToolName,
    defaultDescription: string,
    build: (description: string) => Tool<any, any>
  ) => {
    const override = overrides?.[name];
    if (override?.disabled) return;
    allTools[name] = build(override?.description || defaultDescription);
  };

  register(
    "sheets_create_spreadsheet",
    "Create a new Google Spreadsheet and return its spreadsheet_id and url. " +
      (driveFolderId
        ? "It is created inside the configured shared Drive folder, so the user can open it immediately. "
        : "") +
      "Start here for any calculation task, then write inputs and formulas with sheets_update_values. " +
      "Every spreadsheet starts with one tab named 'Sheet1'; pass extra_sheet_titles to add more tabs up front.",
    (description) =>
      tool({
        description,
        inputSchema: z.object({
          title: z.string().describe("Spreadsheet title shown in Drive."),
          extra_sheet_titles: z
            .array(z.string())
            .optional()
            .describe("Additional tabs to create beyond the default 'Sheet1'."),
        }),
        execute: async ({
          title,
          extra_sheet_titles,
        }: {
          title: string;
          extra_sheet_titles?: string[];
        }) => {
          console.log(`[sheets_create_spreadsheet] title=${title} folder=${driveFolderId ?? "-"}`);
          try {
            const token = await getToken();
            let spreadsheetId: string;
            if (driveFolderId) {
              // Creating through Drive places the file directly in the shared
              // folder (a Sheets-API create would land in the service
              // account's own root Drive, invisible to the user).
              const resp = await apiRequest(
                "post",
                `${DRIVE_API}?supportsAllDrives=true`,
                token,
                {
                  name: title,
                  mimeType: "application/vnd.google-apps.spreadsheet",
                  parents: [driveFolderId],
                }
              );
              if (!resp.ok) return errorResult("sheets_create_spreadsheet", resp.status, resp.body);
              spreadsheetId = resp.body.id;
            } else {
              const resp = await apiRequest("post", SHEETS_API, token, {
                properties: { title },
              });
              if (!resp.ok) return errorResult("sheets_create_spreadsheet", resp.status, resp.body);
              spreadsheetId = resp.body.spreadsheetId;
            }

            if (extra_sheet_titles && extra_sheet_titles.length > 0) {
              const resp = await apiRequest(
                "post",
                `${SHEETS_API}/${encodeURIComponent(spreadsheetId)}:batchUpdate`,
                token,
                {
                  requests: extra_sheet_titles.map((t) => ({
                    addSheet: { properties: { title: t } },
                  })),
                }
              );
              if (!resp.ok) {
                return JSON.stringify({
                  spreadsheet_id: spreadsheetId,
                  url: spreadsheetUrl(spreadsheetId),
                  warning: errorResult("adding extra sheets", resp.status, resp.body),
                });
              }
            }

            return JSON.stringify({
              spreadsheet_id: spreadsheetId,
              url: spreadsheetUrl(spreadsheetId),
              sheets: ["Sheet1", ...(extra_sheet_titles ?? [])],
            });
          } catch (err: any) {
            return `sheets_create_spreadsheet failed: ${err?.message ?? String(err)}`;
          }
        },
      })
  );

  register(
    "sheets_update_values",
    "Write a 2D array of values into a spreadsheet range (A1 notation, e.g. 'Sheet1!A1:C10'). " +
      "Strings starting with '=' are entered as LIVE FORMULAS — prefer building calculations with " +
      "formulas over precomputing numbers yourself, so the sheet stays auditable and recalculates " +
      "when inputs change. Read computed results back with sheets_get_values.",
    (description) =>
      tool({
        description,
        inputSchema: z.object({
          spreadsheet_id: z.string().describe("Spreadsheet id from sheets_create_spreadsheet."),
          range: z.string().describe("A1-notation target range, e.g. 'Sheet1!A1' or 'Model!B2:D20'."),
          values: valuesSchema,
          raw: z
            .boolean()
            .optional()
            .describe("Store strings literally instead of parsing formulas/numbers/dates (default false)."),
        }),
        execute: async ({
          spreadsheet_id,
          range,
          values,
          raw,
        }: {
          spreadsheet_id: string;
          range: string;
          values: (string | number | boolean | null)[][];
          raw?: boolean;
        }) => {
          console.log(`[sheets_update_values] id=${spreadsheet_id} range=${range} rows=${values.length}`);
          try {
            const token = await getToken();
            const input = raw ? "RAW" : "USER_ENTERED";
            const resp = await apiRequest(
              "put",
              `${SHEETS_API}/${encodeURIComponent(spreadsheet_id)}/values/${encodeURIComponent(range)}?valueInputOption=${input}`,
              token,
              { values }
            );
            if (!resp.ok) return errorResult("sheets_update_values", resp.status, resp.body);
            const { updatedRange, updatedCells } = resp.body;
            return JSON.stringify({ updated_range: updatedRange, updated_cells: updatedCells });
          } catch (err: any) {
            return `sheets_update_values failed: ${err?.message ?? String(err)}`;
          }
        },
      })
  );

  register(
    "sheets_batch_update_values",
    "Write multiple ranges of a spreadsheet in one call — use this instead of repeated " +
      "sheets_update_values when laying out a model (e.g. headers, inputs, and a formula block at once). " +
      "Same formula semantics as sheets_update_values.",
    (description) =>
      tool({
        description,
        inputSchema: z.object({
          spreadsheet_id: z.string().describe("Spreadsheet id from sheets_create_spreadsheet."),
          data: z
            .array(z.object({ range: z.string(), values: valuesSchema }))
            .describe("One entry per target range."),
          raw: z
            .boolean()
            .optional()
            .describe("Store strings literally instead of parsing formulas/numbers/dates (default false)."),
        }),
        execute: async ({
          spreadsheet_id,
          data,
          raw,
        }: {
          spreadsheet_id: string;
          data: { range: string; values: (string | number | boolean | null)[][] }[];
          raw?: boolean;
        }) => {
          console.log(`[sheets_batch_update_values] id=${spreadsheet_id} ranges=${data.length}`);
          try {
            const token = await getToken();
            const resp = await apiRequest(
              "post",
              `${SHEETS_API}/${encodeURIComponent(spreadsheet_id)}/values:batchUpdate`,
              token,
              { valueInputOption: raw ? "RAW" : "USER_ENTERED", data }
            );
            if (!resp.ok) return errorResult("sheets_batch_update_values", resp.status, resp.body);
            return JSON.stringify({
              total_updated_cells: resp.body.totalUpdatedCells,
              updated_ranges: (resp.body.responses ?? []).map((r: any) => r.updatedRange),
            });
          } catch (err: any) {
            return `sheets_batch_update_values failed: ${err?.message ?? String(err)}`;
          }
        },
      })
  );

  register(
    "sheets_get_values",
    "Read a range of a spreadsheet (A1 notation). render='computed' (default) returns calculated " +
      "numbers — use it to read the results of formulas you wrote; 'formatted' returns display strings " +
      "(currency symbols, rounding); 'formula' returns the formula text itself for auditing.",
    (description) =>
      tool({
        description,
        inputSchema: z.object({
          spreadsheet_id: z.string().describe("Spreadsheet id."),
          range: z.string().describe("A1-notation range to read, e.g. 'Sheet1!A1:D20'."),
          render: z
            .enum(["computed", "formatted", "formula"])
            .optional()
            .describe("How to render cell values (default 'computed')."),
        }),
        execute: async ({
          spreadsheet_id,
          range,
          render,
        }: {
          spreadsheet_id: string;
          range: string;
          render?: "computed" | "formatted" | "formula";
        }) => {
          console.log(`[sheets_get_values] id=${spreadsheet_id} range=${range} render=${render ?? "computed"}`);
          try {
            const token = await getToken();
            const renderOption =
              render === "formatted"
                ? "FORMATTED_VALUE"
                : render === "formula"
                  ? "FORMULA"
                  : "UNFORMATTED_VALUE";
            const resp = await apiRequest(
              "get",
              `${SHEETS_API}/${encodeURIComponent(spreadsheet_id)}/values/${encodeURIComponent(range)}?valueRenderOption=${renderOption}`,
              token
            );
            if (!resp.ok) return errorResult("sheets_get_values", resp.status, resp.body);
            return JSON.stringify({ range: resp.body.range, values: resp.body.values ?? [] });
          } catch (err: any) {
            return `sheets_get_values failed: ${err?.message ?? String(err)}`;
          }
        },
      })
  );

  register(
    "sheets_add_sheet",
    "Add a new tab to an existing spreadsheet (e.g. a 'Scenarios' tab next to 'Model'). " +
      "Reference its cells from other tabs as 'TabName!A1'.",
    (description) =>
      tool({
        description,
        inputSchema: z.object({
          spreadsheet_id: z.string().describe("Spreadsheet id."),
          title: z.string().describe("Title of the new tab."),
        }),
        execute: async ({ spreadsheet_id, title }: { spreadsheet_id: string; title: string }) => {
          console.log(`[sheets_add_sheet] id=${spreadsheet_id} title=${title}`);
          try {
            const token = await getToken();
            const resp = await apiRequest(
              "post",
              `${SHEETS_API}/${encodeURIComponent(spreadsheet_id)}:batchUpdate`,
              token,
              { requests: [{ addSheet: { properties: { title } } }] }
            );
            if (!resp.ok) return errorResult("sheets_add_sheet", resp.status, resp.body);
            const props = resp.body.replies?.[0]?.addSheet?.properties;
            return JSON.stringify({ sheet_id: props?.sheetId, title: props?.title ?? title });
          } catch (err: any) {
            return `sheets_add_sheet failed: ${err?.message ?? String(err)}`;
          }
        },
      })
  );

  register(
    "sheets_import_spreadsheet",
    "Import every sheet of a source spreadsheet (uploaded .xlsx or existing Google Sheet) into a " +
      "destination spreadsheet as correctly-named tabs. Source may be any Drive file id; non-native " +
      "files (e.g. .xlsx) are converted to Google Sheets format automatically. Each imported tab is " +
      "named 'SOURCE: <label>' (single-sheet source) or 'SOURCE: <label> — <sheet title>' (multi-sheet). " +
      "Collisions with existing tab names are resolved by auto-appending ' (2)', ' (3)', etc. " +
      "Import is best-effort: one sheet failing does not abort the rest. Returns per-sheet status " +
      "(success / failed / copied_unrenamed) and any formula/range warnings.",
    (description) =>
      tool({
        description,
        inputSchema: z.object({
          destination_spreadsheet_id: z
            .string()
            .describe("Spreadsheet id of the destination to copy sheets into."),
          source_file_id: z
            .string()
            .describe(
              "Drive file id of the source — may be an uploaded .xlsx or an existing native Google Sheet."
            ),
          label: z
            .string()
            .optional()
            .describe(
              "Override the label used in tab names. Defaults to the Drive filename (converted) or spreadsheet title (native)."
            ),
          keep_converted_copy: z
            .boolean()
            .optional()
            .describe(
              "When true, the intermediate converted-copy file (for non-native sources) is kept in Drive after import. Default false."
            ),
        }),
        execute: async ({
          destination_spreadsheet_id,
          source_file_id,
          label,
          keep_converted_copy,
        }: {
          destination_spreadsheet_id: string;
          source_file_id: string;
          label?: string;
          keep_converted_copy?: boolean;
        }) => {
          console.log(
            `[sheets_import_spreadsheet] source=${source_file_id} destination=${destination_spreadsheet_id}`
          );
          try {
            const token = await getToken();

            // ── Step 1: Fetch Drive metadata for the source file ──────────────
            const metaResp = await apiRequest(
              "get",
              `${DRIVE_API}/${encodeURIComponent(source_file_id)}?supportsAllDrives=true&fields=name,mimeType,parents`,
              token
            );
            if (!metaResp.ok) {
              return `sheets_import_spreadsheet failed: could not fetch source file metadata: HTTP ${metaResp.status}: ${truncate(metaResp.body?.error?.message ?? metaResp.body, 300)}`;
            }
            const sourceMeta: { name: string; mimeType: string; parents?: string[] } =
              metaResp.body;

            // ── Step 2: Convert if not already a native Google Sheet ──────────
            // tempSpreadsheetId is ONLY the id of the converted copy — never
            // aliased from source_file_id — so the later delete call is
            // unambiguous.
            let tempSpreadsheetId: string | undefined = undefined;
            let workingId: string;

            const SHEETS_MIME = "application/vnd.google-apps.spreadsheet";
            if (sourceMeta.mimeType !== SHEETS_MIME) {
              console.log(
                `[sheets_import_spreadsheet] converting mimeType=${sourceMeta.mimeType} to native Sheets`
              );
              const parentFolder = sourceMeta.parents?.[0];
              const copyBody: Record<string, unknown> = { mimeType: SHEETS_MIME };
              if (parentFolder) copyBody.parents = [parentFolder];
              const copyResp = await apiRequest(
                "post",
                `${DRIVE_API}/${encodeURIComponent(source_file_id)}/copy?supportsAllDrives=true`,
                token,
                copyBody
              );
              if (!copyResp.ok) {
                return `sheets_import_spreadsheet failed: could not convert source file to Google Sheets: HTTP ${copyResp.status}: ${truncate(copyResp.body?.error?.message ?? copyResp.body, 300)}`;
              }
              // Store in a dedicated variable; NEVER reuse source_file_id.
              tempSpreadsheetId = copyResp.body.id as string;
              workingId = tempSpreadsheetId;
              console.log(
                `[sheets_import_spreadsheet] conversion complete, tempSpreadsheetId=${tempSpreadsheetId}`
              );
            } else {
              console.log(`[sheets_import_spreadsheet] source is already native Sheets, no conversion`);
              workingId = source_file_id;
            }

            // ── Step 3: Enumerate source sheets ─────────────────────────────
            const srcSheetsResp = await apiRequest(
              "get",
              `${SHEETS_API}/${encodeURIComponent(workingId)}?fields=properties.title,sheets.properties`,
              token
            );
            if (!srcSheetsResp.ok) {
              return `sheets_import_spreadsheet failed: could not read source spreadsheet: HTTP ${srcSheetsResp.status}: ${truncate(srcSheetsResp.body?.error?.message ?? srcSheetsResp.body, 300)}`;
            }
            const sourceSheets: Array<{ sheetId: number; title: string }> = (
              srcSheetsResp.body.sheets ?? []
            ).map((s: any) => ({
              sheetId: s.properties.sheetId as number,
              title: s.properties.title as string,
            }));
            const sourceSpreadsheetTitle: string =
              srcSheetsResp.body.properties?.title ?? sourceMeta.name;

            // ── Step 4: Seed the live title Set from the destination ──────────
            const dstSheetsResp = await apiRequest(
              "get",
              `${SHEETS_API}/${encodeURIComponent(destination_spreadsheet_id)}?fields=sheets.properties.title`,
              token
            );
            if (!dstSheetsResp.ok) {
              return `sheets_import_spreadsheet failed: could not read destination spreadsheet: HTTP ${dstSheetsResp.status}: ${truncate(dstSheetsResp.body?.error?.message ?? dstSheetsResp.body, 300)}`;
            }
            // Single source of truth for collision detection — updated after each rename.
            const existingTitles = new Set<string>(
              (dstSheetsResp.body.sheets ?? []).map((s: any) => s.properties.title as string)
            );

            // ── Step 5: Per-sheet best-effort copy loop ───────────────────────
            const resolvedLabel = resolveLabel({
              explicitLabel: label,
              driveFileName: sourceMeta.name,
              sourceSpreadsheetTitle,
              isNative: sourceMeta.mimeType === SHEETS_MIME,
            });

            const imported: Array<{
              source_sheet: string;
              tab_name: string | null;
              sheet_id: number | null;
              status: "success" | "failed" | "copied_unrenamed";
              error?: string;
            }> = [];

            for (const sheet of sourceSheets) {
              const targetTabName = resolveCollisionSuffix(
                buildTabName(resolvedLabel, sheet.title, sourceSheets.length),
                existingTitles
              );

              // copyTo
              let copiedSheetId: number;
              try {
                const copyToResp = await apiRequest(
                  "post",
                  `${SHEETS_API}/${encodeURIComponent(workingId)}/sheets/${sheet.sheetId}:copyTo`,
                  token,
                  { destinationSpreadsheetId: destination_spreadsheet_id }
                );
                if (!copyToResp.ok) {
                  const errMsg = `copyTo failed: HTTP ${copyToResp.status}: ${truncate(copyToResp.body?.error?.message ?? copyToResp.body, 200)}`;
                  console.log(`[sheets_import_spreadsheet] sheet="${sheet.title}" ${errMsg}`);
                  imported.push({
                    source_sheet: sheet.title,
                    tab_name: null,
                    sheet_id: null,
                    status: "failed",
                    error: errMsg,
                  });
                  continue;
                }
                copiedSheetId = copyToResp.body.sheetId as number;
              } catch (copyErr: any) {
                const errMsg = `copyTo threw: ${copyErr?.message ?? String(copyErr)}`;
                console.log(`[sheets_import_spreadsheet] sheet="${sheet.title}" ${errMsg}`);
                imported.push({
                  source_sheet: sheet.title,
                  tab_name: null,
                  sheet_id: null,
                  status: "failed",
                  error: errMsg,
                });
                continue;
              }

              // rename via batchUpdate
              try {
                const renameResp = await apiRequest(
                  "post",
                  `${SHEETS_API}/${encodeURIComponent(destination_spreadsheet_id)}:batchUpdate`,
                  token,
                  {
                    requests: [
                      {
                        updateSheetProperties: {
                          properties: { sheetId: copiedSheetId, title: targetTabName },
                          fields: "title",
                        },
                      },
                    ],
                  }
                );
                if (!renameResp.ok) {
                  // Google assigns a default name like "Copy of <original>"
                  const googleDefaultName = `Copy of ${sheet.title}`;
                  const errMsg = `rename failed: HTTP ${renameResp.status}: ${truncate(renameResp.body?.error?.message ?? renameResp.body, 200)}`;
                  console.log(`[sheets_import_spreadsheet] sheet="${sheet.title}" copied_unrenamed: ${errMsg}`);
                  imported.push({
                    source_sheet: sheet.title,
                    tab_name: googleDefaultName,
                    sheet_id: copiedSheetId,
                    status: "copied_unrenamed",
                    error: errMsg,
                  });
                  // Do NOT add to existingTitles — we don't know the actual Google-assigned name precisely.
                } else {
                  console.log(
                    `[sheets_import_spreadsheet] sheet="${sheet.title}" → tab="${targetTabName}" success`
                  );
                  imported.push({
                    source_sheet: sheet.title,
                    tab_name: targetTabName,
                    sheet_id: copiedSheetId,
                    status: "success",
                  });
                  // Update the live title Set immediately so subsequent sheets
                  // in this same run see this name as taken.
                  existingTitles.add(targetTabName);
                }
              } catch (renameErr: any) {
                const googleDefaultName = `Copy of ${sheet.title}`;
                const errMsg = `rename threw: ${renameErr?.message ?? String(renameErr)}`;
                console.log(`[sheets_import_spreadsheet] sheet="${sheet.title}" copied_unrenamed: ${errMsg}`);
                imported.push({
                  source_sheet: sheet.title,
                  tab_name: googleDefaultName,
                  sheet_id: copiedSheetId,
                  status: "copied_unrenamed",
                  error: errMsg,
                });
              }
            }

            // ── Step 6: Cross-sheet formula warnings ─────────────────────────
            const warnings: string[] = [];
            const sourceTitles = sourceSheets.map((s) => s.title);
            const successfulSheets = imported.filter((r) => r.status === "success");

            for (const result of successfulSheets) {
              // Fetch formula cells for this destination sheet
              const formulaResp = await apiRequest(
                "get",
                `${SHEETS_API}/${encodeURIComponent(destination_spreadsheet_id)}/values/${encodeURIComponent(`'${result.tab_name}'`)}?valueRenderOption=FORMULA`,
                token
              );
              if (formulaResp.ok) {
                const cellValues: string[] = (formulaResp.body.values ?? [])
                  .flat()
                  .filter((v: unknown) => typeof v === "string" && v.startsWith("="));
                // Check formulas against other source sheet titles (not its own)
                const otherTitles = sourceTitles.filter((t) => t !== result.source_sheet);
                const referenced = detectCrossSheetFormulaRefs(cellValues, otherTitles);
                for (const ref of referenced) {
                  warnings.push(
                    `Tab "${result.tab_name}" contains formula references to source sheet "${ref}", which has been renamed in the destination — cross-sheet references may not resolve correctly.`
                  );
                }
              }
            }
            // Always warn about named/protected range limitation.
            warnings.push(
              "Named ranges and protected ranges defined in the source workbook are not carried over by the Sheets API copyTo operation and are not recreated by this tool."
            );

            // ── Step 7: Cleanup temp converted file ─────────────────────────
            // converted_copy_deleted is null when no conversion occurred.
            let converted_copy_deleted: boolean | null = null;
            if (tempSpreadsheetId !== undefined) {
              // Safety: tempSpreadsheetId must NEVER equal source_file_id or destination_spreadsheet_id.
              // These are distinct variables by construction (see Step 2).
              if (!keep_converted_copy) {
                try {
                  const deleteResp = await apiRequest(
                    "delete",
                    `${DRIVE_API}/${encodeURIComponent(tempSpreadsheetId)}?supportsAllDrives=true`,
                    token
                  );
                  converted_copy_deleted = deleteResp.ok;
                  console.log(
                    `[sheets_import_spreadsheet] cleanup tempSpreadsheetId=${tempSpreadsheetId} deleted=${converted_copy_deleted}`
                  );
                } catch (delErr: any) {
                  converted_copy_deleted = false;
                  console.log(
                    `[sheets_import_spreadsheet] cleanup failed: ${delErr?.message ?? String(delErr)}`
                  );
                }
              } else {
                converted_copy_deleted = false;
                console.log(
                  `[sheets_import_spreadsheet] keep_converted_copy=true, skipping cleanup of tempSpreadsheetId=${tempSpreadsheetId}`
                );
              }
            } else {
              console.log(`[sheets_import_spreadsheet] no conversion temp file to clean up`);
            }

            return JSON.stringify({
              destination_spreadsheet_id,
              imported,
              converted_copy_deleted,
              warnings,
            });
          } catch (err: any) {
            return `sheets_import_spreadsheet failed: ${err?.message ?? String(err)}`;
          }
        },
      })
  );

  const registered = GOOGLE_SHEETS_TOOL_NAMES.filter((n) => n in allTools);
  console.log(`===> registered google sheets tools: ${registered.join(", ")}`);
}

// ── Pure, exported helper functions (no network calls) ────────────────────────
// These are extracted so they can be unit-tested without live credentials.

/**
 * Resolve the label used in tab names.
 * - An explicit `label` param always wins.
 * - For converted (non-native) sources, fall back to the Drive filename.
 * - For native Google Sheet sources, fall back to the spreadsheet's own title.
 */
export function resolveLabel({
  explicitLabel,
  driveFileName,
  sourceSpreadsheetTitle,
  isNative,
}: {
  explicitLabel?: string;
  driveFileName: string;
  sourceSpreadsheetTitle: string;
  isNative: boolean;
}): string {
  if (explicitLabel) return explicitLabel;
  return isNative ? sourceSpreadsheetTitle : driveFileName;
}

/**
 * Build the raw (pre-collision-check) tab name from label + sheet title.
 * - Single-sheet source → `SOURCE: <label>`
 * - Multi-sheet source  → `SOURCE: <label> — <sheet title>`
 */
export function buildTabName(
  label: string,
  sheetTitle: string,
  totalSheets: number
): string {
  if (totalSheets === 1) return `SOURCE: ${label}`;
  return `SOURCE: ${label} \u2014 ${sheetTitle}`;
}

/**
 * Given a candidate tab name and a Set of already-taken names, return a
 * non-colliding name. If the candidate is already taken, appends " (2)",
 * " (3)", etc., incrementing past any existing suffixes (non-contiguous gaps
 * are handled — e.g. if " (2)" is taken but " (3)" is free, returns " (3)").
 */
export function resolveCollisionSuffix(
  candidate: string,
  existingNames: Set<string>
): string {
  if (!existingNames.has(candidate)) return candidate;
  let n = 2;
  while (existingNames.has(`${candidate} (${n})`)) n++;
  return `${candidate} (${n})`;
}

/**
 * Given a list of formula strings and a list of other sheet titles, returns
 * the subset of titles that are literally referenced in at least one formula.
 * Detection is heuristic: looks for the sheet title followed by `!` (standard
 * A1 cross-sheet reference syntax), with optional single-quote wrapping for
 * titles that contain spaces or special characters.
 */
export function detectCrossSheetFormulaRefs(
  formulas: string[],
  sheetTitles: string[]
): string[] {
  const referenced: string[] = [];
  for (const title of sheetTitles) {
    const escapedTitle = title.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    // Match: 'Title'! or Title! in any formula
    const pattern = new RegExp(
      `(?:'${escapedTitle}'|${escapedTitle})!`,
      "i"
    );
    if (formulas.some((f) => pattern.test(f))) {
      referenced.push(title);
    }
  }
  return referenced;
}
