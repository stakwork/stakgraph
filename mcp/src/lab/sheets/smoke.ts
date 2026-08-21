/**
 * Offline smoke test for the sheets/* steps — no vein server, no live Google.
 * Verifies:
 *   1. seeding: seedSheetsSteps publishes into a temp workspace and
 *      buildRegistry discovers every step from disk;
 *   2. auth: the RS256 service-account JWT is built correctly (signature
 *      verifies against the keypair), exchanged once, then cached;
 *   3. step logic: each step runs against a FAKE `ctx.services.http` that
 *      replays canned Google API responses (correct Authorization headers,
 *      create→update→get round trip shapes, the loud missing-credentials
 *      error, an HTTP-error → teaching string case).
 *
 * Run: npx tsx src/lab/sheets/smoke.ts
 */
import assert from "node:assert/strict";
import { mkdtempSync, rmSync } from "node:fs";
import { join } from "node:path";
import { generateKeyPairSync, createVerify } from "node:crypto";
import { WorkspaceManager, buildRegistry, type StepContext, type HttpResponse } from "vein";
import { seedSheetsSteps } from "./seed.js";

const TOKEN_URI = "https://oauth2.googleapis.com/token";
const SHEETS_API = "https://sheets.googleapis.com/v4/spreadsheets";
const DRIVE_API = "https://www.googleapis.com/drive/v3/files";

// ── service-account keypair (real RSA so the JWT signature is verifiable) ──
const { publicKey, privateKey } = generateKeyPairSync("rsa", { modulusLength: 2048 });
const privPem = privateKey.export({ type: "pkcs8", format: "pem" }) as string;
const SA = { client_email: "sa@test.iam.gserviceaccount.com", private_key: privPem };
const SA_JSON = JSON.stringify(SA);

// ── fake ctx: canned Google APIs over ctx.services.http ────────────────────
type Call = { url: string; opts: any };
const calls: Call[] = [];

type Route = { match: (url: string, opts: any) => boolean; body: unknown; status?: number };

/** Every ctx serves the token endpoint plus the given routes. */
const tokenRoute: Route = {
  match: (u, o) => u === TOKEN_URI && o.method === "POST",
  body: { access_token: "tok-abc", expires_in: 3600, token_type: "Bearer" },
};

function fakeHttp(routes: Route[]) {
  return async (url: string, opts: any = {}): Promise<HttpResponse> => {
    calls.push({ url, opts });
    for (const r of [tokenRoute, ...routes]) {
      if (r.match(url, opts)) {
        return { status: r.status ?? 200, ok: (r.status ?? 200) < 300, headers: {}, body: r.body };
      }
    }
    return { status: 404, ok: false, headers: {}, body: `no fake route for ${opts.method ?? "GET"} ${url}` };
  };
}

function makeCtx(routes: Route[], secrets?: Record<string, string>): StepContext {
  const bag = secrets ?? { GOOGLE_SERVICE_ACCOUNT_JSON: SA_JSON };
  return {
    runId: "smoke",
    path: "smoke",
    scope: {},
    input: undefined,
    emit: async () => {},
    services: {
      http: fakeHttp(routes),
      secrets: { get: async (name: string) => bag[name] },
    },
  } as unknown as StepContext;
}

const tokenCalls = () => calls.filter((c) => c.url === TOKEN_URI);

async function main() {
  // ── 1. seed + discover ───────────────────────────────────────────────────
  // Under the mcp dir (not os tmpdir) so the seeded steps' dynamic
  // `import "vein"` resolves via mcp/node_modules — same as the real
  // lab-workspace location.
  const dir = mkdtempSync(join(process.cwd(), ".sheets-smoke-"));
  try {
    const workspace = new WorkspaceManager(dir);
    await seedSheetsSteps(workspace);
    const { registry } = await buildRegistry(workspace.path);
    const expected = [
      "sheets/create-spreadsheet", "sheets/update-values", "sheets/batch-update-values",
      "sheets/get-values", "sheets/add-sheet", "sheets/import-spreadsheet",
    ];
    for (const t of expected) assert.ok(registry[t], `registry missing ${t}`);
    console.log(`✔ seeded + discovered ${expected.length} sheets steps`);

    // ── 2. missing credentials is a loud error ─────────────────────────────
    await assert.rejects(
      () =>
        registry["sheets/get-values"].run(
          registry["sheets/get-values"].input.parse({ spreadsheet_id: "s1", range: "Sheet1!A1" }),
          makeCtx([], {}),
        ),
      /GOOGLE_SERVICE_ACCOUNT_JSON not configured/,
    );
    console.log("✔ loud error without GOOGLE_SERVICE_ACCOUNT_JSON");

    // ── 3. JWT exchange happens once, then the token is cached ─────────────
    const getRoutes: Route[] = [
      {
        match: (u) => u.includes("/values/") && u.includes("valueRenderOption=UNFORMATTED_VALUE"),
        body: { range: "Sheet1!A1:B2", values: [[1, 2], [3, "=SUM(A1:B1)"]] },
      },
    ];
    let out: any = await registry["sheets/get-values"].run(
      registry["sheets/get-values"].input.parse({ spreadsheet_id: "s1", range: "Sheet1!A1:B2" }),
      makeCtx(getRoutes),
    );
    assert.deepEqual(out, { range: "Sheet1!A1:B2", values: [[1, 2], [3, "=SUM(A1:B1)"]] });
    assert.equal(tokenCalls().length, 1, "expected exactly one token exchange");

    // the assertion is a valid RS256 JWT signed with the SA key
    const tokenCall = tokenCalls()[0];
    assert.equal(tokenCall.opts.headers["Content-Type"], "application/x-www-form-urlencoded");
    const assertion = new URLSearchParams(tokenCall.opts.body as string).get("assertion")!;
    const [h, c, s] = assertion.split(".");
    assert.deepEqual(JSON.parse(Buffer.from(h, "base64url").toString()), { alg: "RS256", typ: "JWT" });
    const claims = JSON.parse(Buffer.from(c, "base64url").toString());
    assert.equal(claims.iss, SA.client_email);
    assert.equal(claims.aud, TOKEN_URI);
    assert.match(claims.scope, /spreadsheets/);
    assert.match(claims.scope, /drive/);
    assert.equal(claims.exp - claims.iat, 3600);
    assert.ok(
      createVerify("RSA-SHA256").update(`${h}.${c}`).verify(publicKey, Buffer.from(s, "base64url")),
      "JWT signature must verify against the SA public key",
    );
    // the API call carried the exchanged bearer token
    const apiCall = calls[calls.length - 1];
    assert.equal(apiCall.opts.headers.Authorization, "Bearer tok-abc");
    console.log("✔ RS256 JWT exchange (header/claims/signature + bearer header)");

    // second run: cached token, no new exchange
    out = await registry["sheets/get-values"].run(
      registry["sheets/get-values"].input.parse({ spreadsheet_id: "s1", range: "Sheet1!A1:B2", render: "formula" }),
      makeCtx([
        { match: (u) => u.includes("valueRenderOption=FORMULA"), body: { range: "Sheet1!A1:B2", values: [["=A1"]] } },
      ]),
    );
    assert.deepEqual(out.values, [["=A1"]]);
    assert.equal(tokenCalls().length, 1, "token must be cached across calls");
    console.log("✔ token cached (no second exchange)");

    // base64 serviceAccount via explicit cfg wins over (absent) secrets
    out = await registry["sheets/get-values"].run(
      registry["sheets/get-values"].input.parse({
        spreadsheet_id: "s1",
        range: "Sheet1!A1",
        serviceAccount: Buffer.from(SA_JSON).toString("base64"),
      }),
      makeCtx(
        [{ match: (u) => u.includes("/values/"), body: { range: "Sheet1!A1", values: [] } }],
        {},
      ),
    );
    assert.deepEqual(out, { range: "Sheet1!A1", values: [] });
    console.log("✔ cfg.serviceAccount (base64) wins over secrets");

    // ── 4. create → update → get round trip shapes ─────────────────────────
    // create without a folder: Sheets API create + extra tabs
    out = await registry["sheets/create-spreadsheet"].run(
      registry["sheets/create-spreadsheet"].input.parse({ title: "Model", extra_sheet_titles: ["Scenarios"] }),
      makeCtx([
        { match: (u, o) => u === SHEETS_API && o.method === "POST", body: { spreadsheetId: "ss1" } },
        { match: (u, o) => u.includes("ss1:batchUpdate") && o.method === "POST", body: { replies: [{}] } },
      ]),
    );
    assert.deepEqual(out, {
      spreadsheet_id: "ss1",
      url: "https://docs.google.com/spreadsheets/d/ss1/edit",
      sheets: ["Sheet1", "Scenarios"],
    });
    let last = calls[calls.length - 1]; // the extra-tabs batchUpdate
    assert.equal(last.opts.body.requests[0].addSheet.properties.title, "Scenarios");
    console.log("✔ create-spreadsheet (Sheets API + extra tabs)");

    // create WITH a drive folder (cfg override): goes through Drive
    out = await registry["sheets/create-spreadsheet"].run(
      registry["sheets/create-spreadsheet"].input.parse({ title: "Model", driveFolderId: "folder1" }),
      makeCtx([
        { match: (u, o) => u.startsWith(DRIVE_API) && o.method === "POST", body: { id: "ss2" } },
      ]),
    );
    assert.equal(out.spreadsheet_id, "ss2");
    last = calls[calls.length - 1];
    assert.ok(last.url.includes("supportsAllDrives=true"));
    assert.deepEqual(last.opts.body.parents, ["folder1"]);
    assert.equal(last.opts.body.mimeType, "application/vnd.google-apps.spreadsheet");
    console.log("✔ create-spreadsheet (Drive folder path)");

    // GOOGLE_DRIVE_FOLDER_ID secret is picked up when cfg has no folder
    out = await registry["sheets/create-spreadsheet"].run(
      registry["sheets/create-spreadsheet"].input.parse({ title: "Model" }),
      makeCtx(
        [{ match: (u, o) => u.startsWith(DRIVE_API) && o.method === "POST", body: { id: "ss3" } }],
        { GOOGLE_SERVICE_ACCOUNT_JSON: SA_JSON, GOOGLE_DRIVE_FOLDER_ID: "folder-env" },
      ),
    );
    assert.equal(out.spreadsheet_id, "ss3");
    assert.deepEqual(calls[calls.length - 1].opts.body.parents, ["folder-env"]);
    console.log("✔ GOOGLE_DRIVE_FOLDER_ID secret fallback");

    // update-values (USER_ENTERED default, PUT)
    out = await registry["sheets/update-values"].run(
      registry["sheets/update-values"].input.parse({
        spreadsheet_id: "ss1",
        range: "Sheet1!A1:B2",
        values: [["Revenue", 100], ["Total", "=SUM(B1:B1)"]],
      }),
      makeCtx([
        {
          match: (u, o) => u.includes("valueInputOption=USER_ENTERED") && o.method === "PUT",
          body: { updatedRange: "Sheet1!A1:B2", updatedCells: 4 },
        },
      ]),
    );
    assert.deepEqual(out, { updated_range: "Sheet1!A1:B2", updated_cells: 4 });
    assert.deepEqual(calls[calls.length - 1].opts.body, {
      values: [["Revenue", 100], ["Total", "=SUM(B1:B1)"]],
    });
    console.log("✔ update-values");

    // raw:true → RAW
    out = await registry["sheets/update-values"].run(
      registry["sheets/update-values"].input.parse({
        spreadsheet_id: "ss1", range: "Sheet1!A1", values: [["=literal"]], raw: true,
      }),
      makeCtx([
        { match: (u, o) => u.includes("valueInputOption=RAW") && o.method === "PUT", body: { updatedRange: "Sheet1!A1", updatedCells: 1 } },
      ]),
    );
    assert.equal(out.updated_cells, 1);
    console.log("✔ update-values raw mode");

    // batch-update-values
    out = await registry["sheets/batch-update-values"].run(
      registry["sheets/batch-update-values"].input.parse({
        spreadsheet_id: "ss1",
        data: [
          { range: "Sheet1!A1", values: [["h1"]] },
          { range: "Sheet1!B1", values: [["=A1"]] },
        ],
      }),
      makeCtx([
        {
          match: (u, o) => u.includes("values:batchUpdate") && o.method === "POST",
          body: { totalUpdatedCells: 2, responses: [{ updatedRange: "Sheet1!A1" }, { updatedRange: "Sheet1!B1" }] },
        },
      ]),
    );
    assert.deepEqual(out, { total_updated_cells: 2, updated_ranges: ["Sheet1!A1", "Sheet1!B1"] });
    assert.equal(calls[calls.length - 1].opts.body.valueInputOption, "USER_ENTERED");
    console.log("✔ batch-update-values");

    // add-sheet
    out = await registry["sheets/add-sheet"].run(
      registry["sheets/add-sheet"].input.parse({ spreadsheet_id: "ss1", title: "Scenarios" }),
      makeCtx([
        {
          match: (u, o) => u.includes("ss1:batchUpdate") && o.method === "POST",
          body: { replies: [{ addSheet: { properties: { sheetId: 42, title: "Scenarios" } } }] },
        },
      ]),
    );
    assert.deepEqual(out, { sheet_id: 42, title: "Scenarios" });
    console.log("✔ add-sheet");

    // ── 5. HTTP error → teaching string ────────────────────────────────────
    out = await registry["sheets/create-spreadsheet"].run(
      registry["sheets/create-spreadsheet"].input.parse({ title: "Model", driveFolderId: "folderX" }),
      makeCtx([
        {
          match: (u, o) => u.startsWith(DRIVE_API) && o.method === "POST",
          status: 403,
          body: { error: { message: "The caller does not have permission" } },
        },
      ]),
    );
    assert.equal(typeof out, "string");
    assert.match(out, /HTTP 403/);
    assert.match(out, /share the Drive folder folderX/);
    assert.ok(out.includes(SA.client_email), "teaching error must name the SA client_email");
    console.log("✔ 403 create → teaching string (share the folder with client_email)");

    // ── 6. import-spreadsheet (xlsx conversion path, 2 sheets) ─────────────
    const dash = "—";
    out = await registry["sheets/import-spreadsheet"].run(
      registry["sheets/import-spreadsheet"].input.parse({
        destination_spreadsheet_id: "dst1",
        source_file_id: "src1",
      }),
      makeCtx([
        {
          match: (u, o) => u.startsWith(`${DRIVE_API}/src1?`) && (o.method ?? "GET") === "GET",
          body: { name: "budget.xlsx", mimeType: "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", parents: ["p1"] },
        },
        { match: (u, o) => u.includes("/src1/copy") && o.method === "POST", body: { id: "conv1" } },
        {
          match: (u) => u.includes("/conv1?fields=properties.title"),
          body: {
            properties: { title: "budget" },
            sheets: [
              { properties: { sheetId: 11, title: "Data" } },
              { properties: { sheetId: 12, title: "Summary" } },
            ],
          },
        },
        {
          match: (u) => u.includes("/dst1?fields=sheets.properties.title"),
          body: { sheets: [{ properties: { title: "Sheet1" } }] },
        },
        { match: (u, o) => u.includes("/sheets/11:copyTo") && o.method === "POST", body: { sheetId: 101 } },
        { match: (u, o) => u.includes("/sheets/12:copyTo") && o.method === "POST", body: { sheetId: 102 } },
        { match: (u, o) => u.includes("dst1:batchUpdate") && o.method === "POST", body: { replies: [{}] } },
        {
          match: (u) => u.includes("valueRenderOption=FORMULA") && decodeURIComponent(u).includes("Data"),
          body: { values: [["=Summary!B2", 1]] },
        },
        { match: (u) => u.includes("valueRenderOption=FORMULA"), body: { values: [] } },
        { match: (u, o) => u.includes("/conv1?supportsAllDrives") && o.method === "DELETE", body: "" },
      ]),
    );
    assert.equal(out.destination_spreadsheet_id, "dst1");
    assert.equal(out.imported.length, 2);
    assert.deepEqual(
      out.imported.map((r: any) => r.tab_name),
      [`SOURCE: budget.xlsx ${dash} Data`, `SOURCE: budget.xlsx ${dash} Summary`],
    );
    assert.ok(out.imported.every((r: any) => r.status === "success"));
    assert.equal(out.converted_copy_deleted, true);
    assert.equal(out.warnings.length, 2); // cross-sheet formula ref + the standing named-ranges warning
    assert.match(out.warnings[0], /formula references to source sheet "Summary"/);
    console.log("✔ import-spreadsheet (convert → copy → rename → warnings → cleanup)");

    console.log("\nALL SHEETS SMOKE CHECKS PASSED");
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
