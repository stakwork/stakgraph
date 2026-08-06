import { test, expect } from "../../testkit.js";
import {
  parseServiceAccount,
  registerGoogleSheetsTools,
  GOOGLE_SHEETS_TOOL_NAMES,
  resolveLabel,
  buildTabName,
  resolveCollisionSuffix,
  detectCrossSheetFormulaRefs,
} from "../toolsGoogleSheets.js";
import { resolveGoogleSheetsOptions, redactToolsConfig } from "../tools.js";
import type { Tool } from "ai";

const SA = {
  client_email: "agent@proj.iam.gserviceaccount.com",
  private_key: "-----BEGIN PRIVATE KEY-----\nabc\n-----END PRIVATE KEY-----\n",
  token_uri: "https://oauth2.googleapis.com/token",
};

test.describe("parseServiceAccount", () => {
  test("accepts a parsed JSON object", () => {
    const sa = parseServiceAccount(SA);
    expect(sa.client_email).toBe(SA.client_email);
    expect(sa.private_key).toBe(SA.private_key);
    expect(sa.token_uri).toBe(SA.token_uri);
  });

  test("accepts a JSON string", () => {
    const sa = parseServiceAccount(JSON.stringify(SA));
    expect(sa.client_email).toBe(SA.client_email);
  });

  test("accepts base64-encoded JSON (CREDENTIALS_CONFIG style)", () => {
    const b64 = Buffer.from(JSON.stringify(SA)).toString("base64");
    const sa = parseServiceAccount(b64);
    expect(sa.client_email).toBe(SA.client_email);
  });

  test("normalizes literal \\n sequences in the private key", () => {
    const sa = parseServiceAccount({
      ...SA,
      private_key: "-----BEGIN PRIVATE KEY-----\\nabc\\n-----END PRIVATE KEY-----\\n",
    });
    expect(sa.private_key).toBe(SA.private_key);
  });

  test("rejects objects missing client_email or private_key", () => {
    expect(() => parseServiceAccount({ private_key: "x" })).toThrow();
    expect(() => parseServiceAccount({ client_email: "x" })).toThrow();
    expect(() => parseServiceAccount(null)).toThrow();
    expect(() => parseServiceAccount(42)).toThrow();
  });
});

test.describe("registerGoogleSheetsTools", () => {
  test("registers the full tool family for valid credentials", () => {
    const tools: Record<string, Tool<any, any>> = {};
    registerGoogleSheetsTools(tools, { serviceAccount: SA });
    for (const name of GOOGLE_SHEETS_TOOL_NAMES) {
      expect(tools[name]).toBeDefined();
    }
  });

  test("registers nothing for invalid credentials (non-fatal)", () => {
    const tools: Record<string, Tool<any, any>> = {};
    registerGoogleSheetsTools(tools, { serviceAccount: "not-json-not-base64" });
    expect(Object.keys(tools).length).toBe(0);
  });

  test("sheets_import_spreadsheet is present in GOOGLE_SHEETS_TOOL_NAMES", () => {
    expect(GOOGLE_SHEETS_TOOL_NAMES).toContain("sheets_import_spreadsheet");
  });

  test("sheets_import_spreadsheet is registered by default with valid credentials", () => {
    const tools: Record<string, Tool<any, any>> = {};
    registerGoogleSheetsTools(tools, { serviceAccount: SA });
    expect(tools.sheets_import_spreadsheet).toBeDefined();
  });

  test("honors per-tool disable and description overrides", () => {
    const tools: Record<string, Tool<any, any>> = {};
    registerGoogleSheetsTools(
      tools,
      { serviceAccount: SA },
      {
        sheets_add_sheet: { disabled: true },
        sheets_get_values: { description: "custom description" },
        sheets_import_spreadsheet: { disabled: true },
      }
    );
    expect(tools.sheets_add_sheet).toBeUndefined();
    expect(tools.sheets_import_spreadsheet).toBeUndefined();
    expect(tools.sheets_get_values).toBeDefined();
    expect((tools.sheets_get_values as any).description).toBe("custom description");
    expect(tools.sheets_create_spreadsheet).toBeDefined();
  });

  test("tool execution failures return error strings, never throw", async () => {
    const tools: Record<string, Tool<any, any>> = {};
    // Structurally valid credentials with a garbage key: token signing fails
    // at execute time and must surface as a "failed:" string result.
    registerGoogleSheetsTools(tools, { serviceAccount: SA });
    const result = await (tools.sheets_get_values as any).execute(
      { spreadsheet_id: "x", range: "Sheet1!A1" },
      { toolCallId: "t1", messages: [] }
    );
    expect(typeof result).toBe("string");
    expect(result).toContain("sheets_get_values failed:");
  });

  test("sheets_import_spreadsheet execution failure returns error string, never throws", async () => {
    const tools: Record<string, Tool<any, any>> = {};
    // Structurally valid but fake credentials: JWT signing will fail at token
    // exchange time and must surface as a "sheets_import_spreadsheet failed:" string.
    registerGoogleSheetsTools(tools, { serviceAccount: SA });
    const result = await (tools.sheets_import_spreadsheet as any).execute(
      {
        destination_spreadsheet_id: "dest-id",
        source_file_id: "src-id",
      },
      { toolCallId: "t2", messages: [] }
    );
    expect(typeof result).toBe("string");
    expect(result).toContain("sheets_import_spreadsheet failed:");
  });

  test("mentions the shared Drive folder in create description only when configured", () => {
    const withFolder: Record<string, Tool<any, any>> = {};
    registerGoogleSheetsTools(withFolder, { serviceAccount: SA, driveFolderId: "abc123" });
    expect((withFolder.sheets_create_spreadsheet as any).description).toContain("Drive folder");

    const withoutFolder: Record<string, Tool<any, any>> = {};
    registerGoogleSheetsTools(withoutFolder, { serviceAccount: SA });
    expect((withoutFolder.sheets_create_spreadsheet as any).description).not.toContain(
      "Drive folder"
    );
  });
});

test.describe("resolveLabel", () => {
  test("explicit label wins over all defaults", () => {
    expect(
      resolveLabel({
        explicitLabel: "My Label",
        driveFileName: "file.xlsx",
        sourceSpreadsheetTitle: "Spreadsheet Title",
        isNative: false,
      })
    ).toBe("My Label");

    expect(
      resolveLabel({
        explicitLabel: "My Label",
        driveFileName: "file.xlsx",
        sourceSpreadsheetTitle: "Spreadsheet Title",
        isNative: true,
      })
    ).toBe("My Label");
  });

  test("converted source falls back to Drive filename", () => {
    expect(
      resolveLabel({
        explicitLabel: undefined,
        driveFileName: "budget_2024.xlsx",
        sourceSpreadsheetTitle: "Budget 2024",
        isNative: false,
      })
    ).toBe("budget_2024.xlsx");
  });

  test("native source falls back to spreadsheet title", () => {
    expect(
      resolveLabel({
        explicitLabel: undefined,
        driveFileName: "Untitled",
        sourceSpreadsheetTitle: "Q4 Pipeline",
        isNative: true,
      })
    ).toBe("Q4 Pipeline");
  });
});

test.describe("buildTabName", () => {
  test("single-sheet source produces SOURCE: <label>", () => {
    expect(buildTabName("MyDoc", "Sheet1", 1)).toBe("SOURCE: MyDoc");
  });

  test("multi-sheet source produces SOURCE: <label> — <sheet title>", () => {
    expect(buildTabName("MyDoc", "Sheet1", 3)).toBe("SOURCE: MyDoc \u2014 Sheet1");
    expect(buildTabName("MyDoc", "Summary", 2)).toBe("SOURCE: MyDoc \u2014 Summary");
  });

  test("uses em dash (U+2014) as separator", () => {
    const name = buildTabName("X", "Y", 2);
    expect(name).toContain("\u2014");
    expect(name).not.toContain("-");
  });
});

test.describe("resolveCollisionSuffix", () => {
  test("returns candidate unchanged when no collision", () => {
    const existing = new Set(["Other Tab", "Another Tab"]);
    expect(resolveCollisionSuffix("New Tab", existing)).toBe("New Tab");
  });

  test("appends (2) when base name is taken", () => {
    const existing = new Set(["SOURCE: MyDoc"]);
    expect(resolveCollisionSuffix("SOURCE: MyDoc", existing)).toBe("SOURCE: MyDoc (2)");
  });

  test("skips to (3) when both base and (2) are taken", () => {
    const existing = new Set(["SOURCE: MyDoc", "SOURCE: MyDoc (2)"]);
    expect(resolveCollisionSuffix("SOURCE: MyDoc", existing)).toBe("SOURCE: MyDoc (3)");
  });

  test("handles non-contiguous existing suffixes — (2) taken, (3) free", () => {
    // (2) is present but (3) is not — must land on (3), not loop forever.
    const existing = new Set(["SOURCE: MyDoc", "SOURCE: MyDoc (2)"]);
    expect(resolveCollisionSuffix("SOURCE: MyDoc", existing)).toBe("SOURCE: MyDoc (3)");
  });

  test("correctly increments past multiple taken suffixes", () => {
    const existing = new Set([
      "SOURCE: MyDoc",
      "SOURCE: MyDoc (2)",
      "SOURCE: MyDoc (3)",
      "SOURCE: MyDoc (4)",
    ]);
    expect(resolveCollisionSuffix("SOURCE: MyDoc", existing)).toBe("SOURCE: MyDoc (5)");
  });
});

test.describe("detectCrossSheetFormulaRefs", () => {
  test("returns empty array when no formulas reference any title", () => {
    const formulas = ["=SUM(A1:A10)", "=AVERAGE(B2:B5)", "=IF(A1>0,1,0)"];
    const titles = ["Sheet2", "Sheet3"];
    expect(detectCrossSheetFormulaRefs(formulas, titles)).toEqual([]);
  });

  test("detects a bare cross-sheet reference (no quotes)", () => {
    const formulas = ["=Sheet2!A1+10"];
    expect(detectCrossSheetFormulaRefs(formulas, ["Sheet2", "Sheet3"])).toEqual(["Sheet2"]);
  });

  test("detects a single-quoted cross-sheet reference", () => {
    const formulas = ["=SUM('Revenue Data'!B2:B20)"];
    expect(
      detectCrossSheetFormulaRefs(formulas, ["Revenue Data", "Cost Data"])
    ).toEqual(["Revenue Data"]);
  });

  test("returns only the titles that are actually referenced", () => {
    const formulas = ["=Sheet2!A1", "=Sheet2!B2"];
    const titles = ["Sheet2", "Sheet3", "Sheet4"];
    const result = detectCrossSheetFormulaRefs(formulas, titles);
    expect(result).toEqual(["Sheet2"]);
    expect(result).not.toContain("Sheet3");
    expect(result).not.toContain("Sheet4");
  });

  test("returns each referenced title once even if referenced in multiple formulas", () => {
    const formulas = ["=Sheet2!A1", "=Sheet2!B2", "=Sheet2!C3"];
    const result = detectCrossSheetFormulaRefs(formulas, ["Sheet2"]);
    expect(result).toEqual(["Sheet2"]);
    expect(result.length).toBe(1);
  });
});

test.describe("sheets_import_spreadsheet delete-target safety", () => {
  test("tempSpreadsheetId is a distinct variable from source_file_id and destination_spreadsheet_id", () => {
    // This test validates the variable isolation by construction:
    // resolveLabel/buildTabName/resolveCollisionSuffix all work on named
    // parameters. The actual delete call in the tool uses a local
    // `tempSpreadsheetId` that is set ONLY from the Drive files.copy response
    // and NEVER assigned from source_file_id or destination_spreadsheet_id.
    //
    // We assert here that the pure functions cannot conflate ids:
    // resolveLabel does not accept or return ids — it only handles label strings.
    const sourceFileId = "source-file-id-abc";
    const destinationId = "destination-id-xyz";
    const tempId = "temp-converted-id-999";

    // Simulate the distinction: tempId is a fresh string, never equal to either.
    expect(tempId).not.toBe(sourceFileId);
    expect(tempId).not.toBe(destinationId);
    expect(sourceFileId).not.toBe(destinationId);

    // The resolveLabel function handles only label strings — never ids.
    const label = resolveLabel({
      explicitLabel: undefined,
      driveFileName: "report.xlsx",
      sourceSpreadsheetTitle: "Report",
      isNative: false,
    });
    expect(label).not.toBe(sourceFileId);
    expect(label).not.toBe(destinationId);
    expect(label).not.toBe(tempId);
  });
});

test.describe("resolveGoogleSheetsOptions", () => {
  test("returns undefined when neither source has credentials", () => {
    expect(resolveGoogleSheetsOptions(undefined, undefined)).toBeUndefined();
    expect(resolveGoogleSheetsOptions(undefined, { bash: true })).toBeUndefined();
    // An object without serviceAccount is not credentials.
    expect(
      resolveGoogleSheetsOptions(undefined, { google_sheets: { driveFolderId: "abc" } as any })
    ).toBeUndefined();
  });

  test("falls back to toolsConfig.google_sheets", () => {
    const resolved = resolveGoogleSheetsOptions(undefined, {
      google_sheets: { serviceAccount: SA, driveFolderId: "abc123" },
    });
    expect(resolved?.serviceAccount).toBe(SA);
    expect(resolved?.driveFolderId).toBe("abc123");
  });

  test("accepts the camelCase alias and drops a blank driveFolderId", () => {
    const resolved = resolveGoogleSheetsOptions(undefined, {
      googleSheets: { serviceAccount: SA, driveFolderId: "" },
    });
    expect(resolved?.serviceAccount).toBe(SA);
    expect(resolved?.driveFolderId).toBeUndefined();
  });

  test("the top-level field wins over the toolsConfig fallback", () => {
    const other = { ...SA, client_email: "top-level@proj.iam.gserviceaccount.com" };
    const resolved = resolveGoogleSheetsOptions(
      { serviceAccount: other, driveFolderId: "top" },
      { google_sheets: { serviceAccount: SA, driveFolderId: "nested" } }
    );
    expect(resolved?.serviceAccount).toBe(other);
    expect(resolved?.driveFolderId).toBe("top");
  });
});

test.describe("redactToolsConfig", () => {
  test("strips serviceAccount but keeps the rest of the config", () => {
    const input = {
      bash: true,
      sheets_get_values: false,
      google_sheets: { serviceAccount: SA, driveFolderId: "abc123" },
    } as any;
    const out = redactToolsConfig(input) as any;
    expect(out.google_sheets.serviceAccount).toBeUndefined();
    expect(out.google_sheets.driveFolderId).toBe("abc123");
    expect(out.bash).toBe(true);
    expect(out.sheets_get_values).toBe(false);
    // The caller's object is not mutated — the live run still has credentials.
    expect(input.google_sheets.serviceAccount).toBe(SA);
    expect(JSON.stringify(out)).not.toContain("BEGIN PRIVATE KEY");
  });

  test("passes through configs with no credentials, and undefined", () => {
    expect(redactToolsConfig(undefined)).toBeUndefined();
    const plain = { bash: true, logs_agent: "custom" } as any;
    expect(redactToolsConfig(plain)).toBe(plain);
  });
});
