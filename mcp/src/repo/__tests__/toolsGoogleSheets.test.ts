import { test, expect } from "../../testkit.js";
import {
  parseServiceAccount,
  registerGoogleSheetsTools,
  GOOGLE_SHEETS_TOOL_NAMES,
} from "../toolsGoogleSheets.js";
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

  test("honors per-tool disable and description overrides", () => {
    const tools: Record<string, Tool<any, any>> = {};
    registerGoogleSheetsTools(
      tools,
      { serviceAccount: SA },
      {
        sheets_add_sheet: { disabled: true },
        sheets_get_values: { description: "custom description" },
      }
    );
    expect(tools.sheets_add_sheet).toBeUndefined();
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
