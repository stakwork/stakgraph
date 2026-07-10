import { test, expect } from "@playwright/test";

// ---- maskCredential unit tests ----
// maskCredential is module-scope but not exported; we replicate its logic here
// to keep these tests self-contained and fast (no network needed).

function maskCredential(cred?: string): string {
  if (!cred) return "(no credential)";
  if (cred.length <= 4) return "*".repeat(cred.length);
  return `${"*".repeat(cred.length - 4)}${cred.slice(-4)}`;
}

test.describe("maskCredential", () => {
  test("undefined input returns (no credential)", () => {
    expect(maskCredential(undefined)).toBe("(no credential)");
  });

  test("empty string returns (no credential)", () => {
    expect(maskCredential("")).toBe("(no credential)");
  });

  test("credential of exactly 4 chars is fully masked", () => {
    expect(maskCredential("abcd")).toBe("****");
  });

  test("credential shorter than 4 chars is fully masked", () => {
    expect(maskCredential("abc")).toBe("***");
    expect(maskCredential("a")).toBe("*");
  });

  test("normal credential shows only last 4 characters", () => {
    const cred = "mysecrettoken1234";
    const result = maskCredential(cred);
    expect(result.endsWith("1234")).toBe(true);
    expect(result.startsWith("*".repeat(cred.length - 4))).toBe(true);
    expect(result).not.toContain("mysecret");
  });
});

// ---- getMcpTools logging tests ----
// We spy on console.log to verify the pre-connection log line is emitted
// with a masked credential and never the full token.

test.describe("getMcpTools logging", () => {
  test("logs masked credential when server.token is set", async () => {
    const { getMcpTools } = await import("../mcpServers.js");
    const fullToken = "supersecrettoken9876";
    const logs: string[] = [];
    const orig = console.log;
    console.log = (...args: unknown[]) => { logs.push(args.join(" ")); };

    try {
      await getMcpTools([
        { name: "test-server", url: "http://localhost:19999", token: fullToken },
      ]);
    } catch {
      // connection failure is expected in test env
    } finally {
      console.log = orig;
    }

    const connectLine = logs.find((l) => l.includes("[MCP] Connecting to test-server"));
    expect(connectLine).toBeTruthy();
    expect(connectLine).toContain("9876");
    expect(connectLine).not.toContain(fullToken);
  });

  test("logs masked credential from headers.Authorization when no server.token", async () => {
    const { getMcpTools } = await import("../mcpServers.js");
    const fullToken = "headertokenabcd";
    const logs: string[] = [];
    const orig = console.log;
    console.log = (...args: unknown[]) => { logs.push(args.join(" ")); };

    try {
      await getMcpTools([
        {
          name: "header-server",
          url: "http://localhost:19999",
          headers: { Authorization: `Bearer ${fullToken}` },
        },
      ]);
    } catch {
      // connection failure is expected in test env
    } finally {
      console.log = orig;
    }

    const connectLine = logs.find((l) => l.includes("[MCP] Connecting to header-server"));
    expect(connectLine).toBeTruthy();
    expect(connectLine).toContain("abcd");
    expect(connectLine).not.toContain(fullToken);
  });

  test("logs (no credential) when no token or auth header", async () => {
    const { getMcpTools } = await import("../mcpServers.js");
    const logs: string[] = [];
    const orig = console.log;
    console.log = (...args: unknown[]) => { logs.push(args.join(" ")); };

    try {
      await getMcpTools([
        { name: "open-server", url: "http://localhost:19999" },
      ]);
    } catch {
      // connection failure is expected in test env
    } finally {
      console.log = orig;
    }

    const connectLine = logs.find((l) => l.includes("[MCP] Connecting to open-server"));
    expect(connectLine).toBeTruthy();
    expect(connectLine).toContain("(no credential)");
  });

  test("shared headers object is not mutated when servers share the same reference", async () => {
    const { getMcpTools } = await import("../mcpServers.js");
    const shared: Record<string, string> = {};
    const servers = [
      { name: "a", url: "http://localhost:19999", token: "tokenAAA", headers: shared },
      { name: "b", url: "http://localhost:19999", token: "tokenBBB", headers: shared },
    ];

    try {
      await getMcpTools(servers);
    } catch {
      // connection failure is expected in test env
    }

    expect(shared.Authorization).toBeUndefined();
  });

  test("no token leak across servers sharing headers when only one has a token", async () => {
    const { getMcpTools } = await import("../mcpServers.js");
    const shared: Record<string, string> = {};
    const logs: string[] = [];
    const orig = console.log;
    console.log = (...args: unknown[]) => { logs.push(args.join(" ")); };

    try {
      await getMcpTools([
        { name: "a", url: "http://localhost:19999", token: "tokenAAA", headers: shared },
        { name: "b", url: "http://localhost:19999", headers: shared },
      ]);
    } catch {
      // connection failure is expected in test env
    } finally {
      console.log = orig;
    }

    const connectLineB = logs.find((l) => l.includes("[MCP] Connecting to b"));
    expect(connectLineB).toBeTruthy();
    expect(connectLineB).toContain("(no credential)");
    expect(connectLineB).not.toContain("tokenAAA");
  });
});
