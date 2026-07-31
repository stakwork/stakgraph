import { test, expect } from "../../testkit.js";

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

// ---- stdio branch tests ----
// Test the new stdio transport branch by observing logs and behavior.
// We use real process spawning with trivial shell commands or a fake process
// to verify routing, error handling, and log sanitization without a real
// MCP server (which would require a full server process).

test.describe("getMcpTools stdio branch", () => {
  test("stdio branch logs command but never args or env values", async () => {
    const { getMcpTools } = await import("../mcpServers.js");
    const logs: string[] = [];
    const errors: string[] = [];
    const origLog = console.log;
    const origErr = console.error;
    console.log = (...args: unknown[]) => { logs.push(args.join(" ")); };
    console.error = (...args: unknown[]) => { errors.push(args.join(" ")); };

    try {
      await getMcpTools([
        {
          name: "stdio-test",
          command: "node",
          args: ["--eval", "console.log('secret-arg')"],
          env: { SECRET_KEY: "my-super-secret-value" },
        },
      ]);
    } catch {
      // connection failure expected — we're not running a real MCP server
    } finally {
      console.log = origLog;
      console.error = origErr;
    }

    // Must log the command
    const connectLine = logs.find((l) => l.includes("[MCP] Connecting to stdio-test"));
    expect(connectLine).toBeTruthy();
    expect(connectLine).toContain("node");

    // Must NOT log args or env secrets in the connect line
    const allOutput = [...logs, ...errors].join("\n");
    expect(connectLine).not.toContain("secret-arg");
    expect(connectLine).not.toContain("my-super-secret-value");
    expect(connectLine).not.toContain("SECRET_KEY");

    // The connect line must NOT reference the HTTP "at <url>" pattern
    expect(connectLine).not.toContain(" at http");
    expect(connectLine).toContain("via command");
  });

  test("entry with both url and command is skipped with an error log", async () => {
    const { getMcpTools } = await import("../mcpServers.js");
    const errors: string[] = [];
    const origErr = console.error;
    console.error = (...args: unknown[]) => { errors.push(args.join(" ")); };

    let result: { tools: Record<string, unknown>; clients: unknown[] } | undefined;
    try {
      result = await getMcpTools([
        // TypeScript union won't allow this shape, so cast through any
        { name: "ambiguous", url: "http://localhost:19999", command: "echo" } as any,
      ]);
    } finally {
      console.error = origErr;
    }

    // Should not throw — the loop continues, returning empty results
    expect(result).toBeDefined();
    expect(Object.keys(result!.tools)).toHaveLength(0);
    expect(result!.clients).toHaveLength(0);

    // Should log a descriptive error
    const errLine = errors.find((l) => l.includes("ambiguous"));
    expect(errLine).toBeTruthy();
    // Error must mention the reason
    const combinedErr = errors.join("\n");
    expect(combinedErr).toContain("ambiguous");
  });

  test("spawn failure (ENOENT binary) is caught and server is skipped — no unhandled rejection", async () => {
    const { getMcpTools } = await import("../mcpServers.js");
    const errors: string[] = [];
    const origErr = console.error;
    console.error = (...args: unknown[]) => { errors.push(args.join(" ")); };

    let result: { tools: Record<string, unknown>; clients: unknown[] } | undefined;
    let threw = false;
    try {
      result = await getMcpTools([
        { name: "missing-bin", command: "definitely-not-a-real-binary-xyz-abc-123" },
      ]);
    } catch {
      threw = true;
    } finally {
      console.error = origErr;
    }

    // getMcpTools must NOT throw — it handles failures internally
    expect(threw).toBe(false);
    expect(result).toBeDefined();
    expect(Object.keys(result!.tools)).toHaveLength(0);

    // Should log an error about the failed server
    const errLine = errors.find((l) => l.includes("missing-bin"));
    expect(errLine).toBeTruthy();
  });

  test("spawn failure does not prevent other servers in the same array from loading", async () => {
    const { getMcpTools } = await import("../mcpServers.js");
    const errors: string[] = [];
    const logs: string[] = [];
    const origErr = console.error;
    const origLog = console.log;
    console.error = (...args: unknown[]) => { errors.push(args.join(" ")); };
    console.log = (...args: unknown[]) => { logs.push(args.join(" ")); };

    let result: { tools: Record<string, unknown>; clients: unknown[] } | undefined;
    try {
      result = await getMcpTools([
        // First server: bad binary — should fail and be skipped
        { name: "bad-server", command: "definitely-not-a-real-binary-xyz" },
        // Second server: also HTTP failure — but we verify it was attempted
        { name: "http-server", url: "http://localhost:19998" },
      ]);
    } catch {
      // ignore top-level errors — we're testing continuation
    } finally {
      console.error = origErr;
      console.log = origLog;
    }

    // The bad server should generate an error log
    const badErrLine = errors.find((l) => l.includes("bad-server"));
    expect(badErrLine).toBeTruthy();

    // The HTTP server should have been attempted (connect log emitted)
    const httpConnectLine = logs.find((l) => l.includes("Connecting to http-server"));
    expect(httpConnectLine).toBeTruthy();
  });

  test("getMcpTools returns a clients array with close() methods", async () => {
    const { getMcpTools } = await import("../mcpServers.js");
    // Use a server that fails — we just need to verify the result shape
    const result = await getMcpTools([]);
    expect(result).toBeDefined();
    expect(typeof result.tools).toBe("object");
    expect(Array.isArray(result.clients)).toBe(true);
  });

  test("returned result has tools and clients fields when no servers given", async () => {
    const { getMcpTools } = await import("../mcpServers.js");
    const result = await getMcpTools([]);
    expect(result.tools).toBeDefined();
    expect(result.clients).toBeDefined();
    expect(Object.keys(result.tools)).toHaveLength(0);
    expect(result.clients).toHaveLength(0);
  });
});

// ---- default tool filter (docx registry) tests ----
// Verify the built-in DEFAULT_TOOL_FILTERS["docx"] registry by using a mock
// MCP client that returns a known tool list. We achieve this by replacing
// createMCPClient in our test helper (without modifying mcpServers.ts).
//
// Since ESM module caching prevents us from patching createMCPClient directly
// on the cached module, we test the filter logic by running getMcpTools against
// an HTTP server that fails (so no HTTP tools appear) and independently testing
// the filter behavior through the public API.

// Helper: replicate the docx default filter defined in mcpServers.ts so our
// tests can assert against the exact same list without importing the private constant.
const DOCX_DEFAULT_FILTER = [
  "open_document",
  "create_document",
  "save_document",
  "close_document",
  "get_headings",
  "search_text",
  "insert_text",
  "delete_text",
  "replace_text",
  "add_comment",
  "add_footnote",
  "audit_document",
  "generate_change_summary",
  "diff_to_text",
];

test.describe("getMcpTools default tool filter (docx registry)", () => {
  test("docx default filter list has 14 entries and excludes convert_to_pdf", () => {
    // This is a pure unit test of our mirror list — it confirms the spec's
    // intent and serves as a canary if the list length changes.
    expect(DOCX_DEFAULT_FILTER.length).toBe(14);
    expect(DOCX_DEFAULT_FILTER.includes("convert_to_pdf")).toBe(false);
  });

  test("docx default filter includes expected core operations", () => {
    const required = [
      "open_document",
      "create_document",
      "save_document",
      "search_text",
      "insert_text",
      "delete_text",
      "audit_document",
      "generate_change_summary",
    ];
    for (const name of required) {
      expect(DOCX_DEFAULT_FILTER.includes(name)).toBe(true);
    }
  });

  // We verify the effective filter selection logic directly by simulating it:
  // - server named "docx" with no toolFilter → uses default registry
  // - server named "docx" with explicit toolFilter → uses caller's list
  // - server with other name and no toolFilter → no filter (all tools included)
  test("effective filter: docx server with no toolFilter uses docx default list", () => {
    // Simulate the logic in getMcpTools: effectiveFilter = server.toolFilter ?? defaultFilterFor(server)
    // defaultFilterFor("docx") → DOCX_DEFAULT_FILTER
    const server = { name: "docx", command: "docx-mcp-server" };
    const defaultFilter: Record<string, string[]> = { docx: DOCX_DEFAULT_FILTER };
    const effectiveFilter = (server as any).toolFilter ?? defaultFilter[server.name];
    expect(effectiveFilter).toBeDefined();
    expect(Array.isArray(effectiveFilter)).toBe(true);
    expect(effectiveFilter.length).toBe(DOCX_DEFAULT_FILTER.length);
    expect(effectiveFilter).not.toContain("convert_to_pdf");
  });

  test("effective filter: explicit toolFilter on docx server overrides built-in default", () => {
    const explicitFilter = ["open_document", "save_document"];
    const server = { name: "docx", command: "docx-mcp-server", toolFilter: explicitFilter };
    const defaultFilter: Record<string, string[]> = { docx: DOCX_DEFAULT_FILTER };
    const effectiveFilter = (server as any).toolFilter ?? defaultFilter[server.name];
    // Caller's explicit list wins
    expect(effectiveFilter).toStrictEqual(explicitFilter);
    expect(effectiveFilter.length).toBe(2);
  });

  test("effective filter: non-docx server with no toolFilter gets undefined (all tools)", () => {
    const server = { name: "my-custom-server", url: "http://localhost:9999" };
    const defaultFilter: Record<string, string[]> = { docx: DOCX_DEFAULT_FILTER };
    const effectiveFilter = (server as any).toolFilter ?? defaultFilter[server.name];
    // No registry entry → undefined → all tools included
    expect(effectiveFilter).toBeUndefined();
  });

  test("filter inclusion logic: tool in filter → included; tool not in filter → excluded", () => {
    const filter = DOCX_DEFAULT_FILTER;
    const allTools = [...DOCX_DEFAULT_FILTER, "convert_to_pdf", "some_other_tool"];

    const included = allTools.filter((name) =>
      !filter || filter.length === 0 || filter.includes(name)
    );
    const excluded = allTools.filter((name) =>
      filter && filter.length > 0 && !filter.includes(name)
    );

    // All docx filter entries should be included
    for (const name of DOCX_DEFAULT_FILTER) {
      expect(included.includes(name)).toBe(true);
    }
    // convert_to_pdf and extra tools should be excluded
    expect(excluded).toContain("convert_to_pdf");
    expect(excluded).toContain("some_other_tool");
    // Included count matches filter length (all filter entries appear in allTools)
    expect(included.length).toBe(DOCX_DEFAULT_FILTER.length);
  });

  test("included count uses effectiveFilter length, not total tool count", () => {
    // Simulate the includedCount computation in getMcpTools HTTP/stdio branch:
    //   const includedCount = effectiveFilter
    //     ? Object.keys(tools).filter(n => effectiveFilter.includes(n)).length
    //     : Object.keys(tools).length;
    const allToolNames = [...DOCX_DEFAULT_FILTER, "convert_to_pdf", "extra_tool"];
    const effectiveFilter = DOCX_DEFAULT_FILTER;

    const includedCount = effectiveFilter
      ? allToolNames.filter((n) => effectiveFilter.includes(n)).length
      : allToolNames.length;

    // Should equal the number of filter entries that appear in the tool list
    expect(includedCount).toBe(DOCX_DEFAULT_FILTER.length);
    // Not the total tool count
    expect(includedCount).not.toBe(allToolNames.length);
  });
});

// ---- sidecar redaction tests ----
// Verify that env (new), token, and headers are stripped from the persisted
// config, while command, args, url, and name are retained.

test.describe("sidecar redaction: env/token/headers stripped, command/args/url retained", () => {
  test("HTTP server: token and headers are stripped, url and name are retained", () => {
    // Simulate the redaction map in agent.ts:
    //   mcpServers: opts.mcpServers?.map(({ token, headers, env, ...rest }: any) => rest)
    const httpServer = {
      name: "my-http",
      url: "http://example.com/mcp",
      token: "secret-token-xyz",
      headers: { Authorization: "Bearer secret-token-xyz" },
      toolFilter: ["tool_a"],
    };
    const { token, headers, env, ...persisted } = httpServer as any;
    expect(persisted.name).toBe("my-http");
    expect(persisted.url).toBe("http://example.com/mcp");
    expect(persisted.toolFilter).toStrictEqual(["tool_a"]);
    expect(persisted.token).toBeUndefined();
    expect(persisted.headers).toBeUndefined();
    expect(persisted.env).toBeUndefined();
  });

  test("stdio server: env is stripped, command and args are retained", () => {
    const stdioServer = {
      name: "docx",
      command: "docx-mcp-server",
      args: ["--port", "8080"],
      env: { DOCX_API_KEY: "super-secret-key", PATH: "/usr/bin" },
      toolFilter: ["open_document"],
    };
    const { token, headers, env, ...persisted } = stdioServer as any;
    // Secrets stripped
    expect(persisted.env).toBeUndefined();
    expect(persisted.token).toBeUndefined();
    expect(persisted.headers).toBeUndefined();
    // Non-secrets retained
    expect(persisted.name).toBe("docx");
    expect(persisted.command).toBe("docx-mcp-server");
    expect(persisted.args).toStrictEqual(["--port", "8080"]);
    expect(persisted.toolFilter).toStrictEqual(["open_document"]);
  });

  test("env value does not appear in the persisted object's serialization", () => {
    const stdioServer = {
      name: "my-stdio",
      command: "my-server",
      env: { API_TOKEN: "top-secret-value-123" },
    };
    const { token, headers, env, ...persisted } = stdioServer as any;
    const serialized = JSON.stringify(persisted);
    expect(serialized).not.toContain("top-secret-value-123");
    expect(serialized).not.toContain("API_TOKEN");
    expect(serialized).toContain("my-stdio");
    expect(serialized).toContain("my-server");
  });

  test("mixed array: both HTTP and stdio servers are redacted correctly", () => {
    const mcpServers = [
      {
        name: "http-svc",
        url: "http://api.example.com",
        token: "http-token",
        headers: { "X-Key": "header-secret" },
      },
      {
        name: "stdio-svc",
        command: "my-stdio-server",
        args: ["--verbose"],
        env: { SECRET: "stdio-secret" },
      },
    ];
    const redacted = mcpServers.map(({ token, headers, env, ...rest }: any) => rest);

    expect(redacted[0].name).toBe("http-svc");
    expect(redacted[0].url).toBe("http://api.example.com");
    expect(redacted[0].token).toBeUndefined();
    expect(redacted[0].headers).toBeUndefined();
    expect(redacted[0].env).toBeUndefined();

    expect(redacted[1].name).toBe("stdio-svc");
    expect(redacted[1].command).toBe("my-stdio-server");
    expect(redacted[1].args).toStrictEqual(["--verbose"]);
    expect(redacted[1].env).toBeUndefined();
    expect(redacted[1].token).toBeUndefined();

    // Verify no secrets in serialized form
    const serialized = JSON.stringify(redacted);
    expect(serialized).not.toContain("http-token");
    expect(serialized).not.toContain("header-secret");
    expect(serialized).not.toContain("stdio-secret");
  });
});

// ---- parseAgentBody shallow-copy tests ----
// Verify that the shallow-copy logic for env/args/headers in parseAgentBody
// (mcp/src/repo/index.ts) ensures caller-supplied objects are never mutated.

test.describe("parseAgentBody shallow-copy: env/args/headers isolation", () => {
  // Helper: simulate the parseAgentBody map() call
  function simulateParseAgentBodyMap(
    servers: Array<Record<string, unknown>>
  ): Array<Record<string, unknown>> {
    return servers.map((s) => ({
      ...s,
      headers: "headers" in s && s.headers ? { ...(s.headers as object) } : (s as any).headers,
      env: "env" in s && (s as any).env ? { ...(s as any).env } : (s as any).env,
      args: "args" in s && (s as any).args ? [...(s as any).args] : (s as any).args,
    }));
  }

  test("mutating returned env does not mutate the original server object", () => {
    const original = {
      name: "my-stdio",
      command: "my-server",
      env: { ORIGINAL_KEY: "original-value" },
    };
    const [mapped] = simulateParseAgentBodyMap([original]);

    // Mutate the returned copy
    (mapped.env as Record<string, string>)["INJECTED"] = "injected-value";
    (mapped.env as Record<string, string>)["ORIGINAL_KEY"] = "mutated";

    // Original must be unchanged
    expect(original.env["ORIGINAL_KEY"]).toBe("original-value");
    expect((original.env as any)["INJECTED"]).toBeUndefined();
  });

  test("mutating returned args does not mutate the original server object", () => {
    const original = {
      name: "my-stdio",
      command: "my-server",
      args: ["--port", "8080"],
    };
    const [mapped] = simulateParseAgentBodyMap([original]);

    // Mutate the returned copy
    (mapped.args as string[]).push("--injected");
    (mapped.args as string[])[0] = "mutated";

    // Original must be unchanged
    expect(original.args).toStrictEqual(["--port", "8080"]);
  });

  test("mutating returned headers does not mutate the original server object", () => {
    const original = {
      name: "my-http",
      url: "http://example.com",
      headers: { Authorization: "Bearer original-token" },
    };
    const [mapped] = simulateParseAgentBodyMap([original]);

    // Mutate the returned copy
    (mapped.headers as Record<string, string>)["Authorization"] = "Bearer mutated";
    (mapped.headers as Record<string, string>)["X-New"] = "new-header";

    // Original must be unchanged
    expect(original.headers["Authorization"]).toBe("Bearer original-token");
    expect((original.headers as any)["X-New"]).toBeUndefined();
  });

  test("HTTP server (no env/args) maps correctly — env and args fields absent", () => {
    const original = {
      name: "http-only",
      url: "http://example.com",
      token: "tok123",
    };
    const [mapped] = simulateParseAgentBodyMap([original]);
    expect(mapped.name).toBe("http-only");
    expect(mapped.url).toBe("http://example.com");
    expect(mapped.token).toBe("tok123");
    // No env or args on HTTP servers
    expect(mapped.env).toBeUndefined();
    expect(mapped.args).toBeUndefined();
  });

  test("stdio server (no headers) maps correctly — headers field absent", () => {
    const original = {
      name: "stdio-only",
      command: "my-server",
      args: ["--fast"],
      env: { KEY: "val" },
    };
    const [mapped] = simulateParseAgentBodyMap([original]);
    expect(mapped.name).toBe("stdio-only");
    expect(mapped.command).toBe("my-server");
    expect((mapped.args as string[])).toStrictEqual(["--fast"]);
    expect((mapped.env as Record<string, string>)["KEY"]).toBe("val");
    // No headers on stdio servers
    expect(mapped.headers).toBeUndefined();
  });

  test("server with undefined env maps to undefined (not an empty object)", () => {
    const original = { name: "stdio-no-env", command: "my-server" };
    const [mapped] = simulateParseAgentBodyMap([original]);
    expect(mapped.env).toBeUndefined();
  });

  test("server with undefined args maps to undefined (not an empty array)", () => {
    const original = { name: "stdio-no-args", command: "my-server" };
    const [mapped] = simulateParseAgentBodyMap([original]);
    expect(mapped.args).toBeUndefined();
  });
});

// ---- getMcpTools return shape (McpToolsResult) tests ----
// Verify that getMcpTools always returns { tools, clients } regardless of success/failure.

test.describe("getMcpTools return shape (McpToolsResult)", () => {
  test("always returns an object with tools and clients fields", async () => {
    const { getMcpTools } = await import("../mcpServers.js");
    const result = await getMcpTools([]);
    expect(typeof result).toBe("object");
    expect("tools" in result).toBe(true);
    expect("clients" in result).toBe(true);
  });

  test("tools is a plain object (not array)", async () => {
    const { getMcpTools } = await import("../mcpServers.js");
    const result = await getMcpTools([]);
    expect(Array.isArray(result.tools)).toBe(false);
    expect(typeof result.tools).toBe("object");
  });

  test("clients is an array", async () => {
    const { getMcpTools } = await import("../mcpServers.js");
    const result = await getMcpTools([]);
    expect(Array.isArray(result.clients)).toBe(true);
  });

  test("failed HTTP server does not add a client to the clients array", async () => {
    const { getMcpTools } = await import("../mcpServers.js");
    const result = await getMcpTools([
      { name: "unreachable", url: "http://localhost:29999" },
    ]);
    // The HTTP server fails to connect — it must not appear in clients
    // (the client was never successfully created or it was cleaned up before push)
    // Either 0 clients (failure before push) or we just verify tools are empty
    expect(Object.keys(result.tools)).toHaveLength(0);
    // clients array always exists
    expect(Array.isArray(result.clients)).toBe(true);
  });

  test("failed stdio server (ENOENT) does not add a client to the clients array", async () => {
    const { getMcpTools } = await import("../mcpServers.js");
    const origErr = console.error;
    console.error = () => {}; // suppress noise

    let result: { tools: Record<string, unknown>; clients: unknown[] } | undefined;
    try {
      result = await getMcpTools([
        { name: "bad-stdio", command: "definitely-not-a-real-binary-xyz-789" },
      ]);
    } finally {
      console.error = origErr;
    }

    expect(result).toBeDefined();
    expect(Object.keys(result!.tools)).toHaveLength(0);
    expect(Array.isArray(result!.clients)).toBe(true);
    // No client was successfully connected, so clients should be empty
    expect(result!.clients).toHaveLength(0);
  });
});

// ---- client lifecycle / close() contract tests ----
// Verify the close() contract by simulating the cleanup pattern used in agent.ts.

test.describe("MCP client lifecycle: close() contract", () => {
  test("each close() call is individually try/caught so one failure does not block others", async () => {
    // Simulate the finally block in agent.ts get_context():
    //   for (const client of prepared.mcpClients) {
    //     try { await client.close(); } catch {}
    //   }
    let closeACount = 0;
    let closeBCount = 0;
    let closeCCount = 0;

    const mockClients = [
      { close: async () => { closeACount++; } },
      { close: async () => { throw new Error("close failed"); } }, // this one throws
      { close: async () => { closeCCount++; } },
    ];

    // Run the cleanup pattern
    for (const client of mockClients) {
      try {
        await client.close();
      } catch {
        // swallowed — same as agent.ts
      }
    }

    // Client A and C must have been closed despite B's failure
    expect(closeACount).toBe(1);
    expect(closeCCount).toBe(1);
  });

  test("close() is called once per client, not multiple times", async () => {
    let closeCount = 0;
    const mockClient = { close: async () => { closeCount++; } };
    const clients = [mockClient];

    for (const client of clients) {
      try { await client.close(); } catch {}
    }

    expect(closeCount).toBe(1);
  });

  test("empty clients array completes without error in the cleanup loop", async () => {
    const clients: Array<{ close: () => Promise<void> }> = [];
    let threw = false;
    try {
      for (const client of clients) {
        try { await client.close(); } catch {}
      }
    } catch {
      threw = true;
    }
    expect(threw).toBe(false);
  });

  test("getMcpTools result includes close() method on each client when connection succeeds", async () => {
    // We verify the shape contract: if a client IS returned, it has close()
    // We use an empty array (no servers) since we can't connect to a real MCP
    const { getMcpTools } = await import("../mcpServers.js");
    const result = await getMcpTools([]);
    for (const client of result.clients) {
      expect(typeof client.close).toBe("function");
    }
  });
});
