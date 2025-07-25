import { expect, test } from "@playwright/test";
import { call } from "../tools.js";
import type { CallToolResult } from "@modelcontextprotocol/sdk/types.js";
import type { NetworkEntry } from "../core.js";

test.describe("Network Activity Monitoring", () => {
  const sessionId = "test-network-session";

  test.afterAll(async () => {
    // Clean up by clearing network entries
    await call("stagehand_network_activity", {}, sessionId);
  });

  test.describe("Basic Network Monitoring", () => {
    test("should capture network requests during navigation", async () => {
      // Navigate to a page that makes network requests
      await call(
        "stagehand_navigate",
        { url: "https://httpbin.org/get" },
        sessionId
      );

      // Wait a moment for network activity to be captured
      await new Promise((resolve) => setTimeout(resolve, 2000));

      // Get network activity
      const result = (await call(
        "stagehand_network_activity",
        {},
        sessionId
      )) as CallToolResult;
      expect(result.isError).toBe(false);

      const responseData = JSON.parse(
        (result.content?.[0] as { type: "text"; text: string }).text
      );
      expect(responseData.entries).toBeDefined();
      expect(responseData.entries.length).toBeGreaterThan(0);

      // Should have both request and response entries
      const hasRequest = responseData.entries.some(
        (entry: NetworkEntry) => entry.type === "request"
      );
      const hasResponse = responseData.entries.some(
        (entry: NetworkEntry) => entry.type === "response"
      );
      expect(hasRequest).toBe(true);
      expect(hasResponse).toBe(true);
    });

    test("should capture timing information for responses", async () => {
      await call(
        "stagehand_navigate",
        { url: "https://httpbin.org/delay/1" },
        sessionId
      );
      await new Promise((resolve) => setTimeout(resolve, 3000));

      const result = (await call(
        "stagehand_network_activity",
        {},
        sessionId
      )) as CallToolResult;
      const responseData = JSON.parse(
        (result.content?.[0] as { type: "text"; text: string }).text
      );

      const responseEntries = responseData.entries.filter(
        (entry: NetworkEntry) => entry.type === "response"
      );
      expect(responseEntries.length).toBeGreaterThan(0);

      // At least one response should have timing data
      const hasTimingData = responseEntries.some(
        (entry: NetworkEntry) =>
          entry.duration !== undefined && entry.duration > 0
      );
      expect(hasTimingData).toBe(true);
    });

    test("should capture status codes correctly", async () => {
      // Test successful request
      await call(
        "stagehand_navigate",
        { url: "https://httpbin.org/status/200" },
        sessionId
      );
      await new Promise((resolve) => setTimeout(resolve, 1500));

      const result = (await call(
        "stagehand_network_activity",
        {},
        sessionId
      )) as CallToolResult;
      const responseData = JSON.parse(
        (result.content?.[0] as { type: "text"; text: string }).text
      );

      const responseEntries = responseData.entries.filter(
        (entry: NetworkEntry) => entry.type === "response"
      );
      const hasSuccessStatus = responseEntries.some(
        (entry: NetworkEntry) => entry.status === 200
      );
      expect(hasSuccessStatus).toBe(true);
    });
  });

  test.describe("API Testing Scenario", () => {
    test("should monitor XHR/fetch requests during SPA interaction", async () => {
      // Create a simple HTML page with API calls
      const testPage = `
        <!DOCTYPE html>
        <html>
        <head><title>API Test Page</title></head>
        <body>
          <button id="fetch-btn">Fetch Data</button>
          <div id="result"></div>
          <script>
            document.getElementById('fetch-btn').addEventListener('click', async () => {
              try {
                const response = await fetch('https://httpbin.org/json');
                const data = await response.json();
                document.getElementById('result').textContent = JSON.stringify(data);
              } catch (error) {
                console.error('Fetch error:', error);
              }
            });
          </script>
        </body>
        </html>
      `;

      // Navigate to a data URL with our test page
      const dataUrl =
        "data:text/html;charset=utf-8," + encodeURIComponent(testPage);
      await call("stagehand_navigate", { url: dataUrl }, sessionId);

      // Clear previous network entries
      await call("stagehand_network_activity", {}, sessionId);

      // Click the button to trigger fetch
      await call(
        "stagehand_act",
        { action: "Click the fetch data button" },
        sessionId
      );
      await new Promise((resolve) => setTimeout(resolve, 2000));

      // Check network activity
      const result = (await call(
        "stagehand_network_activity",
        { filter: "xhr" },
        sessionId
      )) as CallToolResult;
      const responseData = JSON.parse(
        (result.content?.[0] as { type: "text"; text: string }).text
      );

      // Should capture the XHR request
      expect(responseData.entries.length).toBeGreaterThan(0);
      const hasXhrRequest = responseData.entries.some((entry: NetworkEntry) =>
        entry.url.includes("httpbin.org/json")
      );
      expect(hasXhrRequest).toBe(true);
    });
  });

  test.describe("Error Handling Scenarios", () => {
    test("should capture 404 errors correctly", async () => {
      await call(
        "stagehand_navigate",
        { url: "https://httpbin.org/status/404" },
        sessionId
      );
      await new Promise((resolve) => setTimeout(resolve, 1500));

      const result = (await call(
        "stagehand_network_activity",
        {},
        sessionId
      )) as CallToolResult;
      const responseData = JSON.parse(
        (result.content?.[0] as { type: "text"; text: string }).text
      );

      const responseEntries = responseData.entries.filter(
        (entry: NetworkEntry) => entry.type === "response"
      );
      const has404Status = responseEntries.some(
        (entry: NetworkEntry) => entry.status === 404
      );
      expect(has404Status).toBe(true);
    });

    test("should handle network timeouts gracefully", async () => {
      // Navigate to a slow endpoint
      await call(
        "stagehand_navigate",
        { url: "https://httpbin.org/delay/5" },
        sessionId
      );
      await new Promise((resolve) => setTimeout(resolve, 2000));

      const result = (await call(
        "stagehand_network_activity",
        {},
        sessionId
      )) as CallToolResult;
      expect(result.isError).toBe(false);

      // Should still capture the request even if response is slow
      const responseData = JSON.parse(
        (result.content?.[0] as { type: "text"; text: string }).text
      );
      expect(responseData.entries.length).toBeGreaterThan(0);
    });
  });

  test.describe("Filtering Functionality", () => {
    test("should filter by resource type", async () => {
      // Navigate to a page with various resource types
      await call(
        "stagehand_navigate",
        { url: "https://example.com" },
        sessionId
      );
      await new Promise((resolve) => setTimeout(resolve, 2000));

      // Test filtering by document type
      const documentResult = (await call(
        "stagehand_network_activity",
        { filter: "document" },
        sessionId
      )) as CallToolResult;
      const documentData = JSON.parse(
        (documentResult.content?.[0] as { type: "text"; text: string }).text
      );

      const allDocumentEntries = documentData.entries.every(
        (entry: NetworkEntry) => entry.resourceType === "document"
      );
      expect(allDocumentEntries).toBe(true);
    });

    test("should support verbose mode with full details", async () => {
      await call(
        "stagehand_navigate",
        { url: "https://httpbin.org/get" },
        sessionId
      );
      await new Promise((resolve) => setTimeout(resolve, 1500));

      const result = (await call(
        "stagehand_network_activity",
        { verbose: true },
        sessionId
      )) as CallToolResult;
      const responseData = JSON.parse(
        (result.content?.[0] as { type: "text"; text: string }).text
      );

      // Verbose mode should return raw array of entries
      expect(Array.isArray(responseData)).toBe(true);
      if (responseData.length > 0) {
        const entry = responseData[0];
        expect(entry).toHaveProperty("id");
        expect(entry).toHaveProperty("timestamp");
        expect(entry).toHaveProperty("type");
        expect(entry).toHaveProperty("method");
        expect(entry).toHaveProperty("url");
        expect(entry).toHaveProperty("resourceType");
      }
    });
  });

  test.describe("Response Structure", () => {
    test("should provide structured summary in simple mode", async () => {
      await call(
        "stagehand_navigate",
        { url: "https://httpbin.org/get" },
        sessionId
      );
      await new Promise((resolve) => setTimeout(resolve, 1500));

      const result = (await call(
        "stagehand_network_activity",
        {},
        sessionId
      )) as CallToolResult;
      const responseData = JSON.parse(
        (result.content?.[0] as { type: "text"; text: string }).text
      );

      expect(responseData).toHaveProperty("entries");
      expect(responseData).toHaveProperty("summary");
      expect(responseData.summary).toHaveProperty("total_entries");
      expect(responseData.summary).toHaveProperty("requests");
      expect(responseData.summary).toHaveProperty("responses");

      expect(typeof responseData.summary.total_entries).toBe("number");
      expect(typeof responseData.summary.requests).toBe("number");
      expect(typeof responseData.summary.responses).toBe("number");
    });
  });

  test.describe("Performance Monitoring", () => {
    test("should track response sizes when available", async () => {
      await call(
        "stagehand_navigate",
        { url: "https://httpbin.org/json" },
        sessionId
      );
      await new Promise((resolve) => setTimeout(resolve, 2000));

      const result = (await call(
        "stagehand_network_activity",
        { verbose: true },
        sessionId
      )) as CallToolResult;
      const responseData = JSON.parse(
        (result.content?.[0] as { type: "text"; text: string }).text
      );

      const responseEntries = responseData.filter(
        (entry: NetworkEntry) => entry.type === "response"
      );
      const hasSizeData = responseEntries.some(
        (entry: NetworkEntry) => entry.size !== undefined && entry.size > 0
      );
      expect(hasSizeData).toBe(true);
    });
  });
});
