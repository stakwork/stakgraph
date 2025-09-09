/**
 * Tests for stagehand console logs functionality
 * Uses Playwright test framework
 */

import { test, expect } from "@playwright/test";
import { call } from "../tools.js";
import { getOrCreateStagehand, clearConsoleLogs } from "../core.js";
import type { ConsoleLog } from "../core.js";
import type { CallToolResult } from "@modelcontextprotocol/sdk/types.js";
import type { Stagehand } from "@browserbasehq/stagehand";

// Helper function to extract log text from CallToolResult
function extractLogText(result: CallToolResult): string {
  return (result.content as any)[0].text;
}

test.describe("Stagehand Console Logs", () => {
  let stagehand: Stagehand;

  test.beforeAll(async () => {
    console.log("=== Initializing Stagehand for Console Logs Tests ===");
    stagehand = await getOrCreateStagehand();
  });

  test.afterAll(async () => {
    if (stagehand) {
      await stagehand.close();
      console.log("=== Stagehand closed ===");
    }
  });

  test.beforeEach(async () => {
    clearConsoleLogs("default-session-id");
  });

  test("should demonstrate real debugging workflow: find and capture JavaScript errors", async () => {
    console.log(
      "🧪 REAL SCENARIO: Debugging a website with JavaScript errors..."
    );

    // Clear logs to start fresh
    clearConsoleLogs("default-session-id");

    // Navigate to a page that has intentional JavaScript errors (common debugging scenario)
    const buggyPage = `
      <html>
        <head><title>Buggy E-commerce Page</title></head>
        <body>
          <h1>Shopping Cart</h1>
          <button id="add-item">Add Item</button>
          <script>
            console.log('Page loaded - initializing shopping cart');

            // Simulate real application logs
            console.info('User session: guest_user_12345');

            document.getElementById('add-item').addEventListener('click', function() {
              console.log('User clicked add item button');

              // This will cause an error - common debugging scenario
              try {
                nonExistentFunction(); // This will throw an error
              } catch (e) {
                console.error('Failed to add item:', e.message);
                console.warn('Falling back to basic add functionality');
              }

              console.log('Item add process completed');
            });

            // Simulate analytics
            console.log('Analytics: page_view', { page: 'cart', user: 'guest' });
          </script>
        </body>
      </html>
    `;

    await stagehand.page.goto(
      `data:text/html,${encodeURIComponent(buggyPage)}`
    );
    console.log("📄 Navigated to buggy e-commerce page");

    // Wait for page to load and logs to be captured
    await new Promise((resolve) => setTimeout(resolve, 200));

    // Simulate user interaction that triggers the error
    await stagehand.page.click("#add-item");
    console.log(
      '🖱️  Simulated user clicking "Add Item" button (triggers error)'
    );

    // Wait for error logs to be captured
    await new Promise((resolve) => setTimeout(resolve, 300));

    // Now capture the logs - this is what a developer would do when debugging
    const result = (await call("stagehand_logs", {
      verbose: true,
    })) as CallToolResult;
    const logText = extractLogText(result);
    console.log("===> logText", logText);
    const logs: ConsoleLog[] = JSON.parse(logText);

    console.log(
      `🔍 CAPTURED ${logs.length} console logs during debugging session:`
    );
    logs.forEach((log, i) => {
      console.log(`   ${i + 1}. [${log.type.toUpperCase()}] ${log.text}`);
    });

    // Verify we captured meaningful logs
    expect(logs.length).toBeGreaterThan(4);

    // Check for specific debugging-relevant logs
    const pageLoadLog = logs.find((log) =>
      log.text.includes("Page loaded - initializing")
    );
    const userClickLog = logs.find((log) =>
      log.text.includes("User clicked add item")
    );
    const errorLog = logs.find(
      (log) => log.type === "error" && log.text.includes("Failed to add item")
    );
    const warningLog = logs.find(
      (log) => log.type === "warning" && log.text.includes("Falling back")
    );
    const analyticsLog = logs.find((log) =>
      log.text.includes("Analytics: page_view")
    );

    expect(pageLoadLog).toBeDefined();
    expect(userClickLog).toBeDefined();
    expect(errorLog).toBeDefined();
    expect(warningLog).toBeDefined();
    expect(analyticsLog).toBeDefined();

    console.log(
      "✅ SUCCESS: Console logs tool captured real debugging session"
    );
    console.log(
      `   📊 Found page load, user interaction, error, warning, and analytics logs`
    );
    console.log(
      `   🎯 This proves the tool works for real debugging scenarios!`
    );
  });

  test("should capture performance monitoring and API tracking logs from SPA", async () => {
    console.log(
      "🧪 REAL SCENARIO: Monitoring a Single Page Application performance..."
    );

    // Simulate a realistic SPA with performance monitoring
    const spaPage = `
      <html>
        <head><title>Analytics Dashboard</title></head>
        <body>
          <h1>User Analytics Dashboard</h1>
          <div id="loading">Loading...</div>
          <div id="data-container" style="display:none;"></div>
          <script>
            // Performance monitoring start
            const pageStartTime = performance.now();
            console.log('PERF: Page load started', { timestamp: pageStartTime });

            // Simulate API call logging
            console.log('API: Fetching user data...', {
              endpoint: '/api/users/123',
              method: 'GET',
              headers: { 'Content-Type': 'application/json' }
            });

            // Simulate network delay
            setTimeout(() => {
              const apiResponseTime = performance.now();
              console.log('API: Response received', {
                duration: apiResponseTime - pageStartTime + 'ms',
                status: 200,
                dataSize: '2.3KB'
              });

              // Simulate data processing
              console.info('DATA: Processing user analytics...', { recordCount: 1250 });

              // Simulate error during processing
              try {
                // Simulate a third-party service failure
                throw new Error('Third-party analytics service timeout');
              } catch (e) {
                console.error('ANALYTICS: Service error detected', {
                  error: e.message,
                  service: 'analytics-api',
                  fallback: 'enabled'
                });
                console.warn('ANALYTICS: Using cached data instead', {
                  cacheAge: '5 minutes',
                  completeness: '85%'
                });
              }

              // Complete the load
              document.getElementById('loading').style.display = 'none';
              document.getElementById('data-container').style.display = 'block';

              const totalLoadTime = performance.now() - pageStartTime;
              console.log('PERF: Page render complete', {
                totalTime: totalLoadTime + 'ms',
                metrics: {
                  api: '150ms',
                  processing: '75ms',
                  render: '25ms'
                }
              });

            }, 100);
          </script>
        </body>
      </html>
    `;

    await stagehand.page.goto(`data:text/html,${encodeURIComponent(spaPage)}`);
    console.log("📄 Loaded SPA dashboard with performance monitoring");

    // Wait for all async operations to complete
    await new Promise((resolve) => setTimeout(resolve, 500));

    const result = (await call("stagehand_logs", {
      verbose: true,
    })) as CallToolResult;
    const logText = extractLogText(result);
    const logs: ConsoleLog[] = JSON.parse(logText);

    console.log(`🔍 CAPTURED ${logs.length} performance & API logs:`);
    logs.forEach((log, i) => {
      console.log(
        `   ${i + 1}. [${log.type.toUpperCase()}] ${log.text.substring(
          0,
          80
        )}...`
      );
    });

    // Verify we captured all the realistic logging scenarios
    expect(logs.length).toBeGreaterThan(5);

    const perfStartLog = logs.find((log) =>
      log.text.includes("PERF: Page load started")
    );
    const apiRequestLog = logs.find((log) =>
      log.text.includes("API: Fetching user data")
    );
    const apiResponseLog = logs.find((log) =>
      log.text.includes("API: Response received")
    );
    const errorLog = logs.find(
      (log) =>
        log.type === "error" && log.text.includes("ANALYTICS: Service error")
    );
    const warningLog = logs.find(
      (log) => log.type === "warning" && log.text.includes("Using cached data")
    );
    const perfCompleteLog = logs.find((log) =>
      log.text.includes("PERF: Page render complete")
    );

    expect(perfStartLog).toBeDefined();
    expect(apiRequestLog).toBeDefined();
    expect(apiResponseLog).toBeDefined();
    expect(errorLog).toBeDefined();
    expect(warningLog).toBeDefined();
    expect(perfCompleteLog).toBeDefined();

    console.log("✅ SUCCESS: Captured comprehensive SPA monitoring logs");
    console.log(
      "   📊 Performance timing, API calls, errors, and fallback strategies"
    );
    console.log("   🎯 Perfect for debugging production SPA issues!");
  });

  test("should analyze real website console activity and inject custom monitoring", async () => {
    console.log(
      "🧪 REAL SCENARIO: Analyzing GitHub for console activity and adding custom monitoring..."
    );

    await stagehand.page.goto("https://github.com");
    console.log("📄 Navigated to GitHub (real production website)");

    // Wait to capture any existing logs from the site
    await new Promise((resolve) => setTimeout(resolve, 1000));

    // Inject custom monitoring - realistic use case for external agents
    await stagehand.page.evaluate(() => {
      // Custom monitoring that an agent might inject
      console.log("AGENT_MONITOR: Starting GitHub page analysis", {
        url: window.location.href,
        userAgent: navigator.userAgent.substring(0, 50),
        timestamp: new Date().toISOString(),
      });

      // Monitor for any JavaScript errors
      window.addEventListener("error", (e) => {
        console.error("AGENT_ERROR: JavaScript error detected", {
          message: e.message,
          filename: e.filename,
          line: e.lineno,
        });
      });

      // Track performance metrics (using legacy timing API for test purposes)
      const timing = (performance as any).timing;
      console.info("AGENT_PERF: Page timing analysis", {
        domContentLoaded:
          timing.domContentLoadedEventEnd - timing.navigationStart,
        pageLoad: timing.loadEventEnd - timing.navigationStart,
        firstPaint:
          performance.getEntriesByType("paint")[0]?.startTime || "unknown",
      });

      // Monitor network activity
      console.log("AGENT_NETWORK: Monitoring fetch requests");

      // Check for common frameworks/libraries
      const frameworks: string[] = [];
      if ((window as any).jQuery) frameworks.push("jQuery");
      if ((window as any).React) frameworks.push("React");
      if ((window as any).Vue) frameworks.push("Vue");

      console.log("AGENT_FRAMEWORKS: Detected libraries", {
        frameworks: frameworks.length ? frameworks : ["none detected"],
        totalScripts: document.scripts.length,
      });
    });

    console.log("💉 Injected custom monitoring code into GitHub page");

    // Wait for monitoring to collect data
    await new Promise((resolve) => setTimeout(resolve, 500));

    const result = (await call("stagehand_logs", {
      verbose: true,
    })) as CallToolResult;
    const logText = extractLogText(result);
    const logs: ConsoleLog[] = JSON.parse(logText);

    console.log(
      `🔍 CAPTURED ${logs.length} logs from live GitHub page + custom monitoring:`
    );

    // Separate GitHub's logs from our custom monitoring
    const githubLogs = logs.filter((log) => !log.text.includes("AGENT_"));
    const agentLogs = logs.filter((log) => log.text.includes("AGENT_"));

    console.log(`   📊 GitHub native logs: ${githubLogs.length}`);
    console.log(`   🤖 Custom agent logs: ${agentLogs.length}`);

    agentLogs.forEach((log, i) => {
      console.log(
        `   ${i + 1}. [${log.type.toUpperCase()}] ${log.text.substring(
          0,
          100
        )}...`
      );
    });

    // Verify our custom monitoring worked
    expect(agentLogs.length).toBeGreaterThanOrEqual(4);

    const monitorStartLog = agentLogs.find((log) =>
      log.text.includes("AGENT_MONITOR: Starting GitHub")
    );
    const perfLog = agentLogs.find((log) =>
      log.text.includes("AGENT_PERF: Page timing")
    );
    const networkLog = agentLogs.find((log) =>
      log.text.includes("AGENT_NETWORK: Monitoring")
    );
    const frameworkLog = agentLogs.find((log) =>
      log.text.includes("AGENT_FRAMEWORKS: Detected")
    );

    expect(monitorStartLog).toBeDefined();
    expect(perfLog).toBeDefined();
    expect(networkLog).toBeDefined();
    expect(frameworkLog).toBeDefined();

    console.log(
      "✅ SUCCESS: Custom monitoring injected and captured on live website"
    );
    console.log("   🌐 Proved tool works with real production websites");
    console.log("   🤖 Demonstrated external agent monitoring capabilities");
    console.log("   📈 Collected performance and framework detection data");
  });

  test("should demonstrate user behavior tracking and A/B testing log analysis", async () => {
    console.log(
      "🧪 REAL SCENARIO: Tracking user behavior and A/B testing in e-commerce..."
    );

    // Simulate an e-commerce page with A/B testing and user tracking
    const ecommercePage = `
      <html>
        <head><title>ShopApp - Product Page</title></head>
        <body>
          <h1>Premium Headphones</h1>
          <button id="add-to-cart">Add to Cart - $199</button>
          <button id="wishlist">❤️ Add to Wishlist</button>
          <div id="recommendations">Loading recommendations...</div>

          <script>
            // A/B Test initialization
            const testVariant = Math.random() > 0.5 ? 'A' : 'B';
            console.log('AB_TEST: User assigned to variant', {
              variant: testVariant,
              feature: 'checkout_button_color',
              userId: 'user_' + Math.floor(Math.random() * 10000)
            });

            // User session tracking
            console.log('ANALYTICS: Session started', {
              sessionId: 'sess_' + Date.now(),
              referrer: document.referrer || 'direct',
              product: 'premium-headphones-black',
              price: 199
            });

            // Product view tracking
            console.log('ECOMMERCE: Product viewed', {
              productId: 'HEADPHONES_001',
              category: 'Electronics > Audio',
              price: 199,
              inStock: true,
              viewedAt: new Date().toISOString()
            });

            document.getElementById('add-to-cart').addEventListener('click', function() {
              console.log('USER_ACTION: Add to cart clicked', {
                productId: 'HEADPHONES_001',
                quantity: 1,
                price: 199,
                testVariant: testVariant
              });

              // Simulate cart API call
              setTimeout(() => {
                console.log('API: Cart updated successfully', {
                  cartTotal: 199,
                  itemCount: 1,
                  responseTime: '45ms'
                });

                // Track conversion for A/B test
                console.log('AB_TEST: Conversion recorded', {
                  variant: testVariant,
                  event: 'add_to_cart',
                  value: 199
                });

              }, 50);
            });

            document.getElementById('wishlist').addEventListener('click', function() {
              console.log('USER_ACTION: Wishlist clicked', {
                productId: 'HEADPHONES_001',
                action: 'add_to_wishlist'
              });

              // Simulate wishlist error
              setTimeout(() => {
                console.error('WISHLIST_ERROR: Failed to add to wishlist', {
                  error: 'rate_limit_exceeded',
                  userId: 'user_12345',
                  retry: true
                });
                console.warn('WISHLIST: Queuing for retry', {
                  retryIn: '30 seconds',
                  priority: 'low'
                });
              }, 30);
            });

            // Simulate recommendations loading
            setTimeout(() => {
              console.log('RECOMMENDATIONS: Loaded successfully', {
                algorithm: 'collaborative_filtering',
                itemCount: 5,
                loadTime: '120ms',
                personalized: true
              });
            }, 120);
          </script>
        </body>
      </html>
    `;

    await stagehand.page.goto(
      `data:text/html,${encodeURIComponent(ecommercePage)}`
    );
    console.log("📄 Loaded e-commerce product page with A/B testing");

    // Wait for initial logs
    await new Promise((resolve) => setTimeout(resolve, 200));

    // Simulate user interactions
    await stagehand.page.click("#add-to-cart");
    console.log('🛒 Simulated "Add to Cart" click');

    await new Promise((resolve) => setTimeout(resolve, 100));

    await stagehand.page.click("#wishlist");
    console.log('❤️  Simulated "Add to Wishlist" click (will trigger error)');

    // Wait for all async operations
    await new Promise((resolve) => setTimeout(resolve, 300));

    const result = (await call("stagehand_logs", {
      verbose: true,
    })) as CallToolResult;
    const logText = extractLogText(result);
    const logs: ConsoleLog[] = JSON.parse(logText);

    console.log(`🔍 CAPTURED ${logs.length} e-commerce tracking logs:`);

    // Categorize logs by type
    const abTestLogs = logs.filter((log) => log.text.includes("AB_TEST:"));
    const analyticsLogs = logs.filter(
      (log) =>
        log.text.includes("ANALYTICS:") || log.text.includes("ECOMMERCE:")
    );
    const userActionLogs = logs.filter((log) =>
      log.text.includes("USER_ACTION:")
    );
    const errorLogs = logs.filter((log) => log.type === "error");
    const apiLogs = logs.filter((log) => log.text.includes("API:"));

    console.log(`   🧪 A/B Test logs: ${abTestLogs.length}`);
    console.log(`   📊 Analytics logs: ${analyticsLogs.length}`);
    console.log(`   👤 User action logs: ${userActionLogs.length}`);
    console.log(`   ❌ Error logs: ${errorLogs.length}`);
    console.log(`   🔌 API logs: ${apiLogs.length}`);

    // Verify we captured realistic e-commerce scenarios
    expect(logs.length).toBeGreaterThan(8);
    expect(abTestLogs.length).toBeGreaterThanOrEqual(2);
    expect(userActionLogs.length).toBeGreaterThanOrEqual(2);
    expect(errorLogs.length).toBeGreaterThanOrEqual(1);

    const variantAssignment = logs.find((log) =>
      log.text.includes("User assigned to variant")
    );
    const productView = logs.find((log) => log.text.includes("Product viewed"));
    const addToCart = logs.find((log) =>
      log.text.includes("Add to cart clicked")
    );
    const wishlistError = logs.find((log) =>
      log.text.includes("Failed to add to wishlist")
    );
    const conversion = logs.find((log) =>
      log.text.includes("Conversion recorded")
    );

    expect(variantAssignment).toBeDefined();
    expect(productView).toBeDefined();
    expect(addToCart).toBeDefined();
    expect(wishlistError).toBeDefined();
    expect(conversion).toBeDefined();

    console.log(
      "✅ SUCCESS: E-commerce tracking and A/B testing logs captured"
    );
    console.log(
      "   🎯 User behavior, conversions, errors, and API calls tracked"
    );
    console.log(
      "   💼 Perfect for real-world e-commerce debugging and optimization!"
    );
  });

  test("should demonstrate multi-session log management for continuous monitoring", async () => {
    console.log(
      "🧪 REAL SCENARIO: Managing logs across multiple monitoring sessions..."
    );

    // Session 1: Monitor a login flow
    console.log("📱 SESSION 1: Monitoring user login flow...");
    const loginPage = `
      <html><body>
        <form id="login-form">
          <input type="email" placeholder="Email">
          <input type="password" placeholder="Password">
          <button type="submit">Login</button>
        </form>
        <script>
          console.log('AUTH: Login page loaded', { timestamp: Date.now() });
          document.getElementById('login-form').addEventListener('submit', (e) => {
            e.preventDefault();
            console.log('AUTH: Login attempt started', { method: 'email' });
            setTimeout(() => {
              console.log('AUTH: Login successful', { userId: 'user_789', sessionId: 'sess_abc123' });
            }, 100);
          });
        </script>
      </body></html>
    `;

    await stagehand.page.goto(
      `data:text/html,${encodeURIComponent(loginPage)}`
    );
    await stagehand.page.click('button[type="submit"]');
    await new Promise((resolve) => setTimeout(resolve, 200));

    // Check Session 1 logs
    let result = (await call("stagehand_logs", {
      verbose: true,
    })) as CallToolResult;
    let logText = extractLogText(result);
    let logs: ConsoleLog[] = JSON.parse(logText);

    console.log(`   📊 Session 1 captured: ${logs.length} authentication logs`);
    expect(
      logs.some((log: ConsoleLog) =>
        log.text.includes("AUTH: Login page loaded")
      )
    ).toBe(true);
    expect(
      logs.some((log: ConsoleLog) =>
        log.text.includes("AUTH: Login successful")
      )
    ).toBe(true);

    // Clear logs for new session
    console.log("🧹 Clearing logs between monitoring sessions...");
    clearConsoleLogs("default-session-id");

    // Session 2: Monitor dashboard activity
    console.log("📊 SESSION 2: Monitoring dashboard interactions...");
    const dashboardPage = `
      <html><body>
        <div id="dashboard">
          <button id="refresh-data">Refresh</button>
          <div id="charts">Loading charts...</div>
        </div>
        <script>
          console.log('DASHBOARD: Page initialized', { user: 'user_789', widgets: 5 });

          document.getElementById('refresh-data').addEventListener('click', () => {
            console.log('DASHBOARD: Data refresh triggered', { source: 'user_click' });
            setTimeout(() => {
              console.warn('DASHBOARD: Data source slow', { latency: '2.3s', threshold: '1s' });
              console.log('DASHBOARD: Fallback data loaded', { source: 'cache', freshness: '5min' });
            }, 50);
          });

          // Simulate background activity
          setTimeout(() => {
            console.log('DASHBOARD: Real-time update received', { type: 'notification', count: 3 });
          }, 150);
        </script>
      </body></html>
    `;

    await stagehand.page.goto(
      `data:text/html,${encodeURIComponent(dashboardPage)}`
    );
    await stagehand.page.click("#refresh-data");
    await new Promise((resolve) => setTimeout(resolve, 300));

    // Check Session 2 logs (should not include Session 1)
    result = (await call("stagehand_logs", {
      verbose: true,
    })) as CallToolResult;
    logText = extractLogText(result);
    logs = JSON.parse(logText);

    console.log(`   📊 Session 2 captured: ${logs.length} dashboard logs`);
    console.log("   🔍 Verifying session isolation...");

    // Verify no Session 1 logs leaked into Session 2
    const hasAuthLogs = logs.some((log: ConsoleLog) =>
      log.text.includes("AUTH:")
    );
    const hasDashboardLogs = logs.some((log: ConsoleLog) =>
      log.text.includes("DASHBOARD:")
    );

    expect(hasAuthLogs).toBe(false); // Session 1 logs should be cleared
    expect(hasDashboardLogs).toBe(true); // Session 2 logs should be present
    expect(logs.length).toBeGreaterThan(3);

    logs.forEach((log, i) => {
      console.log(
        `   ${i + 1}. [${log.type.toUpperCase()}] ${log.text.substring(
          0,
          80
        )}...`
      );
    });

    // Verify specific dashboard activities
    const dashInit = logs.find((log) =>
      log.text.includes("DASHBOARD: Page initialized")
    );
    const dataRefresh = logs.find((log) =>
      log.text.includes("DASHBOARD: Data refresh triggered")
    );
    const slowWarning = logs.find((log) =>
      log.text.includes("DASHBOARD: Data source slow")
    );
    const realtimeUpdate = logs.find((log) =>
      log.text.includes("DASHBOARD: Real-time update")
    );

    expect(dashInit).toBeDefined();
    expect(dataRefresh).toBeDefined();
    expect(slowWarning).toBeDefined();
    expect(realtimeUpdate).toBeDefined();

    console.log("✅ SUCCESS: Multi-session log management works perfectly");
    console.log(
      "   🎯 Session isolation: Auth logs cleared, dashboard logs captured"
    );
    console.log("   🔄 Perfect for continuous monitoring workflows");
    console.log("   💼 Enables clean separation of monitoring contexts");
  });
});
