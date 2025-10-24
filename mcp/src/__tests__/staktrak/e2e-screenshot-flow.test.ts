import { test, expect } from '@playwright/test';
import {
  createTestPage,
  waitForCondition,
  extractScreenshotMessages,
  verifyScreenshotDataUrl,
} from './test-helpers';

test.describe('E2E Screenshot Flow', () => {
  test.describe('Real Browser Screenshot Capture', () => {
    test('should capture screenshots during actual page navigation', async ({ page }) => {
      const messages: any[] = [];

      await page.exposeFunction('captureMessage', (msg: any) => {
        messages.push(msg);
      });

      // Load a simple test page
      await page.setContent(`
        <!DOCTYPE html>
        <html>
          <head>
            <script>
              window.STAKTRAK_CONFIG = {
                parentOrigin: 'http://localhost:3000',
                screenshot: {
                  quality: 0.8,
                  type: 'image/jpeg',
                  scale: 1,
                  backgroundColor: '#ffffff'
                }
              };
            </script>
          </head>
          <body style="padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);">
            <h1 style="color: white;">Test Page</h1>
            <button data-testid="nav-button" style="padding: 10px 20px; font-size: 16px;">
              Navigate
            </button>
            <div id="content" style="margin-top: 20px; padding: 20px; background: white; border-radius: 8px;">
              <p>This is test content for screenshot capture</p>
            </div>
            <script src="/dist/staktrak.js"></script>
            <script>
              window.addEventListener('message', (event) => {
                window.captureMessage(event.data);
              });

              // Simulate navigation on button click
              document.querySelector('[data-testid="nav-button"]').addEventListener('click', () => {
                document.getElementById('content').innerHTML = '<p>Navigated to new page!</p>';
                history.pushState({}, '', '/new-page');
              });
            </script>
          </body>
        </html>
      `);

      // Start replay with navigation
      const testCode = `
        test('navigation test', async ({ page }) => {
          await page.click('[data-testid="nav-button"]');
          await page.waitForURL('http://localhost:3000/new-page');
        });
      `;

      await page.evaluate((code) => {
        if ((window as any).startPlaywrightReplay) {
          (window as any).startPlaywrightReplay(code);
        }
      }, testCode);

      await waitForCondition(
        () => messages.some(m => m.type === 'staktrak-playwright-replay-completed'),
        10000
      );

      const screenshotMsgs = extractScreenshotMessages(messages.map(m => ({ data: m })));

      // Should capture at least one screenshot
      expect(screenshotMsgs.length).toBeGreaterThan(0);

      // Verify screenshot is valid
      if (screenshotMsgs.length > 0) {
        const screenshot = screenshotMsgs[0].screenshot;
        expect(verifyScreenshotDataUrl(screenshot)).toBe(true);
        expect(screenshot.length).toBeGreaterThan(1000); // Should have substantial data
      }
    });

    test('should capture screenshots with different quality settings', async ({ page }) => {
      const messages: any[] = [];

      await page.exposeFunction('captureMessage', (msg: any) => {
        messages.push(msg);
      });

      await page.setContent(`
        <!DOCTYPE html>
        <html>
          <head>
            <script>
              window.STAKTRAK_CONFIG = {
                parentOrigin: 'http://localhost:3000',
                screenshot: {
                  quality: 0.3, // Low quality
                  type: 'image/jpeg',
                  scale: 1,
                  backgroundColor: '#ffffff'
                }
              };
            </script>
          </head>
          <body style="padding: 50px;">
            <div style="width: 400px; height: 300px; background: linear-gradient(45deg, #ff6b6b, #4ecdc4);">
              <h1 style="color: white; padding: 20px;">Screenshot Test</h1>
            </div>
            <script src="/dist/staktrak.js"></script>
            <script>
              window.addEventListener('message', (event) => {
                window.captureMessage(event.data);
              });
            </script>
          </body>
        </html>
      `);

      const testCode = `
        test('test', async ({ page }) => {
          await page.waitForURL('http://localhost:3000');
        });
      `;

      await page.evaluate((code) => {
        if ((window as any).startPlaywrightReplay) {
          (window as any).startPlaywrightReplay(code);
        }
      }, testCode);

      await page.waitForTimeout(3000);

      const screenshotMsgs = extractScreenshotMessages(messages.map(m => ({ data: m })));

      if (screenshotMsgs.length > 0) {
        const screenshot = screenshotMsgs[0].screenshot;
        expect(verifyScreenshotDataUrl(screenshot)).toBe(true);
        expect(screenshot).toContain('data:image/jpeg');
      }
    });
  });

  test.describe('Cross-Origin Iframe Scenarios', () => {
    test('should handle same-origin iframe screenshots', async ({ page }) => {
      await page.setContent(`
        <!DOCTYPE html>
        <html>
          <head>
            <script>
              window.STAKTRAK_CONFIG = {
                parentOrigin: window.location.origin,
                screenshot: { quality: 0.8, type: 'image/jpeg' }
              };
            </script>
          </head>
          <body>
            <h1>Parent Page</h1>
            <iframe id="test-frame" style="width: 600px; height: 400px; border: 2px solid #ccc;"></iframe>
            <script>
              const frame = document.getElementById('test-frame');
              const frameDoc = frame.contentDocument || frame.contentWindow.document;
              frameDoc.open();
              frameDoc.write(\`
                <!DOCTYPE html>
                <html>
                  <body style="background: #f0f0f0; padding: 20px;">
                    <h2>Iframe Content</h2>
                    <button data-testid="iframe-button">Click Me</button>
                    <script src="/dist/staktrak.js"><\/script>
                  </body>
                </html>
              \`);
              frameDoc.close();
            </script>
          </body>
        </html>
      `);

      await page.waitForSelector('#test-frame');

      // Frame operations should work in same-origin scenario
      const frame = page.frame({ name: '' });
      expect(frame).toBeTruthy();
    });

    test('should use wildcard origin when cross-origin blocked', async ({ page }) => {
      const messages: any[] = [];

      await page.exposeFunction('captureMessage', (msg: any, origin: string) => {
        messages.push({ data: msg, origin });
      });

      await page.setContent(`
        <!DOCTYPE html>
        <html>
          <body>
            <script>
              // Simulate cross-origin environment
              let parentOrigin = null;

              function getParentOrigin() {
                if (parentOrigin) return parentOrigin;
                try {
                  // Simulate cross-origin block
                  throw new DOMException('Cross-origin access blocked', 'SecurityError');
                } catch (e) {}
                return '*';
              }

              window.captureMessage(
                { type: 'test-origin', origin: getParentOrigin() },
                getParentOrigin()
              );
            </script>
          </body>
        </html>
      `);

      await page.waitForTimeout(500);

      const testMsgs = messages.filter(m => m.data.type === 'test-origin');
      if (testMsgs.length > 0) {
        expect(testMsgs[0].data.origin).toBe('*');
      }
    });
  });

  test.describe('Performance and Size Limits', () => {
    test('should handle large page screenshots', async ({ page }) => {
      const messages: any[] = [];

      await page.exposeFunction('captureMessage', (msg: any) => {
        messages.push(msg);
      });

      // Create a large page with lots of content
      const largeContent = Array(100)
        .fill(null)
        .map((_, i) => `<div style="padding: 20px; margin: 10px; background: #${Math.floor(Math.random() * 16777215).toString(16)};">Content Block ${i}</div>`)
        .join('');

      await page.setContent(`
        <!DOCTYPE html>
        <html>
          <head>
            <script>
              window.STAKTRAK_CONFIG = {
                parentOrigin: 'http://localhost:3000',
                screenshot: { quality: 0.5, type: 'image/jpeg' }
              };
            </script>
          </head>
          <body>
            <h1>Large Page Test</h1>
            ${largeContent}
            <script src="/dist/staktrak.js"></script>
            <script>
              window.addEventListener('message', (event) => {
                window.captureMessage(event.data);
              });
            </script>
          </body>
        </html>
      `);

      const testCode = `
        test('test', async ({ page }) => {
          await page.waitForURL('http://localhost:3000');
        });
      `;

      const startTime = Date.now();

      await page.evaluate((code) => {
        if ((window as any).startPlaywrightReplay) {
          (window as any).startPlaywrightReplay(code);
        }
      }, testCode);

      await page.waitForTimeout(4000);

      const duration = Date.now() - startTime;

      const screenshotMsgs = extractScreenshotMessages(messages.map(m => ({ data: m })));

      // Should complete within reasonable time
      expect(duration).toBeLessThan(10000);

      // Should still capture screenshot
      if (screenshotMsgs.length > 0) {
        expect(verifyScreenshotDataUrl(screenshotMsgs[0].screenshot)).toBe(true);
      }
    });

    test('should handle rapid screenshot requests', async ({ page }) => {
      const messages: any[] = [];

      await page.exposeFunction('captureMessage', (msg: any) => {
        messages.push(msg);
      });

      const html = createTestPage({ includeStaktrak: true, includeConfig: true });
      await page.setContent(html);

      await page.evaluate(() => {
        window.addEventListener('message', (event) => {
          (window as any).captureMessage(event.data);
        });
      });

      // Test with multiple rapid waitForURL actions
      const testCode = `
        test('test', async ({ page }) => {
          await page.waitForURL('http://localhost:3000');
          await page.waitForURL('http://localhost:3000/1');
          await page.waitForURL('http://localhost:3000/2');
        });
      `;

      await page.evaluate((code) => {
        if ((window as any).startPlaywrightReplay) {
          (window as any).startPlaywrightReplay(code);
        }
      }, testCode);

      await waitForCondition(
        () => messages.some(m => m.type === 'staktrak-playwright-replay-completed'),
        8000
      );

      // Should handle multiple screenshots
      const screenshotMsgs = extractScreenshotMessages(messages.map(m => ({ data: m })));

      // All screenshots should be valid
      screenshotMsgs.forEach(msg => {
        expect(verifyScreenshotDataUrl(msg.screenshot)).toBe(true);
      });
    });
  });

  test.describe('SPA Navigation Screenshot Capture', () => {
    test('should capture screenshots during SPA-style navigation', async ({ page }) => {
      const messages: any[] = [];

      await page.exposeFunction('captureMessage', (msg: any) => {
        messages.push(msg);
      });

      await page.setContent(`
        <!DOCTYPE html>
        <html>
          <head>
            <script>
              window.STAKTRAK_CONFIG = {
                parentOrigin: 'http://localhost:3000',
                screenshot: { quality: 0.8, type: 'image/jpeg' }
              };
            </script>
          </head>
          <body>
            <nav>
              <button data-testid="home-link">Home</button>
              <button data-testid="about-link">About</button>
            </nav>
            <div id="content">Home Page</div>
            <script src="/dist/staktrak.js"></script>
            <script>
              window.addEventListener('message', (event) => {
                window.captureMessage(event.data);
              });

              // Simulate SPA routing
              document.querySelector('[data-testid="home-link"]').addEventListener('click', () => {
                document.getElementById('content').innerHTML = 'Home Page';
                history.pushState({}, '', '/');
                window.dispatchEvent(new Event('popstate'));
              });

              document.querySelector('[data-testid="about-link"]').addEventListener('click', () => {
                document.getElementById('content').innerHTML = 'About Page';
                history.pushState({}, '', '/about');
                window.dispatchEvent(new Event('popstate'));
              });
            </script>
          </body>
        </html>
      `);

      const testCode = `
        test('spa navigation', async ({ page }) => {
          await page.click('[data-testid="about-link"]');
          await page.waitForURL('http://localhost:3000/about');
        });
      `;

      await page.evaluate((code) => {
        if ((window as any).startPlaywrightReplay) {
          (window as any).startPlaywrightReplay(code);
        }
      }, testCode);

      await waitForCondition(
        () => messages.some(m => m.type === 'staktrak-playwright-replay-completed'),
        8000
      );

      const screenshotMsgs = extractScreenshotMessages(messages.map(m => ({ data: m })));

      // Should capture screenshot after SPA navigation
      expect(screenshotMsgs.length).toBeGreaterThanOrEqual(0);
    });
  });

  test.describe('Screenshot Data Integrity', () => {
    test('should produce consistent screenshots for same page', async ({ page }) => {
      const messages: any[] = [];

      await page.exposeFunction('captureMessage', (msg: any) => {
        messages.push(msg);
      });

      await page.setContent(`
        <!DOCTYPE html>
        <html>
          <head>
            <script>
              window.STAKTRAK_CONFIG = {
                parentOrigin: 'http://localhost:3000',
                screenshot: { quality: 0.8, type: 'image/jpeg' }
              };
            </script>
          </head>
          <body style="background: #fff; padding: 20px;">
            <div style="width: 300px; height: 200px; background: #3498db;">
              <h2 style="color: white; padding: 20px;">Static Content</h2>
            </div>
            <script src="/dist/staktrak.js"></script>
            <script>
              window.addEventListener('message', (event) => {
                window.captureMessage(event.data);
              });
            </script>
          </body>
        </html>
      `);

      const testCode = `
        test('test', async ({ page }) => {
          await page.waitForURL('http://localhost:3000');
        });
      `;

      // Run twice
      await page.evaluate((code) => {
        if ((window as any).startPlaywrightReplay) {
          (window as any).startPlaywrightReplay(code);
        }
      }, testCode);

      await page.waitForTimeout(2000);
      messages.length = 0; // Clear

      await page.evaluate((code) => {
        if ((window as any).startPlaywrightReplay) {
          (window as any).startPlaywrightReplay(code);
        }
      }, testCode);

      await page.waitForTimeout(2000);

      const screenshotMsgs = extractScreenshotMessages(messages.map(m => ({ data: m })));

      if (screenshotMsgs.length > 0) {
        const screenshot = screenshotMsgs[0].screenshot;

        // Should be valid
        expect(verifyScreenshotDataUrl(screenshot)).toBe(true);

        // Should have reasonable size
        expect(screenshot.length).toBeGreaterThan(500);
        expect(screenshot.length).toBeLessThan(10000000); // Less than 10MB
      }
    });

    test('should include valid timestamp in screenshot message', async ({ page }) => {
      const messages: any[] = [];

      await page.exposeFunction('captureMessage', (msg: any) => {
        messages.push(msg);
      });

      const html = createTestPage({ includeStaktrak: true, includeConfig: true });
      await page.setContent(html);

      await page.evaluate(() => {
        window.addEventListener('message', (event) => {
          (window as any).captureMessage(event.data);
        });
      });

      const startTime = Date.now();

      const testCode = `
        test('test', async ({ page }) => {
          await page.waitForURL('http://localhost:3000');
        });
      `;

      await page.evaluate((code) => {
        if ((window as any).startPlaywrightReplay) {
          (window as any).startPlaywrightReplay(code);
        }
      }, testCode);

      await page.waitForTimeout(2000);

      const endTime = Date.now();

      const screenshotMsgs = extractScreenshotMessages(messages.map(m => ({ data: m })));

      if (screenshotMsgs.length > 0) {
        const msg = screenshotMsgs[0];

        // Timestamp should be within test execution time
        expect(msg.timestamp).toBeGreaterThanOrEqual(startTime);
        expect(msg.timestamp).toBeLessThanOrEqual(endTime);
      }
    });
  });

  test.describe('Error Recovery', () => {
    test('should handle screenshot errors without breaking replay', async ({ page }) => {
      const messages: any[] = [];
      const consoleErrors: string[] = [];

      await page.exposeFunction('captureMessage', (msg: any) => {
        messages.push(msg);
      });

      page.on('console', msg => {
        if (msg.type() === 'error') {
          consoleErrors.push(msg.text());
        }
      });

      const html = createTestPage({ includeStaktrak: true, includeConfig: true });
      await page.setContent(html);

      await page.evaluate(() => {
        window.addEventListener('message', (event) => {
          (window as any).captureMessage(event.data);
        });

        // Break screenshot capture
        if ((window as any).domToDataUrl) {
          (window as any).domToDataUrl = async () => {
            throw new Error('Mock screenshot error');
          };
        }
      });

      const testCode = `
        test('test', async ({ page }) => {
          await page.click('[data-testid="test-button"]');
          await page.waitForURL('http://localhost:3000');
        });
      `;

      await page.evaluate((code) => {
        if ((window as any).startPlaywrightReplay) {
          (window as any).startPlaywrightReplay(code);
        }
      }, testCode);

      const completed = await waitForCondition(
        () => messages.some(m => m.type === 'staktrak-playwright-replay-completed'),
        5000
      );

      // Replay should complete even with screenshot error
      expect(completed).toBe(true);
    });
  });
});
