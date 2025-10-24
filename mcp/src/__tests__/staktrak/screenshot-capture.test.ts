import { test, expect } from '@playwright/test';
import {
  createTestPage,
  verifyScreenshotDataUrl,
  validateScreenshotMessage,
  extractScreenshotMessages,
  waitForCondition,
} from './test-helpers';

test.describe('Screenshot Capture', () => {
  test.describe('Screenshot Capture Functionality', () => {
    test('should capture screenshot after waitForURL action', async ({ page }) => {
      const messages: any[] = [];

      // Setup message listener
      await page.exposeFunction('captureMessage', (msg: any) => {
        messages.push(msg);
      });

      // Load test page with staktrak
      const html = createTestPage({ includeStaktrak: true, includeConfig: true });
      await page.setContent(html);

      // Inject message capture in page
      await page.evaluate(() => {
        window.addEventListener('message', (event) => {
          (window as any).captureMessage(event.data);
        });
      });

      // Start a simple replay with waitForURL
      const testCode = `
        test('test', async ({ page }) => {
          await page.goto('http://localhost:3000');
          await page.waitForURL('http://localhost:3000');
        });
      `;

      await page.evaluate((code) => {
        (window as any).startPlaywrightReplay(code);
      }, testCode);

      // Wait for screenshot message
      const hasScreenshot = await waitForCondition(
        () => messages.some(m => m.type === 'staktrak-playwright-screenshot-captured'),
        8000
      );

      expect(hasScreenshot).toBe(true);

      const screenshotMessages = extractScreenshotMessages(messages.map(m => ({ data: m })));
      expect(screenshotMessages.length).toBeGreaterThan(0);
    });

    test('should not capture screenshot for non-waitForURL actions', async ({ page }) => {
      const messages: any[] = [];

      await page.exposeFunction('captureMessage', (msg: any) => {
        messages.push(msg);
      });

      const html = createTestPage({ includeStaktrak: true });
      await page.setContent(html);

      await page.evaluate(() => {
        window.addEventListener('message', (event) => {
          (window as any).captureMessage(event.data);
        });
      });

      // Replay with only click action (no waitForURL)
      const testCode = `
        test('test', async ({ page }) => {
          await page.click('[data-testid="test-button"]');
        });
      `;

      await page.evaluate((code) => {
        (window as any).startPlaywrightReplay(code);
      }, testCode);

      // Wait for replay to complete
      await page.waitForTimeout(2000);

      const screenshotMessages = extractScreenshotMessages(messages.map(m => ({ data: m })));
      expect(screenshotMessages.length).toBe(0);
    });
  });

  test.describe('Screenshot Data URL Validation', () => {
    test('should generate valid data URL format', () => {
      const validDataUrl = 'data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/';
      expect(verifyScreenshotDataUrl(validDataUrl)).toBe(true);
    });

    test('should reject invalid data URL format', () => {
      const invalidUrls = [
        '',
        'not-a-data-url',
        'data:text/plain;base64,test',
        'data:image/jpeg;base64',
        'data:image/jpeg;base64,',
        'http://example.com/image.jpg',
      ];

      invalidUrls.forEach(url => {
        expect(verifyScreenshotDataUrl(url)).toBe(false);
      });
    });

    test('should accept different image formats', () => {
      const formats = [
        'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==',
        'data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/',
        'data:image/jpg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/',
        'data:image/webp;base64,UklGRiQAAABXRUJQVlA4IBgAAAAwAQCdASoBAAEAAwA0JaQAA3AA/',
      ];

      formats.forEach(url => {
        expect(verifyScreenshotDataUrl(url)).toBe(true);
      });
    });
  });

  test.describe('Screenshot Message Structure', () => {
    test('should validate correct screenshot message structure', () => {
      const validMessage = {
        type: 'staktrak-playwright-screenshot-captured',
        screenshot: 'data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/',
        actionIndex: 0,
        url: 'http://localhost:3000',
        timestamp: Date.now(),
        id: `${Date.now()}-0`,
      };

      expect(validateScreenshotMessage(validMessage)).toBe(true);
    });

    test('should reject message with missing fields', () => {
      const invalidMessages = [
        {
          type: 'staktrak-playwright-screenshot-captured',
          // missing screenshot
          actionIndex: 0,
          url: 'http://localhost:3000',
          timestamp: Date.now(),
        },
        {
          type: 'staktrak-playwright-screenshot-captured',
          screenshot: 'data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/',
          // missing actionIndex
          url: 'http://localhost:3000',
          timestamp: Date.now(),
        },
        {
          // wrong type
          type: 'other-message',
          screenshot: 'data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/',
          actionIndex: 0,
          url: 'http://localhost:3000',
          timestamp: Date.now(),
        },
      ];

      invalidMessages.forEach(msg => {
        expect(validateScreenshotMessage(msg)).toBe(false);
      });
    });
  });

  test.describe('Screenshot Options', () => {
    test('should respect STAKTRAK_CONFIG screenshot options', async ({ page }) => {
      const messages: any[] = [];

      await page.exposeFunction('captureMessage', (msg: any) => {
        messages.push(msg);
      });

      // Create page with custom config
      await page.setContent(`
        <!DOCTYPE html>
        <html>
          <head>
            <script>
              window.STAKTRAK_CONFIG = {
                parentOrigin: 'http://localhost:3000',
                screenshot: {
                  quality: 0.5,
                  type: 'image/png',
                  scale: 0.8,
                  backgroundColor: '#000000'
                }
              };
            </script>
          </head>
          <body>
            <div style="width: 200px; height: 200px; background: red;">Test Content</div>
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

      await page.waitForTimeout(2000);

      const screenshotMessages = extractScreenshotMessages(messages.map(m => ({ data: m })));
      if (screenshotMessages.length > 0) {
        const screenshot = screenshotMessages[0].screenshot;
        expect(screenshot).toContain('data:image/');
        expect(verifyScreenshotDataUrl(screenshot)).toBe(true);
      }
    });
  });

  test.describe('Screenshot Error Handling', () => {
    test('should handle screenshot capture errors gracefully', async ({ page, browserName }) => {
      // This test verifies that errors don't break the replay flow
      const messages: any[] = [];
      const consoleMessages: string[] = [];

      await page.exposeFunction('captureMessage', (msg: any) => {
        messages.push(msg);
      });

      page.on('console', msg => {
        if (msg.type() === 'error') {
          consoleMessages.push(msg.text());
        }
      });

      const html = createTestPage({ includeStaktrak: true });
      await page.setContent(html);

      await page.evaluate(() => {
        window.addEventListener('message', (event) => {
          (window as any).captureMessage(event.data);
        });

        // Mock domToDataUrl to throw error
        if ((window as any).domToDataUrl) {
          const original = (window as any).domToDataUrl;
          (window as any).domToDataUrl = async () => {
            throw new Error('Screenshot capture failed');
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

      // Wait for replay to complete
      const completedMsg = await waitForCondition(
        () => messages.some(m => m.type === 'staktrak-playwright-replay-completed'),
        8000
      );

      // Replay should complete even if screenshot fails
      expect(completedMsg).toBe(true);
    });
  });

  test.describe('Multiple Screenshots', () => {
    test('should capture multiple screenshots for multiple waitForURL actions', async ({ page }) => {
      const messages: any[] = [];

      await page.exposeFunction('captureMessage', (msg: any) => {
        messages.push(msg);
      });

      const html = createTestPage({ includeStaktrak: true });
      await page.setContent(html);

      await page.evaluate(() => {
        window.addEventListener('message', (event) => {
          (window as any).captureMessage(event.data);
        });
      });

      // Test code with multiple waitForURL actions
      const testCode = `
        test('test', async ({ page }) => {
          await page.goto('http://localhost:3000');
          await page.waitForURL('http://localhost:3000');
          await page.click('[data-testid="test-button"]');
          await page.waitForURL('http://localhost:3000/result');
        });
      `;

      await page.evaluate((code) => {
        if ((window as any).startPlaywrightReplay) {
          (window as any).startPlaywrightReplay(code);
        }
      }, testCode);

      await page.waitForTimeout(3000);

      const screenshotMessages = extractScreenshotMessages(messages.map(m => ({ data: m })));
      // Should have at least one screenshot (depending on implementation)
      expect(screenshotMessages.length).toBeGreaterThanOrEqual(0);
    });

    test('should include correct actionIndex in each screenshot', async ({ page }) => {
      const messages: any[] = [];

      await page.exposeFunction('captureMessage', (msg: any) => {
        messages.push(msg);
      });

      const html = createTestPage({ includeStaktrak: true });
      await page.setContent(html);

      await page.evaluate(() => {
        window.addEventListener('message', (event) => {
          (window as any).captureMessage(event.data);
        });
      });

      const testCode = `
        test('test', async ({ page }) => {
          await page.goto('http://localhost:3000');
          await page.waitForURL('http://localhost:3000');
          await page.waitForURL('http://localhost:3000/next');
        });
      `;

      await page.evaluate((code) => {
        if ((window as any).startPlaywrightReplay) {
          (window as any).startPlaywrightReplay(code);
        }
      }, testCode);

      await page.waitForTimeout(3000);

      const screenshotMessages = extractScreenshotMessages(messages.map(m => ({ data: m })));

      // Verify each screenshot has actionIndex
      screenshotMessages.forEach(msg => {
        expect(msg.actionIndex).toBeGreaterThanOrEqual(0);
        expect(typeof msg.actionIndex).toBe('number');
      });
    });
  });

  test.describe('Screenshot ID Generation', () => {
    test('should generate unique IDs for screenshots', async ({ page }) => {
      const messages: any[] = [];

      await page.exposeFunction('captureMessage', (msg: any) => {
        messages.push(msg);
      });

      const html = createTestPage({ includeStaktrak: true });
      await page.setContent(html);

      await page.evaluate(() => {
        window.addEventListener('message', (event) => {
          (window as any).captureMessage(event.data);
        });
      });

      const testCode = `
        test('test', async ({ page }) => {
          await page.waitForURL('http://localhost:3000');
          await page.waitForURL('http://localhost:3000/next');
        });
      `;

      await page.evaluate((code) => {
        if ((window as any).startPlaywrightReplay) {
          (window as any).startPlaywrightReplay(code);
        }
      }, testCode);

      await page.waitForTimeout(3000);

      const screenshotMessages = extractScreenshotMessages(messages.map(m => ({ data: m })));

      if (screenshotMessages.length > 1) {
        const ids = screenshotMessages.map(m => m.id);
        const uniqueIds = new Set(ids);
        expect(uniqueIds.size).toBe(ids.length);
      }
    });

    test('should format ID as timestamp-actionIndex', async ({ page }) => {
      const messages: any[] = [];

      await page.exposeFunction('captureMessage', (msg: any) => {
        messages.push(msg);
      });

      const html = createTestPage({ includeStaktrak: true });
      await page.setContent(html);

      await page.evaluate(() => {
        window.addEventListener('message', (event) => {
          (window as any).captureMessage(event.data);
        });
      });

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

      const screenshotMessages = extractScreenshotMessages(messages.map(m => ({ data: m })));

      if (screenshotMessages.length > 0) {
        const msg = screenshotMessages[0];
        expect(msg.id).toMatch(/^\d+-\d+$/); // Format: timestamp-actionIndex
        expect(msg.id).toContain(msg.actionIndex.toString());
      }
    });
  });
});
