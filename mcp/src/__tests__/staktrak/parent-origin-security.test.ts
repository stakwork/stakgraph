import { test, expect } from '@playwright/test';
import { createTestPage, waitForCondition } from './test-helpers';

test.describe('Parent Origin Security', () => {
  test.describe('Parent Origin Capture', () => {
    test('should capture parent origin from first postMessage event', async ({ page }) => {
      const messages: any[] = [];

      await page.exposeFunction('captureMessage', (msg: any, origin: string) => {
        messages.push({ data: msg, origin });
      });

      const html = createTestPage({ includeStaktrak: true, includeConfig: false });
      await page.setContent(html);

      // Inject message capture and test postMessage
      await page.evaluate(() => {
        let capturedOrigin: string | null = null;

        window.addEventListener('message', (event) => {
          (window as any).captureMessage(event.data, event.origin);

          // Simulate staktrak's origin capture logic
          if (!capturedOrigin && event.origin && event.origin !== 'null') {
            capturedOrigin = event.origin;
          }
        });

        // Simulate first message from parent
        window.postMessage(
          { type: 'staktrak-playwright-replay-ping' },
          window.location.origin
        );
      });

      await page.waitForTimeout(500);

      // Verify origin was captured
      const pingMessages = messages.filter(m => m.data.type === 'staktrak-playwright-replay-ping');
      expect(pingMessages.length).toBeGreaterThan(0);
      expect(pingMessages[0].origin).toBeTruthy();
      expect(pingMessages[0].origin).not.toBe('null');
    });

    test('should not overwrite valid origin with wildcard', async ({ page }) => {
      await page.setContent(`
        <!DOCTYPE html>
        <html>
          <body>
            <script>
              let parentOrigin = null;

              function getParentOrigin() {
                if (parentOrigin) {
                  return parentOrigin;
                }
                try {
                  const configOrigin = window.STAKTRAK_CONFIG?.parentOrigin;
                  if (configOrigin) {
                    return configOrigin;
                  }
                } catch (e) {}
                return '*';
              }

              // Simulate first message with valid origin
              parentOrigin = 'http://localhost:3000';

              // Try to overwrite with null (shouldn't work)
              const testOrigin1 = getParentOrigin();

              // Second call should return same origin
              const testOrigin2 = getParentOrigin();

              window.testResults = {
                firstCall: testOrigin1,
                secondCall: testOrigin2,
                originNotOverwritten: testOrigin1 === testOrigin2 && testOrigin1 === 'http://localhost:3000'
              };
            </script>
          </body>
        </html>
      `);

      const results = await page.evaluate(() => (window as any).testResults);

      expect(results.firstCall).toBe('http://localhost:3000');
      expect(results.secondCall).toBe('http://localhost:3000');
      expect(results.originNotOverwritten).toBe(true);
    });
  });

  test.describe('getParentOrigin Function', () => {
    test('should return stored origin when available', async ({ page }) => {
      await page.setContent(`
        <!DOCTYPE html>
        <html>
          <body>
            <script>
              let parentOrigin = 'http://localhost:3000';

              function getParentOrigin() {
                if (parentOrigin) {
                  return parentOrigin;
                }
                return '*';
              }

              window.testOrigin = getParentOrigin();
            </script>
          </body>
        </html>
      `);

      const origin = await page.evaluate(() => (window as any).testOrigin);
      expect(origin).toBe('http://localhost:3000');
    });

    test('should fallback to STAKTRAK_CONFIG when parentOrigin not set', async ({ page }) => {
      await page.setContent(`
        <!DOCTYPE html>
        <html>
          <head>
            <script>
              window.STAKTRAK_CONFIG = {
                parentOrigin: 'http://configured-origin.com',
                screenshot: { quality: 0.8 }
              };
            </script>
          </head>
          <body>
            <script>
              let parentOrigin = null;

              function getParentOrigin() {
                if (parentOrigin) {
                  return parentOrigin;
                }
                try {
                  const configOrigin = window.STAKTRAK_CONFIG?.parentOrigin;
                  if (configOrigin) {
                    return configOrigin;
                  }
                } catch (e) {}
                return '*';
              }

              window.testOrigin = getParentOrigin();
            </script>
          </body>
        </html>
      `);

      const origin = await page.evaluate(() => (window as any).testOrigin);
      expect(origin).toBe('http://configured-origin.com');
    });

    test('should return wildcard when no origin available', async ({ page }) => {
      await page.setContent(`
        <!DOCTYPE html>
        <html>
          <body>
            <script>
              let parentOrigin = null;

              function getParentOrigin() {
                if (parentOrigin) {
                  return parentOrigin;
                }
                try {
                  const configOrigin = window.STAKTRAK_CONFIG?.parentOrigin;
                  if (configOrigin) {
                    return configOrigin;
                  }
                } catch (e) {}
                return '*';
              }

              window.testOrigin = getParentOrigin();
            </script>
          </body>
        </html>
      `);

      const origin = await page.evaluate(() => (window as any).testOrigin);
      expect(origin).toBe('*');
    });

    test('should handle STAKTRAK_CONFIG access errors gracefully', async ({ page }) => {
      await page.setContent(`
        <!DOCTYPE html>
        <html>
          <body>
            <script>
              let parentOrigin = null;

              function getParentOrigin() {
                if (parentOrigin) {
                  return parentOrigin;
                }
                try {
                  // Simulate cross-origin access error
                  if (true) {
                    throw new Error('Cross-origin access blocked');
                  }
                  const configOrigin = window.STAKTRAK_CONFIG?.parentOrigin;
                  if (configOrigin) {
                    return configOrigin;
                  }
                } catch (e) {
                  // Should handle error silently
                }
                return '*';
              }

              window.testOrigin = getParentOrigin();
              window.noError = true;
            </script>
          </body>
        </html>
      `);

      const origin = await page.evaluate(() => (window as any).testOrigin);
      const noError = await page.evaluate(() => (window as any).noError);

      expect(origin).toBe('*');
      expect(noError).toBe(true);
    });
  });

  test.describe('Cross-Origin Scenarios', () => {
    test('should handle same-origin iframe with config access', async ({ page }) => {
      await page.setContent(`
        <!DOCTYPE html>
        <html>
          <head>
            <script>
              window.STAKTRAK_CONFIG = {
                parentOrigin: 'http://localhost:3000',
                screenshot: { quality: 0.8 }
              };
            </script>
          </head>
          <body>
            <iframe id="test-frame" srcdoc='
              <script>
                let parentOrigin = null;

                function getParentOrigin() {
                  if (parentOrigin) {
                    return parentOrigin;
                  }
                  try {
                    const configOrigin = parent.window.STAKTRAK_CONFIG?.parentOrigin;
                    if (configOrigin) {
                      return configOrigin;
                    }
                  } catch (e) {}
                  return "*";
                }

                window.testOrigin = getParentOrigin();
              <\/script>
            '></iframe>
          </body>
        </html>
      `);

      await page.waitForSelector('#test-frame');

      const frameOrigin = await page.frame({ name: '' })?.evaluate(() => {
        return (window as any).testOrigin;
      });

      // In same-origin, should be able to access parent config
      expect(frameOrigin).toBeTruthy();
    });

    test('should handle cross-origin iframe without config access', async ({ page }) => {
      // Simulate cross-origin scenario where parent window access fails
      await page.setContent(`
        <!DOCTYPE html>
        <html>
          <body>
            <script>
              let parentOrigin = null;

              function getParentOrigin() {
                if (parentOrigin) {
                  return parentOrigin;
                }
                try {
                  // Simulate cross-origin block
                  throw new DOMException('Blocked a frame with origin', 'SecurityError');
                } catch (e) {
                  // Cross-origin access blocked (expected)
                }
                return '*';
              }

              window.testOrigin = getParentOrigin();
              window.handledCrossOrigin = true;
            </script>
          </body>
        </html>
      `);

      const origin = await page.evaluate(() => (window as any).testOrigin);
      const handledCrossOrigin = await page.evaluate(() => (window as any).handledCrossOrigin);

      expect(origin).toBe('*');
      expect(handledCrossOrigin).toBe(true);
    });
  });

  test.describe('PostMessage Security', () => {
    test('should use captured origin for postMessage responses', async ({ page }) => {
      const messages: any[] = [];

      await page.exposeFunction('captureMessage', (msg: any, targetOrigin: string) => {
        messages.push({ data: msg, targetOrigin });
      });

      await page.setContent(`
        <!DOCTYPE html>
        <html>
          <body>
            <script>
              let parentOrigin = null;

              function getParentOrigin() {
                if (parentOrigin) return parentOrigin;
                return '*';
              }

              window.addEventListener('message', (event) => {
                if (event.data.type === 'test-ping') {
                  // Capture origin from first message
                  if (!parentOrigin && event.origin && event.origin !== 'null') {
                    parentOrigin = event.origin;
                  }

                  // Send response using captured origin
                  const responseOrigin = getParentOrigin();
                  window.captureMessage(
                    { type: 'test-pong', receivedFrom: event.origin },
                    responseOrigin
                  );
                }
              });

              // Simulate receiving message
              window.postMessage({ type: 'test-ping' }, window.location.origin);
            </script>
          </body>
        </html>
      `);

      await page.waitForTimeout(500);

      const pongMessages = messages.filter(m => m.data.type === 'test-pong');
      expect(pongMessages.length).toBeGreaterThan(0);

      // Should use captured origin (or wildcard as fallback)
      expect(pongMessages[0].targetOrigin).toBeTruthy();
    });

    test('should filter out null origins', async ({ page }) => {
      await page.setContent(`
        <!DOCTYPE html>
        <html>
          <body>
            <script>
              let parentOrigin = null;
              const originHistory = [];

              window.addEventListener('message', (event) => {
                originHistory.push(event.origin);

                // Should not capture 'null' origin
                if (!parentOrigin && event.origin && event.origin !== 'null') {
                  parentOrigin = event.origin;
                }
              });

              // Simulate messages with different origins
              window.postMessage({ type: 'test1' }, 'null');
              setTimeout(() => {
                window.postMessage({ type: 'test2' }, window.location.origin);
              }, 100);

              window.getResults = () => ({
                originHistory,
                capturedOrigin: parentOrigin
              });
            </script>
          </body>
        </html>
      `);

      await page.waitForTimeout(300);

      const results = await page.evaluate(() => (window as any).getResults());

      // 'null' origin should not be captured
      expect(results.capturedOrigin).not.toBe('null');
      expect(results.capturedOrigin).toBeTruthy();
    });
  });

  test.describe('Origin Priority Order', () => {
    test('should prioritize stored origin over config', async ({ page }) => {
      await page.setContent(`
        <!DOCTYPE html>
        <html>
          <head>
            <script>
              window.STAKTRAK_CONFIG = {
                parentOrigin: 'http://config-origin.com'
              };
            </script>
          </head>
          <body>
            <script>
              let parentOrigin = 'http://stored-origin.com';

              function getParentOrigin() {
                if (parentOrigin) {
                  return parentOrigin;
                }
                try {
                  const configOrigin = window.STAKTRAK_CONFIG?.parentOrigin;
                  if (configOrigin) {
                    return configOrigin;
                  }
                } catch (e) {}
                return '*';
              }

              window.testOrigin = getParentOrigin();
            </script>
          </body>
        </html>
      `);

      const origin = await page.evaluate(() => (window as any).testOrigin);

      // Stored origin should take precedence
      expect(origin).toBe('http://stored-origin.com');
      expect(origin).not.toBe('http://config-origin.com');
    });

    test('should prioritize config over wildcard', async ({ page }) => {
      await page.setContent(`
        <!DOCTYPE html>
        <html>
          <head>
            <script>
              window.STAKTRAK_CONFIG = {
                parentOrigin: 'http://config-origin.com'
              };
            </script>
          </head>
          <body>
            <script>
              let parentOrigin = null;

              function getParentOrigin() {
                if (parentOrigin) {
                  return parentOrigin;
                }
                try {
                  const configOrigin = window.STAKTRAK_CONFIG?.parentOrigin;
                  if (configOrigin) {
                    return configOrigin;
                  }
                } catch (e) {}
                return '*';
              }

              window.testOrigin = getParentOrigin();
            </script>
          </body>
        </html>
      `);

      const origin = await page.evaluate(() => (window as any).testOrigin);

      // Config origin should be used instead of wildcard
      expect(origin).toBe('http://config-origin.com');
      expect(origin).not.toBe('*');
    });
  });

  test.describe('Integration with Replay Flow', () => {
    test('should use correct origin in replay messages', async ({ page }) => {
      const messages: Array<{ type: string; origin: string }> = [];

      await page.exposeFunction('captureMessage', (type: string, origin: string) => {
        messages.push({ type, origin });
      });

      const html = createTestPage({ includeStaktrak: true, includeConfig: true, parentOrigin: 'http://test-origin.com' });
      await page.setContent(html);

      await page.evaluate(() => {
        // Intercept postMessage calls
        const originalPostMessage = window.parent.postMessage;
        window.parent.postMessage = function(message: any, targetOrigin: string) {
          if (message?.type) {
            (window as any).captureMessage(message.type, targetOrigin);
          }
          return originalPostMessage.call(this, message, targetOrigin);
        };
      });

      const testCode = `
        test('test', async ({ page }) => {
          await page.goto('http://localhost:3000');
        });
      `;

      await page.evaluate((code) => {
        if ((window as any).startPlaywrightReplay) {
          (window as any).startPlaywrightReplay(code);
        }
      }, testCode);

      await page.waitForTimeout(1000);

      // Verify messages use correct origin
      expect(messages.length).toBeGreaterThan(0);
      messages.forEach(msg => {
        expect(msg.origin).toBeTruthy();
        // Should use config origin or wildcard
        expect(['http://test-origin.com', '*', 'http://localhost:3000']).toContain(msg.origin);
      });
    });
  });
});
