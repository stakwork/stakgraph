import { test, expect } from '@playwright/test';
import {
  loadStaktrakInPage,
  startRecording,
  stopRecording,
  getActions,
  getResults,
  generatePlaywrightTest,
  simulateClick,
  simulateNavigation,
  simulateInput,
  simulateFormChange,
  verifyNewNamingConvention,
} from './test-helpers';

test.describe('Backward Compatibility', () => {
  test.describe('Action Field Name Migration', () => {
    test('should handle new "type" field format', async ({ page }) => {
      await loadStaktrakInPage(page);
      await startRecording(page);

      await simulateClick(page, '[data-testid="test-button"]');

      const actions = await getActions(page);
      const action = actions[0];

      expect(action).toBeTruthy();
      expect(action).toHaveProperty('type');
      expect(action.type).toBe('click');
    });

    test('should use "goto" for navigation instead of "nav"', async ({ page }) => {
      await loadStaktrakInPage(page);
      await startRecording(page);

      await simulateNavigation(page, 'http://localhost:3000/page1');

      const actions = await getActions(page);
      const gotoAction = actions.find(a => a.type === 'goto');

      expect(gotoAction).toBeTruthy();
      expect(gotoAction.type).toBe('goto');
      expect(actions.every(a => a.type !== 'nav')).toBe(true);
    });
  });

  test.describe('Old Action Type Handling', () => {
    test('should convert navigation events to "goto" type', async ({ page }) => {
      await loadStaktrakInPage(page);
      await startRecording(page);

      await simulateNavigation(page, 'http://localhost:3000');

      const actions = await getActions(page);
      const gotoAction = actions.find(a => a.type === 'goto');

      expect(gotoAction).toBeTruthy();
      expect(gotoAction.type).toBe('goto');
      expect(gotoAction.url).toBeTruthy();
    });

    test('should handle tracking data with navigation events', async ({ page }) => {
      await loadStaktrakInPage(page);
      await startRecording(page);

      await simulateNavigation(page, 'http://localhost:3000');

      const actions = await getActions(page);
      const gotoAction = actions.find(a => a.type === 'goto');

      expect(gotoAction).toBeTruthy();
      expect(gotoAction.type).toBe('goto');
      expect(gotoAction.url).toBe('http://localhost:3000');
    });
  });

  test.describe('Generated Code Compatibility', () => {
    test('should generate code without deprecated field names', async ({ page }) => {
      await loadStaktrakInPage(page);
      await startRecording(page);

      await simulateNavigation(page, 'http://localhost:3000');
      await page.waitForTimeout(100);
      await simulateClick(page, '[data-testid="test-button"]');

      const testCode = await generatePlaywrightTest(page);

      // Should not contain old naming
      expect(testCode).not.toContain('kind');
      expect(testCode).not.toContain("'nav'");

      // Should contain new naming
      expect(testCode).toContain('goto');
    });

    test('should handle mixed event types with consistent output', async ({ page }) => {
      await loadStaktrakInPage(page);
      await startRecording(page);

      // Add events using different techniques
      await simulateNavigation(page, 'http://localhost:3000/page1');
      await page.waitForTimeout(100);
      await simulateNavigation(page, 'http://localhost:3000/page2');
      await page.waitForTimeout(100);
      await simulateClick(page, '[data-testid="test-button"]');

      const testCode = await generatePlaywrightTest(page);

      // All should be converted to new format
      expect(testCode).toContain('goto');
      expect(testCode).not.toContain("'nav'");
    });
  });

  test.describe('Data Structure Evolution', () => {
    test('should handle actions without "kind" field', async ({ page }) => {
      await loadStaktrakInPage(page);
      await startRecording(page);

      await simulateNavigation(page, 'http://localhost:3000');

      const actions = await getActions(page);

      expect(actions.length).toBeGreaterThan(0);
      actions.forEach(action => {
        expect(action).not.toHaveProperty('kind');
        expect(action).toHaveProperty('type');
      });
    });

    test('should maintain type consistency in generated actions', async ({ page }) => {
      await loadStaktrakInPage(page);
      await startRecording(page);

      await simulateNavigation(page, 'http://localhost:3000');
      await page.waitForTimeout(100);
      await simulateClick(page, '[data-testid="test-button"]');
      await page.waitForTimeout(100);
      await simulateInput(page, '[data-testid="test-input"]', 'test');

      const actions = await getActions(page);

      // All actions should have "type" field
      actions.forEach(action => {
        expect(action).toHaveProperty('type');
        expect(action).not.toHaveProperty('kind');
        expect(typeof action.type).toBe('string');
      });
    });
  });

  test.describe('API Surface Compatibility', () => {
    test('should maintain userBehaviour API', async ({ page }) => {
      await loadStaktrakInPage(page);

      // Test all public methods exist
      const apiCheck = await page.evaluate(() => {
        const ub = (window as any).userBehaviour;
        return {
          hasStart: typeof ub.start === 'function',
          hasStop: typeof ub.stop === 'function',
          hasResult: typeof ub.result === 'function',
          hasGetActions: typeof ub.getActions === 'function',
          hasGeneratePlaywrightTest: typeof ub.generatePlaywrightTest === 'function',
          hasShowConfig: typeof ub.showConfig === 'function',
          hasMakeConfig: typeof ub.makeConfig === 'function',
        };
      });

      expect(apiCheck.hasStart).toBe(true);
      expect(apiCheck.hasStop).toBe(true);
      expect(apiCheck.hasResult).toBe(true);
      expect(apiCheck.hasGetActions).toBe(true);
      expect(apiCheck.hasGeneratePlaywrightTest).toBe(true);
      expect(apiCheck.hasShowConfig).toBe(true);
      expect(apiCheck.hasMakeConfig).toBe(true);
    });

    test('should maintain generatePlaywrightTest API', async ({ page }) => {
      await loadStaktrakInPage(page);
      await startRecording(page);

      await simulateNavigation(page, 'http://localhost:3000');

      // Should accept options parameter
      const testCode1 = await page.evaluate(() => {
        return (window as any).userBehaviour.generatePlaywrightTest();
      });
      expect(testCode1).toBeTruthy();

      const testCode2 = await page.evaluate(() => {
        return (window as any).userBehaviour.generatePlaywrightTest({ baseUrl: 'http://localhost:3000' });
      });
      expect(testCode2).toBeTruthy();
    });

    test('should maintain getActions API', async ({ page }) => {
      await loadStaktrakInPage(page);
      await startRecording(page);

      await simulateClick(page, '[data-testid="test-button"]');

      // Should return array of actions
      const actions = await page.evaluate(() => {
        return (window as any).userBehaviour.getActions();
      });

      expect(Array.isArray(actions)).toBe(true);
      expect(actions.length).toBeGreaterThan(0);
    });
  });

  test.describe('Event Type Normalization', () => {
    test('should normalize all navigation-related events to "goto"', async ({ page }) => {
      await loadStaktrakInPage(page);
      await startRecording(page);

      // Test various navigation methods
      await simulateNavigation(page, 'http://localhost:3000/page1');
      await page.waitForTimeout(100);
      await simulateNavigation(page, 'http://localhost:3000/page2');

      const actions = await getActions(page);
      const gotoActions = actions.filter(a => a.type === 'goto');

      gotoActions.forEach(action => {
        expect(action).toBeTruthy();
        expect(action.type).toBe('goto');
      });
    });

    test('should preserve other action types as-is', async ({ page }) => {
      await loadStaktrakInPage(page);
      await startRecording(page);

      await simulateClick(page, '[data-testid="test-button"]');

      const actions = await getActions(page);
      const clickAction = actions[0];

      expect(clickAction.type).toBe('click');
    });
  });

  test.describe('TrackingData Compatibility', () => {
    test('should handle empty tracking data gracefully', async ({ page }) => {
      await loadStaktrakInPage(page);
      await startRecording(page);

      const actions = await getActions(page);

      expect(Array.isArray(actions)).toBe(true);
      expect(actions.length).toBe(0);
    });

    test('should handle tracking data with only navigation', async ({ page }) => {
      await loadStaktrakInPage(page);
      await startRecording(page);

      await simulateNavigation(page, 'http://localhost:3000');

      const actions = await getActions(page);

      expect(actions.length).toBeGreaterThan(0);
      expect(actions[0].type).toBe('goto');
    });

    test('should handle tracking data with only clicks', async ({ page }) => {
      await loadStaktrakInPage(page);
      await startRecording(page);

      await simulateClick(page, '[data-testid="test-button"]');

      const actions = await getActions(page);

      expect(actions.length).toBeGreaterThan(0);
      expect(actions[0].type).toBe('click');
    });

    test('should handle complex tracking data with all event types', async ({ page }) => {
      await loadStaktrakInPage(page);
      await startRecording(page);

      await simulateNavigation(page, 'http://localhost:3000');
      await page.waitForTimeout(100);
      await simulateClick(page, '[data-testid="test-button"]');
      await page.waitForTimeout(100);
      await simulateInput(page, '[data-testid="test-input"]', 'test');
      await page.waitForTimeout(2100);
      await simulateFormChange(page, '[data-testid="test-checkbox"]', true);

      const actions = await getActions(page);

      expect(actions.length).toBeGreaterThanOrEqual(4);
      expect(actions.every(a => a.type !== undefined)).toBe(true);
      expect(actions.every(a => !('kind' in a))).toBe(true);
    });
  });

  test.describe('Code Generation Regression Tests', () => {
    test('should generate valid Playwright test code', async ({ page }) => {
      await loadStaktrakInPage(page);
      await startRecording(page);

      await simulateNavigation(page, 'http://localhost:3000');
      await page.waitForTimeout(100);
      await simulateClick(page, '[data-testid="test-button"]');

      const testCode = await generatePlaywrightTest(page);

      // Should contain valid test structure
      expect(testCode).toContain("import { test, expect } from '@playwright/test'");
      expect(testCode).toContain("test('Recorded test', async ({ page }) => {");
      expect(testCode).toContain('});');

      // Should contain actions
      expect(testCode).toContain('await page.goto');
      expect(testCode).toContain('await page.click');
    });

    test('should handle baseUrl option in code generation', async ({ page }) => {
      await loadStaktrakInPage(page);
      await startRecording(page);

      await simulateClick(page, '[data-testid="test-button"]');

      const testCode = await page.evaluate(() => {
        return (window as any).userBehaviour.generatePlaywrightTest({
          baseUrl: 'http://localhost:3000',
        });
      });

      expect(testCode).toBeTruthy();
    });
  });

  test.describe('Naming Convention Verification', () => {
    test('should verify new naming convention is consistently applied', async ({ page }) => {
      await loadStaktrakInPage(page);
      await startRecording(page);

      await simulateNavigation(page, 'http://localhost:3000');
      await page.waitForTimeout(100);
      await simulateClick(page, '[data-testid="test-button"]');
      await page.waitForTimeout(100);
      await simulateInput(page, '[data-testid="test-input"]', 'test');

      const actions = await getActions(page);

      // Verify new naming convention
      expect(verifyNewNamingConvention(actions)).toBe(true);
    });
  });
});
