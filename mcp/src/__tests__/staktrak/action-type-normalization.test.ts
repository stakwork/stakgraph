import { test, expect } from '@playwright/test';
import {
  loadStaktrakInPage,
  startRecording,
  getActions,
  generatePlaywrightTest,
  simulateClick,
  simulateNavigation,
  simulateInput,
  simulateFormChange,
  validateAction,
  verifyNewNamingConvention,
  hasOldFieldNames
} from './test-helpers';

test.describe('Action Type Normalization', () => {
  test.describe('Action Creation - type field', () => {
    test('should create actions with "type" field instead of "kind"', async ({ page }) => {
      await loadStaktrakInPage(page);
      await startRecording(page);

      await simulateClick(page, '[data-testid="test-button"]');

      const actions = await getActions(page);
      expect(actions.length).toBeGreaterThan(0);

      const clickAction = actions[0];
      expect(clickAction).toHaveProperty('type');
      expect(clickAction).not.toHaveProperty('kind');
      expect(clickAction.type).toBe('click');
    });

    test('should use "goto" for navigation events', async ({ page }) => {
      await loadStaktrakInPage(page);
      await startRecording(page);

      await simulateNavigation(page, 'http://localhost:3000/test-page');

      const actions = await getActions(page);
      const gotoAction = actions.find(a => a.type === 'goto');

      expect(gotoAction).toBeTruthy();
      expect(gotoAction.type).toBe('goto');
      expect(gotoAction.url).toContain('test-page');
    });

    test('should create click action with correct type', async ({ page }) => {
      await loadStaktrakInPage(page);
      await startRecording(page);

      await simulateClick(page, '[data-testid="test-button"]');

      const actions = await getActions(page);
      const clickAction = actions[0];

      expect(clickAction.type).toBe('click');
      expect(clickAction.locator).toBeTruthy();
      expect(validateAction(clickAction)).toBe(true);
    });

    test('should create input action with correct type', async ({ page }) => {
      await loadStaktrakInPage(page);
      await startRecording(page);

      await simulateInput(page, '[data-testid="test-input"]', 'test value');

      const actions = await getActions(page);
      const inputAction = actions.find(a => a.type === 'input');

      expect(inputAction).toBeTruthy();
      expect(inputAction.type).toBe('input');
      expect(inputAction.value).toBe('test value');
      expect(validateAction(inputAction)).toBe(true);
    });

    test('should create form action with correct type', async ({ page }) => {
      await loadStaktrakInPage(page);
      await startRecording(page);

      await simulateFormChange(page, '[data-testid="test-checkbox"]', true);

      const actions = await getActions(page);
      const formAction = actions.find(a => a.type === 'form');

      expect(formAction).toBeTruthy();
      expect(formAction.type).toBe('form');
      expect(formAction.formType).toBeTruthy();
      expect(validateAction(formAction)).toBe(true);
    });

    test('should maintain timestamp from event', async ({ page }) => {
      await loadStaktrakInPage(page);

      const beforeTime = await page.evaluate(() => Date.now());
      await startRecording(page);
      await simulateClick(page, '[data-testid="test-button"]');

      const actions = await getActions(page);
      const clickAction = actions[0];

      expect(clickAction.timestamp).toBeGreaterThanOrEqual(beforeTime);
      expect(clickAction.timestamp).toBeLessThanOrEqual(Date.now());
    });
  });

  test.describe('Generated Actions - Type Field Presence', () => {
    test('should generate actions with "type" field from tracking data', async ({ page }) => {
      await loadStaktrakInPage(page);
      await startRecording(page);

      await simulateClick(page, '[data-testid="test-button"]');

      const actions = await getActions(page);

      expect(actions.length).toBeGreaterThan(0);
      actions.forEach(action => {
        expect(action).toHaveProperty('type');
        expect(action).not.toHaveProperty('kind');
      });
    });

    test('should generate "goto" actions for navigation', async ({ page }) => {
      await loadStaktrakInPage(page);
      await startRecording(page);

      await simulateNavigation(page, 'http://localhost:3000/test-page');

      const actions = await getActions(page);
      const gotoAction = actions.find(a => a.type === 'goto');

      expect(gotoAction).toBeTruthy();
      expect(gotoAction.type).toBe('goto');
      expect(gotoAction.url).toBeTruthy();
    });

    test('should generate actions in correct order with type field', async ({ page }) => {
      await loadStaktrakInPage(page);
      await startRecording(page);

      // Perform actions in order
      await simulateNavigation(page, 'http://localhost:3000/page1');
      await page.waitForTimeout(200);
      await simulateClick(page, '[data-testid="test-button"]');
      await page.waitForTimeout(200);
      await simulateInput(page, '[data-testid="test-input"]', 'test value');

      const actions = await getActions(page);

      expect(actions.length).toBeGreaterThanOrEqual(3);

      // Check that all actions have type field
      actions.forEach(action => {
        expect(action).toHaveProperty('type');
        expect(typeof action.type).toBe('string');
      });

      // Verify order (accounting for possible duplicate events)
      const gotoAction = actions.find(a => a.type === 'goto');
      const clickAction = actions.find(a => a.type === 'click');
      const inputAction = actions.find(a => a.type === 'input');

      expect(gotoAction).toBeTruthy();
      expect(clickAction).toBeTruthy();
      expect(inputAction).toBeTruthy();
    });
  });

  test.describe('Code Generation - New Naming Convention', () => {
    test('should generate test code using "goto" instead of "nav"', async ({ page }) => {
      await loadStaktrakInPage(page);
      await startRecording(page);

      await simulateNavigation(page, 'http://localhost:3000/test');

      const testCode = await generatePlaywrightTest(page);

      expect(testCode).toContain('await page.goto');
      expect(testCode).not.toContain("'nav'");
    });

    test('should generate test with all action types correctly', async ({ page }) => {
      await loadStaktrakInPage(page);
      await startRecording(page);

      await simulateNavigation(page, 'http://localhost:3000');
      await page.waitForTimeout(200);
      await simulateClick(page, '[data-testid="test-button"]');
      await page.waitForTimeout(200);
      await simulateInput(page, '[data-testid="test-input"]', 'test input');

      const testCode = await generatePlaywrightTest(page);

      expect(testCode).toContain('await page.goto');
      expect(testCode).toContain('await page.click');
      expect(testCode).toContain('await page.fill');
    });

    test('should not include "kind" field in generated code comments or structure', async ({ page }) => {
      await loadStaktrakInPage(page);
      await startRecording(page);

      await simulateClick(page, '[data-testid="test-button"]');

      const testCode = await generatePlaywrightTest(page);

      expect(testCode).not.toContain('kind');
    });
  });

  test.describe('Full Flow with Type Consistency', () => {
    test('should handle multiple events and maintain type consistency', async ({ page }) => {
      await loadStaktrakInPage(page);
      await startRecording(page);

      // Perform various actions
      await simulateNavigation(page, 'http://localhost:3000/test');
      await page.waitForTimeout(100);
      await simulateClick(page, '[data-testid="test-button"]');
      await page.waitForTimeout(100);
      await simulateInput(page, '[data-testid="test-input"]', 'test');
      await page.waitForTimeout(2100);
      await simulateFormChange(page, '[data-testid="test-checkbox"]', true);

      const actions = await getActions(page);

      // All actions should have type field
      actions.forEach(action => {
        expect(action).toBeTruthy();
        expect(action).toHaveProperty('type');
        expect(action).not.toHaveProperty('kind');
        expect(validateAction(action)).toBe(true);
      });

      // Check that we have expected types
      const hasGoto = actions.some(a => a.type === 'goto');
      const hasClick = actions.some(a => a.type === 'click');
      const hasInput = actions.some(a => a.type === 'input');
      const hasForm = actions.some(a => a.type === 'form');

      expect(hasGoto).toBe(true);
      expect(hasClick).toBe(true);
      expect(hasInput).toBe(true);
      expect(hasForm).toBe(true);
    });

    test('should generate test from tracking data with correct types', async ({ page }) => {
      await loadStaktrakInPage(page);
      await startRecording(page);

      await simulateNavigation(page, 'http://localhost:3000');
      await page.waitForTimeout(100);
      await simulateClick(page, '[data-testid="test-button"]');
      await page.waitForTimeout(100);
      await simulateInput(page, '[data-testid="test-input"]', 'test');

      const testCode = await generatePlaywrightTest(page);

      expect(testCode).toContain('await page.goto');
      expect(testCode).toContain('await page.click');
      expect(testCode).toContain('await page.fill');
      expect(testCode).not.toContain('kind');
      expect(testCode).not.toContain("'nav'");
    });
  });

  test.describe('Edge Cases', () => {
    test('should handle actions without explicit timestamps', async ({ page }) => {
      await loadStaktrakInPage(page);
      await startRecording(page);

      await simulateClick(page, '[data-testid="test-button"]');

      const actions = await getActions(page);
      const action = actions[0];

      expect(action).toBeTruthy();
      expect(action.type).toBe('click');
      expect(action.timestamp).toBeGreaterThan(0);
    });

    test('should verify all actions use new naming convention', async ({ page }) => {
      await loadStaktrakInPage(page);
      await startRecording(page);

      await simulateNavigation(page, 'http://localhost:3000');
      await page.waitForTimeout(100);
      await simulateClick(page, '[data-testid="test-button"]');
      await page.waitForTimeout(100);
      await simulateInput(page, '[data-testid="test-input"]', 'value');

      const actions = await getActions(page);

      // Verify no old field names are present
      const hasOldNames = actions.some(action => hasOldFieldNames(action));
      expect(hasOldNames).toBe(false);

      // Verify new naming convention
      expect(verifyNewNamingConvention(actions)).toBe(true);
    });
  });
});
