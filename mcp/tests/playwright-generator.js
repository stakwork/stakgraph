/**
 * Generates a Playwright test from tracking data
 * @param {Object} trackingData - The tracking data object
 * @returns {string} - Generated Playwright test code
 */
export function generatePlaywrightTest(url, trackingData) {
  const {
    clicks,
    keyboardActivities,
    inputChanges,
    focusChanges,
    assertions,
    userInfo,
    time,
    formElementChanges,
  } = trackingData;

  if (
    (!clicks || !clicks.clickDetails || clicks.clickDetails.length === 0) &&
    (!inputChanges || inputChanges.length === 0) &&
    (!assertions || assertions.length === 0) &&
    (!formElementChanges || formElementChanges.length === 0)
  ) {
    return generateEmptyTest(url);
  }

  const testCode = `import { test, expect } from '@playwright/test';
  
test('User interaction replay', async ({ page }) => {
  // Navigate to the page
  await page.goto('${url}');
  
  // Wait for page to load
  await page.waitForLoadState('networkidle');
  
  // Set viewport size to match recorded session
  await page.setViewportSize({ 
    width: ${userInfo.windowSize[0]}, 
    height: ${userInfo.windowSize[1]} 
  });

${generateUserInteractions(
  clicks,
  inputChanges,
  focusChanges,
  assertions,
  formElementChanges
)}

  await page.waitForTimeout(2500);
});`;

  return testCode;
}

/**
 * Generates code for all user interactions in chronological order
 * @param {Object} clicks - Click data
 * @param {Array} inputChanges - Input change data
 * @param {Array} focusChanges - Focus change data
 * @param {Array} assertions - Assertions to add
 * @param {Array} formElementChanges - Form element changes (checkbox, radio, select)
 * @returns {string} - Generated interactions code
 */
function generateUserInteractions(
  clicks,
  inputChanges,
  focusChanges,
  assertions = [],
  formElementChanges = []
) {
  const allEvents = [];
  const processedSelectors = new Set();
  const formElementTimestamps = {};

  if (formElementChanges && formElementChanges.length > 0) {
    const formElementsBySelector = {};

    formElementChanges.forEach((change) => {
      const selector = change.elementSelector;
      if (!formElementsBySelector[selector]) {
        formElementsBySelector[selector] = [];
      }
      formElementsBySelector[selector].push(change);
    });

    Object.entries(formElementsBySelector).forEach(([selector, changes]) => {
      changes.sort((a, b) => a.timestamp - b.timestamp);

      if (changes[0].type === "checkbox" || changes[0].type === "radio") {
        const latestChange = changes[changes.length - 1];
        allEvents.push({
          type: "form",
          formType: latestChange.type,
          selector: latestChange.elementSelector,
          value: latestChange.value,
          checked: latestChange.checked,
          timestamp: latestChange.timestamp,
        });
        formElementTimestamps[selector] = latestChange.timestamp;
      } else if (changes[0].type === "select") {
        let lastValue = null;
        changes.forEach((change) => {
          if (change.value !== lastValue) {
            allEvents.push({
              type: "form",
              formType: change.type,
              selector: change.elementSelector,
              value: change.value,
              text: change.text,
              timestamp: change.timestamp,
            });
            formElementTimestamps[selector] = change.timestamp;
            lastValue = change.value;
          }
        });
      }

      processedSelectors.add(selector);
    });
  }

  if (clicks && clicks.clickDetails && clicks.clickDetails.length > 0) {
    clicks.clickDetails.forEach((clickDetail) => {
      const [x, y, selector, timestamp] = clickDetail;

      const shouldSkip =
        processedSelectors.has(selector) ||
        Object.entries(formElementTimestamps).some(
          ([formSelector, formTimestamp]) => {
            return (
              (selector.includes(formSelector) ||
                formSelector.includes(selector)) &&
              Math.abs(timestamp - formTimestamp) < 500
            );
          }
        );

      if (!shouldSkip) {
        allEvents.push({
          type: "click",
          x,
          y,
          selector,
          timestamp,
        });
      }
    });
  }

  const inputEvents = [];
  if (inputChanges && inputChanges.length > 0) {
    const completedInputs = inputChanges.filter(
      (change) => change.action === "complete" || !change.action
    );

    completedInputs.forEach((change) => {
      if (
        !processedSelectors.has(change.elementSelector) &&
        !change.elementSelector.includes('type="checkbox"') &&
        !change.elementSelector.includes('type="radio"')
      ) {
        inputEvents.push({
          type: "input",
          selector: change.elementSelector,
          value: change.value,
          timestamp: change.timestamp,
        });
      }
    });

    allEvents.push(...inputEvents);
  }

  if (assertions && assertions.length > 0) {
    assertions.forEach((assertion) => {
      allEvents.push({
        type: "assertion",
        assertionType: assertion.type,
        selector: assertion.selector,
        value: assertion.value,
        timestamp: assertion.timestamp,
      });
    });
  }

  allEvents.sort((a, b) => a.timestamp - b.timestamp);

  const uniqueEvents = [];
  const processedFormActions = new Set();

  allEvents.forEach((event) => {
    if (event.type === "form") {
      const eventKey = `${event.formType}-${event.selector}-${
        event.checked !== undefined ? event.checked : event.value
      }`;
      if (!processedFormActions.has(eventKey)) {
        uniqueEvents.push(event);
        processedFormActions.add(eventKey);
      }
    } else {
      uniqueEvents.push(event);
    }
  });

  let code = "";

  uniqueEvents.forEach((event, index) => {
    switch (event.type) {
      case "click":
        const playwrightSelector = convertToPlaywrightSelector(event.selector);
        code += `  // Click on element: ${event.selector}
  await page.click('${playwrightSelector}');\n\n`;
        break;

      case "input":
        const inputSelector = convertToPlaywrightSelector(event.selector);
        code += `  // Fill input: ${event.selector}
  await page.fill('${inputSelector}', '${event.value.replace(
          /'/g,
          "\\'"
        )}');\n\n`;
        break;

      case "form":
        const formSelector = convertToPlaywrightSelector(event.selector);

        if (event.formType === "checkbox" || event.formType === "radio") {
          if (event.checked) {
            code += `  // Check ${event.formType}: ${event.selector}
  await page.check('${formSelector}');\n\n`;
          } else {
            code += `  // Uncheck ${event.formType}: ${event.selector}
  await page.uncheck('${formSelector}');\n\n`;
          }
        } else if (event.formType === "select") {
          code += `  // Select option: ${event.text || event.value} in ${
            event.selector
          }
  await page.selectOption('${formSelector}', '${event.value.replace(
            /'/g,
            "\\'"
          )}');\n\n`;
        }
        break;

      case "assertion":
        const assertionSelector = convertToPlaywrightSelector(event.selector);
        switch (event.assertionType) {
          case "isVisible":
            code += `  // Assert element is visible: ${event.selector}
  await expect(page.locator('${assertionSelector}')).toBeVisible();\n\n`;
            break;
          case "hasText":
            code += `  // Assert element has text: ${event.selector}
  await expect(page.locator('${assertionSelector}')).toHaveText('${event.value.replace(
              /'/g,
              "\\'"
            )}');\n\n`;
            break;
          case "isChecked":
            code += `  // Assert checkbox/radio is checked: ${event.selector}
  await expect(page.locator('${assertionSelector}')).toBeChecked();\n\n`;
            break;
          case "isNotChecked":
            code += `  // Assert checkbox/radio is not checked: ${event.selector}
  await expect(page.locator('${assertionSelector}')).not.toBeChecked();\n\n`;
            break;
        }
        break;
    }
  });

  return code;
}

/**
 * Convert CSS selector to Playwright selector format
 * @param {string} cssSelector - CSS selector string
 * @returns {string} - Playwright compatible selector
 */
export function convertToPlaywrightSelector(cssSelector) {
  if (!cssSelector) return "";

  let playwrightSelector = cssSelector;

  if (playwrightSelector.includes("[data-testid=")) {
    const testIdPattern = /\[data-testid="([^"]+)"\]/;
    const match = playwrightSelector.match(testIdPattern);
    if (match) {
      return `[data-testid="${match[1]}"]`;
    }
  }

  if (playwrightSelector.includes("html>body>")) {
    playwrightSelector = playwrightSelector.replace("html>body>", "");
  }

  const selectorParts = playwrightSelector.split(">");
  if (selectorParts.length > 2) {
    playwrightSelector = selectorParts.slice(-2).join(" ");
  }

  const idMatch = playwrightSelector.match(/#([a-zA-Z0-9_-]+)/);
  if (idMatch) {
    return `#${idMatch[1]}`;
  }

  return playwrightSelector;
}

/**
 * Generate an empty Playwright test template when no interactions are recorded
 * @param {string} url - The URL to test
 * @returns {string} - Empty test template
 */
function generateEmptyTest(url) {
  return `import { test, expect } from '@playwright/test';

test('Empty test template', async ({ page }) => {
  // Navigate to the page
  await page.goto('${url}');
  
  // Wait for page to load
  await page.waitForLoadState('networkidle');
  
  // No interactions were recorded. Add your test steps here.
  // For example:
  // await page.click('button');
  // await page.fill('input', 'text');
  // await expect(page.locator('.element')).toBeVisible();
});`;
}

if (typeof window !== "undefined") {
  window.PlaywrightGenerator = {
    generatePlaywrightTest,
    convertToPlaywrightSelector,
  };
  console.log("PlaywrightGenerator loaded and attached to window object");
}

export { generatePlaywrightTest, convertToPlaywrightSelector };
