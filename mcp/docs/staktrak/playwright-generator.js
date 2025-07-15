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

  let actionsCode = "";
  let previousTimestamp = null;
  let generatedSelectors = new Set();

  uniqueEvents.forEach((event, index) => {
    if (previousTimestamp !== null) {
      const delay = event.timestamp - previousTimestamp;
      if (delay > 100 && event.type !== "assertion") {
        actionsCode += `  
    // Wait ${delay}ms (matching user timing)
    await page.waitForTimeout(${delay});
  `;
      }
    }

    // Generate code based on event type
    if (event.type === "click") {
      const playwrightSelector = convertToPlaywrightSelector(event.selector);
      const comment = `Click ${index + 1}: ${playwrightSelector}`;

      actionsCode += `  
    // ${comment}
    const element${index + 1} = page.locator('${playwrightSelector}');
    await element${index + 1}.waitFor({ state: 'visible' });
    await element${index + 1}.click();
  `;
    } else if (event.type === "input") {
      const playwrightSelector = convertToPlaywrightSelector(event.selector);
      if (!generatedSelectors.has(playwrightSelector)) {
        const comment = `Input ${index + 1}: Type "${
          event.value
        }" into ${playwrightSelector}`;

        actionsCode += `  
    // ${comment}
    await page.locator('${playwrightSelector}').fill('${event.value.replace(
          /'/g,
          "\\'"
        )}');
  `;

        generatedSelectors.add(playwrightSelector);
      }
    } else if (event.type === "form") {
      const playwrightSelector = convertToPlaywrightSelector(event.selector);

      if (event.formType === "checkbox" || event.formType === "radio") {
        const action = event.checked ? "check" : "uncheck";
        const comment = `${
          event.formType === "checkbox" ? "Checkbox" : "Radio"
        } ${index + 1}: ${action} ${playwrightSelector}`;

        actionsCode += `  
    // ${comment}
    await page.locator('${playwrightSelector}').${action}();
  `;
      } else if (event.formType === "select") {
        const comment = `Select ${index + 1}: Choose option "${
          event.text || event.value
        }" in ${playwrightSelector}`;

        let selectMethod, selectValue;
        if (event.text && event.text.trim() !== "") {
          selectMethod = "label";
          selectValue = event.text;
        } else {
          selectMethod = "value";
          selectValue = event.value;
        }

        actionsCode += `  
    // ${comment}
    await page.locator('${playwrightSelector}').selectOption({ ${selectMethod}: '${selectValue.replace(
          /'/g,
          "\\'"
        )}' });
  `;
      }
    } else if (event.type === "assertion") {
      const playwrightSelector = convertToPlaywrightSelector(event.selector);
      let assertionCode = "";

      switch (event.assertionType) {
        case "hasText":
          assertionCode = `await expect(page.locator('${playwrightSelector}')).toHaveText('${event.value.replace(
            /'/g,
            "\\'"
          )}');`;
          break;
        case "containsText":
          assertionCode = `await expect(page.locator('${playwrightSelector}')).toContainText('${event.value.replace(
            /'/g,
            "\\'"
          )}');`;
          break;
        case "isVisible":
          assertionCode = `await expect(page.locator('${playwrightSelector}')).toBeVisible();`;
          break;
        case "hasValue":
          assertionCode = `await expect(page.locator('${playwrightSelector}')).toHaveValue('${event.value.replace(
            /'/g,
            "\\'"
          )}');`;
          break;
        case "isChecked":
          assertionCode = `await expect(page.locator('${playwrightSelector}')).toBeChecked();`;
          break;
        case "isNotChecked":
          assertionCode = `await expect(page.locator('${playwrightSelector}')).not.toBeChecked();`;
          break;
        default:
          assertionCode = `await expect(page.locator('${playwrightSelector}')).toBeVisible();`;
      }

      actionsCode += `  
    // Assert that ${playwrightSelector} ${event.assertionType}: "${
        event.value || ""
      }"
    ${assertionCode}
  `;
    }

    previousTimestamp = event.timestamp;
  });

  return actionsCode;
}

/**
 * Converts CSS selector to Playwright-friendly selector
 * @param {string} cssSelector - CSS selector string
 * @returns {string} - Playwright selector
 */
export function convertToPlaywrightSelector(cssSelector) {
  // Handle data-testid attributes specially (Playwright best practice)
  if (cssSelector.includes('[data-testid="')) {
    const testIdMatch = cssSelector.match(/\[data-testid="([^"]+)"\]/);
    if (testIdMatch) {
      return `[data-testid="${testIdMatch[1]}"]`;
    }
  }

  // Clean up the selector for Playwright
  let selector = cssSelector;

  // Remove html>body> prefix as it's usually not needed
  selector = selector.replace(/^html>body>/, "");

  selector = selector.replace(/body\.staktrak-selection-active/, "body");
  selector = selector.replace(/\.staktrak-selection-active/, "");

  if (
    /^(p|h[1-6]|div|span|button|a|li|ul|ol|table|tr|td|th|input|textarea|select|form|label)$/i.test(
      selector
    )
  ) {
    return selector;
  }

  if (
    selector.startsWith("#") &&
    !selector.includes(" ") &&
    !selector.includes(">")
  ) {
    return selector;
  }

  // Handle class combinations properly
  selector = selector.replace(/\.([^.#\[]+)/g, ".$1");

  // Handle ID selectors
  selector = selector.replace(/#([^.#\[]+)/g, "#$1");

  const idMatch = selector.match(/#[^.#\[\s>]+/);
  if (idMatch) {
    return idMatch[0];
  }

  return selector;
}

/**
 * Generates an empty test template
 * @returns {string} - Empty test template
 */
function generateEmptyTest(url) {
  return `import { test, expect } from '@playwright/test';
  
  test('User interaction replay', async ({ page }) => {
    // Navigate to the page
    await page.goto('${url || "http://localhost:3000/frame.html"}');
    
    // Wait for page to load
    await page.waitForLoadState('networkidle');
    
    // No interactions were recorded
    console.log('No user interactions to replay');
  });`;
}

// Browser compatibility (for non-module environments)
if (typeof window !== "undefined") {
  window.PlaywrightGenerator = {
    generatePlaywrightTest,
    convertToPlaywrightSelector,
  };
}
