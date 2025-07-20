/**
 * Generates a Playwright test from tracking data
 * @param {string} url - The URL being tested
 * @param {Object} trackingData - The processed tracking data object
 * @returns {string} - Generated Playwright test code
 */
function generatePlaywrightTest(url, trackingData) {
  if (!trackingData) {
    return generateEmptyTest(url);
  }

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
  
  // No interactions were recorded
  console.log('No user interactions to replay');
});`;
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
          isUserAction: true,
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
              isUserAction: true,
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
          isUserAction: true,
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
          isUserAction: true,
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
        isUserAction: false, // Assertions are not user actions
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
  let lastUserActionTimestamp = null;

  uniqueEvents.forEach((event, index) => {
    // Calculate wait time before this event
    if (index > 0) {
      const prevEvent = uniqueEvents[index - 1];

      // Only add waits between user actions or before user actions
      // Never add waits before assertions
      if (event.isUserAction) {
        let waitTime = 0;

        if (lastUserActionTimestamp) {
          // Calculate time difference from last user action
          waitTime = event.timestamp - lastUserActionTimestamp;
        } else if (prevEvent.isUserAction) {
          // If this is the first user action after some assertions
          waitTime = event.timestamp - prevEvent.timestamp;
        }

        // Convert to milliseconds and apply reasonable bounds
        waitTime = Math.max(100, Math.min(5000, waitTime)); // Between 100ms and 5s

        if (waitTime > 100) {
          code += `  // Wait ${waitTime}ms before next action\n`;
          code += `  await page.waitForTimeout(${waitTime});\n\n`;
        }
      }
    }

    switch (event.type) {
      case "click":
        const playwrightSelector = convertToPlaywrightSelector(event.selector);
        code += `  // Click on element: ${event.selector}\n`;
        code += `  await page.click('${playwrightSelector}');\n\n`;
        lastUserActionTimestamp = event.timestamp;
        break;

      case "input":
        const inputSelector = convertToPlaywrightSelector(event.selector);
        code += `  // Fill input: ${event.selector}\n`;
        code += `  await page.fill('${inputSelector}', '${event.value.replace(
          /'/g,
          "\\'"
        )}');\n\n`;
        lastUserActionTimestamp = event.timestamp;
        break;

      case "form":
        const formSelector = convertToPlaywrightSelector(event.selector);

        if (event.formType === "checkbox" || event.formType === "radio") {
          if (event.checked) {
            code += `  // Check ${event.formType}: ${event.selector}\n`;
            code += `  await page.check('${formSelector}');\n\n`;
          } else {
            code += `  // Uncheck ${event.formType}: ${event.selector}\n`;
            code += `  await page.uncheck('${formSelector}');\n\n`;
          }
        } else if (event.formType === "select") {
          code += `  // Select option: ${event.text || event.value} in ${
            event.selector
          }\n`;
          code += `  await page.selectOption('${formSelector}', '${event.value.replace(
            /'/g,
            "\\'"
          )}');\n\n`;
        }
        lastUserActionTimestamp = event.timestamp;
        break;

      case "assertion":
        const assertionSelector = convertToPlaywrightSelector(event.selector);
        switch (event.assertionType) {
          case "isVisible":
            code += `  // Assert element is visible: ${event.selector}\n`;
            code += `  await expect(page.locator('${assertionSelector}')).toBeVisible();\n\n`;
            break;
          case "hasText":
            code += `  // Assert element contains text: ${event.selector}\n`;
            code += `  await expect(page.locator('${assertionSelector}')).toContainText('${event.value.replace(
              /'/g,
              "\\'"
            )}');\n\n`;
            break;
          case "isChecked":
            code += `  // Assert checkbox/radio is checked: ${event.selector}\n`;
            code += `  await expect(page.locator('${assertionSelector}')).toBeChecked();\n\n`;
            break;
          case "isNotChecked":
            code += `  // Assert checkbox/radio is not checked: ${event.selector}\n`;
            code += `  await expect(page.locator('${assertionSelector}')).not.toBeChecked();\n\n`;
            break;
        }
        // Note: We don't update lastUserActionTimestamp for assertions
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
function convertToPlaywrightSelector(cssSelector) {
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

if (typeof window !== "undefined") {
  window.PlaywrightGenerator = {
    generatePlaywrightTest,
    convertToPlaywrightSelector,
  };
  console.log("PlaywrightGenerator loaded and attached to window object");
}

export { generatePlaywrightTest };
