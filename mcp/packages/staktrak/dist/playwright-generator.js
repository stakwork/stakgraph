// src/playwright-generator.ts
function convertToPlaywrightSelector(clickDetail) {
  const { selectors } = clickDetail;
  if (selectors.primary.includes("[data-testid=")) {
    return selectors.primary;
  }
  if (selectors.primary.startsWith("#")) {
    return selectors.primary;
  }
  if (selectors.text && (selectors.tagName === "button" || selectors.tagName === "a" || selectors.role === "button")) {
    const cleanText = selectors.text.trim();
    if (cleanText.length > 0 && cleanText.length <= 50) {
      return `text=${escapeTextForAssertion(cleanText)}`;
    }
  }
  if (selectors.ariaLabel) {
    return `[aria-label="${escapeTextForAssertion(selectors.ariaLabel)}"]`;
  }
  for (const fallback of selectors.fallbacks) {
    if (isValidCSSSelector(fallback)) {
      return fallback;
    }
  }
  if (isValidCSSSelector(selectors.primary)) {
    return selectors.primary;
  }
  if (selectors.role) {
    return `[role="${selectors.role}"]`;
  }
  if (selectors.tagName === "input") {
    const type = clickDetail.elementInfo.attributes.type;
    const name = clickDetail.elementInfo.attributes.name;
    if (type) return `input[type="${type}"]`;
    if (name) return `input[name="${name}"]`;
  }
  if (selectors.xpath) {
    return `xpath=${selectors.xpath}`;
  }
  return selectors.tagName;
}
function isValidCSSSelector(selector) {
  if (!selector || selector.trim() === "") return false;
  try {
    if (typeof document !== "undefined") {
      document.querySelector(selector);
    }
    return true;
  } catch (e) {
    return false;
  }
}
function generatePlaywrightTest(url, trackingData) {
  var _a;
  if (!trackingData) return generateEmptyTest(url);
  const { clicks, inputChanges, assertions, userInfo, formElementChanges } = trackingData;
  if (!((_a = clicks == null ? void 0 : clicks.clickDetails) == null ? void 0 : _a.length) && !(inputChanges == null ? void 0 : inputChanges.length) && !(assertions == null ? void 0 : assertions.length) && !(formElementChanges == null ? void 0 : formElementChanges.length)) {
    return generateEmptyTest(url);
  }
  return `import { test, expect } from '@playwright/test';
    
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
    trackingData.focusChanges,
    assertions,
    formElementChanges
  )}
  
    await page.waitForTimeout(432);
  });`;
}
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
function generateUserInteractions(clicks, inputChanges, focusChanges, assertions = [], formElementChanges = []) {
  var _a;
  const allEvents = [];
  const processedSelectors = /* @__PURE__ */ new Set();
  const formElementTimestamps = {};
  if (formElementChanges == null ? void 0 : formElementChanges.length) {
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
          isUserAction: true
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
              isUserAction: true
            });
            formElementTimestamps[selector] = change.timestamp;
            lastValue = change.value;
          }
        });
      }
      processedSelectors.add(selector);
    });
  }
  if ((_a = clicks == null ? void 0 : clicks.clickDetails) == null ? void 0 : _a.length) {
    clicks.clickDetails.forEach((clickDetail) => {
      const selector = convertToPlaywrightSelector(clickDetail);
      if (!selector || selector.trim() === "") {
        console.warn(
          `Skipping click with invalid selector for element: ${clickDetail.selectors.tagName}`
        );
        return;
      }
      const shouldSkip = processedSelectors.has(clickDetail.selectors.primary) || processedSelectors.has(selector) || Object.entries(formElementTimestamps).some(
        ([formSelector, formTimestamp]) => {
          const isRelatedToForm = clickDetail.selectors.primary.includes(formSelector) || formSelector.includes(clickDetail.selectors.primary) || selector.includes(formSelector) || clickDetail.selectors.fallbacks.some(
            (f) => f.includes(formSelector) || formSelector.includes(f)
          );
          return isRelatedToForm && Math.abs(clickDetail.timestamp - formTimestamp) < 500;
        }
      );
      if (!shouldSkip) {
        allEvents.push({
          type: "click",
          x: clickDetail.x,
          y: clickDetail.y,
          selector,
          timestamp: clickDetail.timestamp,
          isUserAction: true,
          text: clickDetail.selectors.text,
          clickDetail
          // Store for better debugging
        });
      }
    });
  }
  if (inputChanges == null ? void 0 : inputChanges.length) {
    const completedInputs = inputChanges.filter(
      (change) => change.action === "complete" || !change.action
    );
    completedInputs.forEach((change) => {
      const isFormElement = change.elementSelector.includes('type="checkbox"') || change.elementSelector.includes('type="radio"');
      if (!processedSelectors.has(change.elementSelector) && !isFormElement) {
        allEvents.push({
          type: "input",
          selector: change.elementSelector,
          value: change.value,
          timestamp: change.timestamp,
          isUserAction: true
        });
      }
    });
  }
  if (assertions == null ? void 0 : assertions.length) {
    assertions.forEach((assertion) => {
      const text = assertion.value || "";
      const isShortText = text.length < 4;
      const hasValidText = text.trim().length > 0;
      if (!isShortText && hasValidText) {
        allEvents.push({
          type: "assertion",
          assertionType: assertion.type,
          selector: assertion.selector,
          value: assertion.value,
          timestamp: assertion.timestamp,
          isUserAction: false
        });
      }
    });
  }
  allEvents.sort((a, b) => a.timestamp - b.timestamp);
  const uniqueEvents = [];
  const processedFormActions = /* @__PURE__ */ new Set();
  allEvents.forEach((event) => {
    if (event.type === "form") {
      const eventKey = `${event.formType}-${event.selector}-${event.checked !== void 0 ? event.checked : event.value}`;
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
    if (index > 0) {
      const prevEvent = uniqueEvents[index - 1];
      if (event.isUserAction) {
        let waitTime = 0;
        if (lastUserActionTimestamp) {
          waitTime = event.timestamp - lastUserActionTimestamp;
        } else if (prevEvent.isUserAction) {
          waitTime = event.timestamp - prevEvent.timestamp;
        }
        waitTime = Math.max(100, Math.min(5e3, waitTime));
        if (waitTime > 100) {
          code += `  await page.waitForTimeout(${waitTime});

`;
        }
      }
    }
    switch (event.type) {
      case "click":
        code += generateClickCode(event);
        lastUserActionTimestamp = event.timestamp;
        break;
      case "input":
        code += generateInputCode(event);
        lastUserActionTimestamp = event.timestamp;
        break;
      case "form":
        code += generateFormCode(event);
        lastUserActionTimestamp = event.timestamp;
        break;
      case "assertion":
        code += generateAssertionCode(event);
        break;
    }
  });
  return code;
}
function generateClickCode(event) {
  const selectorComment = event.clickDetail ? `${event.clickDetail.selectors.tagName}${event.clickDetail.selectors.text ? ` "${event.clickDetail.selectors.text}"` : ""}` : event.selector;
  let code = `  // Click on ${selectorComment}
`;
  code += `  await page.click('${event.selector}');

`;
  return code;
}
function generateInputCode(event) {
  const escapedValue = escapeTextForAssertion(event.value);
  let code = `  // Fill input: ${event.selector}
`;
  code += `  await page.fill('${event.selector}', '${escapedValue}');

`;
  return code;
}
function generateFormCode(event) {
  let code = "";
  if (event.formType === "checkbox" || event.formType === "radio") {
    if (event.checked) {
      code += `  // Check ${event.formType}: ${event.selector}
`;
      code += `  await page.check('${event.selector}');

`;
    } else {
      code += `  // Uncheck ${event.formType}: ${event.selector}
`;
      code += `  await page.uncheck('${event.selector}');

`;
    }
  } else if (event.formType === "select") {
    const escapedValue = escapeTextForAssertion(event.value);
    code += `  // Select option: ${event.text || event.value} in ${event.selector}
`;
    code += `  await page.selectOption('${event.selector}', '${escapedValue}');

`;
  }
  return code;
}
function generateAssertionCode(event) {
  let code = "";
  switch (event.assertionType) {
    case "isVisible":
      code += `  // Assert element is visible: ${event.selector}
`;
      code += `  await expect(page.locator('${event.selector}')).toBeVisible();

`;
      break;
    case "hasText":
      const genericSelectors = [
        "div",
        "p",
        "span",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6"
      ];
      const isGenericSelector = genericSelectors.includes(event.selector);
      if (isGenericSelector) {
        const cleanedText = cleanTextForGetByText(event.value);
        const isShortText = cleanedText.length < 10 || cleanedText.split(" ").length <= 2;
        code += `  // Assert element contains text: ${event.selector}
`;
        if (isShortText) {
          code += `  await expect(page.locator('${event.selector}').filter({ hasText: '${cleanedText}' })).toBeVisible();

`;
        } else {
          code += `  await expect(page.getByText('${cleanedText}', { exact: false })).toBeVisible();

`;
        }
      } else {
        const escapedText = escapeTextForAssertion(event.value);
        code += `  // Assert element contains text: ${event.selector}
`;
        code += `  await expect(page.locator('${event.selector}')).toContainText('${escapedText}');

`;
      }
      break;
    case "isChecked":
      code += `  // Assert checkbox/radio is checked: ${event.selector}
`;
      code += `  await expect(page.locator('${event.selector}')).toBeChecked();

`;
      break;
    case "isNotChecked":
      code += `  // Assert checkbox/radio is not checked: ${event.selector}
`;
      code += `  await expect(page.locator('${event.selector}')).not.toBeChecked();

`;
      break;
  }
  return code;
}
function escapeTextForAssertion(text) {
  if (!text) return "";
  return text.replace(/\\/g, "\\\\").replace(/'/g, "\\'").replace(/\n/g, "\\n").replace(/\r/g, "\\r").replace(/\t/g, "\\t").trim();
}
function cleanTextForGetByText(text) {
  if (!text) return "";
  return text.replace(/\s+/g, " ").replace(/\n+/g, " ").trim();
}
function isTextAmbiguous(text) {
  if (!text) return true;
  if (text.length < 6) return true;
  if (text.split(/\s+/).length <= 2) return true;
  return false;
}
if (typeof window !== "undefined") {
  window.PlaywrightGenerator = {
    generatePlaywrightTest,
    convertToPlaywrightSelector,
    escapeTextForAssertion,
    cleanTextForGetByText,
    isTextAmbiguous
  };
}
