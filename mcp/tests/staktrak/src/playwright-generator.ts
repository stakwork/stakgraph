// Update these interfaces in the generator file
interface Clicks {
  clickCount: number;
  clickDetails: ClickDetail[];
}

interface ClickDetail {
  x: number;
  y: number;
  timestamp: number;
  selectors: {
    primary: string;
    fallbacks: string[];
    text?: string;
    ariaLabel?: string;
    title?: string;
    role?: string;
    tagName: string;
    xpath?: string;
  };
  elementInfo: {
    tagName: string;
    id?: string;
    className?: string;
    attributes: Record<string, string>;
  };
}

interface InteractionEvent {
  type: "click" | "input" | "form" | "assertion" | "navigation";
  selector?: string;
  timestamp: number;
  isUserAction: boolean;
  x?: number;
  y?: number;
  value?: string;
  formType?: string;
  checked?: boolean;
  text?: string;
  assertionType?: string | 'exact' | 'regex' | 'contains';
  clickDetail?: ClickDetail; // Add this for better click handling
  url?: string;
  urlPattern?: string;
  useRegex?: boolean;
}

interface InputChange {
  elementSelector: string;
  value: string;
  timestamp: number;
  action?: string;
}

interface FormElementChange {
  elementSelector: string;
  type: "checkbox" | "radio" | "select";
  value: string;
  checked?: boolean;
  text?: string;
  timestamp: number;
}

interface Assertion {
  type: "isVisible" | "hasText" | "isChecked" | "isNotChecked";
  selector: string;
  value: string;
  timestamp: number;
}

interface UserInfo {
  windowSize: [number, number];
}

interface PageNavigation {
  type: 'pushState' | 'replaceState' | 'popstate';
  url: string;
  timestamp: number;
}

interface TrackingData {
  clicks?: Clicks;
  keyboardActivities?: any;
  inputChanges?: InputChange[];
  focusChanges?: any;
  assertions?: Assertion[];
  userInfo: UserInfo;
  time?: any;
  formElementChanges?: FormElementChange[];
  pageNavigation?: PageNavigation[];
}

function convertToPlaywrightSelector(clickDetail: ClickDetail): string {
  const { selectors } = clickDetail;

  // Strategy 1: data-testid (highest priority)
  if (selectors.primary.includes("[data-testid=")) {
    return selectors.primary;
  }

  // Strategy 2: ID selectors
  if (selectors.primary.startsWith("#")) {
    return selectors.primary;
  }

  // Strategy 3: Text-based selectors for interactive elements
  if (
    selectors.text &&
    (selectors.tagName === "button" ||
      selectors.tagName === "a" ||
      selectors.role === "button")
  ) {
    const cleanText = selectors.text.trim();
    if (cleanText.length > 0 && cleanText.length <= 50) {
      return `text=${escapeTextForAssertion(cleanText)}`;
    }
  }

  // Strategy 4: aria-label
  if (selectors.ariaLabel) {
    return `[aria-label="${escapeTextForAssertion(selectors.ariaLabel)}"]`;
  }

  // Strategy 5: Try fallback selectors
  for (const fallback of selectors.fallbacks) {
    if (isValidCSSSelector(fallback)) {
      return fallback;
    }
  }

  // Strategy 6: Use primary selector if it's valid
  if (isValidCSSSelector(selectors.primary)) {
    return selectors.primary;
  }

  // Strategy 7: Role-based selector
  if (selectors.role) {
    return `[role="${selectors.role}"]`;
  }

  // Strategy 8: Tag name with attributes
  if (selectors.tagName === "input") {
    const type = clickDetail.elementInfo.attributes.type;
    const name = clickDetail.elementInfo.attributes.name;
    if (type) return `input[type="${type}"]`;
    if (name) return `input[name="${name}"]`;
  }

  // Strategy 9: XPath as last resort
  if (selectors.xpath) {
    return `xpath=${selectors.xpath}`;
  }

  // Final fallback
  return selectors.tagName;
}

function isValidCSSSelector(selector: string): boolean {
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

function generateUrlPattern(url: string, options: { preferRelative?: boolean; preferContains?: boolean } = {}): { pattern: string; useRegex: boolean; assertionType: 'exact' | 'regex' | 'contains' } {
  if (!url) return { pattern: "", useRegex: false, assertionType: 'exact' };

  try {
    const urlObj = new URL(url);
    const pathname = urlObj.pathname;
    const search = urlObj.search;
    
    const dynamicPatterns = [
      /\/\d{4,}(?:\/|$)/,  
      /\/[a-f0-9]{8,}(?:-[a-f0-9]{4,})*(?:\/|$)/i,  
      /\/[A-Z0-9_]{20,}(?:\/|$)/i,  
      /\/[a-z0-9]{25,}(?:\/|$)/i,   
      /\/[a-z0-9]{15,}(?:\/|$)/i,   
    ];
    
    const hasDynamicSegments = dynamicPatterns.some(pattern => pattern.test(pathname));
    
    const workspacePattern = /^\/w\/[^\/]+\/(.+)$/;
    const workspaceMatch = pathname.match(workspacePattern);
    
    if (workspaceMatch && options.preferContains !== false) {
      const section = workspaceMatch[1];
      return { 
        pattern: `/${section}`, 
        useRegex: false, 
        assertionType: 'contains' 
      };
    }
    
    const workspaceRootPattern = /^\/w\/[^\/]+$/;
    if (workspaceRootPattern.test(pathname) && options.preferContains !== false) {
      return { 
        pattern: pathname, 
        useRegex: false, 
        assertionType: 'exact' 
      };
    }
    
    if (hasDynamicSegments) {
      let regexPattern = pathname
        .replace(/\/\d{4,}(?=\/|$)/g, '/\\d+')  
        .replace(/\/[a-f0-9]{8,}(?:-[a-f0-9]{4,})*(?=\/|$)/gi, '/[a-f0-9-]+')  
        .replace(/\/[A-Z0-9_]{20,}(?=\/|$)/gi, '/[A-Z0-9_-]+')  
        .replace(/\/[a-z0-9]{25,}(?=\/|$)/gi, '/[a-z0-9]+')  
        .replace(/\/[a-z0-9]{15,}(?=\/|$)/gi, '/[a-z0-9]+')  
        .replace(/\//g, '\\/')  
        .replace(/\./g, '\\.'); 
      
      if (search) {
        regexPattern += '\\' + search.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
      }
      
      regexPattern += '$';
      
      return { pattern: regexPattern, useRegex: true, assertionType: 'regex' };
    } else {
      if (options.preferRelative !== false && pathname !== '/') {
        return { 
          pattern: pathname + search, 
          useRegex: false, 
          assertionType: 'exact' 
        };
      }
      
      return { 
        pattern: url, 
        useRegex: false, 
        assertionType: 'exact' 
      };
    }
  } catch (e) {
    return { pattern: url, useRegex: false, assertionType: 'exact' };
  }
}

function isSameRoute(url1: string, url2: string): boolean {
  try {
    const url1Obj = new URL(url1);
    const url2Obj = new URL(url2);
    
    return url1Obj.protocol === url2Obj.protocol &&
           url1Obj.host === url2Obj.host &&
           url1Obj.pathname === url2Obj.pathname;
  } catch (e) {
    return url1 === url2;
  }
}

function getUrlDisplayName(url: string): string {
  try {
    const urlObj = new URL(url);
    const pathname = urlObj.pathname;
    
    if (pathname === '/' || pathname === '') {
      return 'homepage';
    }
    
    const segments = pathname.split('/').filter(s => s.length > 0);
    const lastSegment = segments[segments.length - 1];
    
    if (lastSegment && /^\d+$|^[a-f0-9-]{8,}$/i.test(lastSegment) && segments.length > 1) {
      return segments[segments.length - 2];
    }
    
    return lastSegment || 'page';
  } catch (e) {
    return 'page';
  }
}

function generatePlaywrightTest(
  url: string,
  trackingData?: TrackingData
): string {
  if (!trackingData) return generateEmptyTest(url);

  const { clicks, inputChanges, assertions, userInfo, formElementChanges, pageNavigation } =
    trackingData;

  if (
    !clicks?.clickDetails?.length &&
    !inputChanges?.length &&
    !assertions?.length &&
    !formElementChanges?.length &&
    !pageNavigation?.length
  ) {
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
    formElementChanges,
    pageNavigation || []
  )}
  
    await page.waitForTimeout(432);
  });`;
}

function generateEmptyTest(url: string): string {
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

function generateUserInteractions(
  clicks?: Clicks,
  inputChanges?: InputChange[],
  focusChanges?: any,
  assertions: Assertion[] = [],
  formElementChanges: FormElementChange[] = [],
  pageNavigation: PageNavigation[] = []
): string {
  const allEvents: InteractionEvent[] = [];
  const processedSelectors = new Set<string>();
  const formElementTimestamps: Record<string, number> = {};

  // Process form element changes first
  if (formElementChanges?.length) {
    const formElementsBySelector: Record<string, FormElementChange[]> = {};

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
        let lastValue: string | null = null;
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

  if (pageNavigation?.length) {
    const uniqueNavigations: PageNavigation[] = [];
    let lastUrl = '';
    
    pageNavigation
      .sort((a, b) => a.timestamp - b.timestamp)
      .forEach((nav) => {
        if (!isSameRoute(nav.url, lastUrl)) {
          uniqueNavigations.push(nav);
          lastUrl = nav.url;
        }
      });

    uniqueNavigations.forEach((nav) => {
      const { pattern, useRegex, assertionType } = generateUrlPattern(nav.url, { 
        preferRelative: true, 
        preferContains: true 
      });
      const urlDisplayName = getUrlDisplayName(nav.url);
      
      allEvents.push({
        type: "navigation",
        url: nav.url,
        urlPattern: pattern,
        useRegex: useRegex,
        assertionType: assertionType,
        timestamp: nav.timestamp,
        isUserAction: false,
        text: urlDisplayName,
      });
    });
  }

  // Process clicks with new ClickDetail structure
  if (clicks?.clickDetails?.length) {
    clicks.clickDetails.forEach((clickDetail) => {
      const selector = convertToPlaywrightSelector(clickDetail);

      // Skip if conversion failed
      if (!selector || selector.trim() === "") {
        console.warn(
          `Skipping click with invalid selector for element: ${clickDetail.selectors.tagName}`
        );
        return;
      }

      // Check if this click should be skipped due to form interactions
      const shouldSkip =
        processedSelectors.has(clickDetail.selectors.primary) ||
        processedSelectors.has(selector) ||
        Object.entries(formElementTimestamps).some(
          ([formSelector, formTimestamp]) => {
            const isRelatedToForm =
              clickDetail.selectors.primary.includes(formSelector) ||
              formSelector.includes(clickDetail.selectors.primary) ||
              selector.includes(formSelector) ||
              clickDetail.selectors.fallbacks.some(
                (f) => f.includes(formSelector) || formSelector.includes(f)
              );

            return (
              isRelatedToForm &&
              Math.abs(clickDetail.timestamp - formTimestamp) < 500
            );
          }
        );

      if (!shouldSkip) {
        allEvents.push({
          type: "click",
          x: clickDetail.x,
          y: clickDetail.y,
          selector: selector,
          timestamp: clickDetail.timestamp,
          isUserAction: true,
          text: clickDetail.selectors.text,
          clickDetail: clickDetail, // Store for better debugging
        });
      }
    });
  }

  // Process input changes
  if (inputChanges?.length) {
    const completedInputs = inputChanges.filter(
      (change) => change.action === "complete" || !change.action
    );

    completedInputs.forEach((change) => {
      const isFormElement =
        change.elementSelector.includes('type="checkbox"') ||
        change.elementSelector.includes('type="radio"');

      if (!processedSelectors.has(change.elementSelector) && !isFormElement) {
        allEvents.push({
          type: "input",
          selector: change.elementSelector,
          value: change.value,
          timestamp: change.timestamp,
          isUserAction: true,
        });
      }
    });
  }

  // Process assertions
  if (assertions?.length) {
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
          isUserAction: false,
        });
      }
    });
  }

  // Sort all events by timestamp
  allEvents.sort((a, b) => a.timestamp - b.timestamp);

  // Remove duplicate form actions
  const uniqueEvents: InteractionEvent[] = [];
  const processedFormActions = new Set<string>();

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

  // Generate Playwright code
  let code = "";
  let lastUserActionTimestamp: number | null = null;

  uniqueEvents.forEach((event, index) => {
    // Add wait time between user actions
    if (index > 0) {
      const prevEvent = uniqueEvents[index - 1];

      if (event.isUserAction) {
        let waitTime = 0;

        if (lastUserActionTimestamp) {
          waitTime = event.timestamp - lastUserActionTimestamp;
        } else if (prevEvent.isUserAction) {
          waitTime = event.timestamp - prevEvent.timestamp;
        }

        waitTime = Math.max(100, Math.min(5000, waitTime));

        if (waitTime > 100) {
          code += `  await page.waitForTimeout(${waitTime});\n\n`;
        }
      }
    }

    // Generate code based on event type
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

      case "navigation":
        code += generateNavigationCode(event);
        break;
    }
  });

  return code;
}

// Helper functions for code generation
function generateClickCode(event: InteractionEvent): string {
  const selectorComment = event.clickDetail
    ? `${event.clickDetail.selectors.tagName}${event.clickDetail.selectors.text ? ` "${event.clickDetail.selectors.text}"` : ""}`
    : event.selector;

  let code = `  // Click on ${selectorComment}\n`;
  code += `  await page.click('${event.selector}');\n\n`;
  return code;
}

function generateInputCode(event: InteractionEvent): string {
  const escapedValue = escapeTextForAssertion(event.value!);
  let code = `  // Fill input: ${event.selector}\n`;
  code += `  await page.fill('${event.selector}', '${escapedValue}');\n\n`;
  return code;
}

function generateFormCode(event: InteractionEvent): string {
  let code = "";

  if (event.formType === "checkbox" || event.formType === "radio") {
    if (event.checked) {
      code += `  // Check ${event.formType}: ${event.selector}\n`;
      code += `  await page.check('${event.selector}');\n\n`;
    } else {
      code += `  // Uncheck ${event.formType}: ${event.selector}\n`;
      code += `  await page.uncheck('${event.selector}');\n\n`;
    }
  } else if (event.formType === "select") {
    const escapedValue = escapeTextForAssertion(event.value!);
    code += `  // Select option: ${event.text || event.value} in ${event.selector}\n`;
    code += `  await page.selectOption('${event.selector}', '${escapedValue}');\n\n`;
  }

  return code;
}

function generateAssertionCode(event: InteractionEvent): string {
  let code = "";

  switch (event.assertionType) {
    case "isVisible":
      code += `  // Assert element is visible: ${event.selector}\n`;
      code += `  await expect(page.locator('${event.selector}')).toBeVisible();\n\n`;
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
        "h6",
      ];
      const isGenericSelector = genericSelectors.includes(event.selector!);

      if (isGenericSelector) {
        const cleanedText = cleanTextForGetByText(event.value!);
        const isShortText =
          cleanedText.length < 10 || cleanedText.split(" ").length <= 2;

        code += `  // Assert element contains text: ${event.selector}\n`;
        if (isShortText) {
          code += `  await expect(page.locator('${event.selector}').filter({ hasText: '${cleanedText}' })).toBeVisible();\n\n`;
        } else {
          code += `  await expect(page.getByText('${cleanedText}', { exact: false })).toBeVisible();\n\n`;
        }
      } else {
        const escapedText = escapeTextForAssertion(event.value!);
        code += `  // Assert element contains text: ${event.selector}\n`;
        code += `  await expect(page.locator('${event.selector}')).toContainText('${escapedText}');\n\n`;
      }
      break;

    case "isChecked":
      code += `  // Assert checkbox/radio is checked: ${event.selector}\n`;
      code += `  await expect(page.locator('${event.selector}')).toBeChecked();\n\n`;
      break;

    case "isNotChecked":
      code += `  // Assert checkbox/radio is not checked: ${event.selector}\n`;
      code += `  await expect(page.locator('${event.selector}')).not.toBeChecked();\n\n`;
      break;
  }

  return code;
}

function generateNavigationCode(event: InteractionEvent): string {
  let code = "";
  
  if (!event.urlPattern || !event.url) return code;
  
  const urlDisplayName = event.text || getUrlDisplayName(event.url);
  
  switch (event.assertionType) {
    case 'contains':
      code += `  // Assert URL contains ${urlDisplayName} section\n`;
      if (event.urlPattern && !event.urlPattern.match(/[.*+?^${}()|[\]\\]/)) {
        code += `  await expect(page).toHaveURL('${event.urlPattern}');\n\n`;
      } else {
        code += `  await expect(page).toHaveURL(/${event.urlPattern}/);\n\n`;
      }
      break;
      
    case 'regex':
      code += `  // Assert URL matches ${urlDisplayName} page pattern\n`;
      code += `  await expect(page).toHaveURL(/${event.urlPattern}/);\n\n`;
      break;
      
      case 'exact':
      default:
        if (event.urlPattern.startsWith('/') && !event.urlPattern.startsWith('http')) {
          code += `  // Assert URL path is ${urlDisplayName} page\n`;
          code += `  await expect(page).toHaveURL('${event.urlPattern}');\n\n`;
        } else {
          code += `  // Assert exact URL for ${urlDisplayName} page\n`;
          code += `  await expect(page).toHaveURL('${event.urlPattern}');\n\n`;
        }
        break;
  }
  
  return code;
}

function escapeTextForAssertion(text: string): string {
  if (!text) return "";
  return text
    .replace(/\\/g, "\\\\")
    .replace(/'/g, "\\'")
    .replace(/\n/g, "\\n")
    .replace(/\r/g, "\\r")
    .replace(/\t/g, "\\t")
    .trim();
}

function cleanTextForGetByText(text: string): string {
  if (!text) return "";
  return text.replace(/\s+/g, " ").replace(/\n+/g, " ").trim();
}

function isTextAmbiguous(text: string): boolean {
  if (!text) return true;
  if (text.length < 6) return true;
  if (text.split(/\s+/).length <= 2) return true;
  return false;
}

if (typeof window !== "undefined") {
  (window as any).PlaywrightGenerator = {
    generatePlaywrightTest,
    convertToPlaywrightSelector,
    escapeTextForAssertion,
    cleanTextForGetByText,
    isTextAmbiguous,
    generateUrlPattern,
    isSameRoute,
    getUrlDisplayName,
    generateNavigationCode,
  };
}
