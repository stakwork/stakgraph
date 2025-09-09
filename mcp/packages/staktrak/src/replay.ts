import { ActionType, ReplayStatus, ReplayAction, Results } from "./types";

const DEFAULT_SPEED = 1;
const MIN_DELAY = 0;
const MAX_DELAY = 10000;
const INITIAL_DELAY = 500;

interface TimeoutRef {
  current: NodeJS.Timeout | null;
}

interface ElementRef {
  current: HTMLElement | null;
}

interface ActionsRef {
  current: ReplayAction[];
}

interface StatusRef {
  current: ReplayStatus;
}

interface SpeedRef {
  current: number;
}

interface IsTypingRef {
  current: boolean;
}

interface AllTimeoutsRef {
  current: NodeJS.Timeout[];
}

// Add interface for new ClickDetail structure
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

let cursorRef: ElementRef = { current: null };
let statusRef: StatusRef = { current: ReplayStatus.IDLE };
let speedRef: SpeedRef = { current: DEFAULT_SPEED };
let actionsRef: ActionsRef = { current: [] };
let currentActionIndexRef = { current: 0 };
let isTypingRef: IsTypingRef = { current: false };
let timeoutIdsRef: AllTimeoutsRef = { current: [] };
let timeoutRef: TimeoutRef = { current: null };

function registerTimeout(timeoutId: NodeJS.Timeout): NodeJS.Timeout {
  timeoutIdsRef.current.push(timeoutId);
  return timeoutId;
}

function clearAllTimeouts(): void {
  timeoutIdsRef.current.forEach(clearTimeout);
  timeoutIdsRef.current = [];
}

/**
 * Convert Playwright-specific selectors to browser-compatible ones
 */
function convertToBrowserSelector(selector: string): string {
  if (!selector) return selector;

  // Remove :has-text() patterns and replace with data attributes or fallbacks
  if (selector.includes(":has-text(")) {
    const textMatch = selector.match(/:has-text\("([^"]+)"\)/);
    if (textMatch) {
      const text = textMatch[1];
      const tagMatch = selector.match(/^([a-zA-Z]+)/);
      const tagName = tagMatch ? tagMatch[1] : "*";

      // Try to find a better selector for this element
      const elements = Array.from(document.querySelectorAll(tagName));
      for (const element of elements) {
        if (element.textContent?.trim() === text) {
          // Try to create a unique selector for this element
          const uniqueSelector = createUniqueSelector(element);
          if (uniqueSelector && isValidSelector(uniqueSelector)) {
            return uniqueSelector;
          }
        }
      }

      // Fallback to just the tag name
      return tagName;
    }
  }

  // Remove other Playwright-specific patterns
  selector = selector.replace(/:visible/g, "");
  selector = selector.replace(/:enabled/g, "");
  selector = selector.replace(/>>.*$/g, ""); // Remove >> selectors

  return selector.trim();
}

/**
 * Find the best selector from ClickDetail structure (FIXED - browser compatible)
 */
function findBestSelector(clickDetail: ClickDetail): string | null {
  const { selectors } = clickDetail;

  // Try primary selector first (but convert it to browser-compatible)
  if (selectors.primary) {
    const convertedPrimary = convertToBrowserSelector(selectors.primary);
    if (convertedPrimary && isValidSelector(convertedPrimary)) {
      return convertedPrimary;
    }
  }

  // Try text-based selection for interactive elements
  if (
    selectors.text &&
    (selectors.tagName === "button" ||
      selectors.tagName === "a" ||
      selectors.role === "button" ||
      selectors.role === "link")
  ) {
    const textBasedSelector = findElementByText(
      selectors.tagName,
      selectors.text
    );
    if (textBasedSelector) {
      return textBasedSelector;
    }
  }

  // Try aria-label
  if (selectors.ariaLabel) {
    const ariaSelector = `[aria-label="${selectors.ariaLabel}"]`;
    if (isValidSelector(ariaSelector)) {
      return ariaSelector;
    }
  }

  // Try fallback selectors (convert each one)
  for (const fallback of selectors.fallbacks) {
    const convertedFallback = convertToBrowserSelector(fallback);
    if (convertedFallback && isValidSelector(convertedFallback)) {
      return convertedFallback;
    }
  }

  // Try role-based selector
  if (selectors.role) {
    const roleSelector = `[role="${selectors.role}"]`;
    if (isValidSelector(roleSelector)) {
      return roleSelector;
    }
  }

  // Try to create selector from element info
  if (clickDetail.elementInfo) {
    const fromElementInfo = createSelectorFromElementInfo(
      clickDetail.elementInfo
    );
    if (fromElementInfo && isValidSelector(fromElementInfo)) {
      return fromElementInfo;
    }
  }

  // Use tag name as last resort
  if (selectors.tagName && isValidSelector(selectors.tagName)) {
    return selectors.tagName;
  }

  return null;
}

/**
 * Create selector from element info
 */
function createSelectorFromElementInfo(elementInfo: any): string | null {
  const { tagName, id, className, attributes } = elementInfo;

  // Try ID first
  if (id) {
    const idSelector = `#${id}`;
    if (isValidSelector(idSelector)) {
      return idSelector;
    }
  }

  // Try data-testid from attributes
  if (attributes && attributes["data-testid"]) {
    const testIdSelector = `[data-testid="${attributes["data-testid"]}"]`;
    if (isValidSelector(testIdSelector)) {
      return testIdSelector;
    }
  }

  // Try class names
  if (className) {
    const classes = className
      .split(" ")
      .filter((cls: string) => cls && cls.length > 0);
    if (classes.length > 0) {
      // Try with the first stable-looking class
      for (const cls of classes) {
        if (!cls.match(/^[a-zA-Z0-9_-]*[0-9a-f]{6,}/) && cls.length < 30) {
          const classSelector = `${tagName}.${cls}`;
          if (isValidSelector(classSelector)) {
            const matches = document.querySelectorAll(classSelector);
            if (matches.length === 1) {
              return classSelector;
            }
          }
        }
      }
    }
  }

  // Try other attributes
  if (attributes) {
    const priorityAttrs = ["name", "type", "role", "aria-label"];
    for (const attr of priorityAttrs) {
      if (attributes[attr]) {
        const attrSelector = `${tagName}[${attr}="${attributes[attr]}"]`;
        if (isValidSelector(attrSelector)) {
          const matches = document.querySelectorAll(attrSelector);
          if (matches.length <= 3) {
            // Allow some duplicates but not too many
            return attrSelector;
          }
        }
      }
    }
  }

  return null;
}

/**
 * Find element by text content and return a selector for it
 */
function findElementByText(tagName: string, text: string): string | null {
  if (!text || text.length === 0) return null;

  const elements = Array.from(document.querySelectorAll(tagName));
  let matchingElement: Element | null = null;

  // Find exact text match first
  for (const element of elements) {
    const elementText = element.textContent?.trim();
    if (elementText === text) {
      matchingElement = element;
      break;
    }
  }

  // If no exact match, try partial match
  if (!matchingElement) {
    for (const element of elements) {
      const elementText = element.textContent?.trim();
      if (elementText && elementText.includes(text)) {
        matchingElement = element;
        break;
      }
    }
  }

  if (!matchingElement) return null;

  // Try to create a unique selector for this element
  const uniqueSelector = createUniqueSelector(matchingElement);
  if (uniqueSelector && isValidSelector(uniqueSelector)) {
    // Verify it's still the right element
    const foundElement = document.querySelector(uniqueSelector);
    if (foundElement && foundElement.textContent?.trim().includes(text)) {
      return uniqueSelector;
    }
  }

  return null;
}

/**
 * Enhanced findElement function (FIXED - no Playwright selectors)
 */
export function findElement(selector: string): Element | null {
  if (!selector || selector.trim() === "") return null;

  // Convert selector to browser-compatible first
  const browserSelector = convertToBrowserSelector(selector);

  // Try the converted selector first
  if (browserSelector && isValidSelector(browserSelector)) {
    const element = document.querySelector(browserSelector);
    if (element) return element;
  }

  // Enhanced element finding with multiple strategies
  const strategies = [
    () => findByDataTestId(selector),
    () => findByClass(selector),
    () => findById(selector),
    () => findByAriaLabel(selector),
    () => findByRole(selector),
    () => findByTextContent(selector),
    () => findByCoordinates(selector), // New strategy for coordinate-based finding
  ];

  for (const strategy of strategies) {
    try {
      const element = strategy();
      if (element) {
        console.log(
          `Found element using fallback strategy for selector: ${selector}`
        );
        return element;
      }
    } catch (error) {
      console.warn(`Strategy failed for selector ${selector}:`, error);
    }
  }

  console.warn(`Could not find element for selector: ${selector}`);
  return null;
}

/**
 * Find element by coordinates (fallback method)
 */
function findByCoordinates(selector: string): Element | null {
  // This is a very rough fallback - try to find clickable elements at common positions
  const clickableElements = document.querySelectorAll(
    'button, a, input, select, [role="button"], [onclick]'
  );

  // Return the first clickable element as a last resort
  if (clickableElements.length > 0) {
    return clickableElements[0];
  }

  return null;
}

/**
 * Find element by text content (browser-compatible)
 */
function findByTextContent(selector: string): Element | null {
  // Handle text patterns that might be in selectors
  let text: string | null = null;
  let tagName = "*";

  // Extract text from various patterns
  if (selector.includes('text="')) {
    const textMatch = selector.match(/text="([^"]+)"/);
    text = textMatch ? textMatch[1] : null;
  } else if (selector.includes('textContent="')) {
    const textMatch = selector.match(/textContent="([^"]+)"/);
    text = textMatch ? textMatch[1] : null;
  } else if (selector.includes(":has-text(")) {
    const textMatch = selector.match(/:has-text\("([^"]+)"\)/);
    text = textMatch ? textMatch[1] : null;
  }

  // Extract tag name
  const tagMatch = selector.match(/^([a-zA-Z]+)/);
  if (tagMatch) {
    tagName = tagMatch[1];
  }

  if (!text) return null;

  // Find elements by text content
  const elements = Array.from(document.querySelectorAll(tagName));
  for (const element of elements) {
    const elementText = element.textContent?.trim();
    if (elementText === text || elementText?.includes(text)) {
      return element;
    }
  }

  return null;
}

/**
 * Create a unique selector for an element (ENHANCED)
 */
function createUniqueSelector(element: Element): string | null {
  // Try ID first
  if (element.id && /^[a-zA-Z][\w-]*$/.test(element.id)) {
    const idSelector = `#${element.id}`;
    if (document.querySelectorAll(idSelector).length === 1) {
      return idSelector;
    }
  }

  // Try data-testid
  const testId = (element as HTMLElement).dataset?.testid;
  if (testId) {
    const testIdSelector = `[data-testid="${testId}"]`;
    if (document.querySelectorAll(testIdSelector).length === 1) {
      return testIdSelector;
    }
  }

  // Try aria-label
  const ariaLabel = element.getAttribute("aria-label");
  if (ariaLabel) {
    const ariaSelector = `[aria-label="${ariaLabel}"]`;
    if (document.querySelectorAll(ariaSelector).length === 1) {
      return ariaSelector;
    }
  }

  // Try combining tag with classes
  const tagName = element.tagName.toLowerCase();
  const classes = Array.from(element.classList).filter((cls) => {
    // Filter out dynamic/generated classes
    return (
      !cls.match(/^[a-zA-Z0-9_-]*[0-9a-f]{6,}/) && // hash-like patterns
      !cls.includes("emotion-") &&
      !cls.includes("css-") &&
      !cls.includes("module__") &&
      cls.length < 30
    ); // avoid very long class names
  });

  if (classes.length > 0) {
    // Try with different numbers of classes
    for (let i = 1; i <= Math.min(classes.length, 3); i++) {
      const classSelector = `${tagName}.${classes.slice(0, i).join(".")}`;
      if (isValidSelector(classSelector)) {
        const matches = document.querySelectorAll(classSelector);
        if (matches.length === 1) {
          return classSelector;
        }
      }
    }
  }

  // Try combining tag with attributes
  const attributes = ["type", "name", "role", "title"];
  for (const attr of attributes) {
    const value = element.getAttribute(attr);
    if (value) {
      const attrSelector = `${tagName}[${attr}="${value}"]`;
      if (isValidSelector(attrSelector)) {
        const matches = document.querySelectorAll(attrSelector);
        if (matches.length === 1) {
          return attrSelector;
        }
      }
    }
  }

  // Try nth-child approach
  const parent = element.parentElement;
  if (parent) {
    const siblings = Array.from(parent.children);
    const index = siblings.indexOf(element);
    if (index >= 0) {
      const nthSelector = `${tagName}:nth-child(${index + 1})`;
      if (isValidSelector(nthSelector)) {
        return nthSelector;
      }
    }

    // Try nth-of-type
    const typeSiblings = Array.from(parent.children).filter(
      (child) => child.tagName === element.tagName
    );
    const typeIndex = typeSiblings.indexOf(element);
    if (typeIndex >= 0) {
      const nthTypeSelector = `${tagName}:nth-of-type(${typeIndex + 1})`;
      if (isValidSelector(nthTypeSelector)) {
        return nthTypeSelector;
      }
    }
  }

  // Last resort: just the tag name
  return tagName;
}

/**
 * Enhanced convertToReplayActions with better selector handling
 */
export function convertToReplayActions(trackingData: Results): ReplayAction[] {
  if (!trackingData) {
    console.error("No tracking data provided to convertToReplayActions");
    return [];
  }
  console.log("Converting tracking data to replay actions:", trackingData);

  const actions: ReplayAction[] = [];

  try {
    const { clicks, inputChanges, formElementChanges } = trackingData;

    // Handle clicks with new ClickDetail structure
    if (clicks?.clickDetails?.length) {
      clicks.clickDetails.forEach(
        (
          clickDetail: ClickDetail | [number, number, string, number],
          index: number
        ) => {
          // New ClickDetail format
          const detail = clickDetail as ClickDetail;
          let bestSelector = findBestSelector(detail);

          // If we couldn't find a good selector, create a fallback
          if (!bestSelector) {
            console.warn(
              "Could not find valid selector for click detail:",
              detail
            );

            // Try to find element by text if available
            if (detail.selectors.text) {
              const textElement = findElementByText(
                detail.selectors.tagName,
                detail.selectors.text
              );
              if (textElement) {
                bestSelector = textElement;
              }
            }

            // Last resort: use tag name with coordinates for manual targeting
            if (!bestSelector) {
              bestSelector = detail.selectors.tagName || "div";
            }
          }

          if (bestSelector) {
            actions.push({
              type: ActionType.CLICK,
              selector: bestSelector,
              timestamp: detail.timestamp,
              x: detail.x,
              y: detail.y,
            });
          }
        }
      );
    }

    // Handle input changes (unchanged logic)
    if (inputChanges?.length) {
      const completedInputs = inputChanges.filter(
        (change: any) => change.action === "complete" || !change.action
      );

      completedInputs.forEach((change: any) => {
        if (
          !change.elementSelector.includes('type="checkbox"') &&
          !change.elementSelector.includes('type="radio"')
        ) {
          const validSelector = validateAndFixSelector(change.elementSelector);
          if (validSelector) {
            actions.push({
              type: ActionType.INPUT,
              selector: validSelector,
              value: change.value,
              timestamp: change.timestamp,
            });
          }
        }
      });
    }

    // Handle form element changes (unchanged logic)
    if (formElementChanges?.length) {
      formElementChanges.forEach((change: any) => {
        if (!change.elementSelector) return;

        const validSelector = validateAndFixSelector(change.elementSelector);
        if (!validSelector) return;

        if (change.type === "checkbox" || change.type === "radio") {
          actions.push({
            type: change.checked ? ActionType.CHECK : ActionType.UNCHECK,
            selector: validSelector,
            value: change.value,
            timestamp: change.timestamp,
          });
        } else if (change.type === "select") {
          actions.push({
            type: ActionType.SELECT,
            selector: validSelector,
            value: change.value,
            timestamp: change.timestamp,
          });
        }
      });
    }
  } catch (e) {
    console.error("Error processing tracking data", e);
  }

  if (actions.length === 0) {
    console.warn("No actions extracted from tracking data");
  }

  // Sort actions and ensure minimum delays
  actions.sort((a, b) => a.timestamp - b.timestamp);

  for (let i = 1; i < actions.length; i++) {
    if (actions[i].timestamp - actions[i - 1].timestamp < 250) {
      actions[i].timestamp = actions[i - 1].timestamp + 250;
    }
  }

  console.log("Converted replay actions:", actions);
  const cleanedActions = actions.map((action) => {
    return {
      type: action.type || "click",
      selector: action.selector || "[data-testid]",
      timestamp: action.timestamp || Date.now(),
      x: action.x || 100,
      y: action.y || 100,
      value: action.value || "",
    };
  });

  return cleanedActions;
}

/**
 * Validate and fix selectors (enhanced version)
 */
function validateAndFixSelector(selector: string): string | null {
  if (!selector || selector === "undefined" || selector === "null") {
    return null;
  }

  const cleanSelector = selector.trim();

  // Convert to browser-compatible first
  const browserSelector = convertToBrowserSelector(cleanSelector);

  // Test if converted selector is valid
  if (browserSelector && isValidSelector(browserSelector)) {
    return browserSelector;
  }

  console.warn(`Invalid selector: ${cleanSelector}. Attempting to fix.`);

  // Try various fixing strategies
  const fixStrategies = [
    () => fixDataTestIdSelector(cleanSelector),
    () => fixClassSelector(cleanSelector),
    () => fixIdSelector(cleanSelector),
    () => fixTagSelector(cleanSelector),
    () => fixTextSelector(cleanSelector),
  ];

  for (const strategy of fixStrategies) {
    const fixedSelector = strategy();
    if (fixedSelector && isValidSelector(fixedSelector)) {
      console.log(`Fixed selector: ${cleanSelector} -> ${fixedSelector}`);
      return fixedSelector;
    }
  }

  console.error(`Could not fix selector: ${cleanSelector}`);
  return null;
}

function fixTextSelector(selector: string): string | null {
  // Handle :has-text() patterns
  if (selector.includes(":has-text(")) {
    const textMatch = selector.match(/:has-text\("([^"]+)"\)/);
    const tagMatch = selector.match(/^([a-zA-Z]+)/);

    if (textMatch && tagMatch) {
      const text = textMatch[1];
      const tagName = tagMatch[1];

      // Find element by text and create a selector for it
      const elements = Array.from(document.querySelectorAll(tagName));
      for (const element of elements) {
        if (element.textContent?.trim() === text) {
          const uniqueSelector = createUniqueSelector(element);
          if (uniqueSelector && isValidSelector(uniqueSelector)) {
            return uniqueSelector;
          }
        }
      }

      // Fallback to tag name
      return tagName;
    }
  }

  return null;
}

function fixDataTestIdSelector(selector: string): string | null {
  if (!selector.includes("data-testid=")) return null;

  const testIdMatch = selector.match(/data-testid="([^"]+)"/);
  if (testIdMatch && testIdMatch[1]) {
    return `[data-testid="${testIdMatch[1]}"]`;
  }
  return null;
}

function fixClassSelector(selector: string): string | null {
  if (!selector.includes("class=")) return null;

  const classMatch = selector.match(/class="([^"]+)"/);
  if (classMatch && classMatch[1]) {
    const classNames = classMatch[1].split(" ").filter((cls) => cls.length > 0);
    if (classNames.length > 0) {
      return `.${classNames[0]}`;
    }
  }
  return null;
}

function fixIdSelector(selector: string): string | null {
  if (!selector.includes("id=")) return null;

  const idMatch = selector.match(/id="([^"]+)"/);
  if (idMatch && idMatch[1]) {
    return `#${idMatch[1]}`;
  }
  return null;
}

function fixTagSelector(selector: string): string | null {
  const tagMatch = selector.match(/^([a-zA-Z]+)/);
  if (tagMatch && tagMatch[1]) {
    return tagMatch[1];
  }
  return null;
}

function isValidSelector(selector: string): boolean {
  if (!selector || selector.trim() === "") return false;

  try {
    document.querySelector(selector);
    return true;
  } catch (e) {
    return false;
  }
}

function findByDataTestId(selector: string): Element | null {
  if (!selector.includes("data-testid")) return null;

  const testId = selector.match(/data-testid="([^"]+)"/)?.[1];
  if (testId) {
    return document.querySelector(`[data-testid="${testId}"]`);
  }
  return null;
}

function findByClass(selector: string): Element | null {
  if (!selector.includes(".")) return null;

  const classes = selector.match(/\.([^\s.#\[\]]+)/g);
  if (classes && classes.length > 0) {
    const className = classes[0].substring(1);
    return document.querySelector(`.${className}`);
  }
  return null;
}

function findById(selector: string): Element | null {
  if (!selector.includes("#")) return null;

  const ids = selector.match(/#([^\s.#\[\]]+)/g);
  if (ids && ids.length > 0) {
    const id = ids[0].substring(1);
    return document.querySelector(`#${id}`);
  }
  return null;
}

function findByAriaLabel(selector: string): Element | null {
  const ariaMatch = selector.match(/\[aria-label="([^"]+)"\]/);
  if (!ariaMatch) return null;

  return document.querySelector(`[aria-label="${ariaMatch[1]}"]`);
}

function findByRole(selector: string): Element | null {
  const roleMatch = selector.match(/\[role="([^"]+)"\]/);
  if (!roleMatch) return null;

  return document.querySelector(`[role="${roleMatch[1]}"]`);
}

// Rest of the functions remain the same...
export function createCursor(): HTMLElement {
  const cursor = document.createElement("div");
  cursor.className = "replay-cursor";
  cursor.style.position = "fixed";
  cursor.style.width = "24px";
  cursor.style.height = "24px";
  cursor.style.borderRadius = "50%";
  cursor.style.backgroundColor = "rgba(255, 0, 0, 0.6)";
  cursor.style.border = "2px solid white";
  cursor.style.boxShadow = "0 0 10px rgba(255, 0, 0, 0.8)";
  cursor.style.zIndex = "9999";
  cursor.style.pointerEvents = "none";
  cursor.style.display = "none";
  cursor.style.transform = "translate(-50%, -50%)";
  cursor.style.transition =
    "transform 0.1s ease-in-out, left 0.1s ease-in-out, top 0.1s ease-in-out";
  document.body.appendChild(cursor);
  return cursor;
}

export function createReplayStyles(): HTMLStyleElement {
  const style = document.createElement("style");
  style.textContent = `
    @keyframes staktrak-click-ripple {
      0% {
        transform: translate(-50%, -50%) scale(1);
        opacity: 1;
      }
      100% {
        transform: translate(-50%, -50%) scale(8);
        opacity: 0;
      }
    }
    
    @keyframes staktrak-pulse {
      0% { transform: scale(1); }
      50% { transform: scale(1.03); }
      100% { transform: scale(1); }
    }
    
    .staktrak-replay-pulse {
      animation: staktrak-pulse 0.5s ease-in-out infinite;
    }
  `;
  document.head.appendChild(style);
  return style;
}

export function showClickEffect(cursorRef: ElementRef): void {
  if (!cursorRef.current) return;

  const ripple = document.createElement("div");
  ripple.className = "staktrak-click-ripple";
  ripple.style.position = "fixed";
  ripple.style.left = cursorRef.current.style.left;
  ripple.style.top = cursorRef.current.style.top;
  ripple.style.transform = "translate(-50%, -50%)";
  ripple.style.width = "20px";
  ripple.style.height = "20px";
  ripple.style.background = "rgba(255, 0, 0, 0.5)";
  ripple.style.borderRadius = "50%";
  ripple.style.zIndex = "9998";
  ripple.style.pointerEvents = "none";
  ripple.style.animation = "staktrak-click-ripple 1s ease-out forwards";
  document.body.appendChild(ripple);

  cursorRef.current.style.transform = "translate(-50%, -50%) scale(0.8)";
  setTimeout(() => {
    if (cursorRef.current) {
      cursorRef.current.style.transform = "translate(-50%, -50%) scale(1)";
    }
  }, 200);

  setTimeout(() => {
    if (ripple.parentNode) {
      ripple.parentNode.removeChild(ripple);
    }
  }, 1000);
}

export function highlightElement(element: Element, speedRef: SpeedRef): void {
  // const originalOutline = (element as HTMLElement).style.outline;
  // const originalBoxShadow = (element as HTMLElement).style.boxShadow;
  // const originalZIndex = (element as HTMLElement).style.zIndex;
  // const originalTransition = (element as HTMLElement).style.transition;

  // (element as HTMLElement).style.transition = "all 0.3s ease-in-out";
  // (element as HTMLElement).style.outline = "3px solid #ff3333";
  // (element as HTMLElement).style.boxShadow = "0 0 15px rgba(255, 51, 51, 0.7)";
  // (element as HTMLElement).style.zIndex = "1000";

  element.classList.add("staktrak-replay-pulse");

  setTimeout(() => {
    if (element) {
      // (element as HTMLElement).style.outline = originalOutline;
      // (element as HTMLElement).style.boxShadow = originalBoxShadow;
      // (element as HTMLElement).style.zIndex = originalZIndex;
      // (element as HTMLElement).style.transition = originalTransition;
      element.classList.remove("staktrak-replay-pulse");
    }
  }, 200 / speedRef.current);
}

export function moveCursorToElement(
  element: Element | null,
  cursorRef: ElementRef,
  statusRef: StatusRef
): Promise<void> {
  return new Promise((resolve) => {
    if (
      !element ||
      !cursorRef.current ||
      statusRef.current !== ReplayStatus.PLAYING
    ) {
      resolve();
      return;
    }

    const rect = element.getBoundingClientRect();
    const targetX = rect.left + rect.width / 2;
    const targetY = rect.top + rect.height / 2;

    cursorRef.current.style.display = "block";

    element.scrollIntoView({ behavior: "smooth", block: "center" });

    setTimeout(() => {
      if (cursorRef.current) {
        cursorRef.current.style.left = `${targetX}px`;
        cursorRef.current.style.top = `${targetY}px`;
      }

      setTimeout(resolve, 300);
    }, 150);
  });
}

export function typeText(
  element: HTMLInputElement,
  value: string,
  speedRef: SpeedRef,
  statusRef: StatusRef,
  isTypingRef: IsTypingRef,
  registerTimeout: (timeout: NodeJS.Timeout) => NodeJS.Timeout
): Promise<void> {
  return new Promise((resolve) => {
    if (statusRef.current !== ReplayStatus.PLAYING) {
      resolve();
      return;
    }

    isTypingRef.current = true;
    element.focus();
    element.value = "";

    let index = 0;
    const typeChar = () => {
      if (statusRef.current !== ReplayStatus.PLAYING) {
        isTypingRef.current = false;
        resolve();
        return;
      }

      if (index < value.length) {
        element.value += value[index];
        element.dispatchEvent(new Event("input", { bubbles: true }));
        index++;
        registerTimeout(setTimeout(typeChar, 70 / speedRef.current));
      } else {
        element.dispatchEvent(new Event("change", { bubbles: true }));
        isTypingRef.current = false;
        resolve();
      }
    };

    typeChar();
  });
}

export function selectOption(
  element: HTMLSelectElement,
  value: string,
  speedRef: SpeedRef,
  statusRef: StatusRef,
  registerTimeout: (timeout: NodeJS.Timeout) => NodeJS.Timeout
): Promise<void> {
  return new Promise((resolve) => {
    if (statusRef.current !== ReplayStatus.PLAYING) {
      resolve();
      return;
    }

    element.focus();

    registerTimeout(
      setTimeout(() => {
        if (statusRef.current === ReplayStatus.PLAYING) {
          element.value = value;
          element.dispatchEvent(new Event("change", { bubbles: true }));
        }
        resolve();
      }, 50 / speedRef.current)
    );
  });
}

export async function executeAction(
  index: number,
  actionsRef: ActionsRef,
  statusRef: StatusRef,
  cursorRef: ElementRef,
  speedRef: SpeedRef,
  isTypingRef: IsTypingRef,
  registerTimeout: (timeout: NodeJS.Timeout) => NodeJS.Timeout,
  setCurrentActionIndex: (index: number) => void,
  setStatus: (status: ReplayStatus) => void,
  timeoutRef: TimeoutRef
): Promise<void> {
  if (statusRef.current !== ReplayStatus.PLAYING) return;

  if (index >= actionsRef.current.length) {
    setCurrentActionIndex(actionsRef.current.length - 1);
    setStatus(ReplayStatus.COMPLETED);

    window.parent.postMessage(
      {
        type: "staktrak-replay-completed",
        totalActions: actionsRef.current.length,
      },
      "*"
    );

    setTimeout(() => {
      window.parent.postMessage({ type: "staktrak-replay-fadeout" }, "*");
    }, 100);

    if (cursorRef.current) {
      cursorRef.current.style.display = "none";
    }
    return;
  }

  const action = actionsRef.current[index];

  try {
    let element = findElement(action.selector);
    let attempts = 0;
    const maxAttempts = 5;

    while (!element && attempts < maxAttempts) {
      attempts++;
      const delay = Math.min(500 * attempts, 2000);
      await new Promise((resolve) => setTimeout(resolve, delay));
      element = findElement(action.selector);

      if (statusRef.current !== ReplayStatus.PLAYING) return;
    }

    if (!element) {
      console.warn(
        `Could not find element for action ${index}: ${action.type} on ${action.selector} after ${maxAttempts} attempts`
      );

      setCurrentActionIndex(index + 1);
      executeAction(
        index + 1,
        actionsRef,
        statusRef,
        cursorRef,
        speedRef,
        isTypingRef,
        registerTimeout,
        setCurrentActionIndex,
        setStatus,
        timeoutRef
      );
      return;
    }

    setCurrentActionIndex(index);

    window.parent.postMessage(
      {
        type: "staktrak-replay-progress",
        currentAction: index,
        totalActions: actionsRef.current.length,
        action: action,
      },
      "*"
    );

    await moveCursorToElement(element, cursorRef, statusRef);

    if (statusRef.current !== ReplayStatus.PLAYING) return;

    highlightElement(element, speedRef);

    switch (action.type) {
      case ActionType.CLICK:
        showClickEffect(cursorRef);

        element.scrollIntoView({ behavior: "smooth", block: "center" });
        await new Promise((resolve) => setTimeout(resolve, 50));

        try {
          (element as HTMLElement).focus();
        } catch (e) {
          console.warn("Could not focus element:", e);
        }

        try {
          element.dispatchEvent(
            new MouseEvent("mousedown", {
              bubbles: true,
              cancelable: true,
              view: window,
            })
          );

          await new Promise((resolve) => setTimeout(resolve, 10));

          element.dispatchEvent(
            new MouseEvent("mouseup", {
              bubbles: true,
              cancelable: true,
              view: window,
            })
          );

          await new Promise((resolve) => setTimeout(resolve, 10));

          (element as HTMLElement).click();
          element.dispatchEvent(
            new MouseEvent("click", {
              bubbles: true,
              cancelable: true,
              view: window,
            })
          );
        } catch (clickError) {
          console.error("Error during click operation:", clickError);
        }

        await new Promise((resolve) => setTimeout(resolve, 50));
        break;

      case ActionType.INPUT:
        await typeText(
          element as HTMLInputElement,
          action.value || "",
          speedRef,
          statusRef,
          isTypingRef,
          registerTimeout
        );
        break;

      case ActionType.SELECT:
        await selectOption(
          element as HTMLSelectElement,
          action.value || "",
          speedRef,
          statusRef,
          registerTimeout
        );
        break;

      case ActionType.CHECK:
        (element as HTMLInputElement).checked = true;
        element.dispatchEvent(new Event("change", { bubbles: true }));
        break;

      case ActionType.UNCHECK:
        (element as HTMLInputElement).checked = false;
        element.dispatchEvent(new Event("change", { bubbles: true }));
        break;
    }

    if (statusRef.current !== ReplayStatus.PLAYING) return;

    const nextAction = actionsRef.current[index + 1];
    let delay = 500;

    if (nextAction && action.timestamp && nextAction.timestamp) {
      const timeDiff = nextAction.timestamp - action.timestamp;
      delay = Math.min(
        MAX_DELAY,
        Math.max(MIN_DELAY, timeDiff / speedRef.current)
      );
    }

    timeoutRef.current = registerTimeout(
      setTimeout(() => {
        if (statusRef.current === ReplayStatus.PLAYING) {
          setCurrentActionIndex(index + 1);
          executeAction(
            index + 1,
            actionsRef,
            statusRef,
            cursorRef,
            speedRef,
            isTypingRef,
            registerTimeout,
            setCurrentActionIndex,
            setStatus,
            timeoutRef
          );
        } else {
          console.warn(
            `Not moving to next action because status is ${statusRef.current}`
          );
        }
      }, delay)
    );
  } catch (error) {
    console.error("Error executing action:", error);
    window.parent.postMessage(
      {
        type: "staktrak-replay-error",
        error: (error as Error).message,
        action: action,
      },
      "*"
    );

    if (statusRef.current === ReplayStatus.PLAYING) {
      timeoutRef.current = registerTimeout(
        setTimeout(() => {
          setCurrentActionIndex(index + 1);
          executeAction(
            index + 1,
            actionsRef,
            statusRef,
            cursorRef,
            speedRef,
            isTypingRef,
            registerTimeout,
            setCurrentActionIndex,
            setStatus,
            timeoutRef
          );
        }, 2000)
      );
    }
  }
}

export function pauseReplay(): void {
  if (statusRef.current === ReplayStatus.PLAYING) {
    statusRef.current = ReplayStatus.PAUSED;
    window.parent.postMessage({ type: "staktrak-replay-paused" }, "*");
  }
}

export function resumeReplay(): void {
  if (statusRef.current === ReplayStatus.PAUSED) {
    statusRef.current = ReplayStatus.PLAYING;

    executeAction(
      currentActionIndexRef.current,
      actionsRef,
      statusRef,
      cursorRef,
      speedRef,
      isTypingRef,
      registerTimeout,
      (index) => {
        currentActionIndexRef.current = index;
      },
      (status) => {
        statusRef.current = status;
      },
      timeoutRef
    );

    window.parent.postMessage({ type: "staktrak-replay-resumed" }, "*");
  }
}

export function stopReplay(): void {
  clearAllTimeouts();
  statusRef.current = ReplayStatus.IDLE;

  if (cursorRef.current) {
    cursorRef.current.style.display = "none";
  }

  const ripples = document.querySelectorAll(".staktrak-click-ripple");
  ripples.forEach((ripple) => {
    if (ripple.parentNode) {
      ripple.parentNode.removeChild(ripple);
    }
  });

  const highlightedElements = document.querySelectorAll(
    ".staktrak-replay-pulse"
  );
  highlightedElements.forEach((element) => {
    element.classList.remove("staktrak-replay-pulse");
    (element as HTMLElement).style.outline = "";
    (element as HTMLElement).style.boxShadow = "";
    (element as HTMLElement).style.zIndex = "";
    (element as HTMLElement).style.transition = "";
  });

  window.parent.postMessage({ type: "staktrak-replay-stopped" }, "*");
}

export function initReplay(): void {
  const replayStyles = createReplayStyles();
  const cursor = createCursor();

  cursorRef.current = cursor;
  statusRef.current = ReplayStatus.IDLE;
  speedRef.current = DEFAULT_SPEED;
  actionsRef.current = [];
  currentActionIndexRef.current = 0;
  isTypingRef.current = false;
  timeoutIdsRef.current = [];
  timeoutRef.current = null;

  window.addEventListener("message", (event) => {
    const { data } = event;

    if (!data || !data.type) return;

    switch (data.type) {
      case "staktrak-replay-actions":
        const actions = convertToReplayActions(data.actions);
        actionsRef.current = actions || [];
        break;

      case "staktrak-replay-start":
        clearAllTimeouts();
        statusRef.current = ReplayStatus.PLAYING;
        currentActionIndexRef.current = 0;
        speedRef.current = data.speed || DEFAULT_SPEED;

        if (cursorRef.current) {
          cursorRef.current.style.display = "block";
        }

        setTimeout(() => {
          executeAction(
            0,
            actionsRef,
            statusRef,
            cursorRef,
            speedRef,
            isTypingRef,
            registerTimeout,
            (index) => {
              currentActionIndexRef.current = index;
            },
            (status) => {
              statusRef.current = status;
            },
            timeoutRef
          );
        }, INITIAL_DELAY / speedRef.current);
        break;

      case "staktrak-replay-pause":
        pauseReplay();
        break;

      case "staktrak-replay-resume":
        resumeReplay();
        break;

      case "staktrak-replay-stop":
        stopReplay();
        break;

      case "staktrak-replay-speed":
        speedRef.current = data.speed || DEFAULT_SPEED;
        break;

      case "staktrak-replay-ping":
        window.parent.postMessage({ type: "staktrak-replay-ready" }, "*");
        break;
    }
  });

  window.parent.postMessage({ type: "staktrak-replay-ready" }, "*");
}

export { DEFAULT_SPEED, MIN_DELAY, MAX_DELAY, INITIAL_DELAY };

document.addEventListener("DOMContentLoaded", () => {
  initReplay();
});

document.readyState === "loading"
  ? document.addEventListener("DOMContentLoaded", initReplay)
  : initReplay();
