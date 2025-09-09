// Enhanced utils.ts
import { Results, Assertion, Config, ClickDetail } from "./types";

export const getTimeStamp = (): number => Date.now();

export const isInputOrTextarea = (element: Element): boolean =>
  element.tagName === "INPUT" ||
  element.tagName === "TEXTAREA" ||
  (element as HTMLElement).isContentEditable;

/**
 * Generate multiple selector strategies for an element
 */
export const generateSelectorStrategies = (
  element: Element
): {
  primary: string;
  fallbacks: string[];
  text?: string;
  ariaLabel?: string;
  title?: string;
  role?: string;
  tagName: string;
  xpath?: string;
} => {
  const htmlEl = element as HTMLElement;
  const tagName = element.tagName.toLowerCase();
  const fallbacks: string[] = [];

  // Strategy 1: data-testid (highest priority)
  const testId = htmlEl.dataset?.testid;
  if (testId) {
    return {
      primary: `[data-testid="${testId}"]`,
      fallbacks: [],
      tagName,
      text: getElementText(element),
      ariaLabel: htmlEl.getAttribute("aria-label") || undefined,
      title: htmlEl.getAttribute("title") || undefined,
      role: htmlEl.getAttribute("role") || undefined,
    };
  }

  // Strategy 2: ID
  const id = htmlEl.id;
  if (id && /^[a-zA-Z][\w-]*$/.test(id)) {
    return {
      primary: `#${id}`,
      fallbacks: [],
      tagName,
      text: getElementText(element),
      ariaLabel: htmlEl.getAttribute("aria-label") || undefined,
      title: htmlEl.getAttribute("title") || undefined,
      role: htmlEl.getAttribute("role") || undefined,
    };
  }

  // Strategy 3: For interactive elements, try text-based selectors
  const text = getElementText(element);
  if (
    text &&
    (tagName === "button" ||
      tagName === "a" ||
      htmlEl.getAttribute("role") === "button")
  ) {
    const textSelector = generateTextBasedSelector(element, text);
    if (textSelector) {
      fallbacks.push(textSelector);
    }
  }

  // Strategy 4: aria-label
  const ariaLabel = htmlEl.getAttribute("aria-label");
  if (ariaLabel) {
    fallbacks.push(`[aria-label="${ariaLabel}"]`);
  }

  // Strategy 5: role + accessible name
  const role = htmlEl.getAttribute("role");
  if (role && text) {
    fallbacks.push(`[role="${role}"]`);
  }

  // Strategy 6: Clean class-based selector
  const classSelector = generateClassBasedSelector(element);
  if (classSelector && classSelector !== tagName) {
    fallbacks.push(classSelector);
  }

  // Strategy 7: Attribute-based selectors for inputs
  if (tagName === "input") {
    const type = (element as HTMLInputElement).type;
    const name = (element as HTMLInputElement).name;
    if (type) fallbacks.push(`input[type="${type}"]`);
    if (name) fallbacks.push(`input[name="${name}"]`);
  }

  // Strategy 8: Parent-child relationship for specific elements
  const contextualSelector = generateContextualSelector(element);
  if (contextualSelector) {
    fallbacks.push(contextualSelector);
  }

  // Strategy 9: XPath as last resort
  const xpath = generateXPath(element);

  // Choose primary selector
  const primary = fallbacks.length > 0 ? fallbacks[0] : tagName;

  return {
    primary,
    fallbacks: fallbacks.slice(1), // Remove primary from fallbacks
    text,
    ariaLabel: ariaLabel || undefined,
    title: htmlEl.getAttribute("title") || undefined,
    role: role || undefined,
    tagName,
    xpath,
  };
};

/**
 * Get clean text content from element
 */
export const getElementText = (element: Element): string | undefined => {
  const htmlEl = element as HTMLElement;

  // For buttons and links, get the visible text
  if (element.tagName === "BUTTON" || element.tagName === "A") {
    const text = htmlEl.textContent?.trim();
    if (text && text.length > 0 && text.length < 100) {
      return text;
    }
  }

  // For inputs, get placeholder or value
  if (element.tagName === "INPUT") {
    const input = element as HTMLInputElement;
    return input.placeholder || input.value || undefined;
  }

  return undefined;
};

/**
 * Generate text-based selector
 */
export const generateTextBasedSelector = (
  element: Element,
  text: string
): string | null => {
  const tagName = element.tagName.toLowerCase();

  // Clean text for selector use
  const cleanText = text.replace(/"/g, '\\"').trim();
  if (cleanText.length === 0 || cleanText.length > 50) return null;

  if (tagName === "button" || tagName === "a" || 
      (element as HTMLElement).getAttribute("role") === "button") {
    return `text=${cleanText}`;
  }

  return null;
};

/**
 * Generate clean class-based selector
 */
export const generateClassBasedSelector = (element: Element): string => {
  const tagName = element.tagName.toLowerCase();
  const classList = element.classList;

  if (!classList.length) return tagName;

  // Filter out problematic classes
  const safeClasses = Array.from(classList).filter((cls) => {
    // Skip generated/dynamic classes
    if (cls.includes("_") && cls.match(/[0-9a-f]{6}/)) return false;
    if (cls.includes("module__")) return false;
    if (cls.includes("emotion-")) return false;
    if (cls.includes("css-")) return false;
    if (cls.length > 30) return false;

    // Keep semantic classes
    return /^[a-zA-Z][a-zA-Z0-9-]*$/.test(cls);
  });

  if (safeClasses.length === 0) return tagName;

  // Limit to most important classes
  const limitedClasses = safeClasses.slice(0, 3);
  return `${tagName}.${limitedClasses.join(".")}`;
};

/**
 * Generate contextual selector based on parent elements
 */
export const generateContextualSelector = (element: Element): string | null => {
  const tagName = element.tagName.toLowerCase();
  const parent = element.parentElement;

  if (!parent) return null;

  // For buttons in navigation
  if (tagName === "button" && parent.tagName === "NAV") {
    return "nav button";
  }

  // For buttons in header
  if (
    tagName === "button" &&
    (parent.tagName === "HEADER" || parent.closest("header"))
  ) {
    return "header button";
  }

  // For form elements
  if ((tagName === "input" || tagName === "button") && parent.closest("form")) {
    return `form ${tagName}`;
  }

  return null;
};

/**
 * Generate XPath for element (last resort)
 */
export const generateXPath = (element: Element): string => {
  if (element.id) {
    return `//*[@id="${element.id}"]`;
  }

  const parts: string[] = [];
  let current: Element | null = element;

  while (current && current.nodeType === Node.ELEMENT_NODE) {
    let index = 1;
    let sibling = current.previousElementSibling;

    while (sibling) {
      if (sibling.tagName === current.tagName) {
        index++;
      }
      sibling = sibling.previousElementSibling;
    }

    const tagName = current.tagName.toLowerCase();
    const part = index > 1 ? `${tagName}[${index}]` : tagName;
    parts.unshift(part);

    current = current.parentElement;

    // Don't go too deep
    if (parts.length > 10) break;
  }

  return "/" + parts.join("/");
};

/**
 * Create enhanced click detail
 */
export const createClickDetail = (e: MouseEvent): ClickDetail => {
  const target = e.target as Element;
  const selectors = generateSelectorStrategies(target);

  return {
    x: e.clientX,
    y: e.clientY,
    timestamp: getTimeStamp(),
    selectors,
    elementInfo: {
      tagName: target.tagName.toLowerCase(),
      id: (target as HTMLElement).id || undefined,
      className: target.className || undefined,
      attributes: getElementAttributes(target),
    },
  };
};

/**
 * Get relevant attributes from element
 */
export const getElementAttributes = (
  element: Element
): Record<string, string> => {
  const attrs: Record<string, string> = {};
  const htmlEl = element as HTMLElement;

  // Capture important attributes
  const importantAttrs = [
    "type",
    "name",
    "role",
    "aria-label",
    "title",
    "placeholder",
    "value",
  ];

  importantAttrs.forEach((attr) => {
    const value = htmlEl.getAttribute(attr);
    if (value) attrs[attr] = value;
  });

  return attrs;
};

// Keep existing functions for backward compatibility
export const getElementSelector = (element: Element): string => {
  const strategies = generateSelectorStrategies(element);
  return strategies.primary;
};

export const createClickPath = (e: Event): string => {
  // Keep for backward compatibility, but now we use the enhanced method
  const target = e.target as Element;
  return generateSelectorStrategies(target).primary;
};

export const filterClickDetails = (
  clickDetails: ClickDetail[],
  assertions: Assertion[],
  config: Config
): ClickDetail[] => {
  if (!clickDetails.length) return [];

  let filtered = config.filterAssertionClicks
    ? clickDetails.filter(
        (click) =>
          !assertions.some(
            (assertion) =>
              Math.abs(click.timestamp - assertion.timestamp) < 1000 &&
              (click.selectors.primary.includes(assertion.selector) ||
                assertion.selector.includes(click.selectors.primary) ||
                click.selectors.fallbacks.some(
                  (f) =>
                    f.includes(assertion.selector) ||
                    assertion.selector.includes(f)
                ))
          )
      )
    : clickDetails;

  // Remove rapid multi-clicks
  const clicksBySelector: Record<string, ClickDetail[]> = {};

  filtered.forEach((click) => {
    const key = click.selectors.primary;
    if (!clicksBySelector[key]) clicksBySelector[key] = [];
    clicksBySelector[key].push(click);
  });

  const result: ClickDetail[] = [];
  Object.values(clicksBySelector).forEach((clicks) => {
    clicks.sort((a, b) => a.timestamp - b.timestamp);
    let lastClick: ClickDetail | null = null;

    clicks.forEach((click) => {
      if (
        !lastClick ||
        click.timestamp - lastClick.timestamp > config.multiClickInterval
      ) {
        result.push(click);
      }
      lastClick = click;
    });
  });

  return result.sort((a, b) => a.timestamp - b.timestamp);
};
