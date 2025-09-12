// Enhanced utils.ts (cleaned)
import { Results, Assertion, Config, ClickDetail } from "./types";

export const getTimeStamp = (): number => Date.now();

/**
 * Lightweight role derivation (fallback if no explicit role attribute)
 */
export const getElementRole = (el: HTMLElement): string | null => {
  // Explicit role always wins
  const explicit = el.getAttribute("role");
  if (explicit) return explicit;
  const tag = el.tagName.toLowerCase();
  if (tag === "button") return "button";
  if (tag === "a" && el.hasAttribute("href")) return "link";
  if (tag === "input") {
    const type = (el.getAttribute("type") || "text").toLowerCase();
    if (["button", "submit", "reset"].includes(type)) return "button";
    if (type === "checkbox") return "checkbox";
    if (type === "radio") return "radio";
    return "textbox";
  }
  if (tag === "select") return "combobox";
  if (tag === "textarea") return "textbox";
  if (tag === "nav") return "navigation";
  if (tag === "header") return "banner";
  if (tag === "footer") return "contentinfo";
  if (tag === "main") return "main";
  if (tag === "form") return "form";
  return null;
};

/**
 * Enhanced text extraction for interactive elements, especially buttons
 */
export const getEnhancedElementText = (element: HTMLElement): string | null => {
  // Priority order for getting accessible name:
  // 1. aria-label
  // 2. aria-labelledby (resolved)
  // 3. textContent (for buttons, links)
  // 4. title attribute
  // 5. value (for input buttons)
  // 6. placeholder (for inputs)
  
  const ariaLabel = element.getAttribute('aria-label');
  if (ariaLabel) return ariaLabel;
  
  // Try to resolve aria-labelledby
  const resolvedLabel = resolveAriaLabelledBy(element);
  if (resolvedLabel) return resolvedLabel;
  
  const tag = element.tagName.toLowerCase();
  
  // For buttons and links, get the visible text
  if (tag === 'button' || (tag === 'a' && element.hasAttribute('href'))) {
    const text = element.textContent?.trim();
    if (text && text.length > 0 && text.length < 100) {
      return text;
    }
  }
  
  // For inputs, get value, placeholder, or title
  if (tag === 'input') {
    const input = element as HTMLInputElement;
    return input.value || input.placeholder || input.getAttribute('title') || null;
  }
  
  // Fallbacks
  return element.getAttribute('title') || null;
};

/**
 * Get the closest semantic parent element for contextual selection
 */
export const getSemanticParent = (element: HTMLElement): HTMLElement | null => {
  const semanticTags = ['header', 'nav', 'main', 'footer', 'aside', 'section', 'article', 'form', 'dialog'];
  
  let parent = element.parentElement;
  while (parent) {
    const tag = parent.tagName.toLowerCase();
    if (semanticTags.includes(tag)) {
      return parent;
    }
    // Also check for elements with landmark roles
    const role = parent.getAttribute('role');
    if (role && ['navigation', 'banner', 'main', 'contentinfo', 'complementary', 'form', 'search'].includes(role)) {
      return parent;
    }
    parent = parent.parentElement;
  }
  
  return null;
};

/**
 * Detect icon content within an element
 */
export const detectIconContent = (element: HTMLElement): { type: string; selector: string } | null => {
  // Check for SVG icons
  const svg = element.querySelector('svg');
  if (svg) {
    // Check for common icon attributes
    if (svg.getAttribute('data-icon')) {
      return { type: 'svg', selector: `[data-icon="${svg.getAttribute('data-icon')}"]` };
    }
    if (svg.classList.length > 0) {
      const iconClass = Array.from(svg.classList).find(cls => cls.includes('icon'));
      if (iconClass) {
        return { type: 'svg', selector: `.${iconClass}` };
      }
    }
    return { type: 'svg', selector: 'svg' };
  }
  
  // Check for icon fonts (FontAwesome, Material Icons, etc.)
  const iconElement = element.querySelector('[class*="icon"], [class*="fa-"], [class*="material-icons"]');
  if (iconElement) {
    const iconClasses = Array.from(iconElement.classList).filter(cls => 
      cls.includes('icon') || cls.includes('fa-') || cls.includes('material')
    );
    if (iconClasses.length > 0) {
      return { type: 'icon-font', selector: `.${iconClasses[0]}` };
    }
  }
  
  // Check for emoji or unicode icons
  const text = element.textContent?.trim();
  if (text && text.length <= 2 && /[\u{1F300}-\u{1F9FF}]|[\u{2600}-\u{26FF}]|[\u{2700}-\u{27BF}]/u.test(text)) {
    return { type: 'emoji', selector: `text="${text}"` };
  }
  
  return null;
};

/**
 * Resolve aria-labelledby to get the actual text
 */
export const resolveAriaLabelledBy = (element: HTMLElement): string | null => {
  const labelledBy = element.getAttribute('aria-labelledby');
  if (!labelledBy) return null;
  
  // Handle multiple IDs
  const ids = labelledBy.split(' ').filter(id => id.trim());
  const texts: string[] = [];
  
  for (const id of ids) {
    // Note: In browser context, we'd use document.getElementById
    // In this context, we need to traverse the DOM tree
    const referencedEl = findElementById(element.ownerDocument || document, id);
    if (referencedEl) {
      const text = referencedEl.textContent?.trim();
      if (text) texts.push(text);
    }
  }
  
  return texts.length > 0 ? texts.join(' ') : null;
};

/**
 * Helper to find element by ID (for aria-labelledby resolution)
 */
const findElementById = (doc: Document, id: string): Element | null => {
  // In a real browser environment, this would just be document.getElementById
  // This is a fallback implementation for compatibility
  if (typeof doc.getElementById === 'function') {
    return doc.getElementById(id);
  }
  return doc.querySelector(`#${CSS.escape(id)}`);
};

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
      role: getElementRole(htmlEl) || undefined,
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
      role: getElementRole(htmlEl) || undefined,
    };
  }

  // Strategy 3: For interactive elements, try text-based selectors
  const text = getEnhancedElementText(htmlEl);
  const role = getElementRole(htmlEl);
  if (
    text &&
    (tagName === "button" ||
      tagName === "a" ||
      role === "button")
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
    text: text || undefined,
    ariaLabel: ariaLabel || undefined,
    title: htmlEl.getAttribute("title") || undefined,
    role: role || undefined,
    tagName,
    xpath,
  };
};

/**
 * Get clean text content from element (legacy function - use getEnhancedElementText for better results)
 */
export const getElementText = (element: Element): string | undefined => {
  const htmlEl = element as HTMLElement;
  return getEnhancedElementText(htmlEl) || undefined;
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
      getElementRole(element as HTMLElement) === "button") {
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
  // canonical enrichment
  const html = target as HTMLElement;
  const testId = (html.dataset && html.dataset['testid']) || undefined;
  const id = html.id || undefined;
  const accessibleName = getEnhancedElementText(html) || undefined;
  // nth-of-type among same tag siblings
  let nth: number | undefined;
  if (html.parentElement) {
    const same = Array.from(html.parentElement.children).filter(c => c.tagName === html.tagName);
    if (same.length > 1) nth = same.indexOf(html) + 1;
  }
  const ancestors: string[] = [];
  let p: HTMLElement | null = html.parentElement;
  let depth = 0;
  while (p && depth < 4) {
    const role = p.getAttribute('role');
    const tag = p.tagName.toLowerCase();
    if (['main','nav','header','footer','aside','section','form','article'].includes(tag) || role) {
      ancestors.push(role ? `${tag}[role=${role}]` : tag);
    }
    p = p.parentElement; depth++;
  }
    const selAny: any = selectors as any;
    selAny.id = id;
    selAny.testId = testId;
    selAny.accessibleName = accessibleName;
    if (nth) selAny.nth = nth;
    if (ancestors.length) selAny.ancestors = ancestors;

    // Uniqueness stabilization: if primary is generic (tag or simple class) and we have id/testId/text role data, promote a better one
    selectors.primary = chooseStablePrimary(html, selectors.primary, selectors.fallbacks, {
      testId,
      id,
      accessibleName,
      role: getElementRole(html) || undefined,
      nth,
    });

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
    "aria-labelledby",
    "aria-expanded",
    "aria-haspopup",
    "title",
    "placeholder",
    "value",
  ];

  importantAttrs.forEach((attr) => {
    const value = htmlEl.getAttribute(attr);
    if (value) attrs[attr] = value;
  });

  // Add contextual information for Phase 2 improvements
  const semanticParent = getSemanticParent(htmlEl);
  if (semanticParent) {
    attrs.semanticParent = semanticParent.tagName.toLowerCase();
  }

  // Add icon detection information
  const iconInfo = detectIconContent(htmlEl);
  if (iconInfo) {
    attrs.iconContent = iconInfo.selector;
  }

  // Try to resolve aria-labelledby for better accessible name
  const resolvedLabel = resolveAriaLabelledBy(htmlEl);
  if (resolvedLabel) {
    attrs.resolvedAriaLabel = resolvedLabel;
  }

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

// Determine if a selector is likely too weak (e.g., plain tag, tag.class) by simple heuristics
const isWeakSelector = (selector: string): boolean => {
  if (!selector) return true;
  if (/^\w+$/.test(selector)) return true; // just tag
  if (/^\w+\.[^.]+$/.test(selector)) return true; // tag.singleClass
  if (selector.startsWith('text=')) return false;
  if (selector.startsWith('[data-testid=')) return false;
  if (selector.startsWith('#')) return false;
  return false; // default assume okay
};

interface StableInputs {
  testId?: string;
  id?: string;
  accessibleName?: string;
  role?: string;
  nth?: number;
}

const chooseStablePrimary = (
  el: HTMLElement,
  current: string,
  fallbacks: string[],
  meta: StableInputs
): string => {
  if (!isWeakSelector(current)) return current;
  if (meta.testId) return `[data-testid="${meta.testId}"]`;
  if (meta.id && /^[a-zA-Z][\w-]*$/.test(meta.id)) return `#${meta.id}`;
  if (meta.role && meta.accessibleName && meta.accessibleName.length < 80) {
    // Playwright style text selector
    return `text=${meta.accessibleName.replace(/"/g,'\\"')}`;
  }
  // As last resort keep original
  return current;
};
