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
  // scoring additions
  scores?: { selector: string; score: number; reasons: string[] }[];
} => {
  const htmlEl = element as HTMLElement;
  const tagName = element.tagName.toLowerCase();
  const fallbacks: string[] = [];

  // INTERNAL scoring helpers ---------------------------------------------
  const reasonsMap: Record<string,string[]> = {};
  const scored: { selector: string; score: number; reasons: string[] }[] = [];
  const pushCandidate = (sel: string, baseScore: number, reason: string) => {
    if (!sel) return;
    if (!reasonsMap[sel]) reasonsMap[sel] = [];
    reasonsMap[sel].push(reason);
    // compute dynamic penalties
    let score = baseScore;
    // length penalty
    if (sel.length > 60) score -= Math.min(20, Math.floor((sel.length-60)/5));
    // depth penalty
    const depth = sel.split('>').length - 1;
    if (depth > 3) score -= (depth-3)*2;
    // volatility penalty: heuristic for classes with hex fragments
    if (/\.[a-zA-Z0-9_-]*[0-9a-f]{6,}\b/.test(sel)) score -= 25;
    // generic tag penalty
    if (/^\w+$/.test(sel)) score -= 30;
    // text selector slight risk
    if (sel.startsWith('text=')) score -= 5;
    // role selector moderate
    if (sel.startsWith('role:')) score -= 3;
    scored.push({ selector: sel, score, reasons: reasonsMap[sel] });
  };

  const finalizeReturn = (primary: string, fallbacks: string[], extra: any) => {
    // de-duplicate scored entries
    const dedup: Record<string,{ selector: string; score: number; reasons: string[] }> = {};
    for (const c of scored) {
      if (!dedup[c.selector] || dedup[c.selector].score < c.score) dedup[c.selector] = c;
    }
    const ordered = Object.values(dedup).sort((a,b)=>b.score-a.score);
    return { primary, fallbacks, ...extra, scores: ordered };
  };

  // Strategy 1: data-testid (highest priority)
  const testId = htmlEl.dataset?.testid;
  if (testId) {
    const sel = `[data-testid="${testId}"]`;
    pushCandidate(sel, 100, 'data-testid attribute');
    return finalizeReturn(sel, [], {
      tagName,
      text: getElementText(element),
      ariaLabel: htmlEl.getAttribute("aria-label") || undefined,
      title: htmlEl.getAttribute("title") || undefined,
      role: getElementRole(htmlEl) || undefined,
    });
  }

  // Strategy 2: ID
  const id = htmlEl.id;
  if (id && /^[a-zA-Z][\w-]*$/.test(id)) {
    const sel = `#${id}`;
    pushCandidate(sel, 95, 'element id');
    return finalizeReturn(sel, [], {
      tagName,
      text: getElementText(element),
      ariaLabel: htmlEl.getAttribute("aria-label") || undefined,
      title: htmlEl.getAttribute("title") || undefined,
      role: getElementRole(htmlEl) || undefined,
    });
  }

  // Strategy 3: Clean class-based selector first (prefer structural primary)
  const text = getEnhancedElementText(htmlEl);
  const role = getElementRole(htmlEl);
  const classSelector = generateClassBasedSelector(element);
  if (classSelector && classSelector !== tagName) {
    fallbacks.push(classSelector);
    pushCandidate(classSelector, 80, 'class-based selector');
  }

  // Strategy 4: Role+name (only if not already uniquely represented by class)
  if (!testId && !id && role === 'button' && text && text.length < 40) {
    const rsel = `role:button[name="${text.replace(/"/g,'\\"')}"]`;
    fallbacks.push(rsel);
    pushCandidate(rsel, 70, 'role+name');
  }

  // Strategy 5: For interactive elements, text-based selectors (after structural + role)
  const text2 = getEnhancedElementText(htmlEl);
  // recompute not needed but keep variable for clarity
  if (
    text2 &&
    (tagName === "button" ||
      tagName === "a" ||
      role === "button")
  ) {
    const textSelector = generateTextBasedSelector(element, text2);
    if (textSelector) {
      fallbacks.push(textSelector);
      pushCandidate(textSelector, 60, 'text content');
    }
  }

  // Strategy 6: aria-label
  const ariaLabel = htmlEl.getAttribute("aria-label");
  if (ariaLabel) {
    const al = `[aria-label="${ariaLabel}"]`;
    fallbacks.push(al);
    pushCandidate(al, 65, 'aria-label');
  }

  // Strategy 7: generic role fallback only (without name) if not already pushed
  if (role && !fallbacks.find(f=>f.startsWith('role:')||f.startsWith(`[role="${role}`))) {
    const rs = `[role="${role}"]`;
    fallbacks.push(rs);
    pushCandidate(rs, 40, 'generic role');
  }

  // Strategy 8: Attribute-based selectors for inputs
  if (tagName === "input") {
    const type = (element as HTMLInputElement).type;
    const name = (element as HTMLInputElement).name;
    if (type) { const tsel = `input[type="${type}"]`; fallbacks.push(tsel); pushCandidate(tsel, 55, 'input type'); }
    if (name) { const nsel = `input[name="${name}"]`; fallbacks.push(nsel); pushCandidate(nsel, 58, 'input name'); }
  }

  // Strategy 9: Parent-child relationship for specific elements (reduced to avoid broad container > button selectors)
  const contextualSelector = generateContextualSelector(element);
  if (contextualSelector) {
    fallbacks.push(contextualSelector);
    pushCandidate(contextualSelector, 45, 'contextual');
  }

  // Strategy 9: XPath as last resort
  const xpath = generateXPath(element);

  // Choose primary selector
  // Add bare tag candidate (low score) if nothing else
  if (fallbacks.length === 0) {
    pushCandidate(tagName, 10, 'bare tag');
  }

  // choose top scoring unique-ish candidate as primary: sort by score (scored already holds most) but restrict to ones we actually added
  const best = scored.sort((a,b)=>b.score-a.score)[0];
  const primary = best ? best.selector : (fallbacks.length>0? fallbacks[0] : tagName);
  const fb = fallbacks.filter(f=>f!==primary);
  return finalizeReturn(primary, fb, {
    text: text || undefined,
    ariaLabel: ariaLabel || undefined,
    title: htmlEl.getAttribute("title") || undefined,
    role: role || undefined,
    tagName,
    xpath,
  });
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
    const stabilized = chooseStablePrimary(html, selectors.primary, selectors.fallbacks, {
      testId,
      id,
      accessibleName,
      role: getElementRole(html) || undefined,
      nth,
    });
    let uniqueStabilized = ensureStabilizedUnique(html, stabilized);
    // Immediate capture validation loop (simple): if still not unique in DOM (and not text=) try ancestor builder directly
    try {
      if (typeof document !== 'undefined' && !uniqueStabilized.startsWith('text=')) {
        const matches = document.querySelectorAll(uniqueStabilized);
        if (matches.length !== 1) {
          const ancestorOnly = buildAncestorNthSelector(html);
          if (ancestorOnly && ancestorOnly !== uniqueStabilized) {
            const mm = document.querySelectorAll(ancestorOnly);
            if (mm.length === 1) uniqueStabilized = ancestorOnly;
          }
        }
      }
    } catch { /* ignore */ }
  (selectors as any).stabilizedPrimary = uniqueStabilized;
  selectors.primary = uniqueStabilized;
  // Provide a guaranteed CSS visualSelector for highlighter (avoid text=/role: only)
  let visualSelector: string | null = null;
  const isCssResolvable = (s: string) => !s.startsWith('text=') && !s.startsWith('role:');
  if (isCssResolvable(uniqueStabilized)) visualSelector = uniqueStabilized;
  else {
    const fbCss = (selectors.fallbacks || []).find(isCssResolvable);
    if (fbCss) visualSelector = fbCss; else {
      // attempt ancestor nth build
      const anc = buildAncestorNthSelector(html);
      if (anc) visualSelector = anc;
    }
  }
  if (visualSelector) (selectors as any).visualSelector = visualSelector;

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

// Determine if a selector is weak. We now treat a single class selector as strong if it is unique.
const isWeakSelector = (selector: string, el?: HTMLElement): boolean => {
  if (!selector) return true;
  if (selector.startsWith('[data-testid=')) return false;
  if (selector.startsWith('#')) return false;
  if (selector.startsWith('text=')) return false; // text selectors are deliberate
  if (/^\w+$/.test(selector)) return true; // bare tag
  if (/^\w+\.[^.]+$/.test(selector)) {
    // tag.singleClass -> check uniqueness; if unique treat as strong
    if (el && typeof document !== 'undefined') {
      try {
        const count = document.querySelectorAll(selector).length;
        if (count === 1) return false; // strong enough
      } catch {}
    }
    return true;
  }
  return false;
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
  // If already strong, keep it.
  if (!isWeakSelector(current, el)) return current;
  // Prefer testId
  if (meta.testId) return `[data-testid="${meta.testId}"]`;
  // Then ID
  if (meta.id && /^[a-zA-Z][\w-]*$/.test(meta.id)) return `#${meta.id}`;
  // Before using text, see if any fallback structural selector is unique.
  if (typeof document !== 'undefined') {
    const structural = [current, ...fallbacks].filter(s => s && !s.startsWith('text=') && !s.startsWith('[') && !s.startsWith('#'));
    for (const s of structural) {
      try {
        if (document.querySelectorAll(s).length === 1) {
          return s; // use the unique structural selector instead of text
        }
      } catch {}
    }
  }
  // Role + accessibleName -> prefer explicit getByRole semantics (we encode as role: for replay, but for primary keep structural if possible)
  if (meta.role && meta.accessibleName && meta.accessibleName.length < 60) {
    // Use role marker instead of raw text= for traceability; executor already understands role:
    return `role:${meta.role}[name="${meta.accessibleName.replace(/"/g,'\\"')}"]`;
  }
  // Fallback: if accessibleName short, still allow text= but mark exact to narrow matches
  if (meta.accessibleName && meta.accessibleName.length < 40) {
    return `text=${meta.accessibleName.replace(/"/g,'\\"')}:exact`;
  }
  return current;
};

function isSelectorUnique(sel: string): boolean {
  if (typeof document === 'undefined') return false;
  try {
    const n = document.querySelectorAll(sel);
    return n.length === 1;
  } catch { return false; }
}

function buildAncestorNthSelector(el: HTMLElement): string | null {
  if (!el.parentElement) return null;
  const path: string[] = [];
  let current: HTMLElement | null = el;
  let depth = 0;
  while (current && depth < 6) {
    const tag = current.tagName.toLowerCase();
    let part = tag;
    const cur = current as HTMLElement | null;
    if (cur && cur.parentElement) {
      const same = Array.from(cur.parentElement.children).filter(c => (c as HTMLElement).tagName === cur.tagName);
      if (same.length > 1) {
        const idx = same.indexOf(cur) + 1;
        part += `:nth-of-type(${idx})`;
      }
    }
    path.unshift(part);
    const selector = path.join(' > ');
    if (isSelectorUnique(selector)) return selector;
    current = current.parentElement;
    depth++;
  }

  const withBody = 'body > ' + path.join(' > ');
  if (isSelectorUnique(withBody)) return withBody;
  return null;
}

function ensureStabilizedUnique(html: HTMLElement, stabilized: string): string {
  if (stabilized.startsWith('#') || stabilized.startsWith('[data-testid=')) return stabilized;
  // For role: and text:exact selectors validate they point to the original element; if not unique escalate
  if (stabilized.startsWith('role:')) {
    // attempt resolution count
    try {
      const el = findByRoleLike(stabilized);
      if (el) return stabilized;
    } catch {}
  }
  if (stabilized.startsWith('text=') && stabilized.endsWith(':exact')) {
    // verify uniqueness by scanning exact text matches
    const exactTxt = stabilized.slice('text='.length, -(':exact'.length));
    try {
      const candidates = Array.from(document.querySelectorAll('button, a, [role], input, textarea, select'))
        .filter(e => (e.textContent||'').trim() === exactTxt);
      if (candidates.length === 1) return stabilized; // keep it
    } catch {}
  }
  if (isSelectorUnique(stabilized)) return stabilized;
  const ancestor = buildAncestorNthSelector(html);
  if (ancestor && ancestor.length < 180) return ancestor;
  return stabilized;
}

// Minimal role: resolution for uniqueness validation
function findByRoleLike(sel: string): Element | null {
  if (!sel.startsWith('role:')) return null;
  const m = sel.match(/^role:([^\[]+)(?:\[name="(.+?)"\])?/);
  if (!m) return null;
  const role = m[1];
  const name = m[2];
  const candidates = Array.from(document.querySelectorAll('*')).filter(el => {
    const r = (el as HTMLElement).getAttribute('role') || inferRole(el as HTMLElement);
    return r === role;
  });
  if (!name) return candidates[0] || null;
  return candidates.find(c => (c.textContent||'').trim() === name) || null;
}

function inferRole(el: HTMLElement): string | null {
  const tag = el.tagName.toLowerCase();
  if (tag === 'button') return 'button';
  if (tag === 'a' && el.hasAttribute('href')) return 'link';
  if (tag === 'input') return 'textbox';
  return null;
}

/**
 * Extracts the relative URL (pathname + search + hash) from a full URL
 * Returns "/" if the URL is just a domain without a path
 * Returns the original string if it's already a relative URL or can't be parsed
 */
export function getRelativeUrl(url: string): string {
  if (!url) return "/";

  let pathname: string;
  let search = "";
  let hash = "";

  try {
    // Try to parse as URL
    const urlObj = new URL(url);
    pathname = urlObj.pathname;
    search = urlObj.search;
    hash = urlObj.hash;
  } catch {
    // If parsing fails, it might already be a relative URL
    if (!url.startsWith("/")) {
      return url;
    }
    // Extract parts manually for relative URLs
    const hashIndex = url.indexOf("#");
    const searchIndex = url.indexOf("?");

    if (hashIndex !== -1) {
      pathname = url.substring(0, hashIndex);
      hash = url.substring(hashIndex);
    } else if (searchIndex !== -1) {
      pathname = url.substring(0, searchIndex);
      search = url.substring(searchIndex);
    } else {
      pathname = url;
    }
  }

  // If pathname is now empty, default to "/"
  if (!pathname) {
    pathname = "/";
  }

  return pathname + search + hash;
}
