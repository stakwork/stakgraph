import { PlaywrightAction, ReplayStatus } from "../types";

// Replay match tracking (stores last matched selector + requested selector + text)
const __stakReplayMatch = (window as any).__stakTrakReplayMatch || { last: null as null | { requested: string; matched: string; text?: string; time: number; element?: Element } };
(window as any).__stakTrakReplayMatch = __stakReplayMatch;

// Maintain last structurally-unique selector to stabilize transitions
const __stakReplayState = (window as any).__stakTrakReplayState || { lastStructural: null as string | null, lastEl: null as Element | null };
(window as any).__stakTrakReplayState = __stakReplayState;

// Map for primary selector -> metadata (visualSelector) populated externally before replay
(window as any).__stakTrakSelectorMap = (window as any).__stakTrakSelectorMap || {};

// Simple one-shot warning cache to avoid log spam
const __stakWarned: Record<string, boolean> = (window as any).__stakTrakWarned || {};
(window as any).__stakTrakWarned = __stakWarned;

function highlight(element: Element, actionType: string = "action"): void {
  try { ensureStylesInDocument(document); } catch {}
  const htmlElement = element as HTMLElement;

  const original = {
    border: htmlElement.style.border,
    boxShadow: htmlElement.style.boxShadow,
    backgroundColor: htmlElement.style.backgroundColor,
  };

  htmlElement.style.border = "3px solid #ff6b6b";
  htmlElement.style.boxShadow = "0 0 20px rgba(255, 107, 107, 0.8)";
  htmlElement.style.backgroundColor = "rgba(255, 107, 107, 0.2)";
  htmlElement.style.transition = "all 0.3s ease";

  // Attach match metadata if present
  const last = __stakReplayMatch.last;
  if (last && last.element === element && Date.now() - last.time < 4000) {
    htmlElement.setAttribute('data-staktrak-matched-selector', last.matched);
    htmlElement.setAttribute('data-staktrak-requested-selector', last.requested);
    if (last.text) htmlElement.setAttribute('data-staktrak-matched-text', last.text);
  }

  element.scrollIntoView({ behavior: "smooth", block: "center" });

  setTimeout(() => {
    htmlElement.style.border = original.border;
    htmlElement.style.boxShadow = original.boxShadow;
    htmlElement.style.backgroundColor = original.backgroundColor;
    htmlElement.style.transition = "";
  }, 1500);
}

function normalizeUrl(u: string): string {
  try {
    const url = new URL(u, window.location.origin);
    return url.href.replace(/[#?].*$/, '').replace(/\/$/, '');
  } catch {
    return u.replace(/[#?].*$/, '').replace(/\/$/, '');
  }
}

enum PlaywrightActionType {
  GOTO = "goto",
  CLICK = "click",
  FILL = "fill",
  CHECK = "check",
  UNCHECK = "uncheck",
  SELECT_OPTION = "selectOption",
  WAIT_FOR_TIMEOUT = "waitForTimeout",
  WAIT_FOR_SELECTOR = "waitForSelector",
  WAIT_FOR_URL = "waitForURL",
  WAIT_FOR = "waitFor",
  WAIT_FOR_LOAD_STATE = "waitForLoadState",
  SET_VIEWPORT_SIZE = "setViewportSize",
  HOVER = "hover",
  FOCUS = "focus",
  BLUR = "blur",
  SCROLL_INTO_VIEW = "scrollIntoView",
  EXPECT = "expect",
}

function getRoleSelector(role: string): string {
  const roleMap: Record<string, string> = {
    button:
      'button, [role="button"], input[type="button"], input[type="submit"]',
    heading: 'h1, h2, h3, h4, h5, h6, [role="heading"]',
    link: 'a, [role="link"]',
    textbox:
      'input[type="text"], input[type="email"], input[type="password"], textarea, [role="textbox"]',
    checkbox: 'input[type="checkbox"], [role="checkbox"]',
    radio: 'input[type="radio"], [role="radio"]',
    listitem: 'li, [role="listitem"]',
    list: 'ul, ol, [role="list"]',
    img: 'img, [role="img"]',
    table: 'table, [role="table"]',
    row: 'tr, [role="row"]',
    cell: 'td, th, [role="cell"], [role="gridcell"]',
    menu: '[role="menu"]',
    menuitem: '[role="menuitem"]',
    dialog: '[role="dialog"]',
    alert: '[role="alert"]',
    tab: '[role="tab"]',
    tabpanel: '[role="tabpanel"]',
  };

  return roleMap[role] || `[role="${role}"]`;
}

export async function executePlaywrightAction(
  action: PlaywrightAction
): Promise<void> {
  try {
    switch (action.type) {
      case PlaywrightActionType.GOTO:
        if (action.value && typeof action.value === "string") {
          window.parent.postMessage(
            {
              type: "staktrak-iframe-navigate",
              url: action.value,
            },
            "*"
          );
        }
        break;

      case PlaywrightActionType.SET_VIEWPORT_SIZE:
        if (action.options) {
          try {
            if (window.top === window) {
              window.resizeTo(action.options.width, action.options.height);
            }
          } catch (e) {
            console.warn("Cannot resize viewport in iframe context:", e);
          }
        }
        break;

      case PlaywrightActionType.WAIT_FOR_LOAD_STATE:
        break;

      case PlaywrightActionType.WAIT_FOR_SELECTOR:
        if (action.selector) {
          await waitForElement(action.selector);
        }
        break;
      case PlaywrightActionType.WAIT_FOR_URL:
        if (action.value && typeof action.value === 'string') {
          const target = normalizeUrl(action.value);
          const start = Date.now();
          let matched = false;
          let lastPulse = 0;
          const maxMs = 8000;
          const stopSignals: Array<() => void> = [];
          const tryMatch = () => {
            const current = normalizeUrl(window.location.href);
            
            if (current === target) { matched = true; return true; }
            
            try {
              const currentUrl = new URL(window.location.href);
              const currentPath = currentUrl.pathname + currentUrl.search;
              
              if (target.startsWith('/')) {
                if (currentPath === target) { matched = true; return true; }
                const currentNoQuery = currentUrl.pathname;
                const targetNoQuery = target.replace(/[?#].*$/, '');
                if (currentNoQuery === targetNoQuery) { matched = true; return true; }
              } else {
                const targetUrl = new URL(target, window.location.origin);
                const targetPath = targetUrl.pathname + targetUrl.search;
                if (currentPath === targetPath) { matched = true; return true; }
              }
              
              const curNoHash = current.replace(/#.*/,'');
              const tgtNoHash = target.replace(/#.*/,'');
              if (curNoHash === tgtNoHash) { matched = true; return true; }
            } catch {}
            return false;
          };
          // Event-driven shortcut via SPA history instrumentation
          const onHist = (e: any) => {
            if (!matched && tryMatch()) {}
          };
          try { window.addEventListener('staktrak-history-change', onHist as any); stopSignals.push(()=>window.removeEventListener('staktrak-history-change', onHist as any)); } catch {}
          // Also listen to hashchange (some routers rely on it)
          const onHash = () => { if (!matched && tryMatch()) {} };
          try { window.addEventListener('hashchange', onHash); stopSignals.push(()=>window.removeEventListener('hashchange', onHash)); } catch {}
          // Initial immediate check
          tryMatch();
          while (!matched && Date.now() - start < maxMs) {
            if (Date.now() - lastPulse > 1000) {
              try {
                document.body.style.outline = '3px dashed #ff6b6b';
                setTimeout(()=>{ document.body.style.outline = ''; }, 400);
              } catch {}
              lastPulse = Date.now();
            }
            await new Promise(r=>setTimeout(r,120));
            if (tryMatch()) break;
          }
          stopSignals.forEach(fn=>{ try { fn(); } catch {} });
          try { highlight(document.body, matched ? 'nav' : 'nav-timeout'); } catch {}
          try { ensureStylesInDocument(document); } catch {}
          if (!matched && !(window as any).__stakTrakWarnedNav) {
            console.warn('[staktrak] waitForURL timeout — last, expected', window.location.href, target);
            (window as any).__stakTrakWarnedNav = true;
          }
        }
        break;

      case PlaywrightActionType.CLICK:
        if (action.selector) {
          const element = await waitForElement(action.selector);
          if (element) {
            const htmlElement = element as HTMLElement;

            highlight(element, "click");

            try {
              htmlElement.focus();
            } catch {}

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

              htmlElement.click();
              element.dispatchEvent(
                new MouseEvent("click", {
                  bubbles: true,
                  cancelable: true,
                  view: window,
                })
              );
            } catch (clickError) { throw clickError; }

            await new Promise((resolve) => setTimeout(resolve, 50));
          } else {
            throw new Error(`Element not found: ${action.selector}`);
          }
        }
        break;

      case PlaywrightActionType.FILL:
        if (action.selector && action.value !== undefined) {
          const element = (await waitForElement(action.selector)) as
            | HTMLInputElement
            | HTMLTextAreaElement;
          if (element) {
            highlight(element, "fill");
            element.focus();
            element.value = "";
            element.value = String(action.value);
            element.dispatchEvent(new Event("input", { bubbles: true }));
            element.dispatchEvent(new Event("change", { bubbles: true }));
          } else {
            throw new Error(`Input element not found: ${action.selector}`);
          }
        }
        break;

      case PlaywrightActionType.CHECK:
        if (action.selector) {
          const element = (await waitForElement(
            action.selector
          )) as HTMLInputElement;
          if (
            element &&
            (element.type === "checkbox" || element.type === "radio")
          ) {
            highlight(element, "check");
            if (!element.checked) {
              element.click();
            }
          } else {
            throw new Error(
              `Checkbox/radio element not found: ${action.selector}`
            );
          }
        }
        break;

      case PlaywrightActionType.UNCHECK:
        if (action.selector) {
          const element = (await waitForElement(
            action.selector
          )) as HTMLInputElement;
          if (element && element.type === "checkbox") {
            highlight(element, "uncheck");
            if (element.checked) {
              element.click();
            }
          } else {
            throw new Error(`Checkbox element not found: ${action.selector}`);
          }
        }
        break;

      case PlaywrightActionType.SELECT_OPTION:
        if (action.selector && action.value !== undefined) {
          const element = (await waitForElement(
            action.selector
          )) as HTMLSelectElement;
          if (element && element.tagName === "SELECT") {
            highlight(element, "select");
            element.value = String(action.value);
            element.dispatchEvent(new Event("change", { bubbles: true }));
          } else {
            throw new Error(`Select element not found: ${action.selector}`);
          }
        }
        break;

      case PlaywrightActionType.WAIT_FOR_TIMEOUT:
        const shortDelay = Math.min(action.value as number, 500);
        await new Promise((resolve) => setTimeout(resolve, shortDelay));
        break;

      case PlaywrightActionType.WAIT_FOR:
        if (action.selector) {
          const element = await waitForElement(action.selector);
          if (!element) {
            throw new Error(
              `Element not found for waitFor: ${action.selector}`
            );
          }
          if (action.options?.state === "visible") {
            if (!isElementVisible(element)) {
              throw new Error(`Element is not visible: ${action.selector}`);
            }
          }
        }
        break;

      case PlaywrightActionType.HOVER:
        if (action.selector) {
          const element = await waitForElement(action.selector);
          if (element) {
            highlight(element, "hover");
            element.dispatchEvent(
              new MouseEvent("mouseover", { bubbles: true })
            );
            element.dispatchEvent(
              new MouseEvent("mouseenter", { bubbles: true })
            );
          } else {
            throw new Error(`Element not found for hover: ${action.selector}`);
          }
        }
        break;

      case PlaywrightActionType.FOCUS:
        if (action.selector) {
          const element = (await waitForElement(
            action.selector
          )) as HTMLElement;
          if (element && typeof element.focus === "function") {
            highlight(element, "focus");
            element.focus();
          } else {
            throw new Error(
              `Element not found or not focusable: ${action.selector}`
            );
          }
        }
        break;

      case PlaywrightActionType.BLUR:
        if (action.selector) {
          const element = (await waitForElement(
            action.selector
          )) as HTMLElement;
          if (element && typeof element.blur === "function") {
            highlight(element, "blur");
            element.blur();
          } else {
            throw new Error(
              `Element not found or not blurable: ${action.selector}`
            );
          }
        }
        break;

      case PlaywrightActionType.SCROLL_INTO_VIEW:
        if (action.selector) {
          const element = await waitForElement(action.selector);
          if (element) {
            highlight(element, "scroll");
            element.scrollIntoView({
              behavior: "smooth",
              block: "center",
              inline: "center",
            });
          } else {
            throw new Error(
              `Element not found for scrollIntoView: ${action.selector}`
            );
          }
        }
        break;

      case PlaywrightActionType.EXPECT:
        if (action.selector) {
          await verifyExpectation(action);
        }
        break;

      default:
  // unknown action type ignored
        break;
    }
  } catch (error) {
    throw error;
  }
}

async function waitForElements(
  selector: string,
  timeout = 5000
): Promise<Element[]> {
  const startTime = Date.now();

  while (Date.now() - startTime < timeout) {
    try {
      const elements = findElements(selector);
      if (elements.length > 0) {
        return elements;
      }
    } catch {}

    await new Promise((resolve) => setTimeout(resolve, 100));
  }

  return [];
}

function findElements(selector: string): Element[] {
  const element = findElementWithFallbacks(selector);
  return element ? [element] : [];
}

function findElementWithFallbacks(selector: string): Element | null {
  if (!selector || selector.trim() === "") return null;

  if (selector.includes('data-staktrak-id=')) {
    const stakIdMatch = selector.match(/\[data-staktrak-id="([^"]+)"\]/);
    if (stakIdMatch) {
      const element = document.querySelector(`[data-staktrak-id="${stakIdMatch[1]}"]`);
      if (element) {
        return element;
      }
    }
  }

  if (selector.includes('data-testid=')) {
    const testIdMatch = selector.match(/\[data-testid="([^"]+)"\]/);
    if (testIdMatch) {
      const element = document.querySelector(`[data-testid="${testIdMatch[1]}"]`);
      if (element) {
        return element;
      }
    }
  }

  // If selector is a DSL (text=/role:) AND we have a stored mapping (window.__stakTrakSelectorMap) use its visualSelector for highlighting attempt
  try {
    if ((selector.startsWith('text=') || selector.startsWith('role:')) && (window as any).__stakTrakSelectorMap) {
      const map = (window as any).__stakTrakSelectorMap as Record<string, { visualSelector?: string }>;
      const entry = map[selector];
      if (entry?.visualSelector) {
        try {
          const cssEl = document.querySelector(entry.visualSelector);
          if (cssEl) return cssEl;
        } catch {}
      }
    }
  } catch {}

  // Fast path: unique structural tag.class selector
  if (/^[a-zA-Z]+\.[a-zA-Z0-9_-]+(\.[a-zA-Z0-9_-]+)*$/.test(selector)) {
    try {
      const matches = document.querySelectorAll(selector);
      if (matches.length === 1) {
        __stakReplayState.lastStructural = selector;
        __stakReplayState.lastEl = matches[0];
        return matches[0];
      }
    } catch {}
  }

  // If selector is role/text and we have a previous unique structural pointing to same accessible name, reuse it to avoid container highlight flicker
  if ((selector.startsWith('role:') || selector.startsWith('text=')) && __stakReplayState.lastStructural && __stakReplayState.lastEl) {
    // verify element still present
    try {
      if (document.contains(__stakReplayState.lastEl)) {
        const acc = getAccessibleName(__stakReplayState.lastEl as HTMLElement);
        const nameMatch = selector.includes('name="') ? selector.includes(`name="${acc}`) : selector.includes(acc || '');
        if (acc && nameMatch) {
          return __stakReplayState.lastEl;
        }
      }
    } catch {}
  }

  const noteMatch = (el: Element | null, matched: string, text?: string) => {
    if (el) {
      __stakReplayMatch.last = { requested: selector, matched, text, time: Date.now(), element: el };
      if (text) (el as any).__stakTrakMatchedText = text;
    }
    return el;
  };

  if (selector.startsWith('role:')) {
    const roleMatch = selector.match(/^role:([^\[]+)(?:\[name(?:-regex)?="(.+?)"\])?/);
    if (roleMatch) {
      const role = roleMatch[1];
      const nameRaw = roleMatch[2];
      const nameRegex = selector.includes('[name-regex=');
      const candidates = Array.from(queryByRole(role.trim()));
      if (!nameRaw) {
        return noteMatch((candidates[0] as Element) || null, selector);
      }
      let matcher: (s: string) => boolean;
      if (nameRegex) {
        const rx = nameRaw.match(/^\/(.*)\/(.*)$/);
        if (rx) {
          try { const r = new RegExp(rx[1], rx[2]); matcher = (s) => r.test(s); } catch { matcher = (s)=>s.includes(nameRaw); }
        } else {
          matcher = (s) => s.includes(nameRaw);
        }
      } else {
        const target = nameRaw;
        matcher = (s) => s === target;
      }
      for (const el of candidates) {
        const acc = getAccessibleName(el as HTMLElement);
        if (acc && matcher(acc)) {
          return noteMatch(el as Element, selector, acc);
        }
      }
      return noteMatch(null, selector);
    }
  }

  if (selector.startsWith('text=') && selector.endsWith(':exact')) {
    const core = selector.slice('text='.length, -(':exact'.length));
    const norm = core.trim();
    const interactive = Array.from(document.querySelectorAll('button, a, [role], input, textarea, select')) as HTMLElement[];
    const exact = interactive.filter(el => (el.textContent||'').trim() === norm);
    if (exact.length === 1) return noteMatch(exact[0], selector, norm);
    if (exact.length > 1) {
      // take the deepest (most specific) element to avoid container highlight
      const deepest = exact.sort((a,b) => depth(b)-depth(a))[0];
      return noteMatch(deepest, selector, norm);
    }
    return noteMatch(null, selector);
  }

  if (selector.startsWith('getByStakTrakId:')) {
    const val = selector.substring('getByStakTrakId:'.length);
    return noteMatch(document.querySelector(`[data-staktrak-id="${cssEscape(val)}"]`), `[data-staktrak-id="${val}"]`);
  }
  if (selector.startsWith('getByTestId:')) {
    const val = selector.substring('getByTestId:'.length);
    return noteMatch(document.querySelector(`[data-testid="${cssEscape(val)}"]`), `[data-testid="${val}"]`);
  }
  if (selector.startsWith('getByText-regex:')) {
    const body = selector.substring('getByText-regex:'.length);
    const rx = body.match(/^\/(.*)\/(.*)$/);
    let r: RegExp | null = null;
    if (rx) try { r = new RegExp(rx[1], rx[2]); } catch {}
    const all = textSearchCandidates();
    for (const el of all) {
      const txt = el.textContent?.trim() || '';
      if (r && r.test(txt)) { return noteMatch(el, selector, txt); }
    }
    return noteMatch(null, selector);
  }
  if (selector.startsWith('getByText:')) {
    // format getByText:Text or getByText:Text:exact
    const exact = selector.endsWith(':exact');
    const core = exact ? selector.slice('getByText:'.length, -(':exact'.length)) : selector.slice('getByText:'.length);
    const norm = core.trim();
    const all = textSearchCandidates();
    for (const el of all) {
      const txt = el.textContent?.trim() || '';
      if ((exact && txt === norm) || (!exact && txt.includes(norm))) {
        return noteMatch(el, selector, txt);
      }
    }
    return noteMatch(null, selector);
  }
  if (selector.startsWith('getByLabel:')) {
    const label = selector.substring('getByLabel:'.length).trim();
    // label[for]
    const labels = Array.from(document.querySelectorAll('label')).filter(l => l.textContent?.trim() === label);
    for (const lab of labels) {
      const forId = lab.getAttribute('for');
      if (forId) {
        const ctl = document.getElementById(forId);
        if (ctl) return noteMatch(ctl, selector);
      }
      // nested control
      const nested = lab.querySelector('input,select,textarea,button');
      if (nested) return noteMatch(nested, selector);
    }
    // aria-label fallback
    return noteMatch(document.querySelector(`[aria-label="${cssEscape(label)}"]`), selector);
  }
  if (selector.startsWith('getByPlaceholder:')) {
    const ph = selector.substring('getByPlaceholder:'.length);
    return noteMatch(document.querySelector(`[placeholder="${cssEscape(ph)}"]`), selector);
  }
  if (selector.startsWith('getByTitle:')) {
    const t = selector.substring('getByTitle:'.length);
    return noteMatch(document.querySelector(`[title="${cssEscape(t)}"]`), selector);
  }
  if (selector.startsWith('getByAltText:')) {
    const alt = selector.substring('getByAltText:'.length);
    return noteMatch(document.querySelector(`[alt="${cssEscape(alt)}"]`), selector);
  }

  const browserSelector = convertToBrowserSelector(selector);

  if (browserSelector && isValidSelector(browserSelector)) {
    const element = document.querySelector(browserSelector);
    if (element) return noteMatch(element, browserSelector);
  }

  const strategies = [
    () => findByStakTrakId(selector),
    () => findByDataTestId(selector),
    () => findById(selector),
    () => findByClassUnique(selector),
    () => findByAriaLabel(selector),
    () => findByRole(selector),
    () => findByTextContentTight(selector),
  ];

  for (const strategy of strategies) {
    try {
      const element = strategy();
      if (element) {
        return noteMatch(element, selector);
      }
    } catch {}
  }

  return noteMatch(null, selector);
}

// --- Helper utilities added for enhanced selector resolution ---
function cssEscape(value: string): string {
  // Basic CSS.escape polyfill (not full spec)
  return value.replace(/[^a-zA-Z0-9_-]/g, (c) => `\\${c}`);
}

function queryByRole(role: string): HTMLElement[] {
  const selector = getRoleSelector(role);
  return Array.from(document.querySelectorAll(selector))
    .filter(el => el instanceof HTMLElement) as HTMLElement[];
}

function getAccessibleName(el: HTMLElement): string | null {
  // priority: aria-label, aria-labelledby, text content, title, value/placeholder
  const aria = el.getAttribute('aria-label');
  if (aria) return aria.trim();
  const labelled = el.getAttribute('aria-labelledby');
  if (labelled) {
    const parts = labelled.split(/\s+/).map(id => document.getElementById(id)?.textContent?.trim()).filter(Boolean) as string[];
    if (parts.length) return parts.join(' ');
  }
  const tag = el.tagName.toLowerCase();
  if (tag === 'input' || tag === 'textarea') {
    const val = (el as HTMLInputElement).value || el.getAttribute('placeholder');
    if (val) return val.trim();
  }
  const txt = el.textContent?.trim();
  if (txt) return txt.slice(0,120);
  const title = el.getAttribute('title');
  if (title) return title.trim();
  return null;
}

function textSearchCandidates(): HTMLElement[] {
  return Array.from(document.querySelectorAll('button, a, [role], input, textarea, select, label, div, span')) as HTMLElement[];
}

function convertToBrowserSelector(selector: string): string {
  if (!selector) return selector;

  if (selector.includes(":has-text(")) {
    const textMatch = selector.match(/:has-text\("([^"]+)"\)/);
    if (textMatch) {
      const text = textMatch[1];
      const tagMatch = selector.match(/^([a-zA-Z]+)/);
      const tagName = tagMatch ? tagMatch[1] : "*";

      const elements = Array.from(document.querySelectorAll(tagName));
      for (const element of elements) {
        if (element.textContent?.trim() === text) {
          const uniqueSelector = createUniqueSelector(element);
          if (uniqueSelector && isValidSelector(uniqueSelector)) {
            return uniqueSelector;
          }
        }
      }
      return tagName;
    }
  }

  selector = selector.replace(/:visible/g, "");
  selector = selector.replace(/:enabled/g, "");
  selector = selector.replace(/>>.*$/g, "");

  return selector.trim();
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

function findByStakTrakId(selector: string): Element | null {
  if (!selector.includes("data-staktrak-id")) return null;
  const stakId = selector.match(/data-staktrak-id="([^"]+)"/)?.[1];
  if (stakId) {
    return document.querySelector(`[data-staktrak-id="${stakId}"]`);
  }
  return null;
}

function findByDataTestId(selector: string): Element | null {
  if (!selector.includes("data-testid")) return null;
  const testId = selector.match(/data-testid="([^"]+)"/)?.[1];
  if (testId) {
    return document.querySelector(`[data-testid="${testId}"]`);
  }
  return null;
}

function findByClassUnique(selector: string): Element | null {
  if (!selector.includes('.')) return null;
  if (selector.startsWith('text=') || selector.startsWith('role:')) return null;
  // only accept if unique to avoid selecting container ancestor
  try {
    const els = document.querySelectorAll(selector);
    if (els.length === 1) return els[0];
  } catch {}
  // fallback: first exact class-only element if unique among interactive
  const classOnly = selector.match(/^\w+\.[^.]+$/);
  if (classOnly) {
    const els = Array.from(document.querySelectorAll(selector)) as HTMLElement[];
    const interactive = els.filter(isInteractive);
    if (interactive.length === 1) return interactive[0];
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

function findByTextContentTight(selector: string): Element | null {
  if (!selector.startsWith('text=')) return null;
  const exact = selector.endsWith(':exact');
  const core = exact ? selector.slice('text='.length, -(':exact'.length)) : selector.slice('text='.length);
  const norm = core.trim();
  const candidates = textSearchCandidates().filter(isInteractiveOrSmall);
  for (const el of candidates) {
    const txt = (el.textContent||'').trim();
    if ((exact && txt === norm) || (!exact && txt.includes(norm))) return el;
  }
  return null;
}

function isInteractive(el: HTMLElement): boolean {
  const tag = el.tagName.toLowerCase();
  if (['button','a','input','textarea','select','option'].includes(tag)) return true;
  const role = el.getAttribute('role');
  if (role && ['button','link','menuitem','option','tab'].includes(role)) return true;
  return false;
}

function isInteractiveOrSmall(el: HTMLElement): boolean {
  if (isInteractive(el)) return true;
  const rect = el.getBoundingClientRect();
  if (rect.width < 400 && rect.height < 200) return true;
  return false;
}

function depth(el: HTMLElement): number {
  let d=0; let p=el.parentElement; while (p){d++;p=p.parentElement;} return d;
}

function findByCoordinates(selector: string): Element | null {
  const clickableElements = document.querySelectorAll(
    'button, a, input, select, [role="button"], [onclick]'
  );
  return clickableElements.length > 0 ? clickableElements[0] : null;
}

function createUniqueSelector(element: Element): string | null {
  if (element.id && /^[a-zA-Z][\w-]*$/.test(element.id)) {
    const idSelector = `#${element.id}`;
    if (document.querySelectorAll(idSelector).length === 1) {
      return idSelector;
    }
  }

  const testId = (element as HTMLElement).dataset?.testid;
  if (testId) {
    const testIdSelector = `[data-testid="${testId}"]`;
    if (document.querySelectorAll(testIdSelector).length === 1) {
      return testIdSelector;
    }
  }

  const ariaLabel = element.getAttribute("aria-label");
  if (ariaLabel) {
    const ariaSelector = `[aria-label="${ariaLabel}"]`;
    if (document.querySelectorAll(ariaSelector).length === 1) {
      return ariaSelector;
    }
  }

  const tagName = element.tagName.toLowerCase();
  const classes = Array.from(element.classList).filter((cls) => {
    return (
      !cls.match(/^[a-zA-Z0-9_-]*[0-9a-f]{6,}/) &&
      !cls.includes("emotion-") &&
      !cls.includes("css-") &&
      !cls.includes("module__") &&
      cls.length < 30
    );
  });

  if (classes.length > 0) {
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

  return tagName;
}

async function waitForElement(
  selector: string,
  matchedText?: string
): Promise<Element | null> {
  const startTime = Date.now();
  const timeout = 5000;
  const backoffs = [50, 80, 120, 180, 250, 350, 500, 650, 800];
  let attempt = 0;
  while (Date.now() - startTime < timeout) {
    try {
      const elements = findElements(selector);
      if (elements.length > 0) {
        const element = elements[0];
        if (matchedText) {
          (element as any).__stakTrakMatchedText = matchedText;
        }
        return element;
      }
    } catch (error) {
      if (!__stakWarned[selector]) {
        console.warn('[staktrak] resolution error', selector, error);
        __stakWarned[selector] = true;
      }
    }
    const delay = backoffs[Math.min(attempt, backoffs.length - 1)];
    attempt++;
    await new Promise(r=>setTimeout(r, delay));
  }
  if (!__stakWarned[selector]) {
    console.warn('[staktrak] highlight failed: not found', selector);
    __stakWarned[selector] = true;
  }
  return null;
}

function ensureStylesInDocument(doc: Document): void {
  if (doc.querySelector("#staktrak-highlight-styles")) return;

  const style = doc.createElement("style");
  style.id = "staktrak-highlight-styles";
  style.textContent = `
    .staktrak-text-highlight {
      background-color: #3b82f6 !important;
      color: white !important;
      padding: 2px 4px !important;
      border-radius: 3px !important;
      font-weight: bold !important;
      box-shadow: 0 0 8px rgba(59, 130, 246, 0.6) !important;
      animation: staktrak-text-pulse 2s ease-in-out !important;
    }

    @keyframes staktrak-text-pulse {
      0% { background-color: #3b82f6; box-shadow: 0 0 8px rgba(59, 130, 246, 0.6); }
      50% { background-color: #1d4ed8; box-shadow: 0 0 15px rgba(29, 78, 216, 0.8); }
      100% { background-color: #3b82f6; box-shadow: 0 0 8px rgba(59, 130, 246, 0.6); }
    }
  `;

  doc.head.appendChild(style);
}

function highlightElement(element: Element, matchedText?: string): void {
  try {
    ensureStylesInDocument(document);

    const textToHighlight =
      matchedText || (element as any).__stakTrakMatchedText;

    if (textToHighlight) {
      highlightTextInElement(element, textToHighlight);
    } else {
      highlight(element, "element");
    }
  } catch (error) {
    console.warn("Error highlighting element:", error);
  }
}

function highlightTextInElement(
  element: Element,
  textToHighlight: string
): void {
  try {
    ensureStylesInDocument(document);

    function wrapTextNodes(node: Node): void {
      if (node.nodeType === Node.TEXT_NODE) {
        const textContent = node.textContent || "";
        if (textContent.includes(textToHighlight)) {
          const parent = node.parentNode;
          if (parent) {
            const tempDiv = document.createElement("div");
            tempDiv.innerHTML = textContent.replace(
              new RegExp(
                `(${textToHighlight.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")})`,
                "gi"
              ),
              '<span class="staktrak-text-highlight">$1</span>'
            );

            while (tempDiv.firstChild) {
              parent.insertBefore(tempDiv.firstChild, node);
            }
            parent.removeChild(node);
          }
        }
      } else if (
        node.nodeType === Node.ELEMENT_NODE &&
        !(node as Element).classList?.contains("staktrak-text-highlight")
      ) {
        const children = Array.from(node.childNodes);
        children.forEach((child) => wrapTextNodes(child));
      }
    }

    wrapTextNodes(element);

    element.setAttribute("data-staktrak-processed", "true");

    setTimeout(() => {
      const highlights = element.querySelectorAll(".staktrak-text-highlight");
      highlights.forEach((highlight) => {
        const parent = highlight.parentNode;
        if (parent) {
          parent.insertBefore(
            document.createTextNode(highlight.textContent || ""),
            highlight
          );
          parent.removeChild(highlight);
        }
      });

      element.removeAttribute("data-staktrak-processed");

      element.normalize();
    }, 3000);
  } catch (error) {
    console.warn("Error highlighting text:", error);
  }
}

async function verifyExpectation(action: PlaywrightAction): Promise<void> {
  if (!action.selector) return;

  switch (action.expectation) {
    case "toBeVisible":
      const element = await waitForElement(action.selector);
      if (!element || !isElementVisible(element)) {
        throw new Error(`Element is not visible: ${action.selector}`);
      }
      break;

    case "toContainText":
      const textElement = await waitForElement(
        action.selector,
        String(action.value)
      );
      if (
        !textElement ||
        !textElement.textContent?.includes(String(action.value || ""))
      ) {
        throw new Error(
          `Element does not contain text "${action.value}": ${action.selector}`
        );
      }
      break;

    case "toHaveText":
      const exactTextElement = await waitForElement(
        action.selector,
        String(action.value)
      );
      if (
        !exactTextElement ||
        exactTextElement.textContent?.trim() !== String(action.value || "")
      ) {
        throw new Error(
          `Element does not have exact text "${action.value}": ${action.selector}`
        );
      }
      break;

    case "toBeChecked":
      const checkedElement = (await waitForElement(
        action.selector
      )) as HTMLInputElement;
      if (!checkedElement || !checkedElement.checked) {
        throw new Error(`Element is not checked: ${action.selector}`);
      }
      break;

    case "not.toBeChecked":
      const uncheckedElement = (await waitForElement(
        action.selector
      )) as HTMLInputElement;
      if (!uncheckedElement || uncheckedElement.checked) {
        throw new Error(`Element should not be checked: ${action.selector}`);
      }
      break;

    case "toHaveCount":
      const elements = await waitForElements(action.selector);
      const expectedCount = Number(action.value);
      if (elements.length !== expectedCount) {
        throw new Error(
          `Expected ${expectedCount} elements, but found ${elements.length}: ${action.selector}`
        );
      }
      break;

    default:
      // ignore unknown expectation
  }
}
function isElementVisible(element: Element): boolean {
  const style = window.getComputedStyle(element);
  return (
    style.display !== "none" &&
    style.visibility !== "hidden" &&
    style.opacity !== "0" &&
    element.getBoundingClientRect().width > 0 &&
    element.getBoundingClientRect().height > 0
  );
}

export function getActionDescription(action: PlaywrightAction): string {
  switch (action.type) {
    case PlaywrightActionType.GOTO:
      return `Navigate to ${action.value}`;
    case PlaywrightActionType.CLICK:
      return `Click element: ${action.selector}`;
    case PlaywrightActionType.FILL:
      return `Fill "${action.value}" in ${action.selector}`;
    case PlaywrightActionType.CHECK:
      return `Check checkbox: ${action.selector}`;
    case PlaywrightActionType.UNCHECK:
      return `Uncheck checkbox: ${action.selector}`;
    case PlaywrightActionType.SELECT_OPTION:
      return `Select "${action.value}" in ${action.selector}`;
    case PlaywrightActionType.HOVER:
      return `Hover over element: ${action.selector}`;
    case PlaywrightActionType.FOCUS:
      return `Focus element: ${action.selector}`;
    case PlaywrightActionType.BLUR:
      return `Blur element: ${action.selector}`;
    case PlaywrightActionType.SCROLL_INTO_VIEW:
      return `Scroll element into view: ${action.selector}`;
    case PlaywrightActionType.WAIT_FOR:
      return `Wait for element: ${action.selector}`;
    case PlaywrightActionType.EXPECT:
      return `Verify ${action.selector} ${action.expectation}`;
    case PlaywrightActionType.SET_VIEWPORT_SIZE:
      return `Set viewport size to ${action.options?.width}x${action.options?.height}`;
    case PlaywrightActionType.WAIT_FOR_TIMEOUT:
      return `Wait ${action.value}ms`;
    case PlaywrightActionType.WAIT_FOR_LOAD_STATE:
      return "Wait for page to load";
    case PlaywrightActionType.WAIT_FOR_SELECTOR:
      return `Wait for element: ${action.selector}`;
    case PlaywrightActionType.WAIT_FOR_URL:
      return `Wait for URL: ${action.value}`;
    default:
      return `Execute ${action.type}`;
  }
}