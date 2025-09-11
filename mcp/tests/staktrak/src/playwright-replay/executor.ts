import { PlaywrightAction, ReplayStatus } from "../types";

enum PlaywrightActionType {
  GOTO = "goto",
  CLICK = "click",
  FILL = "fill",
  CHECK = "check",
  UNCHECK = "uncheck",
  SELECT_OPTION = "selectOption",
  WAIT_FOR_TIMEOUT = "waitForTimeout",
  WAIT_FOR_SELECTOR = "waitForSelector",
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

      case PlaywrightActionType.CLICK:
        if (action.selector) {
          const element = await waitForElement(action.selector);
          if (element) {
            const htmlElement = element as HTMLElement;

            const originalBorder = htmlElement.style.border;
            htmlElement.style.border = "3px solid #ff6b6b";
            htmlElement.style.boxShadow = "0 0 10px rgba(255, 107, 107, 0.5)";

            element.scrollIntoView({ behavior: "smooth", block: "center" });
            await new Promise((resolve) => setTimeout(resolve, 50));

            try {
              htmlElement.focus();
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

              htmlElement.click();
              element.dispatchEvent(
                new MouseEvent("click", {
                  bubbles: true,
                  cancelable: true,
                  view: window,
                })
              );
            } catch (clickError) {
              console.warn("Error during click simulation:", clickError);
              throw clickError;
            }

            await new Promise((resolve) => setTimeout(resolve, 50));

            setTimeout(() => {
              htmlElement.style.border = originalBorder;
              htmlElement.style.boxShadow = "";
            }, 300);
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
            // element.scrollIntoView({ behavior: "auto", block: "center" });

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
            // element.scrollIntoView({ behavior: "auto", block: "center" });

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
        console.warn(`Unknown action type: ${action.type}`);
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
    } catch (error) {
      console.warn(`Error finding elements with selector: ${selector}`, error);
    }

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

  const browserSelector = convertToBrowserSelector(selector);

  if (browserSelector && isValidSelector(browserSelector)) {
    const element = document.querySelector(browserSelector);
    if (element) return element;
  }

  const strategies = [
    () => findByDataTestId(selector),
    () => findByClass(selector),
    () => findById(selector),
    () => findByAriaLabel(selector),
    () => findByRole(selector),
    () => findByTextContent(selector),
    () => findByCoordinates(selector),
  ];

  for (const strategy of strategies) {
    try {
      const element = strategy();
      if (element) {
        return element;
      }
    } catch (error) {
      console.warn(`Strategy failed for ${selector}:`, error);
    }
  }

  return null;
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

function findByTextContent(selector: string): Element | null {
  let text: string | null = null;
  let tagName = "*";

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

  const tagMatch = selector.match(/^([a-zA-Z]+)/);
  if (tagMatch) {
    tagName = tagMatch[1];
  }

  if (!text) return null;

  const elements = Array.from(document.querySelectorAll(tagName));
  for (const element of elements) {
    const elementText = element.textContent?.trim();
    if (elementText === text || elementText?.includes(text)) {
      return element;
    }
  }
  return null;
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
  while (Date.now() - startTime < timeout) {
    try {
      const elements = findElements(selector);
      if (elements.length > 0) {
        const element = elements[0];
        if (matchedText) {
          (element as any).__stakTrakMatchedText = matchedText;
        }
        setTimeout(() => highlightElement(element), 100);
        return element;
      }
    } catch (error) {
      console.warn("Error finding element with selector:", selector, error);
    }

    await new Promise((resolve) => setTimeout(resolve, 100));
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

    // element.scrollIntoView({
    //   behavior: "smooth",
    //   block: "center",
    //   inline: "center",
    // });

    const textToHighlight =
      matchedText || (element as any).__stakTrakMatchedText;

    if (textToHighlight) {
      highlightTextInElement(element, textToHighlight);
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
      console.warn(`Unknown expectation: ${action.expectation}`);
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
      return `Set viewport size to ${action.value}`;
    case PlaywrightActionType.WAIT_FOR_TIMEOUT:
      return `Wait ${action.value}ms`;
    case PlaywrightActionType.WAIT_FOR_LOAD_STATE:
      return "Wait for page to load";
    case PlaywrightActionType.WAIT_FOR_SELECTOR:
      return `Wait for element: ${action.selector}`;
    default:
      return `Execute ${action.type}`;
  }
}