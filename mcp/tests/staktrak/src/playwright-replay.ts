import { ReplayStatus, PlaywrightAction, PlaywrightReplayState } from "./types";

let playwrightReplayRef = {
  current: null as {
    actions: PlaywrightAction[];
    status: ReplayStatus;
    currentActionIndex: number;
    testCode: string;
    errors: string[];
    timeouts: number[];
  } | null,
};

export function parsePlaywrightTest(testCode: string): PlaywrightAction[] {
  const actions: PlaywrightAction[] = [];
  const lines = testCode.split("\n");

  let lineNumber = 0;
  const variables: Map<string, string> = new Map();

  for (const line of lines) {
    lineNumber++;
    const trimmed = line.trim();

    if (
      !trimmed ||
      trimmed.startsWith("//") ||
      trimmed.startsWith("import") ||
      trimmed.startsWith("test(") ||
      trimmed.includes("async ({ page })") ||
      trimmed === "}); " ||
      trimmed === "});"
    ) {
      continue;
    }

    const commentMatch = line.match(/^\s*\/\/\s*(.+)/);
    const comment = commentMatch ? commentMatch[1] : undefined;

    try {
      const variableMatch = trimmed.match(/^const\s+(\w+)\s*=\s*page\.(.+);$/);
      if (variableMatch) {
        const [, varName, locatorCall] = variableMatch;
        const selector = parseLocatorCall(locatorCall);
        variables.set(varName, selector);
        continue;
      }

      const chainedVariableMatch = trimmed.match(
        /^const\s+(\w+)\s*=\s*(\w+)\.(.+);$/
      );
      if (chainedVariableMatch) {
        const [, newVarName, baseVarName, chainCall] = chainedVariableMatch;
        if (variables.has(baseVarName)) {
          const baseSelector = variables.get(baseVarName)!;
          const chainedSelector = parseChainedCall(baseSelector, chainCall);
          variables.set(newVarName, chainedSelector);
          continue;
        }
      }

      const awaitVariableCallMatch = trimmed.match(
        /^await\s+(\w+)\.(\w+)\((.*?)\);?$/
      );
      if (awaitVariableCallMatch) {
        const [, varName, method, args] = awaitVariableCallMatch;
        if (variables.has(varName)) {
          const selector = variables.get(varName)!;
          const action = parseVariableMethodCall(
            varName,
            method,
            args,
            comment,
            lineNumber,
            selector
          );
          if (action) {
            actions.push(action);
          }
          continue;
        }
      }

      const variableCallMatch = trimmed.match(/^(\w+)\.(\w+)\((.*?)\);?$/);
      if (variableCallMatch) {
        const [, varName, method, args] = variableCallMatch;
        if (variables.has(varName)) {
          const selector = variables.get(varName)!;
          const action = parseVariableMethodCall(
            varName,
            method,
            args,
            comment,
            lineNumber,
            selector
          );
          if (action) {
            actions.push(action);
          }
          continue;
        }
      }

      const pageLocatorActionMatch = trimmed.match(
        /^(?:await\s+)?page\.locator\(([^)]+)\)\.(\w+)\((.*?)\);?$/
      );
      if (pageLocatorActionMatch) {
        const [, selectorArg, method, args] = pageLocatorActionMatch;
        const selector = extractSelectorFromArg(selectorArg);
        const action = parseDirectAction(
          method,
          args,
          comment,
          lineNumber,
          selector
        );
        if (action) {
          actions.push(action);
        }
        continue;
      }

      const expectVariableMatch = trimmed.match(
        /^(?:await\s+)?expect\((\w+)\)\.(.+)$/
      );
      if (expectVariableMatch) {
        const [, varName, expectation] = expectVariableMatch;
        if (variables.has(varName)) {
          const selector = variables.get(varName)!;
          const action = parseExpectStatement(
            expectation,
            comment,
            lineNumber,
            selector
          );
          if (action) {
            actions.push(action);
          }
          continue;
        }
      }

      const expectLocatorMatch = trimmed.match(
        /^(?:await\s+)?expect\(page\.locator\(([^)]+)\)\)\.(.+)$/
      );
      if (expectLocatorMatch) {
        const [, selectorArg, expectation] = expectLocatorMatch;
        const selector = extractSelectorFromArg(selectorArg);
        const action = parseExpectStatement(
          expectation,
          comment,
          lineNumber,
          selector
        );
        if (action) {
          actions.push(action);
        }
        continue;
      }

      const waitForSelectorMatch = trimmed.match(
        /^(?:await\s+)?page\.waitForSelector\(['"](.*?)['"]\);?$/
      );
      if (waitForSelectorMatch) {
        actions.push({
          type: "waitForSelector",
          selector: waitForSelectorMatch[1],
          comment,
          lineNumber,
        });
        continue;
      }

      if (trimmed.includes("page.goto(")) {
        const urlMatch = trimmed.match(/page\.goto\(['"](.*?)['"]\)/);
        if (urlMatch) {
          actions.push({
            type: "goto",
            value: urlMatch[1],
            comment,
            lineNumber,
          });
        }
      } else if (trimmed.includes("page.setViewportSize(")) {
        const sizeMatch = trimmed.match(
          /page\.setViewportSize\(\s*{\s*width:\s*(\d+),\s*height:\s*(\d+)\s*}\s*\)/
        );
        if (sizeMatch) {
          actions.push({
            type: "setViewportSize",
            options: {
              width: parseInt(sizeMatch[1]),
              height: parseInt(sizeMatch[2]),
            },
            comment,
            lineNumber,
          });
        }
      } else if (trimmed.includes("page.waitForLoadState(")) {
        const stateMatch = trimmed.match(
          /page\.waitForLoadState\(['"](.*?)['"]\)/
        );
        actions.push({
          type: "waitForLoadState",
          value: stateMatch ? stateMatch[1] : "networkidle",
          comment,
          lineNumber,
        });
      } else if (trimmed.includes("page.click(")) {
        const selectorMatch = trimmed.match(/page\.click\(['"](.*?)['"]\)/);
        if (selectorMatch) {
          actions.push({
            type: "click",
            selector: selectorMatch[1],
            comment,
            lineNumber,
          });
        }
      } else if (trimmed.includes("page.fill(")) {
        const fillMatch = trimmed.match(
          /page\.fill\(['"](.*?)['"],\s*['"](.*?)['"]\)/
        );
        if (fillMatch) {
          actions.push({
            type: "fill",
            selector: fillMatch[1],
            value: fillMatch[2],
            comment,
            lineNumber,
          });
        }
      } else if (trimmed.includes("page.check(")) {
        const selectorMatch = trimmed.match(/page\.check\(['"](.*?)['"]\)/);
        if (selectorMatch) {
          actions.push({
            type: "check",
            selector: selectorMatch[1],
            comment,
            lineNumber,
          });
        }
      } else if (trimmed.includes("page.uncheck(")) {
        const selectorMatch = trimmed.match(/page\.uncheck\(['"](.*?)['"]\)/);
        if (selectorMatch) {
          actions.push({
            type: "uncheck",
            selector: selectorMatch[1],
            comment,
            lineNumber,
          });
        }
      } else if (trimmed.includes("page.selectOption(")) {
        const selectMatch = trimmed.match(
          /page\.selectOption\(['"](.*?)['"],\s*['"](.*?)['"]\)/
        );
        if (selectMatch) {
          actions.push({
            type: "selectOption",
            selector: selectMatch[1],
            value: selectMatch[2],
            comment,
            lineNumber,
          });
        }
      } else if (trimmed.includes("page.waitForTimeout(")) {
        const timeoutMatch = trimmed.match(/page\.waitForTimeout\((\d+)\)/);
        if (timeoutMatch) {
          actions.push({
            type: "waitForTimeout",
            value: parseInt(timeoutMatch[1]),
            comment,
            lineNumber,
          });
        }
      } else if (trimmed.includes("page.waitForSelector(")) {
        const selectorMatch = trimmed.match(
          /page\.waitForSelector\(['"](.*?)['"]\)/
        );
        if (selectorMatch) {
          actions.push({
            type: "waitForSelector",
            selector: selectorMatch[1],
            comment,
            lineNumber,
          });
        }
      } else if (trimmed.includes("page.getByRole(")) {
        const roleMatch = trimmed.match(
          /page\.getByRole\(['"](.*?)['"](?:,\s*\{\s*name:\s*['"](.*?)['"]\s*\})?\)/
        );
        if (roleMatch) {
          const [, role, name] = roleMatch;
          const selector = name
            ? `role:${role}[name="${name}"]`
            : `role:${role}`;
          actions.push({
            type: "click",
            selector,
            comment,
            lineNumber,
          });
        }
      } else if (trimmed.includes("page.getByLabel(")) {
        const labelMatch = trimmed.match(/page\.getByLabel\(['"](.*?)['"]\)/);
        if (labelMatch) {
          actions.push({
            type: "click",
            selector: `getByLabel:${labelMatch[1]}`,
            comment,
            lineNumber,
          });
        }
      } else if (trimmed.includes("page.getByPlaceholder(")) {
        const placeholderMatch = trimmed.match(
          /page\.getByPlaceholder\(['"](.*?)['"]\)/
        );
        if (placeholderMatch) {
          actions.push({
            type: "click",
            selector: `getByPlaceholder:${placeholderMatch[1]}`,
            comment,
            lineNumber,
          });
        }
      } else if (trimmed.includes("page.getByTestId(")) {
        const testIdMatch = trimmed.match(/page\.getByTestId\(['"](.*?)['"]\)/);
        if (testIdMatch) {
          actions.push({
            type: "click",
            selector: `getByTestId:${testIdMatch[1]}`,
            comment,
            lineNumber,
          });
        }
      } else if (trimmed.includes("page.getByTitle(")) {
        const titleMatch = trimmed.match(/page\.getByTitle\(['"](.*?)['"]\)/);
        if (titleMatch) {
          actions.push({
            type: "click",
            selector: `getByTitle:${titleMatch[1]}`,
            comment,
            lineNumber,
          });
        }
      } else if (trimmed.includes("page.getByAltText(")) {
        const altMatch = trimmed.match(/page\.getByAltText\(['"](.*?)['"]\)/);
        if (altMatch) {
          actions.push({
            type: "click",
            selector: `getByAltText:${altMatch[1]}`,
            comment,
            lineNumber,
          });
        }
      } else if (
        trimmed.includes("expect(") &&
        trimmed.includes("toBeVisible()")
      ) {
        const getByTextMatch = trimmed.match(
          /expect\(page\.getByText\(['"](.*?)['"](?:,\s*\{\s*exact:\s*(true|false)\s*\})?\)\)\.toBeVisible\(\)/
        );
        if (getByTextMatch) {
          const text = getByTextMatch[1];
          const exact = getByTextMatch[2] === "true";
          actions.push({
            type: "expect",
            selector: `getByText:${text}`,
            expectation: "toBeVisible",
            options: { exact },
            comment,
            lineNumber,
          });
        } else {
          const locatorFilterMatch = trimmed.match(
            /expect\(page\.locator\(['"](.*?)['"]\)\.filter\(\{\s*hasText:\s*['"](.*?)['"]\s*\}\)\)\.toBeVisible\(\)/
          );
          if (locatorFilterMatch) {
            const selector = locatorFilterMatch[1];
            const filterText = locatorFilterMatch[2];
            actions.push({
              type: "expect",
              selector: `${selector}:has-text("${filterText}")`,
              expectation: "toBeVisible",
              comment,
              lineNumber,
            });
          } else {
            const expectMatch = trimmed.match(
              /expect\(page\.locator\(['"](.*?)['"]\)\)\.toBeVisible\(\)/
            );
            if (expectMatch) {
              actions.push({
                type: "expect",
                selector: expectMatch[1],
                expectation: "toBeVisible",
                comment,
                lineNumber,
              });
            }
          }
        }
      } else if (
        trimmed.includes("expect(") &&
        trimmed.includes("toContainText(")
      ) {
        const expectMatch = trimmed.match(
          /expect\(page\.locator\(['"](.*?)['"]\)\)\.toContainText\(['"](.*?)['"]\)/
        );
        if (expectMatch) {
          actions.push({
            type: "expect",
            selector: expectMatch[1],
            value: expectMatch[2],
            expectation: "toContainText",
            comment,
            lineNumber,
          });
        }
      } else if (
        trimmed.includes("expect(") &&
        trimmed.includes("toBeChecked()")
      ) {
        const expectMatch = trimmed.match(
          /expect\(page\.locator\(['"](.*?)['"]\)\)\.toBeChecked\(\)/
        );
        if (expectMatch) {
          actions.push({
            type: "expect",
            selector: expectMatch[1],
            expectation: "toBeChecked",
            comment,
            lineNumber,
          });
        }
      } else if (
        trimmed.includes("expect(") &&
        trimmed.includes("not.toBeChecked()")
      ) {
        const expectMatch = trimmed.match(
          /expect\(page\.locator\(['"](.*?)['"]\)\)\.not\.toBeChecked\(\)/
        );
        if (expectMatch) {
          actions.push({
            type: "expect",
            selector: expectMatch[1],
            expectation: "not.toBeChecked",
            comment,
            lineNumber,
          });
        }
      }
    } catch (error) {
      console.warn(`Failed to parse line ${lineNumber}: ${trimmed}`, error);
    }
  }

  return actions;
}

function parseVariableMethodCall(
  varName: string,
  method: string,
  args: string,
  comment?: string,
  lineNumber?: number,
  selector?: string
): PlaywrightAction | null {
  const actualSelector = selector || `variable:${varName}`;

  switch (method) {
    case "click":
      return { type: "click", selector: actualSelector, comment, lineNumber };
    case "fill":
      const fillValue = args.match(/['"](.*?)['"]/)?.[1] || "";
      return {
        type: "fill",
        selector: actualSelector,
        value: fillValue,
        comment,
        lineNumber,
      };
    case "check":
      return { type: "check", selector: actualSelector, comment, lineNumber };
    case "uncheck":
      return { type: "uncheck", selector: actualSelector, comment, lineNumber };
    case "selectOption":
      const optionValue = args.match(/['"](.*?)['"]/)?.[1] || "";
      return {
        type: "selectOption",
        selector: actualSelector,
        value: optionValue,
        comment,
        lineNumber,
      };
    case "waitFor":
      const stateMatch = args.match(/{\s*state:\s*['"](.*?)['"]\s*}/);
      return {
        type: "waitFor",
        selector: actualSelector,
        options: stateMatch ? { state: stateMatch[1] } : {},
        comment,
        lineNumber,
      };
    case "hover":
      return { type: "hover", selector: actualSelector, comment, lineNumber };
    case "focus":
      return { type: "focus", selector: actualSelector, comment, lineNumber };
    case "blur":
      return { type: "blur", selector: actualSelector, comment, lineNumber };
    case "scrollIntoViewIfNeeded":
      return {
        type: "scrollIntoView",
        selector: actualSelector,
        comment,
        lineNumber,
      };
    default:
      return null;
  }
}

function parseLocatorCall(locatorCall: string): string {
  const roleMatch = locatorCall.match(
    /getByRole\(['"](.*?)['"](?:,\s*\{\s*name:\s*([^}]+)\s*\})?\)/
  );
  if (roleMatch) {
    const [, role, nameArg] = roleMatch;
    if (nameArg) {
      const regexMatch = nameArg.match(/\/(.*?)\/([gimuy]*)/);
      if (regexMatch) {
        return `role:${role}[name-regex="/${regexMatch[1]}/${regexMatch[2]}"]`;
      }
      const stringMatch = nameArg.match(/['"](.*?)['"]/);
      if (stringMatch) {
        return `role:${role}[name="${stringMatch[1]}"]`;
      }
    }
    return `role:${role}`;
  }

  const textMatch = locatorCall.match(/getByText\(([^)]+)\)/);
  if (textMatch) {
    const args = textMatch[1];
    const regexMatch = args.match(/\/(.*?)\/([gimuy]*)/);
    if (regexMatch) {
      return `getByText-regex:/${regexMatch[1]}/${regexMatch[2]}`;
    }
    const stringMatch = args.match(
      /['"](.*?)['"](?:,\s*\{\s*exact:\s*(true|false)\s*\})?/
    );
    if (stringMatch) {
      const [, text, exact] = stringMatch;
      return `getByText:${text}${exact === "true" ? ":exact" : ""}`;
    }
  }

  const labelMatch = locatorCall.match(/getByLabel\(['"](.*?)['"]\)/);
  if (labelMatch) return `getByLabel:${labelMatch[1]}`;

  const placeholderMatch = locatorCall.match(
    /getByPlaceholder\(['"](.*?)['"]\)/
  );
  if (placeholderMatch) return `getByPlaceholder:${placeholderMatch[1]}`;

  const testIdMatch = locatorCall.match(/getByTestId\(['"](.*?)['"]\)/);
  if (testIdMatch) return `getByTestId:${testIdMatch[1]}`;

  const titleMatch = locatorCall.match(/getByTitle\(['"](.*?)['"]\)/);
  if (titleMatch) return `getByTitle:${titleMatch[1]}`;

  const altMatch = locatorCall.match(/getByAltText\(['"](.*?)['"]\)/);
  if (altMatch) return `getByAltText:${altMatch[1]}`;

  const locatorMatch = locatorCall.match(/locator\(['"](.*?)['"]\)/);
  if (locatorMatch) return locatorMatch[1];

  const locatorWithOptionsMatch = locatorCall.match(
    /locator\(['"](.*?)['"],\s*\{\s*hasText:\s*['"](.*?)['"]\s*\}/
  );
  if (locatorWithOptionsMatch) {
    const [, selector, text] = locatorWithOptionsMatch;
    return `${selector}:has-text("${text}")`;
  }

  return locatorCall;
}

function parseChainedCall(baseSelector: string, chainCall: string): string {
  const filterTextMatch = chainCall.match(
    /filter\(\{\s*hasText:\s*['"](.*?)['"]\s*\}/
  );
  if (filterTextMatch)
    return `${baseSelector}:filter-text("${filterTextMatch[1]}")`;

  const filterRegexMatch = chainCall.match(
    /filter\(\{\s*hasText:\s*\/(.*?)\/([gimuy]*)\s*\}/
  );
  if (filterRegexMatch)
    return `${baseSelector}:filter-regex("/${filterRegexMatch[1]}/${filterRegexMatch[2]}")`;

  const filterHasMatch = chainCall.match(
    /filter\(\{\s*has:\s*page\.(.+?)\s*\}/
  );
  if (filterHasMatch) {
    const innerSelector = parseLocatorCall(filterHasMatch[1]);
    return `${baseSelector}:filter-has("${innerSelector}")`;
  }

  const filterHasNotMatch = chainCall.match(
    /filter\(\{\s*hasNot:\s*page\.(.+?)\s*\}/
  );
  if (filterHasNotMatch) {
    const innerSelector = parseLocatorCall(filterHasNotMatch[1]);
    return `${baseSelector}:filter-has-not("${innerSelector}")`;
  }

  if (chainCall.includes("first()")) return `${baseSelector}:first`;
  if (chainCall.includes("last()")) return `${baseSelector}:last`;

  const nthMatch = chainCall.match(/nth\((\d+)\)/);
  if (nthMatch) return `${baseSelector}:nth(${nthMatch[1]})`;

  const andMatch = chainCall.match(/and\(page\.(.+?)\)/);
  if (andMatch) {
    const otherSelector = parseLocatorCall(andMatch[1]);
    return `${baseSelector}:and("${otherSelector}")`;
  }

  const orMatch = chainCall.match(/or\(page\.(.+?)\)/);
  if (orMatch) {
    const otherSelector = parseLocatorCall(orMatch[1]);
    return `${baseSelector}:or("${otherSelector}")`;
  }

  const getByMatch = chainCall.match(/^(getBy\w+\([^)]+\))/);
  if (getByMatch) {
    const innerSelector = parseLocatorCall(getByMatch[1]);
    return `${baseSelector} >> ${innerSelector}`;
  }

  const locatorChainMatch = chainCall.match(/^locator\(['"](.*?)['"]\)/);
  if (locatorChainMatch) return `${baseSelector} >> ${locatorChainMatch[1]}`;

  return `${baseSelector}:${chainCall}`;
}

function extractSelectorFromArg(selectorArg: string): string {
  return selectorArg.trim().replace(/^['"]|['"]$/g, "");
}

function parseDirectAction(
  method: string,
  args: string,
  comment?: string,
  lineNumber?: number,
  selector?: string
): PlaywrightAction | null {
  switch (method) {
    case "click":
      return { type: "click", selector, comment, lineNumber };
    case "fill":
      const fillValue = args.match(/['"](.*?)['"]/)?.[1] || "";
      return { type: "fill", selector, value: fillValue, comment, lineNumber };
    case "check":
      return { type: "check", selector, comment, lineNumber };
    case "uncheck":
      return { type: "uncheck", selector, comment, lineNumber };
    case "selectOption":
      const optionValue = args.match(/['"](.*?)['"]/)?.[1] || "";
      return {
        type: "selectOption",
        selector,
        value: optionValue,
        comment,
        lineNumber,
      };
    case "waitFor":
      const stateMatch = args.match(/{\s*state:\s*['"](.*?)['"]\s*}/);
      return {
        type: "waitFor",
        selector,
        options: stateMatch ? { state: stateMatch[1] } : {},
        comment,
        lineNumber,
      };
    case "hover":
      return { type: "hover", selector, comment, lineNumber };
    case "focus":
      return { type: "focus", selector, comment, lineNumber };
    case "blur":
      return { type: "blur", selector, comment, lineNumber };
    case "scrollIntoViewIfNeeded":
      return { type: "scrollIntoView", selector, comment, lineNumber };
    default:
      return null;
  }
}

function parseExpectStatement(
  expectation: string,
  comment?: string,
  lineNumber?: number,
  selector?: string
): PlaywrightAction | null {
  if (expectation.includes("toBeVisible()")) {
    return {
      type: "expect",
      selector,
      expectation: "toBeVisible",
      comment,
      lineNumber,
    };
  }

  const toContainTextMatch = expectation.match(
    /toContainText\(['"](.*?)['"]\)/
  );
  if (toContainTextMatch) {
    return {
      type: "expect",
      selector,
      expectation: "toContainText",
      value: toContainTextMatch[1],
      comment,
      lineNumber,
    };
  }

  const toHaveTextMatch = expectation.match(/toHaveText\(['"](.*?)['"]\)/);
  if (toHaveTextMatch) {
    return {
      type: "expect",
      selector,
      expectation: "toHaveText",
      value: toHaveTextMatch[1],
      comment,
      lineNumber,
    };
  }

  if (expectation.includes("toBeChecked()")) {
    return {
      type: "expect",
      selector,
      expectation: "toBeChecked",
      comment,
      lineNumber,
    };
  }

  if (expectation.includes("not.toBeChecked()")) {
    return {
      type: "expect",
      selector,
      expectation: "not.toBeChecked",
      comment,
      lineNumber,
    };
  }

  const toHaveCountMatch = expectation.match(/toHaveCount\((\d+)\)/);
  if (toHaveCountMatch) {
    return {
      type: "expect",
      selector,
      expectation: "toHaveCount",
      value: parseInt(toHaveCountMatch[1]),
      comment,
      lineNumber,
    };
  }

  return null;
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

async function executePlaywrightAction(
  action: PlaywrightAction
): Promise<void> {
  try {
    switch (action.type) {
      case "goto":
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

      case "setViewportSize":
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

      case "waitForLoadState":
        break;

      case "waitForSelector":
        if (action.selector) {
          await waitForElement(action.selector);
        }
        break;

      case "click":
        if (action.selector) {
          const element = await waitForElement(action.selector);
                      if (element) {
              const htmlElement = element as HTMLElement;
              const originalBorder = htmlElement.style.border;
              const originalBoxShadow = htmlElement.style.boxShadow;
              const originalOutline = htmlElement.style.outline;
              
              htmlElement.style.border = "4px solid #ff6b6b";
              htmlElement.style.boxShadow = "0 0 20px rgba(255, 107, 107, 0.8), 0 0 40px rgba(255, 107, 107, 0.4)";
              htmlElement.style.outline = "2px solid #ff6b6b";
              htmlElement.style.transform = "scale(1.02)";
              htmlElement.style.transition = "all 0.3s ease-in-out";
              
              htmlElement.style.animation = "staktrak-click-pulse 0.6s ease-in-out";

              htmlElement.click();

              setTimeout(() => {
                htmlElement.style.border = originalBorder;
                htmlElement.style.boxShadow = originalBoxShadow;
                htmlElement.style.outline = originalOutline;
                htmlElement.style.transform = "";
                htmlElement.style.animation = "";
              }, 600);
          } else {
            throw new Error(`Element not found: ${action.selector}`);
          }
        }
        break;

      case "fill":
        if (action.selector && action.value !== undefined) {
          const element = (await waitForElement(action.selector)) as
            | HTMLInputElement
            | HTMLTextAreaElement;
          if (element) {
            const originalBorder = element.style.border;
            const originalBoxShadow = element.style.boxShadow;
            const originalOutline = element.style.outline;
            
            element.style.border = "3px solid #3b82f6";
            element.style.boxShadow = "0 0 15px rgba(59, 130, 246, 0.6), inset 0 0 10px rgba(59, 130, 246, 0.2)";
            element.style.outline = "2px solid #3b82f6";
            element.style.transform = "scale(1.01)";
            element.style.transition = "all 0.3s ease-in-out";
            element.style.animation = "staktrak-fill-pulse 0.8s ease-in-out";

            element.focus();
            element.value = "";
            element.value = String(action.value);
            element.dispatchEvent(new Event("input", { bubbles: true }));
            element.dispatchEvent(new Event("change", { bubbles: true }));
            
            setTimeout(() => {
              element.style.border = originalBorder;
              element.style.boxShadow = originalBoxShadow;
              element.style.outline = originalOutline;
              element.style.transform = "";
              element.style.animation = "";
            }, 800);
          } else {
            throw new Error(`Input element not found: ${action.selector}`);
          }
        }
        break;

      case "check":
        if (action.selector) {
          const element = (await waitForElement(
            action.selector
          )) as HTMLInputElement;
          if (
            element &&
            (element.type === "checkbox" || element.type === "radio")
          ) {
            const originalBorder = element.style.border;
            const originalBoxShadow = element.style.boxShadow;
            const originalOutline = element.style.outline;
            
            element.style.border = "3px solid #10b981";
            element.style.boxShadow = "0 0 15px rgba(16, 185, 129, 0.6)";
            element.style.outline = "2px solid #10b981";
            element.style.transform = "scale(1.05)";
            element.style.transition = "all 0.3s ease-in-out";
            element.style.animation = "staktrak-check-pulse 0.5s ease-in-out";

            if (!element.checked) {
              element.click();
            }
            
            setTimeout(() => {
              element.style.border = originalBorder;
              element.style.boxShadow = originalBoxShadow;
              element.style.outline = originalOutline;
              element.style.transform = "";
              element.style.animation = "";
            }, 500);
          } else {
            throw new Error(
              `Checkbox/radio element not found: ${action.selector}`
            );
          }
        }
        break;

      case "uncheck":
        if (action.selector) {
          const element = (await waitForElement(
            action.selector
          )) as HTMLInputElement;
          if (element && element.type === "checkbox") {
            const originalBorder = element.style.border;
            const originalBoxShadow = element.style.boxShadow;
            const originalOutline = element.style.outline;
            
            element.style.border = "3px solid #f59e0b";
            element.style.boxShadow = "0 0 15px rgba(245, 158, 11, 0.6)";
            element.style.outline = "2px solid #f59e0b";
            element.style.transform = "scale(1.05)";
            element.style.transition = "all 0.3s ease-in-out";
            element.style.animation = "staktrak-uncheck-pulse 0.5s ease-in-out";

            if (element.checked) {
              element.click();
            }
            
            setTimeout(() => {
              element.style.border = originalBorder;
              element.style.boxShadow = originalBoxShadow;
              element.style.outline = originalOutline;
              element.style.transform = "";
              element.style.animation = "";
            }, 500);
          } else {
            throw new Error(`Checkbox element not found: ${action.selector}`);
          }
        }
        break;

      case "selectOption":
        if (action.selector && action.value !== undefined) {
          const element = (await waitForElement(
            action.selector
          )) as HTMLSelectElement;
          if (element && element.tagName === "SELECT") {
            const originalBorder = element.style.border;
            const originalBoxShadow = element.style.boxShadow;
            const originalOutline = element.style.outline;
            
            element.style.border = "3px solid #8b5cf6";
            element.style.boxShadow = "0 0 15px rgba(139, 92, 246, 0.6)";
            element.style.outline = "2px solid #8b5cf6";
            element.style.transform = "scale(1.02)";
            element.style.transition = "all 0.3s ease-in-out";
            element.style.animation = "staktrak-select-pulse 0.7s ease-in-out";

            element.value = String(action.value);
            element.dispatchEvent(new Event("change", { bubbles: true }));
            
            setTimeout(() => {
              element.style.border = originalBorder;
              element.style.boxShadow = originalBoxShadow;
              element.style.outline = originalOutline;
              element.style.transform = "";
              element.style.animation = "";
            }, 700);
          } else {
            throw new Error(`Select element not found: ${action.selector}`);
          }
        }
        break;

      case "waitForTimeout":
        const shortDelay = Math.min(action.value as number, 500);
        await new Promise((resolve) => setTimeout(resolve, shortDelay));
        break;

      case "waitFor":
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

      case "hover":
        if (action.selector) {
          const element = await waitForElement(action.selector);
          if (element) {
            const htmlElement = element as HTMLElement;
            const originalBorder = htmlElement.style.border;
            const originalBoxShadow = htmlElement.style.boxShadow;
            const originalOutline = htmlElement.style.outline;
            
            htmlElement.style.border = "2px solid #06b6d4";
            htmlElement.style.boxShadow = "0 0 10px rgba(6, 182, 212, 0.5)";
            htmlElement.style.outline = "1px solid #06b6d4";
            htmlElement.style.transform = "scale(1.01)";
            htmlElement.style.transition = "all 0.2s ease-in-out";
            htmlElement.style.animation = "staktrak-hover-pulse 1s ease-in-out";
            
            element.dispatchEvent(
              new MouseEvent("mouseover", { bubbles: true })
            );
            element.dispatchEvent(
              new MouseEvent("mouseenter", { bubbles: true })
            );
            
            setTimeout(() => {
              htmlElement.style.border = originalBorder;
              htmlElement.style.boxShadow = originalBoxShadow;
              htmlElement.style.outline = originalOutline;
              htmlElement.style.transform = "";
              htmlElement.style.animation = "";
            }, 1000);
          } else {
            throw new Error(`Element not found for hover: ${action.selector}`);
          }
        }
        break;

      case "focus":
        if (action.selector) {
          const element = (await waitForElement(
            action.selector
          )) as HTMLElement;
          if (element && typeof element.focus === "function") {
            const originalBorder = element.style.border;
            const originalBoxShadow = element.style.boxShadow;
            const originalOutline = element.style.outline;
            
            element.style.border = "3px solid #6366f1";
            element.style.boxShadow = "0 0 15px rgba(99, 102, 241, 0.6), inset 0 0 10px rgba(99, 102, 241, 0.2)";
            element.style.outline = "2px solid #6366f1";
            element.style.transform = "scale(1.01)";
            element.style.transition = "all 0.3s ease-in-out";
            element.style.animation = "staktrak-focus-pulse 0.6s ease-in-out";
            
            element.focus();
            
            setTimeout(() => {
              element.style.border = originalBorder;
              element.style.boxShadow = originalBoxShadow;
              element.style.outline = originalOutline;
              element.style.transform = "";
              element.style.animation = "";
            }, 600);
          } else {
            throw new Error(
              `Element not found or not focusable: ${action.selector}`
            );
          }
        }
        break;

      case "blur":
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

      case "scrollIntoView":
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

      case "expect":
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
  return findElementsInContext(selector, document);
}

function findElementsInContext(
  selector: string,
  searchContext: Document
): Element[] {
  if (selector.includes(" >> ")) {
    const parts = selector.split(" >> ");
    let elements = findElementsInContext(parts[0], searchContext);

    for (let i = 1; i < parts.length; i++) {
      const newElements: Element[] = [];
      for (const element of elements) {
        const subElements = findElementsInContext(
          parts[i],
          element.ownerDocument || document
        );
        newElements.push(...subElements.filter((el) => element.contains(el)));
      }
      elements = newElements;
    }
    return elements;
  }

  if (selector.includes(":filter-text(")) {
    const match = selector.match(/^(.+?):filter-text\("(.+?)"\)$/);
    if (match) {
      const [, baseSelector, filterText] = match;
      const baseElements = findElementsInContext(baseSelector, searchContext);

      return baseElements.filter((el) => {
        const elementText = el.textContent?.trim() || "";
        return elementText.includes(filterText);
      });
    }
  }

  if (selector.includes(":filter-regex(")) {
    const match = selector.match(
      /^(.+?):filter-regex\("\/(.+?)\/([gimuy]*)"\)$/
    );
    if (match) {
      const [, baseSelector, pattern, flags] = match;
      const regex = new RegExp(pattern, flags);
      const baseElements = findElementsInContext(baseSelector, searchContext);

      return baseElements.filter((el) => {
        const elementText = el.textContent?.trim() || "";
        return regex.test(elementText);
      });
    }
  }

  if (selector.includes(":filter-has(")) {
    const match = selector.match(/^(.+?):filter-has\("(.+?)"\)$/);
    if (match) {
      const [, baseSelector, hasSelector] = match;
      const baseElements = findElementsInContext(baseSelector, searchContext);

      return baseElements.filter((el) => {
        const childElements = findElementsInContext(
          hasSelector,
          el.ownerDocument || document
        );
        return childElements.some((child) => el.contains(child));
      });
    }
  }

  if (selector.includes(":filter-has-not(")) {
    const match = selector.match(/^(.+?):filter-has-not\("(.+?)"\)$/);
    if (match) {
      const [, baseSelector, hasNotSelector] = match;
      const baseElements = findElementsInContext(baseSelector, searchContext);

      return baseElements.filter((el) => {
        const childElements = findElementsInContext(
          hasNotSelector,
          el.ownerDocument || document
        );
        return !childElements.some((child) => el.contains(child));
      });
    }
  }

  if (selector.includes(":first")) {
    const baseSelector = selector.replace(":first", "");
    const elements = findElementsInContext(baseSelector, searchContext);
    return elements.length > 0 ? [elements[0]] : [];
  }

  if (selector.includes(":last")) {
    const baseSelector = selector.replace(":last", "");
    const elements = findElementsInContext(baseSelector, searchContext);
    return elements.length > 0 ? [elements[elements.length - 1]] : [];
  }

  const nthMatch = selector.match(/^(.+?):nth\((\d+)\)$/);
  if (nthMatch) {
    const [, baseSelector, index] = nthMatch;
    const elements = findElementsInContext(baseSelector, searchContext);
    const idx = parseInt(index);
    return idx < elements.length ? [elements[idx]] : [];
  }

  if (selector.includes(":and(")) {
    const match = selector.match(/^(.+?):and\("(.+?)"\)$/);
    if (match) {
      const [, baseSelector, andSelector] = match;
      const baseElements = findElementsInContext(baseSelector, searchContext);
      const andElements = findElementsInContext(andSelector, searchContext);

      return baseElements.filter((el) => andElements.includes(el));
    }
  }

  if (selector.includes(":or(")) {
    const match = selector.match(/^(.+?):or\("(.+?)"\)$/);
    if (match) {
      const [, baseSelector, orSelector] = match;
      const baseElements = findElementsInContext(baseSelector, searchContext);
      const orElements = findElementsInContext(orSelector, searchContext);

      const allElements = [...baseElements, ...orElements];
      return Array.from(new Set(allElements));
    }
  }

  if (selector.startsWith("getByText:")) {
    const parts = selector.substring(10).split(":");
    const text = parts[0];
    const exact = parts[1] === "exact";
    const allElements = searchContext.querySelectorAll("*");

    const matches: Element[] = [];

    for (const el of Array.from(allElements)) {
      const elementText = el.textContent?.trim() || "";
      const elementOwnText =
        el.childNodes.length === 1 &&
        el.childNodes[0].nodeType === Node.TEXT_NODE
          ? el.childNodes[0].textContent?.trim() || ""
          : elementText;

      const matchesText = exact
        ? elementOwnText === text.trim() || elementText === text.trim()
        : elementOwnText.includes(text.trim()) ||
          elementText.includes(text.trim());

      if (matchesText) {
        matches.push(el);
        (el as any).__stakTrakMatchedText = text.trim();
      }
    }

    return matches.sort(
      (a, b) => (a.textContent?.length || 0) - (b.textContent?.length || 0)
    );
  } else if (selector.startsWith("getByText-regex:")) {
    const regexPattern = selector.substring(16);
    const regexMatch = regexPattern.match(/^\/(.+?)\/([gimuy]*)$/);
    if (regexMatch) {
      const [, pattern, flags] = regexMatch;
      const regex = new RegExp(pattern, flags);
      const allElements = searchContext.querySelectorAll("*");

      const matches: Element[] = [];
      for (const el of Array.from(allElements)) {
        const elementText = el.textContent?.trim() || "";
        if (regex.test(elementText)) {
          matches.push(el);
          (el as any).__stakTrakMatchedText = elementText;
        }
      }
      return matches.sort(
        (a, b) => (a.textContent?.length || 0) - (b.textContent?.length || 0)
      );
    }
  } else if (selector.startsWith("role:")) {
    const roleRegexMatch = selector.match(
      /^role:(\w+)\[name-regex="\/(.+?)\/([gimuy]*)"\]$/
    );
    if (roleRegexMatch) {
      const [, role, pattern, flags] = roleRegexMatch;
      const regex = new RegExp(pattern, flags);
      const roleElements = searchContext.querySelectorAll(
        `[role="${role}"], ${getRoleSelector(role)}`
      );

      const matches: Element[] = [];
      for (const el of Array.from(roleElements)) {
        const elementText = el.textContent?.trim() || "";
        const ariaLabel = el.getAttribute("aria-label") || "";
        if (regex.test(elementText) || regex.test(ariaLabel)) {
          matches.push(el);
        }
      }
      return matches;
    }

    const roleMatch = selector.match(/^role:(\w+)(?:\[name="([^"]+)"\])?$/);
    if (roleMatch) {
      const [, role, name] = roleMatch;
      const roleElements = searchContext.querySelectorAll(
        `[role="${role}"], ${getRoleSelector(role)}`
      );

      const matches: Element[] = [];
      for (const el of Array.from(roleElements)) {
        if (name) {
          const elementText = el.textContent?.trim() || "";
          const ariaLabel = el.getAttribute("aria-label") || "";
          if (elementText.includes(name) || ariaLabel.includes(name)) {
            matches.push(el);
          }
        } else {
          matches.push(el);
        }
      }
      return matches;
    }
  } else if (selector.includes(":has-text(")) {
    const match =
      selector.match(/^(.+?):has-text\("(.+?)"\)$/) ||
      selector.match(/^(.+?):has-text\((.+?)\)$/);
    if (match) {
      const [, baseSelector, text] = match;
      const elements = searchContext.querySelectorAll(baseSelector);

      const matches: Element[] = [];

      for (const el of Array.from(elements)) {
        const elementText = el.textContent?.trim() || "";
        const elementOwnText =
          el.childNodes.length === 1 &&
          el.childNodes[0].nodeType === Node.TEXT_NODE
            ? el.childNodes[0].textContent?.trim() || ""
            : elementText;

        if (
          elementOwnText.includes(text.trim()) ||
          elementText.includes(text.trim())
        ) {
          matches.push(el);
          (el as any).__stakTrakMatchedText = text.trim();
        }
      }

      return matches.sort(
        (a, b) => (a.textContent?.length || 0) - (b.textContent?.length || 0)
      );
    }
  } else if (selector.startsWith("getByLabel:")) {
    const labelText = selector.substring(11);
    const labels = searchContext.querySelectorAll("label");

    const matches: Element[] = [];
    for (const label of Array.from(labels)) {
      if (label.textContent?.includes(labelText)) {
        const forAttr = label.getAttribute("for");
        let element: Element | null = null;
        if (forAttr) {
          element = searchContext.querySelector(`#${forAttr}`);
        } else {
          element = label.querySelector("input, textarea, select");
        }
        if (element) matches.push(element);
      }
    }
    return matches;
  } else if (selector.startsWith("getByPlaceholder:")) {
    const placeholder = selector.substring(17);
    return Array.from(
      searchContext.querySelectorAll(`[placeholder*="${placeholder}"]`)
    );
  } else if (selector.startsWith("getByTestId:")) {
    const testId = selector.substring(12);
    return Array.from(
      searchContext.querySelectorAll(`[data-testid="${testId}"]`)
    );
  } else if (selector.startsWith("getByTitle:")) {
    const title = selector.substring(11);
    return Array.from(searchContext.querySelectorAll(`[title*="${title}"]`));
  } else if (selector.startsWith("getByAltText:")) {
    const altText = selector.substring(13);
    return Array.from(searchContext.querySelectorAll(`[alt*="${altText}"]`));
  } else if (selector.startsWith("variable:")) {
    return [];
  } else if (selector.startsWith("text=")) {
    const regexMatch = selector.match(/^text=\/(.+?)\/([gimuy]*)$/);
    if (regexMatch) {
      const [, pattern, flags] = regexMatch;
      const regex = new RegExp(pattern, flags);
      const allElements = searchContext.querySelectorAll("*");
      const matches: Element[] = [];
      for (const el of Array.from(allElements)) {
        const elementText = el.textContent?.trim() || "";
        if (regex.test(elementText)) {
          matches.push(el);
        }
      }
      return matches;
    }

    const exactMatch = selector.match(/^text="([^"]+)"$/);
    if (exactMatch) {
      const text = exactMatch[1];
      const allElements = searchContext.querySelectorAll("*");
      const matches: Element[] = [];
      for (const el of Array.from(allElements)) {
        const elementOwnText =
          el.childNodes.length === 1 &&
          el.childNodes[0].nodeType === Node.TEXT_NODE
            ? el.childNodes[0].textContent?.trim() || ""
            : "";
        if (elementOwnText === text.trim()) {
          matches.push(el);
        }
      }
      return matches;
    }

    const textMatch = selector.match(/text=["']?([^"']+)["']?/);
    if (textMatch) {
      const text = textMatch[1];
      const allElements = searchContext.querySelectorAll("*");
      const matches: Element[] = [];
      for (const el of Array.from(allElements)) {
        const elementText = el.textContent?.trim() || "";
        if (elementText.includes(text.trim())) {
          matches.push(el);
        }
      }
      return matches.sort(
        (a, b) => (a.textContent?.length || 0) - (b.textContent?.length || 0)
      );
    }
  } else if (selector.startsWith("id=")) {
    const id = selector.substring(3);
    return Array.from(searchContext.querySelectorAll(`#${id}`));
  } else if (selector.startsWith("data-testid=")) {
    const testId = selector.substring(13);
    return Array.from(
      searchContext.querySelectorAll(`[data-testid="${testId}"]`)
    );
  } else if (selector.startsWith("data-test-id=")) {
    const testId = selector.substring(14);
    return Array.from(
      searchContext.querySelectorAll(`[data-test-id="${testId}"]`)
    );
  } else if (selector.startsWith("data-test=")) {
    const test = selector.substring(11);
    return Array.from(searchContext.querySelectorAll(`[data-test="${test}"]`));
  } else if (selector.startsWith("xpath=")) {
    const xpath = selector.substring(6);
    const result = searchContext.evaluate(
      xpath,
      searchContext,
      null,
      XPathResult.ORDERED_NODE_SNAPSHOT_TYPE,
      null
    );
    const matches: Element[] = [];
    for (let i = 0; i < result.snapshotLength; i++) {
      const node = result.snapshotItem(i);
      if (node && node.nodeType === Node.ELEMENT_NODE) {
        matches.push(node as Element);
      }
    }
    return matches;
  } else if (selector.includes(",")) {
    const selectors = selector.split(",").map((s) => s.trim());
    const allMatches: Element[] = [];

    for (const sel of selectors) {
      const matches = findElementsInContext(sel, searchContext);
      allMatches.push(...matches);
    }

    return Array.from(new Set(allMatches));
  } else if (selector.includes(":visible")) {
    const baseSelector = selector.replace(":visible", "");
    const elements = baseSelector
      ? Array.from(searchContext.querySelectorAll(baseSelector))
      : [];

    return elements.filter((el) => {
      const style = window.getComputedStyle(el);
      return (
        style.display !== "none" &&
        style.visibility !== "hidden" &&
        style.opacity !== "0" &&
        el.getBoundingClientRect().width > 0 &&
        el.getBoundingClientRect().height > 0
      );
    });
  } else if (selector.includes(":has(")) {
    const match = selector.match(/^(.+?):has\((.+?)\)$/);
    if (match) {
      const [, baseSelector, hasSelector] = match;
      const baseElements = baseSelector
        ? Array.from(searchContext.querySelectorAll(baseSelector))
        : [];

      return baseElements.filter((el) => {
        const childElements = Array.from(el.querySelectorAll(hasSelector));
        return childElements.length > 0;
      });
    }
  } else if (selector.includes(":nth-match(")) {
    const match = selector.match(/^:nth-match\((.+?),\s*(\d+)\)$/);
    if (match) {
      const [, innerSelector, nthStr] = match;
      const nth = parseInt(nthStr);
      const elements = findElementsInContext(innerSelector, searchContext);
      return nth <= elements.length ? [elements[nth - 1]] : [];
    }
  } else if (selector.includes(":text(")) {
    const match =
      selector.match(/^(.+?):text\("([^"]+)"\)$/) ||
      selector.match(/^(.+?):text\(([^)]+)\)$/);
    if (match) {
      const [, baseSelector, text] = match;
      const baseElements = baseSelector
        ? Array.from(searchContext.querySelectorAll(baseSelector))
        : Array.from(searchContext.querySelectorAll("*"));

      let shortestMatch: Element | null = null;
      let shortestLength = Infinity;

      for (const el of baseElements) {
        const elementText = el.textContent?.trim() || "";
        if (elementText.includes(text.trim())) {
          if (elementText.length < shortestLength) {
            shortestMatch = el;
            shortestLength = elementText.length;
          }
        }
      }

      return shortestMatch ? [shortestMatch] : [];
    }
  } else if (selector.includes(":text-is(")) {
    const match =
      selector.match(/^(.+?):text-is\("([^"]+)"\)$/) ||
      selector.match(/^(.+?):text-is\(([^)]+)\)$/);
    if (match) {
      const [, baseSelector, text] = match;
      const baseElements = baseSelector
        ? Array.from(searchContext.querySelectorAll(baseSelector))
        : Array.from(searchContext.querySelectorAll("*"));

      return baseElements.filter((el) => {
        const elementOwnText =
          el.childNodes.length === 1 &&
          el.childNodes[0].nodeType === Node.TEXT_NODE
            ? el.childNodes[0].textContent?.trim() || ""
            : "";
        return elementOwnText === text.trim();
      });
    }
  } else if (
    selector.includes(":right-of(") ||
    selector.includes(":left-of(") ||
    selector.includes(":above(") ||
    selector.includes(":below(") ||
    selector.includes(":near(")
  ) {
    return [];
  } else {
    try {
      return Array.from(searchContext.querySelectorAll(selector));
    } catch (error) {
      return [];
    }
  }

  return [];
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
      background: linear-gradient(135deg, #3b82f6, #1d4ed8) !important;
      color: white !important;
      padding: 3px 6px !important;
      border-radius: 6px !important;
      font-weight: bold !important;
      box-shadow: 0 0 12px rgba(59, 130, 246, 0.8), 0 0 24px rgba(29, 78, 216, 0.4) !important;
      animation: staktrak-text-pulse 2s ease-in-out !important;
      text-shadow: 0 1px 2px rgba(0, 0, 0, 0.3) !important;
      border: 1px solid rgba(255, 255, 255, 0.2) !important;
    }

    @keyframes staktrak-text-pulse {
      0% { 
        background: linear-gradient(135deg, #3b82f6, #1d4ed8);
        box-shadow: 0 0 12px rgba(59, 130, 246, 0.8), 0 0 24px rgba(29, 78, 216, 0.4);
        transform: scale(1);
      }
      50% { 
        background: linear-gradient(135deg, #1d4ed8, #1e40af);
        box-shadow: 0 0 20px rgba(29, 78, 216, 1), 0 0 40px rgba(30, 64, 175, 0.6);
        transform: scale(1.05);
      }
      100% { 
        background: linear-gradient(135deg, #3b82f6, #1d4ed8);
        box-shadow: 0 0 12px rgba(59, 130, 246, 0.8), 0 0 24px rgba(29, 78, 216, 0.4);
        transform: scale(1);
      }
    }

    @keyframes staktrak-click-pulse {
      0% { transform: scale(1.02); box-shadow: 0 0 20px rgba(255, 107, 107, 0.8), 0 0 40px rgba(255, 107, 107, 0.4); }
      50% { transform: scale(1.05); box-shadow: 0 0 30px rgba(255, 107, 107, 1), 0 0 60px rgba(255, 107, 107, 0.6); }
      100% { transform: scale(1.02); box-shadow: 0 0 20px rgba(255, 107, 107, 0.8), 0 0 40px rgba(255, 107, 107, 0.4); }
    }

    @keyframes staktrak-fill-pulse {
      0% { transform: scale(1.01); box-shadow: 0 0 15px rgba(59, 130, 246, 0.6), inset 0 0 10px rgba(59, 130, 246, 0.2); }
      50% { transform: scale(1.03); box-shadow: 0 0 25px rgba(59, 130, 246, 0.9), inset 0 0 15px rgba(59, 130, 246, 0.4); }
      100% { transform: scale(1.01); box-shadow: 0 0 15px rgba(59, 130, 246, 0.6), inset 0 0 10px rgba(59, 130, 246, 0.2); }
    }

    @keyframes staktrak-check-pulse {
      0% { transform: scale(1.05); box-shadow: 0 0 15px rgba(16, 185, 129, 0.6); }
      50% { transform: scale(1.08); box-shadow: 0 0 25px rgba(16, 185, 129, 0.9); }
      100% { transform: scale(1.05); box-shadow: 0 0 15px rgba(16, 185, 129, 0.6); }
    }

    @keyframes staktrak-uncheck-pulse {
      0% { transform: scale(1.05); box-shadow: 0 0 15px rgba(245, 158, 11, 0.6); }
      50% { transform: scale(1.08); box-shadow: 0 0 25px rgba(245, 158, 11, 0.9); }
      100% { transform: scale(1.05); box-shadow: 0 0 15px rgba(245, 158, 11, 0.6); }
    }

    @keyframes staktrak-select-pulse {
      0% { transform: scale(1.02); box-shadow: 0 0 15px rgba(139, 92, 246, 0.6); }
      50% { transform: scale(1.04); box-shadow: 0 0 25px rgba(139, 92, 246, 0.9); }
      100% { transform: scale(1.02); box-shadow: 0 0 15px rgba(139, 92, 246, 0.6); }
    }

    @keyframes staktrak-hover-pulse {
      0% { transform: scale(1.01); box-shadow: 0 0 10px rgba(6, 182, 212, 0.5); }
      50% { transform: scale(1.02); box-shadow: 0 0 20px rgba(6, 182, 212, 0.8); }
      100% { transform: scale(1.01); box-shadow: 0 0 10px rgba(6, 182, 212, 0.5); }
    }

    @keyframes staktrak-focus-pulse {
      0% { transform: scale(1.01); box-shadow: 0 0 15px rgba(99, 102, 241, 0.6), inset 0 0 10px rgba(99, 102, 241, 0.2); }
      50% { transform: scale(1.03); box-shadow: 0 0 25px rgba(99, 102, 241, 0.9), inset 0 0 15px rgba(99, 102, 241, 0.4); }
      100% { transform: scale(1.01); box-shadow: 0 0 15px rgba(99, 102, 241, 0.6), inset 0 0 10px rgba(99, 102, 241, 0.2); }
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
        const textElement = await waitForElement(action.selector, String(action.value || ""));
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
          String(action.value || "")
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

function getActionDescription(action: PlaywrightAction): string {
  switch (action.type) {
    case "goto":
      return `Navigate to ${action.value}`;
    case "click":
      return `Click element: ${action.selector}`;
    case "fill":
      return `Fill "${action.value}" in ${action.selector}`;
    case "check":
      return `Check checkbox: ${action.selector}`;
    case "uncheck":
      return `Uncheck checkbox: ${action.selector}`;
    case "selectOption":
      return `Select "${action.value}" in ${action.selector}`;
    case "hover":
      return `Hover over element: ${action.selector}`;
    case "focus":
      return `Focus element: ${action.selector}`;
    case "blur":
      return `Blur element: ${action.selector}`;
    case "scrollIntoView":
      return `Scroll element into view: ${action.selector}`;
    case "waitFor":
      return `Wait for element: ${action.selector}`;
    case "expect":
      return `Verify ${action.selector} ${action.expectation}`;
    case "setViewportSize":
      return `Set viewport size to ${action.value}`;
    case "waitForTimeout":
      return `Wait ${action.value}ms`;
    case "waitForLoadState":
      return "Wait for page to load";
    case "waitForSelector":
      return `Wait for element: ${action.selector}`;
    default:
      return `Execute ${action.type}`;
  }
}

export function startPlaywrightReplay(testCode: string): void {
  try {
    const actions = parsePlaywrightTest(testCode);

    if (actions.length === 0) {
      throw new Error("No valid actions found in test code");
    }

    playwrightReplayRef.current = {
      actions,
      status: ReplayStatus.PLAYING,
      currentActionIndex: 0,
      testCode,
      errors: [],
      timeouts: [],
    };

    window.parent.postMessage(
      {
        type: "staktrak-playwright-replay-started",
        totalActions: actions.length,
        actions: actions,
      },
      "*"
    );

    executeNextPlaywrightAction();
  } catch (error) {
    window.parent.postMessage(
      {
        type: "staktrak-playwright-replay-error",
        error: error instanceof Error ? error.message : "Unknown error",
      },
      "*"
    );
  }
}

async function executeNextPlaywrightAction(): Promise<void> {
  const state = playwrightReplayRef.current;
  if (!state || state.status !== ReplayStatus.PLAYING) {
    return;
  }

  if (state.currentActionIndex >= state.actions.length) {
    state.status = ReplayStatus.COMPLETED;
    window.parent.postMessage(
      {
        type: "staktrak-playwright-replay-completed",
      },
      "*"
    );
    return;
  }

  const action = state.actions[state.currentActionIndex];

  try {
    window.parent.postMessage(
      {
        type: "staktrak-playwright-replay-progress",
        current: state.currentActionIndex + 1,
        total: state.actions.length,
        currentAction: {
          ...action,
          description: getActionDescription(action),
        },
      },
      "*"
    );

    await executePlaywrightAction(action);

    state.currentActionIndex++;

    setTimeout(() => {
      executeNextPlaywrightAction();
    }, 300);
  } catch (error) {
    state.errors.push(
      `Action ${state.currentActionIndex + 1}: ${error instanceof Error ? error.message : "Unknown error"}`
    );

    state.currentActionIndex++;

    window.parent.postMessage(
      {
        type: "staktrak-playwright-replay-error",
        error: error instanceof Error ? error.message : "Unknown error",
        actionIndex: state.currentActionIndex - 1,
        action: action,
      },
      "*"
    );

    executeNextPlaywrightAction();
  }
}

export function pausePlaywrightReplay(): void {
  const state = playwrightReplayRef.current;
  if (state) {
    state.status = ReplayStatus.PAUSED;

    state.timeouts.forEach((id) => clearTimeout(id as any));
    state.timeouts = [];

    window.parent.postMessage(
      { type: "staktrak-playwright-replay-paused" },
      "*"
    );
  }
}

export function resumePlaywrightReplay(): void {
  const state = playwrightReplayRef.current;
  if (state && state.status === ReplayStatus.PAUSED) {
    state.status = ReplayStatus.PLAYING;

    executeNextPlaywrightAction();

    window.parent.postMessage(
      { type: "staktrak-playwright-replay-resumed" },
      "*"
    );
  }
}

export function stopPlaywrightReplay(): void {
  const state = playwrightReplayRef.current;
  if (state) {
    state.status = ReplayStatus.IDLE;

    state.timeouts.forEach((id) => clearTimeout(id as any));
    state.timeouts = [];

    window.parent.postMessage(
      { type: "staktrak-playwright-replay-stopped" },
      "*"
    );
  }
}

export function getPlaywrightReplayState(): PlaywrightReplayState | null {
  const state = playwrightReplayRef.current;
  if (!state) return null;

  return {
    actions: state.actions,
    status: state.status,
    currentActionIndex: state.currentActionIndex,
    testCode: state.testCode,
    errors: state.errors,
  };
}

export function initPlaywrightReplay(): void {
  window.addEventListener("message", (event) => {
    const { data } = event;

    if (!data || !data.type) return;

    switch (data.type) {
      case "staktrak-playwright-replay-start":
        if (data.testCode) {
          startPlaywrightReplay(data.testCode);
        }
        break;

      case "staktrak-playwright-replay-pause":
        pausePlaywrightReplay();
        break;

      case "staktrak-playwright-replay-resume":
        resumePlaywrightReplay();
        break;

      case "staktrak-playwright-replay-stop":
        stopPlaywrightReplay();
        break;

      case "staktrak-playwright-replay-ping":
        const currentState = getPlaywrightReplayState();
        window.parent.postMessage(
          {
            type: "staktrak-playwright-replay-pong",
            state: currentState,
          },
          "*"
        );
        break;
    }
  });
}
