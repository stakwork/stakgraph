import { PlaywrightAction } from "../types";
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
      } else if (trimmed.includes("page.locator('[data-staktrak-id=") && trimmed.includes(".fill(")) {
        const stakFillMatch = trimmed.match(/page\.locator\('\[data-staktrak-id="([^"]+)"\]'\)\.fill\(['"]([^'"]*)['"]\)/);
        if (stakFillMatch) {
          actions.push({
            type: "fill",
            selector: `[data-staktrak-id="${stakFillMatch[1]}"]`,
            value: stakFillMatch[2],
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
      } else if (trimmed.includes("page.locator('[data-staktrak-id=") && trimmed.includes(".check(")) {
        const stakCheckMatch = trimmed.match(/page\.locator\('\[data-staktrak-id="([^"]+)"\]'\)\.check\(\)/);
        if (stakCheckMatch) {
          actions.push({
            type: "check",
            selector: `[data-staktrak-id="${stakCheckMatch[1]}"]`,
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
      } else if (trimmed.includes("page.locator('[data-staktrak-id=") && trimmed.includes(".uncheck(")) {
        const stakUncheckMatch = trimmed.match(/page\.locator\('\[data-staktrak-id="([^"]+)"\]'\)\.uncheck\(\)/);
        if (stakUncheckMatch) {
          actions.push({
            type: "uncheck",
            selector: `[data-staktrak-id="${stakUncheckMatch[1]}"]`,
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
      } else if (trimmed.includes("page.locator('[data-staktrak-id=") && trimmed.includes(".selectOption(")) {
        const stakSelectMatch = trimmed.match(/page\.locator\('\[data-staktrak-id="([^"]+)"\]'\)\.selectOption\(['"]([^'"]*)['"]\)/);
        if (stakSelectMatch) {
          actions.push({
            type: "selectOption",
            selector: `[data-staktrak-id="${stakSelectMatch[1]}"]`,
            value: stakSelectMatch[2],
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
      } else if (trimmed.includes('page.waitForURL(')) {
        const urlMatch = trimmed.match(/page\.waitForURL\(['\"](.*?)['\"]\)/);
        if (urlMatch) {
          actions.push({
            type: 'waitForURL',
            value: urlMatch[1],
            comment,
            lineNumber,
          });
        }
      } else if (/page\.[a-zA-Z]+\([^)]*\)\.click\([^)]*\)\s*;?$/.test(trimmed)) {
        const locatorCallMatch = trimmed.match(/page\.([a-zA-Z]+\([^)]*\))\.click\([^)]*\)/);
        if (locatorCallMatch) {
          const selector = parseLocatorCall(locatorCallMatch[1]);
          actions.push({
            type: 'click',
            selector,
            comment,
            lineNumber,
          });
        }
      } else if (trimmed.startsWith('await Promise.all([') && trimmed.includes('waitForURL')) {
        // Attempt to capture compound click+waitForURL pattern
        // Collect lines until closing ']);'
        const blockLines: string[] = [trimmed];
        let j = lineNumber; // current line number in outer context
        // We'll peek ahead in the original testCode string; fallback if not available
        // Simpler approach: scan next 6 lines from 'lines'
        for (let k = 1; k <= 6 && lineNumber + k - 1 < lines.length; k++) {
          const peek = lines[lineNumber + k - 1].trim();
            blockLines.push(peek);
            if (peek.endsWith(']);')) break;
        }
        const block = blockLines.join(' ');
        const url = block.match(/page\.waitForURL\(['\"](.*?)['\"]\)/)?.[1];
        const clickSelector = block.match(/page\.(getBy[^.]+\([^)]*\)|locator\([^)]*\))\.click\(\)/)?.[1];
        if (url) {
          actions.push({ type: 'waitForURL', value: url, comment: (comment?comment+' ':'')+'(compound)', lineNumber });
        }
        if (clickSelector) {
          const selector = parseLocatorCall(clickSelector);
          actions.push({ type: 'click', selector, comment, lineNumber });
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
      } else if (trimmed.includes("page.locator('[data-staktrak-id=")) {
        const stakIdMatch = trimmed.match(/page\.locator\('\[data-staktrak-id="([^"]+)"\]'\)/);
        if (stakIdMatch) {
          actions.push({
            type: "click",
            selector: `[data-staktrak-id="${stakIdMatch[1]}"]`,
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
            
            const stakExpectMatch = trimmed.match(
              /expect\(page\.locator\('\[data-staktrak-id="([^"]+)"\]'\)\)\.toBeVisible\(\)/
            );
            if (stakExpectMatch) {
              actions.push({
                type: "expect",
                selector: `[data-staktrak-id="${stakExpectMatch[1]}"]`,
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