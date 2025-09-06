"use strict";
var userBehaviour = (() => {
  var __defProp = Object.defineProperty;
  var __defProps = Object.defineProperties;
  var __getOwnPropDesc = Object.getOwnPropertyDescriptor;
  var __getOwnPropDescs = Object.getOwnPropertyDescriptors;
  var __getOwnPropNames = Object.getOwnPropertyNames;
  var __getOwnPropSymbols = Object.getOwnPropertySymbols;
  var __hasOwnProp = Object.prototype.hasOwnProperty;
  var __propIsEnum = Object.prototype.propertyIsEnumerable;
  var __defNormalProp = (obj, key, value) => key in obj ? __defProp(obj, key, { enumerable: true, configurable: true, writable: true, value }) : obj[key] = value;
  var __spreadValues = (a, b) => {
    for (var prop in b || (b = {}))
      if (__hasOwnProp.call(b, prop))
        __defNormalProp(a, prop, b[prop]);
    if (__getOwnPropSymbols)
      for (var prop of __getOwnPropSymbols(b)) {
        if (__propIsEnum.call(b, prop))
          __defNormalProp(a, prop, b[prop]);
      }
    return a;
  };
  var __spreadProps = (a, b) => __defProps(a, __getOwnPropDescs(b));
  var __export = (target, all) => {
    for (var name in all)
      __defProp(target, name, { get: all[name], enumerable: true });
  };
  var __copyProps = (to, from, except, desc) => {
    if (from && typeof from === "object" || typeof from === "function") {
      for (let key of __getOwnPropNames(from))
        if (!__hasOwnProp.call(to, key) && key !== except)
          __defProp(to, key, { get: () => from[key], enumerable: !(desc = __getOwnPropDesc(from, key)) || desc.enumerable });
    }
    return to;
  };
  var __toCommonJS = (mod) => __copyProps(__defProp({}, "__esModule", { value: true }), mod);

  // src/index.ts
  var index_exports = {};
  __export(index_exports, {
    default: () => index_default
  });

  // src/utils.ts
  var getTimeStamp = () => Date.now();
  var isInputOrTextarea = (element) => element.tagName === "INPUT" || element.tagName === "TEXTAREA" || element.isContentEditable;
  var generateSelectorStrategies = (element) => {
    var _a;
    const htmlEl = element;
    const tagName = element.tagName.toLowerCase();
    const fallbacks = [];
    const testId = (_a = htmlEl.dataset) == null ? void 0 : _a.testid;
    if (testId) {
      return {
        primary: `[data-testid="${testId}"]`,
        fallbacks: [],
        tagName,
        text: getElementText(element),
        ariaLabel: htmlEl.getAttribute("aria-label") || void 0,
        title: htmlEl.getAttribute("title") || void 0,
        role: htmlEl.getAttribute("role") || void 0
      };
    }
    const id = htmlEl.id;
    if (id && /^[a-zA-Z][\w-]*$/.test(id)) {
      return {
        primary: `#${id}`,
        fallbacks: [],
        tagName,
        text: getElementText(element),
        ariaLabel: htmlEl.getAttribute("aria-label") || void 0,
        title: htmlEl.getAttribute("title") || void 0,
        role: htmlEl.getAttribute("role") || void 0
      };
    }
    const text = getElementText(element);
    if (text && (tagName === "button" || tagName === "a" || htmlEl.getAttribute("role") === "button")) {
      const textSelector = generateTextBasedSelector(element, text);
      if (textSelector) {
        fallbacks.push(textSelector);
      }
    }
    const ariaLabel = htmlEl.getAttribute("aria-label");
    if (ariaLabel) {
      fallbacks.push(`[aria-label="${ariaLabel}"]`);
    }
    const role = htmlEl.getAttribute("role");
    if (role && text) {
      fallbacks.push(`[role="${role}"]`);
    }
    const classSelector = generateClassBasedSelector(element);
    if (classSelector && classSelector !== tagName) {
      fallbacks.push(classSelector);
    }
    if (tagName === "input") {
      const type = element.type;
      const name = element.name;
      if (type) fallbacks.push(`input[type="${type}"]`);
      if (name) fallbacks.push(`input[name="${name}"]`);
    }
    const contextualSelector = generateContextualSelector(element);
    if (contextualSelector) {
      fallbacks.push(contextualSelector);
    }
    const xpath = generateXPath(element);
    const primary = fallbacks.length > 0 ? fallbacks[0] : tagName;
    return {
      primary,
      fallbacks: fallbacks.slice(1),
      // Remove primary from fallbacks
      text,
      ariaLabel: ariaLabel || void 0,
      title: htmlEl.getAttribute("title") || void 0,
      role: role || void 0,
      tagName,
      xpath
    };
  };
  var getElementText = (element) => {
    var _a;
    const htmlEl = element;
    if (element.tagName === "BUTTON" || element.tagName === "A") {
      const text = (_a = htmlEl.textContent) == null ? void 0 : _a.trim();
      if (text && text.length > 0 && text.length < 100) {
        return text;
      }
    }
    if (element.tagName === "INPUT") {
      const input = element;
      return input.placeholder || input.value || void 0;
    }
    return void 0;
  };
  var generateTextBasedSelector = (element, text) => {
    const tagName = element.tagName.toLowerCase();
    const cleanText = text.replace(/"/g, '\\"').trim();
    if (cleanText.length === 0 || cleanText.length > 50) return null;
    if (tagName === "button" || tagName === "a" || element.getAttribute("role") === "button") {
      return `text=${cleanText}`;
    }
    return null;
  };
  var generateClassBasedSelector = (element) => {
    const tagName = element.tagName.toLowerCase();
    const classList = element.classList;
    if (!classList.length) return tagName;
    const safeClasses = Array.from(classList).filter((cls) => {
      if (cls.includes("_") && cls.match(/[0-9a-f]{6}/)) return false;
      if (cls.includes("module__")) return false;
      if (cls.includes("emotion-")) return false;
      if (cls.includes("css-")) return false;
      if (cls.length > 30) return false;
      return /^[a-zA-Z][a-zA-Z0-9-]*$/.test(cls);
    });
    if (safeClasses.length === 0) return tagName;
    const limitedClasses = safeClasses.slice(0, 3);
    return `${tagName}.${limitedClasses.join(".")}`;
  };
  var generateContextualSelector = (element) => {
    const tagName = element.tagName.toLowerCase();
    const parent = element.parentElement;
    if (!parent) return null;
    if (tagName === "button" && parent.tagName === "NAV") {
      return "nav button";
    }
    if (tagName === "button" && (parent.tagName === "HEADER" || parent.closest("header"))) {
      return "header button";
    }
    if ((tagName === "input" || tagName === "button") && parent.closest("form")) {
      return `form ${tagName}`;
    }
    return null;
  };
  var generateXPath = (element) => {
    if (element.id) {
      return `//*[@id="${element.id}"]`;
    }
    const parts = [];
    let current = element;
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
      if (parts.length > 10) break;
    }
    return "/" + parts.join("/");
  };
  var createClickDetail = (e) => {
    const target = e.target;
    const selectors = generateSelectorStrategies(target);
    return {
      x: e.clientX,
      y: e.clientY,
      timestamp: getTimeStamp(),
      selectors,
      elementInfo: {
        tagName: target.tagName.toLowerCase(),
        id: target.id || void 0,
        className: target.className || void 0,
        attributes: getElementAttributes(target)
      }
    };
  };
  var getElementAttributes = (element) => {
    const attrs = {};
    const htmlEl = element;
    const importantAttrs = [
      "type",
      "name",
      "role",
      "aria-label",
      "title",
      "placeholder",
      "value"
    ];
    importantAttrs.forEach((attr) => {
      const value = htmlEl.getAttribute(attr);
      if (value) attrs[attr] = value;
    });
    return attrs;
  };
  var getElementSelector = (element) => {
    const strategies = generateSelectorStrategies(element);
    return strategies.primary;
  };
  var filterClickDetails = (clickDetails, assertions, config) => {
    if (!clickDetails.length) return [];
    let filtered = config.filterAssertionClicks ? clickDetails.filter(
      (click) => !assertions.some(
        (assertion) => Math.abs(click.timestamp - assertion.timestamp) < 1e3 && (click.selectors.primary.includes(assertion.selector) || assertion.selector.includes(click.selectors.primary) || click.selectors.fallbacks.some(
          (f) => f.includes(assertion.selector) || assertion.selector.includes(f)
        ))
      )
    ) : clickDetails;
    const clicksBySelector = {};
    filtered.forEach((click) => {
      const key = click.selectors.primary;
      if (!clicksBySelector[key]) clicksBySelector[key] = [];
      clicksBySelector[key].push(click);
    });
    const result = [];
    Object.values(clicksBySelector).forEach((clicks) => {
      clicks.sort((a, b) => a.timestamp - b.timestamp);
      let lastClick = null;
      clicks.forEach((click) => {
        if (!lastClick || click.timestamp - lastClick.timestamp > config.multiClickInterval) {
          result.push(click);
        }
        lastClick = click;
      });
    });
    return result.sort((a, b) => a.timestamp - b.timestamp);
  };

  // src/debug.ts
  function rectsIntersect(rect1, rect2) {
    return !(rect1.x + rect1.width < rect2.x || rect2.x + rect2.width < rect1.x || rect1.y + rect1.height < rect2.y || rect2.y + rect2.height < rect1.y);
  }
  function isReactDevModeActive() {
    try {
      const allElements = Array.from(document.querySelectorAll("*"));
      for (const element of allElements) {
        const fiberKey = Object.keys(element).find(
          (key) => key.startsWith("__reactFiber$") || key.startsWith("__reactInternalInstance$")
        );
        if (fiberKey) {
          return true;
        }
      }
      return false;
    } catch (error) {
      console.error("Error checking React dev mode:", error);
      return false;
    }
  }
  function getComponentNameFromFiber(element) {
    try {
      const fiberKey = Object.keys(element).find(
        (key) => key.startsWith("__reactFiber$") || key.startsWith("__reactInternalInstance$")
      );
      if (!fiberKey) {
        return null;
      }
      let fiber = element[fiberKey];
      let level = 0;
      const maxTraversalDepth = 10;
      while (fiber && level < maxTraversalDepth) {
        if (typeof fiber.type === "string") {
          fiber = fiber.return;
          level++;
          continue;
        }
        if (fiber.type) {
          let componentName = null;
          if (fiber.type.displayName) {
            componentName = fiber.type.displayName;
          } else if (fiber.type.name) {
            componentName = fiber.type.name;
          } else if (fiber.type.render) {
            componentName = fiber.type.render.displayName || fiber.type.render.name || "ForwardRef";
          } else if (fiber.type.type) {
            componentName = fiber.type.type.displayName || fiber.type.type.name || "Memo";
          } else if (fiber.type._payload && fiber.type._payload._result) {
            componentName = fiber.type._payload._result.name || "LazyComponent";
          } else if (fiber.type._context) {
            componentName = fiber.type._context.displayName || "Context";
          } else if (fiber.type === Symbol.for("react.fragment")) {
            fiber = fiber.return;
            level++;
            continue;
          }
          if (componentName) {
            return {
              name: componentName,
              level,
              type: typeof fiber.type === "function" ? "function" : "class"
            };
          }
        }
        fiber = fiber.return;
        level++;
      }
      return null;
    } catch (error) {
      console.error("Error extracting component name:", error);
      return null;
    }
  }
  function extractReactDebugSource(element) {
    var _a, _b, _c;
    try {
      const fiberKey = Object.keys(element).find(
        (key) => key.startsWith("__reactFiber$") || key.startsWith("__reactInternalInstance$")
      );
      if (!fiberKey) {
        return null;
      }
      let fiber = element[fiberKey];
      let level = 0;
      const maxTraversalDepth = Number((_a = window.STAKTRAK_CONFIG) == null ? void 0 : _a.maxTraversalDepth) || 10;
      const extractSource = (source) => {
        if (!source) return null;
        return {
          fileName: source.fileName,
          lineNumber: source.lineNumber,
          columnNumber: source.columnNumber
        };
      };
      while (fiber && level < maxTraversalDepth) {
        const source = fiber._debugSource || ((_b = fiber.memoizedProps) == null ? void 0 : _b.__source) || ((_c = fiber.pendingProps) == null ? void 0 : _c.__source);
        if (source) {
          return extractSource(source);
        }
        fiber = fiber.return;
        level++;
      }
      return null;
    } catch (error) {
      console.error("Error extracting React debug source:", error);
      return null;
    }
  }
  function debugMsg(data) {
    var _a, _b;
    const { messageId, coordinates } = data;
    try {
      const sourceFiles = [];
      const processedFiles = /* @__PURE__ */ new Map();
      const componentNames = [];
      const processedComponents = /* @__PURE__ */ new Set();
      let elementsToProcess = [];
      if (coordinates.width === 0 && coordinates.height === 0) {
        const element = document.elementFromPoint(coordinates.x, coordinates.y);
        if (element) {
          elementsToProcess = [element];
          let parent = element.parentElement;
          while (parent && parent !== document.body) {
            elementsToProcess.push(parent);
            parent = parent.parentElement;
          }
        }
      } else {
        const allElements = document.querySelectorAll("*");
        elementsToProcess = Array.from(allElements).filter((el) => {
          const rect = el.getBoundingClientRect();
          return rectsIntersect(
            {
              x: rect.left,
              y: rect.top,
              width: rect.width,
              height: rect.height
            },
            coordinates
          );
        });
      }
      for (const element of elementsToProcess) {
        const componentInfo = getComponentNameFromFiber(element);
        if (componentInfo && !processedComponents.has(componentInfo.name)) {
          processedComponents.add(componentInfo.name);
          componentNames.push({
            name: componentInfo.name,
            level: componentInfo.level,
            type: componentInfo.type,
            element: element.tagName.toLowerCase()
          });
        }
        const dataSource = element.getAttribute("data-source") || element.getAttribute("data-inspector-relative-path");
        const dataLine = element.getAttribute("data-line") || element.getAttribute("data-inspector-line");
        if (dataSource && dataLine) {
          const lineNum = parseInt(dataLine, 10);
          if (!processedFiles.has(dataSource) || !((_a = processedFiles.get(dataSource)) == null ? void 0 : _a.has(lineNum))) {
            if (!processedFiles.has(dataSource)) {
              processedFiles.set(dataSource, /* @__PURE__ */ new Set());
            }
            processedFiles.get(dataSource).add(lineNum);
            let fileEntry = sourceFiles.find((f) => f.file === dataSource);
            if (!fileEntry) {
              fileEntry = { file: dataSource, lines: [] };
              sourceFiles.push(fileEntry);
            }
            fileEntry.lines.push(lineNum);
          }
        } else {
          const debugSource = extractReactDebugSource(element);
          if (debugSource && debugSource.fileName && debugSource.lineNumber) {
            const fileName = debugSource.fileName;
            const lineNum = debugSource.lineNumber;
            if (!processedFiles.has(fileName) || !((_b = processedFiles.get(fileName)) == null ? void 0 : _b.has(lineNum))) {
              if (!processedFiles.has(fileName)) {
                processedFiles.set(fileName, /* @__PURE__ */ new Set());
              }
              processedFiles.get(fileName).add(lineNum);
              let fileEntry = sourceFiles.find((f) => f.file === fileName);
              if (!fileEntry) {
                fileEntry = { file: fileName, lines: [] };
                sourceFiles.push(fileEntry);
              }
              fileEntry.lines.push(lineNum);
              const tagName = element.tagName.toLowerCase();
              const className = element.className ? `.${element.className.split(" ")[0]}` : "";
              fileEntry.context = `${tagName}${className}`;
            }
          }
        }
      }
      sourceFiles.forEach((file) => {
        file.lines.sort((a, b) => a - b);
      });
      const formatComponentsForChat = (components) => {
        if (components.length === 0) return void 0;
        const sortedComponents = components.sort((a, b) => a.level - b.level).slice(0, 3);
        const componentLines = sortedComponents.map((c) => {
          const nameToUse = c.name || "Unknown";
          return `&lt;${nameToUse}&gt; (${c.level} level${c.level !== 1 ? "s" : ""} up)`;
        });
        return "React Components Found:\n" + componentLines.join("\n");
      };
      if (sourceFiles.length === 0) {
        if (componentNames.length > 0) {
          const formattedMessage = formatComponentsForChat(componentNames);
          sourceFiles.push({
            file: "React component detected",
            lines: [],
            context: `Components found: ${componentNames.map((c) => c.name).join(", ")}`,
            componentNames,
            message: formattedMessage
          });
        } else {
          sourceFiles.push({
            file: "No React components detected",
            lines: [],
            context: "The selected element may not be a React component or may be a native DOM element",
            message: "Try selecting an interactive element like a button or link"
          });
        }
      } else {
        sourceFiles.forEach((file) => {
          if (!file.componentNames && componentNames.length > 0) {
            file.componentNames = componentNames;
            const formattedMessage = formatComponentsForChat(componentNames);
            if (formattedMessage) {
              file.message = formattedMessage;
            }
          }
        });
      }
      window.parent.postMessage(
        {
          type: "staktrak-debug-response",
          messageId,
          success: true,
          sourceFiles
        },
        "*"
      );
    } catch (error) {
      console.error("Error processing debug request:", error);
      window.parent.postMessage(
        {
          type: "staktrak-debug-response",
          messageId,
          success: false,
          error: error instanceof Error ? error.message : "Unknown error",
          sourceFiles: []
        },
        "*"
      );
    }
  }

  // src/playwright-replay.ts
  var playwrightReplayRef = {
    current: null
  };
  function parsePlaywrightTest(testCode) {
    const actions = [];
    const lines = testCode.split("\n");
    let lineNumber = 0;
    const variables = /* @__PURE__ */ new Map();
    for (const line of lines) {
      lineNumber++;
      const trimmed = line.trim();
      if (!trimmed || trimmed.startsWith("//") || trimmed.startsWith("import") || trimmed.startsWith("test(") || trimmed.includes("async ({ page })") || trimmed === "}); " || trimmed === "});") {
        continue;
      }
      const commentMatch = line.match(/^\s*\/\/\s*(.+)/);
      const comment = commentMatch ? commentMatch[1] : void 0;
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
            const baseSelector = variables.get(baseVarName);
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
            const selector = variables.get(varName);
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
            const selector = variables.get(varName);
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
            const selector = variables.get(varName);
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
            lineNumber
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
              lineNumber
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
                height: parseInt(sizeMatch[2])
              },
              comment,
              lineNumber
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
            lineNumber
          });
        } else if (trimmed.includes("page.click(")) {
          const selectorMatch = trimmed.match(/page\.click\(['"](.*?)['"]\)/);
          if (selectorMatch) {
            actions.push({
              type: "click",
              selector: selectorMatch[1],
              comment,
              lineNumber
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
              lineNumber
            });
          }
        } else if (trimmed.includes("page.check(")) {
          const selectorMatch = trimmed.match(/page\.check\(['"](.*?)['"]\)/);
          if (selectorMatch) {
            actions.push({
              type: "check",
              selector: selectorMatch[1],
              comment,
              lineNumber
            });
          }
        } else if (trimmed.includes("page.uncheck(")) {
          const selectorMatch = trimmed.match(/page\.uncheck\(['"](.*?)['"]\)/);
          if (selectorMatch) {
            actions.push({
              type: "uncheck",
              selector: selectorMatch[1],
              comment,
              lineNumber
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
              lineNumber
            });
          }
        } else if (trimmed.includes("page.waitForTimeout(")) {
          const timeoutMatch = trimmed.match(/page\.waitForTimeout\((\d+)\)/);
          if (timeoutMatch) {
            actions.push({
              type: "waitForTimeout",
              value: parseInt(timeoutMatch[1]),
              comment,
              lineNumber
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
              lineNumber
            });
          }
        } else if (trimmed.includes("page.getByRole(")) {
          const roleMatch = trimmed.match(
            /page\.getByRole\(['"](.*?)['"](?:,\s*\{\s*name:\s*['"](.*?)['"]\s*\})?\)/
          );
          if (roleMatch) {
            const [, role, name] = roleMatch;
            const selector = name ? `role:${role}[name="${name}"]` : `role:${role}`;
            actions.push({
              type: "click",
              selector,
              comment,
              lineNumber
            });
          }
        } else if (trimmed.includes("page.getByLabel(")) {
          const labelMatch = trimmed.match(/page\.getByLabel\(['"](.*?)['"]\)/);
          if (labelMatch) {
            actions.push({
              type: "click",
              selector: `getByLabel:${labelMatch[1]}`,
              comment,
              lineNumber
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
              lineNumber
            });
          }
        } else if (trimmed.includes("page.getByTestId(")) {
          const testIdMatch = trimmed.match(/page\.getByTestId\(['"](.*?)['"]\)/);
          if (testIdMatch) {
            actions.push({
              type: "click",
              selector: `getByTestId:${testIdMatch[1]}`,
              comment,
              lineNumber
            });
          }
        } else if (trimmed.includes("page.getByTitle(")) {
          const titleMatch = trimmed.match(/page\.getByTitle\(['"](.*?)['"]\)/);
          if (titleMatch) {
            actions.push({
              type: "click",
              selector: `getByTitle:${titleMatch[1]}`,
              comment,
              lineNumber
            });
          }
        } else if (trimmed.includes("page.getByAltText(")) {
          const altMatch = trimmed.match(/page\.getByAltText\(['"](.*?)['"]\)/);
          if (altMatch) {
            actions.push({
              type: "click",
              selector: `getByAltText:${altMatch[1]}`,
              comment,
              lineNumber
            });
          }
        } else if (trimmed.includes("expect(") && trimmed.includes("toBeVisible()")) {
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
              lineNumber
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
                lineNumber
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
                  lineNumber
                });
              }
            }
          }
        } else if (trimmed.includes("expect(") && trimmed.includes("toContainText(")) {
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
              lineNumber
            });
          }
        } else if (trimmed.includes("expect(") && trimmed.includes("toBeChecked()")) {
          const expectMatch = trimmed.match(
            /expect\(page\.locator\(['"](.*?)['"]\)\)\.toBeChecked\(\)/
          );
          if (expectMatch) {
            actions.push({
              type: "expect",
              selector: expectMatch[1],
              expectation: "toBeChecked",
              comment,
              lineNumber
            });
          }
        } else if (trimmed.includes("expect(") && trimmed.includes("not.toBeChecked()")) {
          const expectMatch = trimmed.match(
            /expect\(page\.locator\(['"](.*?)['"]\)\)\.not\.toBeChecked\(\)/
          );
          if (expectMatch) {
            actions.push({
              type: "expect",
              selector: expectMatch[1],
              expectation: "not.toBeChecked",
              comment,
              lineNumber
            });
          }
        }
      } catch (error) {
        console.warn(`Failed to parse line ${lineNumber}: ${trimmed}`, error);
      }
    }
    return actions;
  }
  function parseVariableMethodCall(varName, method, args, comment, lineNumber, selector) {
    var _a, _b;
    const actualSelector = selector || `variable:${varName}`;
    switch (method) {
      case "click":
        return { type: "click", selector: actualSelector, comment, lineNumber };
      case "fill":
        const fillValue = ((_a = args.match(/['"](.*?)['"]/)) == null ? void 0 : _a[1]) || "";
        return {
          type: "fill",
          selector: actualSelector,
          value: fillValue,
          comment,
          lineNumber
        };
      case "check":
        return { type: "check", selector: actualSelector, comment, lineNumber };
      case "uncheck":
        return { type: "uncheck", selector: actualSelector, comment, lineNumber };
      case "selectOption":
        const optionValue = ((_b = args.match(/['"](.*?)['"]/)) == null ? void 0 : _b[1]) || "";
        return {
          type: "selectOption",
          selector: actualSelector,
          value: optionValue,
          comment,
          lineNumber
        };
      case "waitFor":
        const stateMatch = args.match(/{\s*state:\s*['"](.*?)['"]\s*}/);
        return {
          type: "waitFor",
          selector: actualSelector,
          options: stateMatch ? { state: stateMatch[1] } : {},
          comment,
          lineNumber
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
          lineNumber
        };
      default:
        return null;
    }
  }
  function parseLocatorCall(locatorCall) {
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
  function parseChainedCall(baseSelector, chainCall) {
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
  function extractSelectorFromArg(selectorArg) {
    return selectorArg.trim().replace(/^['"]|['"]$/g, "");
  }
  function parseDirectAction(method, args, comment, lineNumber, selector) {
    var _a, _b;
    switch (method) {
      case "click":
        return { type: "click", selector, comment, lineNumber };
      case "fill":
        const fillValue = ((_a = args.match(/['"](.*?)['"]/)) == null ? void 0 : _a[1]) || "";
        return { type: "fill", selector, value: fillValue, comment, lineNumber };
      case "check":
        return { type: "check", selector, comment, lineNumber };
      case "uncheck":
        return { type: "uncheck", selector, comment, lineNumber };
      case "selectOption":
        const optionValue = ((_b = args.match(/['"](.*?)['"]/)) == null ? void 0 : _b[1]) || "";
        return {
          type: "selectOption",
          selector,
          value: optionValue,
          comment,
          lineNumber
        };
      case "waitFor":
        const stateMatch = args.match(/{\s*state:\s*['"](.*?)['"]\s*}/);
        return {
          type: "waitFor",
          selector,
          options: stateMatch ? { state: stateMatch[1] } : {},
          comment,
          lineNumber
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
  function parseExpectStatement(expectation, comment, lineNumber, selector) {
    if (expectation.includes("toBeVisible()")) {
      return {
        type: "expect",
        selector,
        expectation: "toBeVisible",
        comment,
        lineNumber
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
        lineNumber
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
        lineNumber
      };
    }
    if (expectation.includes("toBeChecked()")) {
      return {
        type: "expect",
        selector,
        expectation: "toBeChecked",
        comment,
        lineNumber
      };
    }
    if (expectation.includes("not.toBeChecked()")) {
      return {
        type: "expect",
        selector,
        expectation: "not.toBeChecked",
        comment,
        lineNumber
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
        lineNumber
      };
    }
    return null;
  }
  function getRoleSelector(role) {
    const roleMap = {
      button: 'button, [role="button"], input[type="button"], input[type="submit"]',
      heading: 'h1, h2, h3, h4, h5, h6, [role="heading"]',
      link: 'a, [role="link"]',
      textbox: 'input[type="text"], input[type="email"], input[type="password"], textarea, [role="textbox"]',
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
      tabpanel: '[role="tabpanel"]'
    };
    return roleMap[role] || `[role="${role}"]`;
  }
  async function executePlaywrightAction(action) {
    var _a;
    try {
      switch (action.type) {
        case "goto":
          if (action.value && typeof action.value === "string") {
            window.parent.postMessage(
              {
                type: "staktrak-iframe-navigate",
                url: action.value
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
              const htmlElement = element;
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
          if (action.selector && action.value !== void 0) {
            const element = await waitForElement(action.selector);
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
            const element = await waitForElement(
              action.selector
            );
            if (element && (element.type === "checkbox" || element.type === "radio")) {
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
            const element = await waitForElement(
              action.selector
            );
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
          if (action.selector && action.value !== void 0) {
            const element = await waitForElement(
              action.selector
            );
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
          const shortDelay = Math.min(action.value, 500);
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
            if (((_a = action.options) == null ? void 0 : _a.state) === "visible") {
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
              const htmlElement = element;
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
              }, 1e3);
            } else {
              throw new Error(`Element not found for hover: ${action.selector}`);
            }
          }
          break;
        case "focus":
          if (action.selector) {
            const element = await waitForElement(
              action.selector
            );
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
            const element = await waitForElement(
              action.selector
            );
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
                inline: "center"
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
  async function waitForElements(selector, timeout = 5e3) {
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
  function findElements(selector) {
    return findElementsInContext(selector, document);
  }
  function findElementsInContext(selector, searchContext) {
    var _a, _b, _c, _d, _e, _f, _g, _h, _i, _j, _k, _l;
    if (selector.includes(" >> ")) {
      const parts = selector.split(" >> ");
      let elements = findElementsInContext(parts[0], searchContext);
      for (let i = 1; i < parts.length; i++) {
        const newElements = [];
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
          var _a2;
          const elementText = ((_a2 = el.textContent) == null ? void 0 : _a2.trim()) || "";
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
          var _a2;
          const elementText = ((_a2 = el.textContent) == null ? void 0 : _a2.trim()) || "";
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
      const matches = [];
      for (const el of Array.from(allElements)) {
        const elementText = ((_a = el.textContent) == null ? void 0 : _a.trim()) || "";
        const elementOwnText = el.childNodes.length === 1 && el.childNodes[0].nodeType === Node.TEXT_NODE ? ((_b = el.childNodes[0].textContent) == null ? void 0 : _b.trim()) || "" : elementText;
        const matchesText = exact ? elementOwnText === text.trim() || elementText === text.trim() : elementOwnText.includes(text.trim()) || elementText.includes(text.trim());
        if (matchesText) {
          matches.push(el);
          el.__stakTrakMatchedText = text.trim();
        }
      }
      return matches.sort(
        (a, b) => {
          var _a2, _b2;
          return (((_a2 = a.textContent) == null ? void 0 : _a2.length) || 0) - (((_b2 = b.textContent) == null ? void 0 : _b2.length) || 0);
        }
      );
    } else if (selector.startsWith("getByText-regex:")) {
      const regexPattern = selector.substring(16);
      const regexMatch = regexPattern.match(/^\/(.+?)\/([gimuy]*)$/);
      if (regexMatch) {
        const [, pattern, flags] = regexMatch;
        const regex = new RegExp(pattern, flags);
        const allElements = searchContext.querySelectorAll("*");
        const matches = [];
        for (const el of Array.from(allElements)) {
          const elementText = ((_c = el.textContent) == null ? void 0 : _c.trim()) || "";
          if (regex.test(elementText)) {
            matches.push(el);
            el.__stakTrakMatchedText = elementText;
          }
        }
        return matches.sort(
          (a, b) => {
            var _a2, _b2;
            return (((_a2 = a.textContent) == null ? void 0 : _a2.length) || 0) - (((_b2 = b.textContent) == null ? void 0 : _b2.length) || 0);
          }
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
        const matches = [];
        for (const el of Array.from(roleElements)) {
          const elementText = ((_d = el.textContent) == null ? void 0 : _d.trim()) || "";
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
        const matches = [];
        for (const el of Array.from(roleElements)) {
          if (name) {
            const elementText = ((_e = el.textContent) == null ? void 0 : _e.trim()) || "";
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
      const match = selector.match(/^(.+?):has-text\("(.+?)"\)$/) || selector.match(/^(.+?):has-text\((.+?)\)$/);
      if (match) {
        const [, baseSelector, text] = match;
        const elements = searchContext.querySelectorAll(baseSelector);
        const matches = [];
        for (const el of Array.from(elements)) {
          const elementText = ((_f = el.textContent) == null ? void 0 : _f.trim()) || "";
          const elementOwnText = el.childNodes.length === 1 && el.childNodes[0].nodeType === Node.TEXT_NODE ? ((_g = el.childNodes[0].textContent) == null ? void 0 : _g.trim()) || "" : elementText;
          if (elementOwnText.includes(text.trim()) || elementText.includes(text.trim())) {
            matches.push(el);
            el.__stakTrakMatchedText = text.trim();
          }
        }
        return matches.sort(
          (a, b) => {
            var _a2, _b2;
            return (((_a2 = a.textContent) == null ? void 0 : _a2.length) || 0) - (((_b2 = b.textContent) == null ? void 0 : _b2.length) || 0);
          }
        );
      }
    } else if (selector.startsWith("getByLabel:")) {
      const labelText = selector.substring(11);
      const labels = searchContext.querySelectorAll("label");
      const matches = [];
      for (const label of Array.from(labels)) {
        if ((_h = label.textContent) == null ? void 0 : _h.includes(labelText)) {
          const forAttr = label.getAttribute("for");
          let element = null;
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
        const matches = [];
        for (const el of Array.from(allElements)) {
          const elementText = ((_i = el.textContent) == null ? void 0 : _i.trim()) || "";
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
        const matches = [];
        for (const el of Array.from(allElements)) {
          const elementOwnText = el.childNodes.length === 1 && el.childNodes[0].nodeType === Node.TEXT_NODE ? ((_j = el.childNodes[0].textContent) == null ? void 0 : _j.trim()) || "" : "";
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
        const matches = [];
        for (const el of Array.from(allElements)) {
          const elementText = ((_k = el.textContent) == null ? void 0 : _k.trim()) || "";
          if (elementText.includes(text.trim())) {
            matches.push(el);
          }
        }
        return matches.sort(
          (a, b) => {
            var _a2, _b2;
            return (((_a2 = a.textContent) == null ? void 0 : _a2.length) || 0) - (((_b2 = b.textContent) == null ? void 0 : _b2.length) || 0);
          }
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
      const matches = [];
      for (let i = 0; i < result.snapshotLength; i++) {
        const node = result.snapshotItem(i);
        if (node && node.nodeType === Node.ELEMENT_NODE) {
          matches.push(node);
        }
      }
      return matches;
    } else if (selector.includes(",")) {
      const selectors = selector.split(",").map((s) => s.trim());
      const allMatches = [];
      for (const sel of selectors) {
        const matches = findElementsInContext(sel, searchContext);
        allMatches.push(...matches);
      }
      return Array.from(new Set(allMatches));
    } else if (selector.includes(":visible")) {
      const baseSelector = selector.replace(":visible", "");
      const elements = baseSelector ? Array.from(searchContext.querySelectorAll(baseSelector)) : [];
      return elements.filter((el) => {
        const style = window.getComputedStyle(el);
        return style.display !== "none" && style.visibility !== "hidden" && style.opacity !== "0" && el.getBoundingClientRect().width > 0 && el.getBoundingClientRect().height > 0;
      });
    } else if (selector.includes(":has(")) {
      const match = selector.match(/^(.+?):has\((.+?)\)$/);
      if (match) {
        const [, baseSelector, hasSelector] = match;
        const baseElements = baseSelector ? Array.from(searchContext.querySelectorAll(baseSelector)) : [];
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
      const match = selector.match(/^(.+?):text\("([^"]+)"\)$/) || selector.match(/^(.+?):text\(([^)]+)\)$/);
      if (match) {
        const [, baseSelector, text] = match;
        const baseElements = baseSelector ? Array.from(searchContext.querySelectorAll(baseSelector)) : Array.from(searchContext.querySelectorAll("*"));
        let shortestMatch = null;
        let shortestLength = Infinity;
        for (const el of baseElements) {
          const elementText = ((_l = el.textContent) == null ? void 0 : _l.trim()) || "";
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
      const match = selector.match(/^(.+?):text-is\("([^"]+)"\)$/) || selector.match(/^(.+?):text-is\(([^)]+)\)$/);
      if (match) {
        const [, baseSelector, text] = match;
        const baseElements = baseSelector ? Array.from(searchContext.querySelectorAll(baseSelector)) : Array.from(searchContext.querySelectorAll("*"));
        return baseElements.filter((el) => {
          var _a2;
          const elementOwnText = el.childNodes.length === 1 && el.childNodes[0].nodeType === Node.TEXT_NODE ? ((_a2 = el.childNodes[0].textContent) == null ? void 0 : _a2.trim()) || "" : "";
          return elementOwnText === text.trim();
        });
      }
    } else if (selector.includes(":right-of(") || selector.includes(":left-of(") || selector.includes(":above(") || selector.includes(":below(") || selector.includes(":near(")) {
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
  async function waitForElement(selector, matchedText) {
    const startTime = Date.now();
    const timeout = 5e3;
    while (Date.now() - startTime < timeout) {
      try {
        const elements = findElements(selector);
        if (elements.length > 0) {
          const element = elements[0];
          if (matchedText) {
            element.__stakTrakMatchedText = matchedText;
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
  function ensureStylesInDocument(doc) {
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
  function highlightElement(element, matchedText) {
    try {
      ensureStylesInDocument(document);
      const textToHighlight = matchedText || element.__stakTrakMatchedText;
      if (textToHighlight) {
        highlightTextInElement(element, textToHighlight);
      }
    } catch (error) {
      console.warn("Error highlighting element:", error);
    }
  }
  function highlightTextInElement(element, textToHighlight) {
    try {
      let wrapTextNodes2 = function(node) {
        var _a;
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
        } else if (node.nodeType === Node.ELEMENT_NODE && !((_a = node.classList) == null ? void 0 : _a.contains("staktrak-text-highlight"))) {
          const children = Array.from(node.childNodes);
          children.forEach((child) => wrapTextNodes2(child));
        }
      };
      var wrapTextNodes = wrapTextNodes2;
      ensureStylesInDocument(document);
      wrapTextNodes2(element);
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
      }, 3e3);
    } catch (error) {
      console.warn("Error highlighting text:", error);
    }
  }
  async function verifyExpectation(action) {
    var _a, _b;
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
        if (!textElement || !((_a = textElement.textContent) == null ? void 0 : _a.includes(String(action.value || "")))) {
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
        if (!exactTextElement || ((_b = exactTextElement.textContent) == null ? void 0 : _b.trim()) !== String(action.value || "")) {
          throw new Error(
            `Element does not have exact text "${action.value}": ${action.selector}`
          );
        }
        break;
      case "toBeChecked":
        const checkedElement = await waitForElement(
          action.selector
        );
        if (!checkedElement || !checkedElement.checked) {
          throw new Error(`Element is not checked: ${action.selector}`);
        }
        break;
      case "not.toBeChecked":
        const uncheckedElement = await waitForElement(
          action.selector
        );
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
  function isElementVisible(element) {
    const style = window.getComputedStyle(element);
    return style.display !== "none" && style.visibility !== "hidden" && style.opacity !== "0" && element.getBoundingClientRect().width > 0 && element.getBoundingClientRect().height > 0;
  }
  function getActionDescription(action) {
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
  function startPlaywrightReplay(testCode) {
    try {
      const actions = parsePlaywrightTest(testCode);
      if (actions.length === 0) {
        throw new Error("No valid actions found in test code");
      }
      playwrightReplayRef.current = {
        actions,
        status: "playing" /* PLAYING */,
        currentActionIndex: 0,
        testCode,
        errors: [],
        timeouts: []
      };
      window.parent.postMessage(
        {
          type: "staktrak-playwright-replay-started",
          totalActions: actions.length,
          actions
        },
        "*"
      );
      executeNextPlaywrightAction();
    } catch (error) {
      window.parent.postMessage(
        {
          type: "staktrak-playwright-replay-error",
          error: error instanceof Error ? error.message : "Unknown error"
        },
        "*"
      );
    }
  }
  async function executeNextPlaywrightAction() {
    const state = playwrightReplayRef.current;
    if (!state || state.status !== "playing" /* PLAYING */) {
      return;
    }
    if (state.currentActionIndex >= state.actions.length) {
      state.status = "completed" /* COMPLETED */;
      window.parent.postMessage(
        {
          type: "staktrak-playwright-replay-completed"
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
          currentAction: __spreadProps(__spreadValues({}, action), {
            description: getActionDescription(action)
          })
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
          action
        },
        "*"
      );
      executeNextPlaywrightAction();
    }
  }
  function pausePlaywrightReplay() {
    const state = playwrightReplayRef.current;
    if (state) {
      state.status = "paused" /* PAUSED */;
      state.timeouts.forEach((id) => clearTimeout(id));
      state.timeouts = [];
      window.parent.postMessage(
        { type: "staktrak-playwright-replay-paused" },
        "*"
      );
    }
  }
  function resumePlaywrightReplay() {
    const state = playwrightReplayRef.current;
    if (state && state.status === "paused" /* PAUSED */) {
      state.status = "playing" /* PLAYING */;
      executeNextPlaywrightAction();
      window.parent.postMessage(
        { type: "staktrak-playwright-replay-resumed" },
        "*"
      );
    }
  }
  function stopPlaywrightReplay() {
    const state = playwrightReplayRef.current;
    if (state) {
      state.status = "idle" /* IDLE */;
      state.timeouts.forEach((id) => clearTimeout(id));
      state.timeouts = [];
      window.parent.postMessage(
        { type: "staktrak-playwright-replay-stopped" },
        "*"
      );
    }
  }
  function getPlaywrightReplayState() {
    const state = playwrightReplayRef.current;
    if (!state) return null;
    return {
      actions: state.actions,
      status: state.status,
      currentActionIndex: state.currentActionIndex,
      testCode: state.testCode,
      errors: state.errors
    };
  }
  function initPlaywrightReplay() {
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
              state: currentState
            },
            "*"
          );
          break;
      }
    });
  }

  // src/index.ts
  var defaultConfig = {
    userInfo: true,
    clicks: true,
    mouseMovement: false,
    mouseMovementInterval: 1,
    mouseScroll: true,
    timeCount: true,
    clearAfterProcess: true,
    windowResize: true,
    visibilitychange: true,
    keyboardActivity: true,
    formInteractions: true,
    touchEvents: true,
    audioVideoInteraction: true,
    customEventRegistration: true,
    inputDebounceDelay: 2e3,
    multiClickInterval: 300,
    filterAssertionClicks: true,
    processData: (results) => console.log(results)
  };
  var UserBehaviorTracker = class {
    constructor() {
      this.config = defaultConfig;
      this.results = this.createEmptyResults();
      this.memory = {
        mousePosition: [0, 0, 0],
        inputDebounceTimers: {},
        selectionMode: false,
        assertionDebounceTimer: null,
        assertions: [],
        mutationObserver: null,
        mouseInterval: null,
        listeners: [],
        alwaysListeners: []
      };
      this.isRunning = false;
    }
    createEmptyResults() {
      return {
        pageNavigation: [],
        clicks: { clickCount: 0, clickDetails: [] },
        keyboardActivities: [],
        mouseMovement: [],
        mouseScroll: [],
        inputChanges: [],
        focusChanges: [],
        visibilitychanges: [],
        windowSizes: [],
        formElementChanges: [],
        touchEvents: [],
        audioVideoInteractions: [],
        assertions: []
      };
    }
    makeConfig(newConfig) {
      this.config = __spreadValues(__spreadValues({}, this.config), newConfig);
      return this;
    }
    listen() {
      this.setupMessageHandling();
      this.setupPageNavigation();
      window.parent.postMessage({ type: "staktrak-setup" }, "*");
      this.checkDebugInfo();
    }
    start() {
      this.cleanup();
      this.resetResults();
      this.setupEventListeners();
      this.isRunning = true;
      return this;
    }
    resetResults() {
      this.memory.assertions = [];
      this.results = this.createEmptyResults();
      if (this.config.userInfo) {
        this.results.userInfo = {
          url: document.URL,
          userAgent: navigator.userAgent,
          platform: navigator.platform,
          windowSize: [window.innerWidth, window.innerHeight]
        };
      }
      if (this.config.timeCount) {
        this.results.time = {
          startedAt: getTimeStamp(),
          completedAt: 0,
          totalSeconds: 0
        };
      }
    }
    cleanup() {
      this.memory.listeners.forEach((cleanup) => cleanup());
      this.memory.listeners = [];
      if (this.memory.mutationObserver) {
        this.memory.mutationObserver.disconnect();
        this.memory.mutationObserver = null;
      }
      if (this.memory.mouseInterval) {
        clearInterval(this.memory.mouseInterval);
        this.memory.mouseInterval = null;
      }
      Object.values(this.memory.inputDebounceTimers).forEach(
        (timer) => clearTimeout(timer)
      );
      this.memory.inputDebounceTimers = {};
      if (this.memory.assertionDebounceTimer) {
        clearTimeout(this.memory.assertionDebounceTimer);
        this.memory.assertionDebounceTimer = null;
      }
      if (this.memory.selectionMode) {
        this.setSelectionMode(false);
      }
    }
    setupEventListeners() {
      if (this.config.clicks) {
        const clickHandler = (e) => {
          this.results.clicks.clickCount++;
          const clickDetail = createClickDetail(e);
          this.results.clicks.clickDetails.push(clickDetail);
          const target = e.target;
          if (target.tagName === "INPUT" && (target.type === "checkbox" || target.type === "radio")) {
            this.results.formElementChanges.push({
              elementSelector: clickDetail.selectors.primary,
              type: target.type,
              checked: target.checked,
              value: target.value,
              timestamp: getTimeStamp()
            });
          }
        };
        document.addEventListener("click", clickHandler);
        this.memory.listeners.push(
          () => document.removeEventListener("click", clickHandler)
        );
      }
      if (this.config.mouseScroll) {
        const scrollHandler = () => {
          this.results.mouseScroll.push([
            window.scrollX,
            window.scrollY,
            getTimeStamp()
          ]);
        };
        window.addEventListener("scroll", scrollHandler);
        this.memory.listeners.push(
          () => window.removeEventListener("scroll", scrollHandler)
        );
      }
      if (this.config.mouseMovement) {
        const mouseMoveHandler = (e) => {
          this.memory.mousePosition = [e.clientX, e.clientY, getTimeStamp()];
        };
        document.addEventListener("mousemove", mouseMoveHandler);
        this.memory.mouseInterval = setInterval(() => {
          if (this.memory.mousePosition[2] + 500 > getTimeStamp()) {
            this.results.mouseMovement.push(this.memory.mousePosition);
          }
        }, this.config.mouseMovementInterval * 1e3);
        this.memory.listeners.push(() => {
          document.removeEventListener("mousemove", mouseMoveHandler);
          if (this.memory.mouseInterval) {
            clearInterval(this.memory.mouseInterval);
            this.memory.mouseInterval = null;
          }
        });
      }
      if (this.config.windowResize) {
        const resizeHandler = () => {
          this.results.windowSizes.push([
            window.innerWidth,
            window.innerHeight,
            getTimeStamp()
          ]);
        };
        window.addEventListener("resize", resizeHandler);
        this.memory.listeners.push(
          () => window.removeEventListener("resize", resizeHandler)
        );
      }
      if (this.config.visibilitychange) {
        const visibilityHandler = () => {
          this.results.visibilitychanges.push([
            document.visibilityState,
            getTimeStamp()
          ]);
        };
        document.addEventListener("visibilitychange", visibilityHandler);
        this.memory.listeners.push(
          () => document.removeEventListener("visibilitychange", visibilityHandler)
        );
      }
      if (this.config.keyboardActivity) {
        const keyHandler = (e) => {
          if (!isInputOrTextarea(e.target)) {
            this.results.keyboardActivities.push([e.key, getTimeStamp()]);
          }
        };
        document.addEventListener("keypress", keyHandler);
        this.memory.listeners.push(
          () => document.removeEventListener("keypress", keyHandler)
        );
      }
      if (this.config.formInteractions) {
        this.setupFormInteractions();
      }
      if (this.config.touchEvents) {
        const touchHandler = (e) => {
          if (e.touches.length > 0) {
            const touch = e.touches[0];
            this.results.touchEvents.push({
              type: "touchstart",
              x: touch.clientX,
              y: touch.clientY,
              timestamp: getTimeStamp()
            });
          }
        };
        document.addEventListener("touchstart", touchHandler);
        this.memory.listeners.push(
          () => document.removeEventListener("touchstart", touchHandler)
        );
      }
    }
    setupFormInteractions() {
      const attachFormListeners = (element) => {
        const htmlEl = element;
        if (htmlEl.tagName === "INPUT" || htmlEl.tagName === "SELECT" || htmlEl.tagName === "TEXTAREA") {
          const inputEl = htmlEl;
          if (inputEl.type === "checkbox" || inputEl.type === "radio" || htmlEl.tagName === "SELECT") {
            const changeHandler = () => {
              const selector = getElementSelector(htmlEl);
              if (htmlEl.tagName === "SELECT") {
                const selectEl = htmlEl;
                const selectedOption = selectEl.options[selectEl.selectedIndex];
                this.results.formElementChanges.push({
                  elementSelector: selector,
                  type: "select",
                  value: selectEl.value,
                  text: (selectedOption == null ? void 0 : selectedOption.text) || "",
                  timestamp: getTimeStamp()
                });
              } else {
                this.results.formElementChanges.push({
                  elementSelector: selector,
                  type: inputEl.type,
                  checked: inputEl.checked,
                  value: inputEl.value,
                  timestamp: getTimeStamp()
                });
              }
            };
            htmlEl.addEventListener("change", changeHandler);
          } else {
            const inputHandler = () => {
              const selector = getElementSelector(htmlEl);
              const elementId = inputEl.id || selector;
              if (this.memory.inputDebounceTimers[elementId]) {
                clearTimeout(this.memory.inputDebounceTimers[elementId]);
              }
              this.memory.inputDebounceTimers[elementId] = setTimeout(() => {
                this.results.inputChanges.push({
                  elementSelector: selector,
                  value: inputEl.value,
                  timestamp: getTimeStamp(),
                  action: "complete"
                });
                delete this.memory.inputDebounceTimers[elementId];
              }, this.config.inputDebounceDelay);
              this.results.inputChanges.push({
                elementSelector: selector,
                value: inputEl.value,
                timestamp: getTimeStamp(),
                action: "intermediate"
              });
            };
            const focusHandler = (e) => {
              const selector = getElementSelector(htmlEl);
              this.results.focusChanges.push({
                elementSelector: selector,
                type: e.type,
                timestamp: getTimeStamp()
              });
              if (e.type === "blur") {
                const elementId = inputEl.id || selector;
                if (this.memory.inputDebounceTimers[elementId]) {
                  clearTimeout(this.memory.inputDebounceTimers[elementId]);
                  delete this.memory.inputDebounceTimers[elementId];
                }
                this.results.inputChanges.push({
                  elementSelector: selector,
                  value: inputEl.value,
                  timestamp: getTimeStamp(),
                  action: "complete"
                });
              }
            };
            htmlEl.addEventListener("input", inputHandler);
            htmlEl.addEventListener("focus", focusHandler);
            htmlEl.addEventListener("blur", focusHandler);
          }
        }
      };
      document.querySelectorAll("input, select, textarea").forEach(attachFormListeners);
      this.memory.mutationObserver = new MutationObserver((mutations) => {
        mutations.forEach((mutation) => {
          mutation.addedNodes.forEach((node) => {
            if (node.nodeType === 1) {
              attachFormListeners(node);
              node.querySelectorAll("input, select, textarea").forEach(attachFormListeners);
            }
          });
        });
      });
      this.memory.mutationObserver.observe(document.body, {
        childList: true,
        subtree: true
      });
      this.memory.listeners.push(() => {
        if (this.memory.mutationObserver) {
          this.memory.mutationObserver.disconnect();
          this.memory.mutationObserver = null;
        }
      });
    }
    setupPageNavigation() {
      const originalPushState = history.pushState;
      const originalReplaceState = history.replaceState;
      const recordStateChange = (type) => {
        this.results.pageNavigation.push({
          type,
          url: document.URL,
          timestamp: getTimeStamp()
        });
        window.parent.postMessage(
          { type: "staktrak-page-navigation", data: document.URL },
          "*"
        );
      };
      history.pushState = (...args) => {
        originalPushState.apply(history, args);
        recordStateChange("pushState");
      };
      history.replaceState = (...args) => {
        originalReplaceState.apply(history, args);
        recordStateChange("replaceState");
      };
      const popstateHandler = () => {
        recordStateChange("popstate");
      };
      window.addEventListener("popstate", popstateHandler);
      this.memory.alwaysListeners.push(
        () => window.removeEventListener("popstate", popstateHandler)
      );
    }
    setupMessageHandling() {
      if (this.memory.alwaysListeners.length > 0) return;
      const messageHandler = (event) => {
        var _a;
        if (!((_a = event.data) == null ? void 0 : _a.type)) return;
        switch (event.data.type) {
          case "staktrak-start":
            this.resetResults();
            this.start();
            break;
          case "staktrak-stop":
            this.stop();
            break;
          case "staktrak-enable-selection":
            this.setSelectionMode(true);
            break;
          case "staktrak-disable-selection":
            this.setSelectionMode(false);
            break;
          case "staktrak-add-assertion":
            if (event.data.assertion) {
              this.memory.assertions.push({
                type: event.data.assertion.type || "hasText",
                selector: event.data.assertion.selector,
                value: event.data.assertion.value || "",
                timestamp: getTimeStamp()
              });
            }
            break;
          case "staktrak-debug-request":
            debugMsg({
              messageId: event.data.messageId,
              coordinates: event.data.coordinates
            });
        }
      };
      window.addEventListener("message", messageHandler);
      this.memory.alwaysListeners.push(
        () => window.removeEventListener("message", messageHandler)
      );
    }
    checkDebugInfo() {
      setTimeout(() => {
        if (isReactDevModeActive()) {
          window.parent.postMessage({ type: "staktrak-debug-init" }, "*");
        }
      }, 1500);
    }
    setSelectionMode(isActive) {
      var _a;
      this.memory.selectionMode = isActive;
      if (isActive) {
        document.body.classList.add("staktrak-selection-active");
        const mouseUpHandler = () => {
          const selection = window.getSelection();
          if (selection == null ? void 0 : selection.toString().trim()) {
            const text = selection.toString();
            let container = selection.getRangeAt(0).commonAncestorContainer;
            if (container.nodeType === 3)
              container = container.parentNode;
            if (this.memory.assertionDebounceTimer)
              clearTimeout(this.memory.assertionDebounceTimer);
            this.memory.assertionDebounceTimer = setTimeout(() => {
              const selector = getElementSelector(container);
              const assertion = {
                type: "hasText",
                selector,
                value: text,
                timestamp: getTimeStamp()
              };
              this.memory.assertions.push(assertion);
              window.parent.postMessage(
                { type: "staktrak-selection", text, selector },
                "*"
              );
            }, 300);
          }
        };
        document.addEventListener("mouseup", mouseUpHandler);
        this.memory.listeners.push(
          () => document.removeEventListener("mouseup", mouseUpHandler)
        );
      } else {
        document.body.classList.remove("staktrak-selection-active");
        (_a = window.getSelection()) == null ? void 0 : _a.removeAllRanges();
      }
      window.parent.postMessage(
        {
          type: `staktrak-selection-mode-${isActive ? "started" : "ended"}`
        },
        "*"
      );
    }
    processResults() {
      if (this.config.timeCount && this.results.time) {
        this.results.time.completedAt = getTimeStamp();
        this.results.time.totalSeconds = (this.results.time.completedAt - this.results.time.startedAt) / 1e3;
      }
      this.results.clicks.clickDetails = filterClickDetails(
        this.results.clicks.clickDetails,
        this.memory.assertions,
        this.config
      );
      this.results.assertions = this.memory.assertions;
      window.parent.postMessage(
        { type: "staktrak-results", data: this.results },
        "*"
      );
      this.config.processData(this.results);
      if (this.config.clearAfterProcess) {
        this.resetResults();
      }
    }
    stop() {
      if (!this.isRunning) {
        console.log("StakTrak is not running");
        return this;
      }
      this.cleanup();
      this.processResults();
      this.isRunning = false;
      return this;
    }
    result() {
      return this.results;
    }
    showConfig() {
      return this.config;
    }
    addAssertion(type, selector, value = "") {
      this.memory.assertions.push({
        type,
        selector,
        value,
        timestamp: getTimeStamp()
      });
    }
  };
  var userBehaviour = new UserBehaviorTracker();
  var initializeStakTrak = () => {
    userBehaviour.makeConfig({
      processData: (results) => console.log("StakTrak recording processed:", results)
    }).listen();
    initPlaywrightReplay();
  };
  document.readyState === "loading" ? document.addEventListener("DOMContentLoaded", initializeStakTrak) : initializeStakTrak();
  var index_default = userBehaviour;
  return __toCommonJS(index_exports);
})();
