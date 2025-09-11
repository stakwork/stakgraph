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
  var getElementRole = (element) => {
    const explicit = element.getAttribute("role");
    if (explicit) return explicit;
    const tag = element.tagName.toLowerCase();
    if (tag === "button") return "button";
    if (tag === "a" && element.hasAttribute("href")) return "link";
    if (tag === "input") {
      const type = element.getAttribute("type");
      if (["button", "submit", "reset"].includes(type || "text")) return "button";
      return "textbox";
    }
    if (tag === "nav") return "navigation";
    if (tag === "main") return "main";
    if (tag === "header") return "banner";
    if (tag === "footer") return "contentinfo";
    if (tag === "aside") return "complementary";
    if (tag === "section") return "region";
    return null;
  };
  var getEnhancedElementText = (element) => {
    var _a;
    const ariaLabel = element.getAttribute("aria-label");
    if (ariaLabel) return ariaLabel;
    const resolvedLabel = resolveAriaLabelledBy(element);
    if (resolvedLabel) return resolvedLabel;
    const tag = element.tagName.toLowerCase();
    if (tag === "button" || tag === "a" && element.hasAttribute("href")) {
      const text = (_a = element.textContent) == null ? void 0 : _a.trim();
      if (text && text.length > 0 && text.length < 100) {
        return text;
      }
    }
    if (tag === "input") {
      const input = element;
      return input.value || input.placeholder || input.getAttribute("title") || null;
    }
    return element.getAttribute("title") || null;
  };
  var getSemanticParent = (element) => {
    const semanticTags = ["header", "nav", "main", "footer", "aside", "section", "article", "form", "dialog"];
    let parent = element.parentElement;
    while (parent) {
      const tag = parent.tagName.toLowerCase();
      if (semanticTags.includes(tag)) {
        return parent;
      }
      const role = parent.getAttribute("role");
      if (role && ["navigation", "banner", "main", "contentinfo", "complementary", "form", "search"].includes(role)) {
        return parent;
      }
      parent = parent.parentElement;
    }
    return null;
  };
  var detectIconContent = (element) => {
    var _a;
    const svg = element.querySelector("svg");
    if (svg) {
      if (svg.getAttribute("data-icon")) {
        return { type: "svg", selector: `[data-icon="${svg.getAttribute("data-icon")}"]` };
      }
      if (svg.classList.length > 0) {
        const iconClass = Array.from(svg.classList).find((cls) => cls.includes("icon"));
        if (iconClass) {
          return { type: "svg", selector: `.${iconClass}` };
        }
      }
      return { type: "svg", selector: "svg" };
    }
    const iconElement = element.querySelector('[class*="icon"], [class*="fa-"], [class*="material-icons"]');
    if (iconElement) {
      const iconClasses = Array.from(iconElement.classList).filter(
        (cls) => cls.includes("icon") || cls.includes("fa-") || cls.includes("material")
      );
      if (iconClasses.length > 0) {
        return { type: "icon-font", selector: `.${iconClasses[0]}` };
      }
    }
    const text = (_a = element.textContent) == null ? void 0 : _a.trim();
    if (text && text.length <= 2 && /[\u{1F300}-\u{1F9FF}]|[\u{2600}-\u{26FF}]|[\u{2700}-\u{27BF}]/u.test(text)) {
      return { type: "emoji", selector: `text="${text}"` };
    }
    return null;
  };
  var resolveAriaLabelledBy = (element) => {
    var _a;
    const labelledBy = element.getAttribute("aria-labelledby");
    if (!labelledBy) return null;
    const ids = labelledBy.split(" ").filter((id) => id.trim());
    const texts = [];
    for (const id of ids) {
      const referencedEl = findElementById(element.ownerDocument || document, id);
      if (referencedEl) {
        const text = (_a = referencedEl.textContent) == null ? void 0 : _a.trim();
        if (text) texts.push(text);
      }
    }
    return texts.length > 0 ? texts.join(" ") : null;
  };
  var findElementById = (doc, id) => {
    if (typeof doc.getElementById === "function") {
      return doc.getElementById(id);
    }
    return doc.querySelector(`#${CSS.escape(id)}`);
  };
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
        role: getElementRole(htmlEl) || void 0
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
        role: getElementRole(htmlEl) || void 0
      };
    }
    const text = getEnhancedElementText(htmlEl);
    const role = getElementRole(htmlEl);
    if (text && (tagName === "button" || tagName === "a" || role === "button")) {
      const textSelector = generateTextBasedSelector(element, text);
      if (textSelector) {
        fallbacks.push(textSelector);
      }
    }
    const ariaLabel = htmlEl.getAttribute("aria-label");
    if (ariaLabel) {
      fallbacks.push(`[aria-label="${ariaLabel}"]`);
    }
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
      text: text || void 0,
      ariaLabel: ariaLabel || void 0,
      title: htmlEl.getAttribute("title") || void 0,
      role: role || void 0,
      tagName,
      xpath
    };
  };
  var getElementText = (element) => {
    const htmlEl = element;
    return getEnhancedElementText(htmlEl) || void 0;
  };
  var generateTextBasedSelector = (element, text) => {
    const tagName = element.tagName.toLowerCase();
    const cleanText = text.replace(/"/g, '\\"').trim();
    if (cleanText.length === 0 || cleanText.length > 50) return null;
    if (tagName === "button" || tagName === "a" || getElementRole(element) === "button") {
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
      "aria-labelledby",
      "aria-expanded",
      "aria-haspopup",
      "title",
      "placeholder",
      "value"
    ];
    importantAttrs.forEach((attr) => {
      const value = htmlEl.getAttribute(attr);
      if (value) attrs[attr] = value;
    });
    const semanticParent = getSemanticParent(htmlEl);
    if (semanticParent) {
      attrs.semanticParent = semanticParent.tagName.toLowerCase();
    }
    const iconInfo = detectIconContent(htmlEl);
    if (iconInfo) {
      attrs.iconContent = iconInfo.selector;
    }
    const resolvedLabel = resolveAriaLabelledBy(htmlEl);
    if (resolvedLabel) {
      attrs.resolvedAriaLabel = resolvedLabel;
    }
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

  // src/playwright-replay/parser.ts
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

  // src/playwright-replay/executor.ts
  async function executePlaywrightAction(action) {
    var _a, _b, _c, _d;
    window.parent.postMessage({
      type: "staktrak-debug-click",
      message: `===>>> \u{1F680} Executing action: ${action.type}`,
      data: {
        actionType: action.type,
        selector: action.selector,
        value: action.value
      }
    }, "*");
    try {
      switch (action.type) {
        case "goto" /* GOTO */:
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
        case "setViewportSize" /* SET_VIEWPORT_SIZE */:
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
        case "waitForLoadState" /* WAIT_FOR_LOAD_STATE */:
          break;
        case "waitForSelector" /* WAIT_FOR_SELECTOR */:
          if (action.selector) {
            await waitForElement(action.selector);
          }
          break;
        case "click" /* CLICK */:
          if (action.selector) {
            console.log(`===>>> \u{1F4CD} CLICK case reached for selector:`, action.selector);
            const element = await waitForElement(action.selector);
            if (element) {
              console.log(`===>>> \u2705 Element found!`);
              const htmlElement = element;
              window.parent.postMessage({
                type: "staktrak-debug-click",
                message: "===>>> \u{1F3AF} Clicking element",
                data: {
                  selector: action.selector,
                  tagName: element.tagName,
                  textContent: (_a = element.textContent) == null ? void 0 : _a.trim(),
                  hasOnClick: !!htmlElement.onclick,
                  hasEventListeners: typeof window.getEventListeners === "function" ? window.getEventListeners(element) : "DevTools required",
                  classList: element.classList.toString(),
                  parentElement: (_b = element.parentElement) == null ? void 0 : _b.tagName,
                  href: htmlElement.getAttribute("href"),
                  role: htmlElement.getAttribute("role"),
                  isButton: element.tagName === "BUTTON",
                  isLink: element.tagName === "A",
                  boundingRect: element.getBoundingClientRect()
                }
              }, "*");
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
                    view: window
                  })
                );
                await new Promise((resolve) => setTimeout(resolve, 10));
                element.dispatchEvent(
                  new MouseEvent("mouseup", {
                    bubbles: true,
                    cancelable: true,
                    view: window
                  })
                );
                await new Promise((resolve) => setTimeout(resolve, 10));
                htmlElement.click();
                element.dispatchEvent(
                  new MouseEvent("click", {
                    bubbles: true,
                    cancelable: true,
                    view: window
                  })
                );
                window.parent.postMessage({
                  type: "staktrak-debug-click",
                  message: "===>>> \u2705 Click executed successfully",
                  data: {
                    tagName: element.tagName,
                    currentUrl: window.location.href,
                    textContent: (_c = element.textContent) == null ? void 0 : _c.trim(),
                    success: true
                  }
                }, "*");
              } catch (clickError) {
                window.parent.postMessage({
                  type: "staktrak-debug-click",
                  message: "===>>> \u274C Error during click operation",
                  data: {
                    error: clickError instanceof Error ? clickError.message : String(clickError),
                    success: false
                  }
                }, "*");
              }
              await new Promise((resolve) => setTimeout(resolve, 50));
              setTimeout(() => {
                htmlElement.style.border = originalBorder;
                htmlElement.style.boxShadow = "";
              }, 300);
            } else {
              window.parent.postMessage({
                type: "staktrak-debug-click",
                message: "===>>> \u274C Element not found",
                data: {
                  selector: action.selector,
                  success: false,
                  error: "Element not found"
                }
              }, "*");
              console.log(`===>>> \u274C Element not found for selector:`, action.selector);
              throw new Error(`Element not found: ${action.selector}`);
            }
          } else {
            console.log(`===>>> \u26A0\uFE0F No selector provided for CLICK action`);
          }
          break;
        case "fill" /* FILL */:
          if (action.selector && action.value !== void 0) {
            const element = await waitForElement(action.selector);
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
        case "check" /* CHECK */:
          if (action.selector) {
            const element = await waitForElement(
              action.selector
            );
            if (element && (element.type === "checkbox" || element.type === "radio")) {
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
        case "uncheck" /* UNCHECK */:
          if (action.selector) {
            const element = await waitForElement(
              action.selector
            );
            if (element && element.type === "checkbox") {
              if (element.checked) {
                element.click();
              }
            } else {
              throw new Error(`Checkbox element not found: ${action.selector}`);
            }
          }
          break;
        case "selectOption" /* SELECT_OPTION */:
          if (action.selector && action.value !== void 0) {
            const element = await waitForElement(
              action.selector
            );
            if (element && element.tagName === "SELECT") {
              element.value = String(action.value);
              element.dispatchEvent(new Event("change", { bubbles: true }));
            } else {
              throw new Error(`Select element not found: ${action.selector}`);
            }
          }
          break;
        case "waitForTimeout" /* WAIT_FOR_TIMEOUT */:
          const shortDelay = Math.min(action.value, 500);
          await new Promise((resolve) => setTimeout(resolve, shortDelay));
          break;
        case "waitFor" /* WAIT_FOR */:
          if (action.selector) {
            const element = await waitForElement(action.selector);
            if (!element) {
              throw new Error(
                `Element not found for waitFor: ${action.selector}`
              );
            }
            if (((_d = action.options) == null ? void 0 : _d.state) === "visible") {
              if (!isElementVisible(element)) {
                throw new Error(`Element is not visible: ${action.selector}`);
              }
            }
          }
          break;
        case "hover" /* HOVER */:
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
        case "focus" /* FOCUS */:
          if (action.selector) {
            const element = await waitForElement(
              action.selector
            );
            if (element && typeof element.focus === "function") {
              element.focus();
            } else {
              throw new Error(
                `Element not found or not focusable: ${action.selector}`
              );
            }
          }
          break;
        case "blur" /* BLUR */:
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
        case "scrollIntoView" /* SCROLL_INTO_VIEW */:
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
        case "expect" /* EXPECT */:
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
    const element = findElementWithFallbacks(selector);
    return element ? [element] : [];
  }
  function findElementWithFallbacks(selector) {
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
      () => findByCoordinates(selector)
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
  function convertToBrowserSelector(selector) {
    var _a;
    if (!selector) return selector;
    if (selector.includes(":has-text(")) {
      const textMatch = selector.match(/:has-text\("([^"]+)"\)/);
      if (textMatch) {
        const text = textMatch[1];
        const tagMatch = selector.match(/^([a-zA-Z]+)/);
        const tagName = tagMatch ? tagMatch[1] : "*";
        const elements = Array.from(document.querySelectorAll(tagName));
        for (const element of elements) {
          if (((_a = element.textContent) == null ? void 0 : _a.trim()) === text) {
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
  function isValidSelector(selector) {
    if (!selector || selector.trim() === "") return false;
    try {
      document.querySelector(selector);
      return true;
    } catch (e) {
      return false;
    }
  }
  function findByDataTestId(selector) {
    var _a;
    if (!selector.includes("data-testid")) return null;
    const testId = (_a = selector.match(/data-testid="([^"]+)"/)) == null ? void 0 : _a[1];
    if (testId) {
      return document.querySelector(`[data-testid="${testId}"]`);
    }
    return null;
  }
  function findByClass(selector) {
    if (!selector.includes(".")) return null;
    const classes = selector.match(/\.([^\s.#\[\]]+)/g);
    if (classes && classes.length > 0) {
      const className = classes[0].substring(1);
      return document.querySelector(`.${className}`);
    }
    return null;
  }
  function findById(selector) {
    if (!selector.includes("#")) return null;
    const ids = selector.match(/#([^\s.#\[\]]+)/g);
    if (ids && ids.length > 0) {
      const id = ids[0].substring(1);
      return document.querySelector(`#${id}`);
    }
    return null;
  }
  function findByAriaLabel(selector) {
    const ariaMatch = selector.match(/\[aria-label="([^"]+)"\]/);
    if (!ariaMatch) return null;
    return document.querySelector(`[aria-label="${ariaMatch[1]}"]`);
  }
  function findByRole(selector) {
    const roleMatch = selector.match(/\[role="([^"]+)"\]/);
    if (!roleMatch) return null;
    return document.querySelector(`[role="${roleMatch[1]}"]`);
  }
  function findByTextContent(selector) {
    var _a;
    let text = null;
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
      const elementText = (_a = element.textContent) == null ? void 0 : _a.trim();
      if (elementText === text || (elementText == null ? void 0 : elementText.includes(text))) {
        return element;
      }
    }
    return null;
  }
  function findByCoordinates(selector) {
    const clickableElements = document.querySelectorAll(
      'button, a, input, select, [role="button"], [onclick]'
    );
    return clickableElements.length > 0 ? clickableElements[0] : null;
  }
  function createUniqueSelector(element) {
    var _a;
    if (element.id && /^[a-zA-Z][\w-]*$/.test(element.id)) {
      const idSelector = `#${element.id}`;
      if (document.querySelectorAll(idSelector).length === 1) {
        return idSelector;
      }
    }
    const testId = (_a = element.dataset) == null ? void 0 : _a.testid;
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
      return !cls.match(/^[a-zA-Z0-9_-]*[0-9a-f]{6,}/) && !cls.includes("emotion-") && !cls.includes("css-") && !cls.includes("module__") && cls.length < 30;
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
        const textElement = await waitForElement(
          action.selector,
          String(action.value)
        );
        if (!textElement || !((_a = textElement.textContent) == null ? void 0 : _a.includes(String(action.value || "")))) {
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
      case "goto" /* GOTO */:
        return `Navigate to ${action.value}`;
      case "click" /* CLICK */:
        return `Click element: ${action.selector}`;
      case "fill" /* FILL */:
        return `Fill "${action.value}" in ${action.selector}`;
      case "check" /* CHECK */:
        return `Check checkbox: ${action.selector}`;
      case "uncheck" /* UNCHECK */:
        return `Uncheck checkbox: ${action.selector}`;
      case "selectOption" /* SELECT_OPTION */:
        return `Select "${action.value}" in ${action.selector}`;
      case "hover" /* HOVER */:
        return `Hover over element: ${action.selector}`;
      case "focus" /* FOCUS */:
        return `Focus element: ${action.selector}`;
      case "blur" /* BLUR */:
        return `Blur element: ${action.selector}`;
      case "scrollIntoView" /* SCROLL_INTO_VIEW */:
        return `Scroll element into view: ${action.selector}`;
      case "waitFor" /* WAIT_FOR */:
        return `Wait for element: ${action.selector}`;
      case "expect" /* EXPECT */:
        return `Verify ${action.selector} ${action.expectation}`;
      case "setViewportSize" /* SET_VIEWPORT_SIZE */:
        return `Set viewport size to ${action.value}`;
      case "waitForTimeout" /* WAIT_FOR_TIMEOUT */:
        return `Wait ${action.value}ms`;
      case "waitForLoadState" /* WAIT_FOR_LOAD_STATE */:
        return "Wait for page to load";
      case "waitForSelector" /* WAIT_FOR_SELECTOR */:
        return `Wait for element: ${action.selector}`;
      default:
        return `Execute ${action.type}`;
    }
  }

  // src/playwright-replay/index.ts
  var playwrightReplayRef = {
    current: null
  };
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
        alwaysListeners: [],
        healthCheckInterval: null
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
      this.startHealthCheck();
      this.saveSessionState();
      console.log("\u{1F50D} STAKTRAK: Recording state saved to sessionStorage");
      return this;
    }
    saveSessionState() {
      try {
        const sessionData = {
          isRecording: true,
          startTime: Date.now(),
          lastSaved: Date.now(),
          results: this.results,
          memory: {
            assertions: this.memory.assertions,
            selectionMode: this.memory.selectionMode
          },
          version: "1.0"
        };
        sessionStorage.setItem("stakTrakActiveRecording", JSON.stringify(sessionData));
      } catch (error) {
        console.warn("\u{1F50D} STAKTRAK: Failed to save session state:", error);
      }
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
      if (this.memory.healthCheckInterval) {
        clearInterval(this.memory.healthCheckInterval);
        this.memory.healthCheckInterval = null;
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
      console.log("\u{1F50D} STAKTRAK: Setting up event listeners", { isRunning: this.isRunning });
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
          this.saveSessionState();
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
              this.saveSessionState();
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
                this.saveSessionState();
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
            break;
          case "staktrak-recover":
            this.recoverRecording();
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
      sessionStorage.removeItem("stakTrakActiveRecording");
      console.log("\u{1F50D} STAKTRAK: Recording state cleared from sessionStorage");
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
    attemptSessionRestoration() {
      try {
        const activeRecording = sessionStorage.getItem("stakTrakActiveRecording");
        if (!activeRecording) {
          console.log("\u{1F50D} STAKTRAK: No previous session to restore");
          return;
        }
        const recordingData = JSON.parse(activeRecording);
        console.log("\u{1F50D} STAKTRAK: Found previous session data in sessionStorage");
        if (recordingData && recordingData.isRecording && recordingData.version === "1.0") {
          console.log("\u{1F50D} STAKTRAK: Attempting session restoration...");
          const timeSinceLastSave = Date.now() - (recordingData.lastSaved || 0);
          const isLikelyIframeReload = timeSinceLastSave < 1e4;
          if (isLikelyIframeReload) {
            console.log("\u{1F50D} STAKTRAK: Detected iframe reload, restoring recording state");
            if (recordingData.results) {
              this.results = __spreadValues(__spreadValues({}, this.createEmptyResults()), recordingData.results);
            }
            if (recordingData.memory) {
              this.memory.assertions = recordingData.memory.assertions || [];
              this.memory.selectionMode = recordingData.memory.selectionMode || false;
            }
            this.isRunning = true;
            this.setupEventListeners();
            this.startHealthCheck();
            console.log("\u{1F50D} STAKTRAK: Session restored successfully", {
              clicks: this.results.clicks.clickCount,
              inputs: this.results.inputChanges.length,
              assertions: this.memory.assertions.length
            });
            this.verifyEventListeners();
            window.parent.postMessage({ type: "staktrak-replay-ready" }, "*");
          } else {
            console.log("\u{1F50D} STAKTRAK: Session data is too old, starting fresh");
            sessionStorage.removeItem("stakTrakActiveRecording");
          }
        } else {
          console.log("\u{1F50D} STAKTRAK: Invalid session data, starting fresh");
          sessionStorage.removeItem("stakTrakActiveRecording");
        }
      } catch (error) {
        console.warn("\u{1F50D} STAKTRAK: Session restoration failed:", error);
        sessionStorage.removeItem("stakTrakActiveRecording");
      }
    }
    verifyEventListeners() {
      console.log("\u{1F50D} STAKTRAK: Verifying event listeners", {
        isRunning: this.isRunning,
        listenersCount: this.memory.listeners.length,
        mutationObserver: !!this.memory.mutationObserver
      });
      if (this.isRunning && this.memory.listeners.length === 0) {
        console.warn("\u{1F50D} STAKTRAK: No listeners found, re-establishing...");
        this.setupEventListeners();
      }
    }
    recoverRecording() {
      console.log("\u{1F50D} STAKTRAK: Attempting recording recovery");
      if (!this.isRunning) {
        console.log("\u{1F50D} STAKTRAK: Recording was not active, starting fresh");
        return;
      }
      this.verifyEventListeners();
      this.saveSessionState();
      console.log("\u{1F50D} STAKTRAK: Recording recovery completed");
    }
    startHealthCheck() {
      this.memory.healthCheckInterval = setInterval(() => {
        if (this.isRunning) {
          if (this.memory.listeners.length === 0) {
            console.warn("\u{1F50D} STAKTRAK: Health check failed - no listeners, attempting recovery");
            this.recoverRecording();
          }
          this.saveSessionState();
        }
      }, 5e3);
      console.log("\u{1F50D} STAKTRAK: Health check started");
    }
  };
  var userBehaviour = new UserBehaviorTracker();
  var initializeStakTrak = () => {
    userBehaviour.makeConfig({
      processData: (results) => console.log("StakTrak recording processed:", results)
    }).listen();
    userBehaviour.attemptSessionRestoration();
    initPlaywrightReplay();
  };
  document.readyState === "loading" ? document.addEventListener("DOMContentLoaded", initializeStakTrak) : initializeStakTrak();
  userBehaviour.createClickDetail = createClickDetail;
  var index_default = userBehaviour;
  return __toCommonJS(index_exports);
})();
