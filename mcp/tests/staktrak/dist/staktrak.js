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
  var stakTrakIdCounter = 0;
  var stakTrakIdPrefix = "stk";
  var generateStakTrakId = (element) => {
    var _a;
    if (element) {
      const tagName = element.tagName.toLowerCase();
      const className = element.className || "";
      const text = (element.textContent || "").replace(/\s+/g, " ").trim().slice(0, 20);
      const position = Array.from(((_a = element.parentElement) == null ? void 0 : _a.children) || []).indexOf(element);
      const hash = simpleHash(`${tagName}-${className}-${text}-${position}`);
      return `${stakTrakIdPrefix}-${hash}`;
    }
    return `${stakTrakIdPrefix}-${++stakTrakIdCounter}`;
  };
  var simpleHash = (str) => {
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
      const char = str.charCodeAt(i);
      hash = (hash << 5) - hash + char;
      hash = hash & hash;
    }
    return Math.abs(hash).toString(36);
  };
  var ensureElementId = (element) => {
    var _a;
    const existingTestId = ((_a = element.dataset) == null ? void 0 : _a.testid) || element.getAttribute("data-testid");
    if (existingTestId) {
      return `[data-testid="${existingTestId}"]`;
    }
    const existingId = element.id;
    if (existingId && /^[a-zA-Z][\w-]*$/.test(existingId)) {
      return `#${existingId}`;
    }
    const existingStakTrakId = element.getAttribute("data-staktrak-id");
    if (existingStakTrakId) {
      return `[data-staktrak-id="${existingStakTrakId}"]`;
    }
    const uniqueId = generateStakTrakId(element);
    try {
      element.setAttribute("data-staktrak-id", uniqueId);
      return `[data-staktrak-id="${uniqueId}"]`;
    } catch (error) {
      return generateClassBasedSelector(element);
    }
  };
  var isInteractiveElement = (element) => {
    const tagName = element.tagName.toLowerCase();
    const role = element.getAttribute("role");
    return tagName === "button" || tagName === "a" && element.hasAttribute("href") || tagName === "input" || tagName === "select" || tagName === "textarea" || role === "button" || role === "link" || role === "menuitem" || role === "tab" || element.hasAttribute("onclick");
  };
  var getInteractiveSelectors = () => {
    return [
      "button",
      "a[href]",
      'input[type="button"], input[type="submit"], input[type="reset"]',
      'input[type="text"], input[type="email"], input[type="password"], input[type="search"]',
      'input[type="checkbox"], input[type="radio"]',
      "select",
      "textarea",
      '[role="button"]',
      '[role="link"]',
      '[role="menuitem"]',
      '[role="tab"]',
      "[onclick]",
      "[data-testid]"
    ];
  };
  var hasStableIdentifier = (element) => {
    return !!(element.getAttribute("data-testid") || element.getAttribute("data-staktrak-id") || element.id && /^[a-zA-Z][\w-]*$/.test(element.id));
  };
  var annotateInteractiveElements = () => {
    const startTime = Date.now();
    let annotatedCount = 0;
    try {
      const selectors = getInteractiveSelectors();
      const allInteractive = document.querySelectorAll(selectors.join(", "));
      allInteractive.forEach((element) => {
        const htmlEl = element;
        if (!hasStableIdentifier(htmlEl)) {
          ensureElementId(htmlEl);
          annotatedCount++;
        }
      });
      const duration = Date.now() - startTime;
      return { count: annotatedCount, duration };
    } catch (error) {
      const duration = Date.now() - startTime;
      return { count: 0, duration };
    }
  };
  var getAnnotationStats = () => {
    try {
      const selectors = getInteractiveSelectors();
      const allInteractive = document.querySelectorAll(selectors.join(", "));
      const stakTrakAnnotated = document.querySelectorAll("[data-staktrak-id]");
      const existingStable = Array.from(allInteractive).filter(
        (el) => hasStableIdentifier(el)
      );
      return {
        total: allInteractive.length,
        annotated: stakTrakAnnotated.length,
        existing: existingStable.length - stakTrakAnnotated.length
      };
    } catch (error) {
      return { total: 0, annotated: 0, existing: 0 };
    }
  };
  var cleanupAnnotations = () => {
    try {
      const stakTrakElements = document.querySelectorAll("[data-staktrak-id]");
      stakTrakElements.forEach((element) => {
        element.removeAttribute("data-staktrak-id");
      });
      return { removed: stakTrakElements.length };
    } catch (error) {
      return { removed: 0 };
    }
  };
  var stakTrakObserver = null;
  var observerThrottle = null;
  var pendingElements = /* @__PURE__ */ new Set();
  var processPendingElements = () => {
    if (pendingElements.size === 0) return;
    const elementsToProcess = Array.from(pendingElements);
    pendingElements.clear();
    elementsToProcess.forEach((element) => {
      if (document.contains(element) && !hasStableIdentifier(element)) {
        ensureElementId(element);
      }
    });
  };
  var handleMutation = (mutations) => {
    mutations.forEach((mutation) => {
      mutation.addedNodes.forEach((node) => {
        if (node.nodeType === Node.ELEMENT_NODE) {
          const element = node;
          if (isInteractiveElement(element)) {
            pendingElements.add(element);
          }
          const interactiveChildren = element.querySelectorAll(
            getInteractiveSelectors().join(", ")
          );
          interactiveChildren.forEach((child) => {
            pendingElements.add(child);
          });
        }
      });
    });
    if (observerThrottle) {
      clearTimeout(observerThrottle);
    }
    observerThrottle = setTimeout(processPendingElements, 100);
  };
  var startDynamicAnnotation = () => {
    if (stakTrakObserver) return false;
    try {
      stakTrakObserver = new MutationObserver(handleMutation);
      stakTrakObserver.observe(document.body, {
        childList: true,
        subtree: true
      });
      return true;
    } catch (error) {
      return false;
    }
  };
  var stopDynamicAnnotation = () => {
    const processedCount = pendingElements.size;
    if (stakTrakObserver) {
      stakTrakObserver.disconnect();
      stakTrakObserver = null;
    }
    if (observerThrottle) {
      clearTimeout(observerThrottle);
      observerThrottle = null;
    }
    processPendingElements();
    pendingElements.clear();
    return { processed: processedCount };
  };
  var getDynamicAnnotationStatus = () => {
    return {
      active: stakTrakObserver !== null,
      pending: pendingElements.size
    };
  };
  var getElementRole = (el) => {
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
    const reasonsMap = {};
    const scored = [];
    const pushCandidate = (sel, baseScore, reason) => {
      if (!sel) return;
      if (!reasonsMap[sel]) reasonsMap[sel] = [];
      reasonsMap[sel].push(reason);
      let score = baseScore;
      if (sel.length > 60) score -= Math.min(20, Math.floor((sel.length - 60) / 5));
      const depth2 = sel.split(">").length - 1;
      if (depth2 > 3) score -= (depth2 - 3) * 2;
      if (/\.[a-zA-Z0-9_-]*[0-9a-f]{6,}\b/.test(sel)) score -= 25;
      if (/^\w+$/.test(sel)) score -= 30;
      if (sel.startsWith("text=")) score -= 5;
      if (sel.startsWith("role:")) score -= 3;
      scored.push({ selector: sel, score, reasons: reasonsMap[sel] });
    };
    const finalizeReturn = (primary2, fallbacks2, extra) => {
      const dedup = {};
      for (const c of scored) {
        if (!dedup[c.selector] || dedup[c.selector].score < c.score) dedup[c.selector] = c;
      }
      const ordered = Object.values(dedup).sort((a, b) => b.score - a.score);
      return __spreadProps(__spreadValues({ primary: primary2, fallbacks: fallbacks2 }, extra), { scores: ordered });
    };
    const testId = (_a = htmlEl.dataset) == null ? void 0 : _a.testid;
    if (testId) {
      const sel = `[data-testid="${testId}"]`;
      pushCandidate(sel, 100, "data-testid attribute");
      return finalizeReturn(sel, [], {
        tagName,
        text: getElementText(element),
        ariaLabel: htmlEl.getAttribute("aria-label") || void 0,
        title: htmlEl.getAttribute("title") || void 0,
        role: getElementRole(htmlEl) || void 0
      });
    }
    const id = htmlEl.id;
    if (id && /^[a-zA-Z][\w-]*$/.test(id)) {
      const sel = `#${id}`;
      pushCandidate(sel, 95, "element id");
      return finalizeReturn(sel, [], {
        tagName,
        text: getElementText(element),
        ariaLabel: htmlEl.getAttribute("aria-label") || void 0,
        title: htmlEl.getAttribute("title") || void 0,
        role: getElementRole(htmlEl) || void 0
      });
    }
    const text = getEnhancedElementText(htmlEl);
    const role = getElementRole(htmlEl);
    const classSelector = generateClassBasedSelector(element);
    if (classSelector && classSelector !== tagName) {
      fallbacks.push(classSelector);
      pushCandidate(classSelector, 80, "class-based selector");
    }
    if (!testId && !id && role === "button" && text && text.length < 40) {
      const rsel = `role:button[name="${text.replace(/"/g, '\\"')}"]`;
      fallbacks.push(rsel);
      pushCandidate(rsel, 70, "role+name");
    }
    const text2 = getEnhancedElementText(htmlEl);
    if (text2 && (tagName === "button" || tagName === "a" || role === "button")) {
      const textSelector = generateTextBasedSelector(element, text2);
      if (textSelector) {
        fallbacks.push(textSelector);
        pushCandidate(textSelector, 60, "text content");
      }
    }
    const ariaLabel = htmlEl.getAttribute("aria-label");
    if (ariaLabel) {
      const al = `[aria-label="${ariaLabel}"]`;
      fallbacks.push(al);
      pushCandidate(al, 65, "aria-label");
    }
    if (role && !fallbacks.find((f) => f.startsWith("role:") || f.startsWith(`[role="${role}`))) {
      const rs = `[role="${role}"]`;
      fallbacks.push(rs);
      pushCandidate(rs, 40, "generic role");
    }
    if (tagName === "input") {
      const type = element.type;
      const name = element.name;
      if (type) {
        const tsel = `input[type="${type}"]`;
        fallbacks.push(tsel);
        pushCandidate(tsel, 55, "input type");
      }
      if (name) {
        const nsel = `input[name="${name}"]`;
        fallbacks.push(nsel);
        pushCandidate(nsel, 58, "input name");
      }
    }
    const contextualSelector = generateContextualSelector(element);
    if (contextualSelector) {
      fallbacks.push(contextualSelector);
      pushCandidate(contextualSelector, 45, "contextual");
    }
    const xpath = generateXPath(element);
    if (fallbacks.length === 0) {
      pushCandidate(tagName, 10, "bare tag");
    }
    const best = scored.sort((a, b) => b.score - a.score)[0];
    const primary = best ? best.selector : fallbacks.length > 0 ? fallbacks[0] : tagName;
    const fb = fallbacks.filter((f) => f !== primary);
    return finalizeReturn(primary, fb, {
      text: text || void 0,
      ariaLabel: ariaLabel || void 0,
      title: htmlEl.getAttribute("title") || void 0,
      role: role || void 0,
      tagName,
      xpath
    });
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
    const html = target;
    const injectedSelector = ensureElementId(html);
    const fallbackSelectors = generateSelectorStrategies(target);
    const enhancedSelectors = {
      primary: injectedSelector,
      fallbacks: [
        fallbackSelectors.primary,
        ...fallbackSelectors.fallbacks
      ].filter((s) => s && s !== injectedSelector),
      text: fallbackSelectors.text,
      ariaLabel: fallbackSelectors.ariaLabel,
      title: fallbackSelectors.title,
      role: fallbackSelectors.role,
      tagName: fallbackSelectors.tagName,
      xpath: fallbackSelectors.xpath
    };
    const testId = html.getAttribute("data-testid") || html.getAttribute("data-staktrak-id") || void 0;
    const id = html.id || void 0;
    const accessibleName = getEnhancedElementText(html) || void 0;
    let nth;
    if (html.parentElement) {
      const same = Array.from(html.parentElement.children).filter((c) => c.tagName === html.tagName);
      if (same.length > 1) nth = same.indexOf(html) + 1;
    }
    const ancestors = [];
    let p = html.parentElement;
    let depth2 = 0;
    while (p && depth2 < 4) {
      const role = p.getAttribute("role");
      const tag = p.tagName.toLowerCase();
      if (["main", "nav", "header", "footer", "aside", "section", "form", "article"].includes(tag) || role) {
        ancestors.push(role ? `${tag}[role=${role}]` : tag);
      }
      p = p.parentElement;
      depth2++;
    }
    const selAny = enhancedSelectors;
    selAny.id = id;
    selAny.testId = testId;
    selAny.accessibleName = accessibleName;
    if (nth) selAny.nth = nth;
    if (ancestors.length) selAny.ancestors = ancestors;
    selAny.stabilizedPrimary = injectedSelector;
    selAny.visualSelector = injectedSelector;
    return {
      x: e.clientX,
      y: e.clientY,
      timestamp: getTimeStamp(),
      selectors: enhancedSelectors,
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
    return ensureElementId(element);
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
    var _a, _b;
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
        } else if (trimmed.includes("page.locator('[data-staktrak-id=") && trimmed.includes(".fill(")) {
          const stakFillMatch = trimmed.match(/page\.locator\('\[data-staktrak-id="([^"]+)"\]'\)\.fill\(['"]([^'"]*)['"]\)/);
          if (stakFillMatch) {
            actions.push({
              type: "fill",
              selector: `[data-staktrak-id="${stakFillMatch[1]}"]`,
              value: stakFillMatch[2],
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
        } else if (trimmed.includes("page.locator('[data-staktrak-id=") && trimmed.includes(".check(")) {
          const stakCheckMatch = trimmed.match(/page\.locator\('\[data-staktrak-id="([^"]+)"\]'\)\.check\(\)/);
          if (stakCheckMatch) {
            actions.push({
              type: "check",
              selector: `[data-staktrak-id="${stakCheckMatch[1]}"]`,
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
        } else if (trimmed.includes("page.locator('[data-staktrak-id=") && trimmed.includes(".uncheck(")) {
          const stakUncheckMatch = trimmed.match(/page\.locator\('\[data-staktrak-id="([^"]+)"\]'\)\.uncheck\(\)/);
          if (stakUncheckMatch) {
            actions.push({
              type: "uncheck",
              selector: `[data-staktrak-id="${stakUncheckMatch[1]}"]`,
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
        } else if (trimmed.includes("page.locator('[data-staktrak-id=") && trimmed.includes(".selectOption(")) {
          const stakSelectMatch = trimmed.match(/page\.locator\('\[data-staktrak-id="([^"]+)"\]'\)\.selectOption\(['"]([^'"]*)['"]\)/);
          if (stakSelectMatch) {
            actions.push({
              type: "selectOption",
              selector: `[data-staktrak-id="${stakSelectMatch[1]}"]`,
              value: stakSelectMatch[2],
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
        } else if (trimmed.includes("page.waitForURL(")) {
          const urlMatch = trimmed.match(/page\.waitForURL\(['\"](.*?)['\"]\)/);
          if (urlMatch) {
            actions.push({
              type: "waitForURL",
              value: urlMatch[1],
              comment,
              lineNumber
            });
          }
        } else if (/page\.[a-zA-Z]+\([^)]*\)\.click\([^)]*\)\s*;?$/.test(trimmed)) {
          const locatorCallMatch = trimmed.match(/page\.([a-zA-Z]+\([^)]*\))\.click\([^)]*\)/);
          if (locatorCallMatch) {
            const selector = parseLocatorCall(locatorCallMatch[1]);
            actions.push({
              type: "click",
              selector,
              comment,
              lineNumber
            });
          }
        } else if (trimmed.startsWith("await Promise.all([") && trimmed.includes("waitForURL")) {
          const blockLines = [trimmed];
          let j = lineNumber;
          for (let k = 1; k <= 6 && lineNumber + k - 1 < lines.length; k++) {
            const peek = lines[lineNumber + k - 1].trim();
            blockLines.push(peek);
            if (peek.endsWith("]);")) break;
          }
          const block = blockLines.join(" ");
          const url = (_a = block.match(/page\.waitForURL\(['\"](.*?)['\"]\)/)) == null ? void 0 : _a[1];
          const clickSelector = (_b = block.match(/page\.(getBy[^.]+\([^)]*\)|locator\([^)]*\))\.click\(\)/)) == null ? void 0 : _b[1];
          if (url) {
            actions.push({ type: "waitForURL", value: url, comment: (comment ? comment + " " : "") + "(compound)", lineNumber });
          }
          if (clickSelector) {
            const selector = parseLocatorCall(clickSelector);
            actions.push({ type: "click", selector, comment, lineNumber });
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
        } else if (trimmed.includes("page.locator('[data-staktrak-id=")) {
          const stakIdMatch = trimmed.match(/page\.locator\('\[data-staktrak-id="([^"]+)"\]'\)/);
          if (stakIdMatch) {
            actions.push({
              type: "click",
              selector: `[data-staktrak-id="${stakIdMatch[1]}"]`,
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
              const stakExpectMatch = trimmed.match(
                /expect\(page\.locator\('\[data-staktrak-id="([^"]+)"\]'\)\)\.toBeVisible\(\)/
              );
              if (stakExpectMatch) {
                actions.push({
                  type: "expect",
                  selector: `[data-staktrak-id="${stakExpectMatch[1]}"]`,
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
  var __stakReplayMatch = window.__stakTrakReplayMatch || { last: null };
  window.__stakTrakReplayMatch = __stakReplayMatch;
  var __stakReplayState = window.__stakTrakReplayState || { lastStructural: null, lastEl: null };
  window.__stakTrakReplayState = __stakReplayState;
  window.__stakTrakSelectorMap = window.__stakTrakSelectorMap || {};
  var __stakWarned = window.__stakTrakWarned || {};
  window.__stakTrakWarned = __stakWarned;
  function highlight(element, actionType = "action") {
    try {
      ensureStylesInDocument(document);
    } catch (e) {
    }
    const htmlElement = element;
    const original = {
      border: htmlElement.style.border,
      boxShadow: htmlElement.style.boxShadow,
      backgroundColor: htmlElement.style.backgroundColor
    };
    htmlElement.style.border = "3px solid #ff6b6b";
    htmlElement.style.boxShadow = "0 0 20px rgba(255, 107, 107, 0.8)";
    htmlElement.style.backgroundColor = "rgba(255, 107, 107, 0.2)";
    htmlElement.style.transition = "all 0.3s ease";
    const last = __stakReplayMatch.last;
    if (last && last.element === element && Date.now() - last.time < 4e3) {
      htmlElement.setAttribute("data-staktrak-matched-selector", last.matched);
      htmlElement.setAttribute("data-staktrak-requested-selector", last.requested);
      if (last.text) htmlElement.setAttribute("data-staktrak-matched-text", last.text);
    }
    element.scrollIntoView({ behavior: "smooth", block: "center" });
    setTimeout(() => {
      htmlElement.style.border = original.border;
      htmlElement.style.boxShadow = original.boxShadow;
      htmlElement.style.backgroundColor = original.backgroundColor;
      htmlElement.style.transition = "";
    }, 1500);
  }
  function normalizeUrl(u) {
    try {
      const url = new URL(u, window.location.origin);
      return url.href.replace(/[#?].*$/, "").replace(/\/$/, "");
    } catch (e) {
      return u.replace(/[#?].*$/, "").replace(/\/$/, "");
    }
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
        case "waitForURL" /* WAIT_FOR_URL */:
          if (action.value && typeof action.value === "string") {
            const target = normalizeUrl(action.value);
            const start = Date.now();
            let matched = false;
            let lastPulse = 0;
            const maxMs = 8e3;
            const stopSignals = [];
            const tryMatch = () => {
              const current = normalizeUrl(window.location.href);
              if (current === target) {
                matched = true;
                return true;
              }
              try {
                const curNoHash = current.replace(/#.*/, "");
                const tgtNoHash = target.replace(/#.*/, "");
                if (curNoHash === tgtNoHash) {
                  matched = true;
                  return true;
                }
              } catch (e) {
              }
              return false;
            };
            const onHist = (e) => {
              if (!matched && tryMatch()) {
              }
            };
            try {
              window.addEventListener("staktrak-history-change", onHist);
              stopSignals.push(() => window.removeEventListener("staktrak-history-change", onHist));
            } catch (e) {
            }
            const onHash = () => {
              if (!matched && tryMatch()) {
              }
            };
            try {
              window.addEventListener("hashchange", onHash);
              stopSignals.push(() => window.removeEventListener("hashchange", onHash));
            } catch (e) {
            }
            tryMatch();
            while (!matched && Date.now() - start < maxMs) {
              if (Date.now() - lastPulse > 1e3) {
                try {
                  document.body.style.outline = "3px dashed #ff6b6b";
                  setTimeout(() => {
                    document.body.style.outline = "";
                  }, 400);
                } catch (e) {
                }
                lastPulse = Date.now();
              }
              await new Promise((r) => setTimeout(r, 120));
              if (tryMatch()) break;
            }
            stopSignals.forEach((fn) => {
              try {
                fn();
              } catch (e) {
              }
            });
            try {
              highlight(document.body, matched ? "nav" : "nav-timeout");
            } catch (e) {
            }
            try {
              ensureStylesInDocument(document);
            } catch (e) {
            }
            if (!matched && !window.__stakTrakWarnedNav) {
              console.warn("[staktrak] waitForURL timeout \u2014 last, expected", window.location.href, target);
              window.__stakTrakWarnedNav = true;
            }
          }
          break;
        case "click" /* CLICK */:
          if (action.selector) {
            const element = await waitForElement(action.selector);
            if (element) {
              const htmlElement = element;
              highlight(element, "click");
              try {
                htmlElement.focus();
              } catch (e) {
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
              } catch (clickError) {
                throw clickError;
              }
              await new Promise((resolve) => setTimeout(resolve, 50));
            } else {
              throw new Error(`Element not found: ${action.selector}`);
            }
          }
          break;
        case "fill" /* FILL */:
          if (action.selector && action.value !== void 0) {
            const element = await waitForElement(action.selector);
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
        case "check" /* CHECK */:
          if (action.selector) {
            const element = await waitForElement(
              action.selector
            );
            if (element && (element.type === "checkbox" || element.type === "radio")) {
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
        case "uncheck" /* UNCHECK */:
          if (action.selector) {
            const element = await waitForElement(
              action.selector
            );
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
        case "selectOption" /* SELECT_OPTION */:
          if (action.selector && action.value !== void 0) {
            const element = await waitForElement(
              action.selector
            );
            if (element && element.tagName === "SELECT") {
              highlight(element, "select");
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
            if (((_a = action.options) == null ? void 0 : _a.state) === "visible") {
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
        case "focus" /* FOCUS */:
          if (action.selector) {
            const element = await waitForElement(
              action.selector
            );
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
        case "blur" /* BLUR */:
          if (action.selector) {
            const element = await waitForElement(
              action.selector
            );
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
        case "scrollIntoView" /* SCROLL_INTO_VIEW */:
          if (action.selector) {
            const element = await waitForElement(action.selector);
            if (element) {
              highlight(element, "scroll");
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
      } catch (e) {
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
    var _a, _b;
    if (!selector || selector.trim() === "") return null;
    if (selector.includes("data-staktrak-id=")) {
      const stakIdMatch = selector.match(/\[data-staktrak-id="([^"]+)"\]/);
      if (stakIdMatch) {
        const element = document.querySelector(`[data-staktrak-id="${stakIdMatch[1]}"]`);
        if (element) {
          return element;
        }
        console.warn(`[staktrak-replay] Element not found with data-staktrak-id="${stakIdMatch[1]}", trying fallback strategies`);
      }
    }
    if (selector.includes("data-testid=")) {
      const testIdMatch = selector.match(/\[data-testid="([^"]+)"\]/);
      if (testIdMatch) {
        const element = document.querySelector(`[data-testid="${testIdMatch[1]}"]`);
        if (element) {
          return element;
        }
      }
    }
    try {
      if ((selector.startsWith("text=") || selector.startsWith("role:")) && window.__stakTrakSelectorMap) {
        const map = window.__stakTrakSelectorMap;
        const entry = map[selector];
        if (entry == null ? void 0 : entry.visualSelector) {
          try {
            const cssEl = document.querySelector(entry.visualSelector);
            if (cssEl) return cssEl;
          } catch (e) {
          }
        }
      }
    } catch (e) {
    }
    if (/^[a-zA-Z]+\.[a-zA-Z0-9_-]+(\.[a-zA-Z0-9_-]+)*$/.test(selector)) {
      try {
        const matches = document.querySelectorAll(selector);
        if (matches.length === 1) {
          __stakReplayState.lastStructural = selector;
          __stakReplayState.lastEl = matches[0];
          return matches[0];
        }
      } catch (e) {
      }
    }
    if ((selector.startsWith("role:") || selector.startsWith("text=")) && __stakReplayState.lastStructural && __stakReplayState.lastEl) {
      try {
        if (document.contains(__stakReplayState.lastEl)) {
          const acc = getAccessibleName(__stakReplayState.lastEl);
          const nameMatch = selector.includes('name="') ? selector.includes(`name="${acc}`) : selector.includes(acc || "");
          if (acc && nameMatch) {
            return __stakReplayState.lastEl;
          }
        }
      } catch (e) {
      }
    }
    const noteMatch = (el, matched, text) => {
      if (el) {
        __stakReplayMatch.last = { requested: selector, matched, text, time: Date.now(), element: el };
        if (text) el.__stakTrakMatchedText = text;
      }
      return el;
    };
    if (selector.startsWith("role:")) {
      const roleMatch = selector.match(/^role:([^\[]+)(?:\[name(?:-regex)?="(.+?)"\])?/);
      if (roleMatch) {
        const role = roleMatch[1];
        const nameRaw = roleMatch[2];
        const nameRegex = selector.includes("[name-regex=");
        const candidates = Array.from(queryByRole(role.trim()));
        if (!nameRaw) {
          return noteMatch(candidates[0] || null, selector);
        }
        let matcher;
        if (nameRegex) {
          const rx = nameRaw.match(/^\/(.*)\/(.*)$/);
          if (rx) {
            try {
              const r = new RegExp(rx[1], rx[2]);
              matcher = (s) => r.test(s);
            } catch (e) {
              matcher = (s) => s.includes(nameRaw);
            }
          } else {
            matcher = (s) => s.includes(nameRaw);
          }
        } else {
          const target = nameRaw;
          matcher = (s) => s === target;
        }
        for (const el of candidates) {
          const acc = getAccessibleName(el);
          if (acc && matcher(acc)) {
            return noteMatch(el, selector, acc);
          }
        }
        return noteMatch(null, selector);
      }
    }
    if (selector.startsWith("text=") && selector.endsWith(":exact")) {
      const core = selector.slice("text=".length, -":exact".length);
      const norm = core.trim();
      const interactive = Array.from(document.querySelectorAll("button, a, [role], input, textarea, select"));
      const exact = interactive.filter((el) => (el.textContent || "").trim() === norm);
      if (exact.length === 1) return noteMatch(exact[0], selector, norm);
      if (exact.length > 1) {
        const deepest = exact.sort((a, b) => depth(b) - depth(a))[0];
        return noteMatch(deepest, selector, norm);
      }
      return noteMatch(null, selector);
    }
    if (selector.startsWith("getByStakTrakId:")) {
      const val = selector.substring("getByStakTrakId:".length);
      return noteMatch(document.querySelector(`[data-staktrak-id="${cssEscape(val)}"]`), `[data-staktrak-id="${val}"]`);
    }
    if (selector.startsWith("getByTestId:")) {
      const val = selector.substring("getByTestId:".length);
      return noteMatch(document.querySelector(`[data-testid="${cssEscape(val)}"]`), `[data-testid="${val}"]`);
    }
    if (selector.startsWith("getByText-regex:")) {
      const body = selector.substring("getByText-regex:".length);
      const rx = body.match(/^\/(.*)\/(.*)$/);
      let r = null;
      if (rx) try {
        r = new RegExp(rx[1], rx[2]);
      } catch (e) {
      }
      const all = textSearchCandidates();
      for (const el of all) {
        const txt = ((_a = el.textContent) == null ? void 0 : _a.trim()) || "";
        if (r && r.test(txt)) {
          return noteMatch(el, selector, txt);
        }
      }
      return noteMatch(null, selector);
    }
    if (selector.startsWith("getByText:")) {
      const exact = selector.endsWith(":exact");
      const core = exact ? selector.slice("getByText:".length, -":exact".length) : selector.slice("getByText:".length);
      const norm = core.trim();
      const all = textSearchCandidates();
      for (const el of all) {
        const txt = ((_b = el.textContent) == null ? void 0 : _b.trim()) || "";
        if (exact && txt === norm || !exact && txt.includes(norm)) {
          return noteMatch(el, selector, txt);
        }
      }
      return noteMatch(null, selector);
    }
    if (selector.startsWith("getByLabel:")) {
      const label = selector.substring("getByLabel:".length).trim();
      const labels = Array.from(document.querySelectorAll("label")).filter((l) => {
        var _a2;
        return ((_a2 = l.textContent) == null ? void 0 : _a2.trim()) === label;
      });
      for (const lab of labels) {
        const forId = lab.getAttribute("for");
        if (forId) {
          const ctl = document.getElementById(forId);
          if (ctl) return noteMatch(ctl, selector);
        }
        const nested = lab.querySelector("input,select,textarea,button");
        if (nested) return noteMatch(nested, selector);
      }
      return noteMatch(document.querySelector(`[aria-label="${cssEscape(label)}"]`), selector);
    }
    if (selector.startsWith("getByPlaceholder:")) {
      const ph = selector.substring("getByPlaceholder:".length);
      return noteMatch(document.querySelector(`[placeholder="${cssEscape(ph)}"]`), selector);
    }
    if (selector.startsWith("getByTitle:")) {
      const t = selector.substring("getByTitle:".length);
      return noteMatch(document.querySelector(`[title="${cssEscape(t)}"]`), selector);
    }
    if (selector.startsWith("getByAltText:")) {
      const alt = selector.substring("getByAltText:".length);
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
      () => findByTextContentTight(selector)
    ];
    for (const strategy of strategies) {
      try {
        const element = strategy();
        if (element) {
          return noteMatch(element, selector);
        }
      } catch (e) {
      }
    }
    return noteMatch(null, selector);
  }
  function cssEscape(value) {
    return value.replace(/[^a-zA-Z0-9_-]/g, (c) => `\\${c}`);
  }
  function queryByRole(role) {
    const selector = getRoleSelector(role);
    return Array.from(document.querySelectorAll(selector)).filter((el) => el instanceof HTMLElement);
  }
  function getAccessibleName(el) {
    var _a;
    const aria = el.getAttribute("aria-label");
    if (aria) return aria.trim();
    const labelled = el.getAttribute("aria-labelledby");
    if (labelled) {
      const parts = labelled.split(/\s+/).map((id) => {
        var _a2, _b;
        return (_b = (_a2 = document.getElementById(id)) == null ? void 0 : _a2.textContent) == null ? void 0 : _b.trim();
      }).filter(Boolean);
      if (parts.length) return parts.join(" ");
    }
    const tag = el.tagName.toLowerCase();
    if (tag === "input" || tag === "textarea") {
      const val = el.value || el.getAttribute("placeholder");
      if (val) return val.trim();
    }
    const txt = (_a = el.textContent) == null ? void 0 : _a.trim();
    if (txt) return txt.slice(0, 120);
    const title = el.getAttribute("title");
    if (title) return title.trim();
    return null;
  }
  function textSearchCandidates() {
    return Array.from(document.querySelectorAll("button, a, [role], input, textarea, select, label, div, span"));
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
  function findByStakTrakId(selector) {
    var _a;
    if (!selector.includes("data-staktrak-id")) return null;
    const stakId = (_a = selector.match(/data-staktrak-id="([^"]+)"/)) == null ? void 0 : _a[1];
    if (stakId) {
      return document.querySelector(`[data-staktrak-id="${stakId}"]`);
    }
    return null;
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
  function findByClassUnique(selector) {
    if (!selector.includes(".")) return null;
    if (selector.startsWith("text=") || selector.startsWith("role:")) return null;
    try {
      const els = document.querySelectorAll(selector);
      if (els.length === 1) return els[0];
    } catch (e) {
    }
    const classOnly = selector.match(/^\w+\.[^.]+$/);
    if (classOnly) {
      const els = Array.from(document.querySelectorAll(selector));
      const interactive = els.filter(isInteractive);
      if (interactive.length === 1) return interactive[0];
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
  function findByTextContentTight(selector) {
    if (!selector.startsWith("text=")) return null;
    const exact = selector.endsWith(":exact");
    const core = exact ? selector.slice("text=".length, -":exact".length) : selector.slice("text=".length);
    const norm = core.trim();
    const candidates = textSearchCandidates().filter(isInteractiveOrSmall);
    for (const el of candidates) {
      const txt = (el.textContent || "").trim();
      if (exact && txt === norm || !exact && txt.includes(norm)) return el;
    }
    return null;
  }
  function isInteractive(el) {
    const tag = el.tagName.toLowerCase();
    if (["button", "a", "input", "textarea", "select", "option"].includes(tag)) return true;
    const role = el.getAttribute("role");
    if (role && ["button", "link", "menuitem", "option", "tab"].includes(role)) return true;
    return false;
  }
  function isInteractiveOrSmall(el) {
    if (isInteractive(el)) return true;
    const rect = el.getBoundingClientRect();
    if (rect.width < 400 && rect.height < 200) return true;
    return false;
  }
  function depth(el) {
    let d = 0;
    let p = el.parentElement;
    while (p) {
      d++;
      p = p.parentElement;
    }
    return d;
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
    const backoffs = [50, 80, 120, 180, 250, 350, 500, 650, 800];
    let attempt = 0;
    while (Date.now() - startTime < timeout) {
      try {
        const elements = findElements(selector);
        if (elements.length > 0) {
          const element = elements[0];
          if (matchedText) {
            element.__stakTrakMatchedText = matchedText;
          }
          return element;
        }
      } catch (error) {
        if (!__stakWarned[selector]) {
          console.warn("[staktrak] resolution error", selector, error);
          __stakWarned[selector] = true;
        }
      }
      const delay = backoffs[Math.min(attempt, backoffs.length - 1)];
      attempt++;
      await new Promise((r) => setTimeout(r, delay));
    }
    if (!__stakWarned[selector]) {
      console.warn("[staktrak] highlight failed: not found", selector);
      __stakWarned[selector] = true;
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
    }
  }
  function isElementVisible(element) {
    const style = window.getComputedStyle(element);
    return style.display !== "none" && style.visibility !== "hidden" && style.opacity !== "0" && element.getBoundingClientRect().width > 0 && element.getBoundingClientRect().height > 0;
  }
  function getActionDescription(action) {
    var _a, _b;
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
        return `Set viewport size to ${(_a = action.options) == null ? void 0 : _a.width}x${(_b = action.options) == null ? void 0 : _b.height}`;
      case "waitForTimeout" /* WAIT_FOR_TIMEOUT */:
        return `Wait ${action.value}ms`;
      case "waitForLoadState" /* WAIT_FOR_LOAD_STATE */:
        return "Wait for page to load";
      case "waitForSelector" /* WAIT_FOR_SELECTOR */:
        return `Wait for element: ${action.selector}`;
      case "waitForURL" /* WAIT_FOR_URL */:
        return `Wait for URL: ${action.value}`;
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
      try {
        if (typeof window.__stakTrakInitAnnotations === "function") {
          setTimeout(() => {
            try {
              window.__stakTrakInitAnnotations();
              console.log("[staktrak-replay] Annotations initialized for replay");
            } catch (e) {
              console.warn("[staktrak-replay] Failed to initialize annotations:", e);
            }
          }, 100);
        }
      } catch (e) {
        console.warn("[staktrak-replay] Failed to initialize annotations:", e);
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
      setTimeout(() => {
        executeNextPlaywrightAction();
      }, 200);
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
    try {
      if (!window.__stakTrakHistoryInstrumented) {
        const fire = () => {
          try {
            const detail = { href: window.location.href, path: window.location.pathname, ts: Date.now() };
            const ev = new CustomEvent("staktrak-history-change", { detail });
            window.dispatchEvent(ev);
          } catch (e) {
          }
        };
        const origPush = history.pushState;
        const origReplace = history.replaceState;
        history.pushState = function(...args) {
          const ret = origPush.apply(this, args);
          setTimeout(fire, 0);
          return ret;
        };
        history.replaceState = function(...args) {
          const ret = origReplace.apply(this, args);
          setTimeout(fire, 0);
          return ret;
        };
        window.addEventListener("popstate", fire, { passive: true });
        setTimeout(fire, 0);
        window.__stakTrakHistoryInstrumented = true;
      }
    } catch (e) {
    }
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

  // src/actionModel.ts
  function resultsToActions(results) {
    var _a;
    const actions = [];
    const navigations = (results.pageNavigation || []).slice().sort((a, b) => a.timestamp - b.timestamp);
    const normalize = (u) => {
      var _a2;
      try {
        const url = new URL(u, ((_a2 = results.userInfo) == null ? void 0 : _a2.url) || "http://localhost");
        return url.origin + url.pathname.replace(/\/$/, "");
      } catch (e) {
        return u.replace(/[?#].*$/, "").replace(/\/$/, "");
      }
    };
    for (const nav of navigations) {
      actions.push({ kind: "nav", timestamp: nav.timestamp, url: nav.url, normalizedUrl: normalize(nav.url) });
    }
    const clicks = ((_a = results.clicks) == null ? void 0 : _a.clickDetails) || [];
    for (let i = 0; i < clicks.length; i++) {
      const cd = clicks[i];
      actions.push({
        kind: "click",
        timestamp: cd.timestamp,
        locator: {
          primary: cd.selectors.stabilizedPrimary || cd.selectors.primary,
          fallbacks: cd.selectors.fallbacks || [],
          role: cd.selectors.role,
          text: cd.selectors.text,
          tagName: cd.selectors.tagName,
          stableSelector: cd.selectors.stabilizedPrimary || cd.selectors.primary,
          candidates: cd.selectors.scores || void 0
        }
      });
      const nav = navigations.find((n) => n.timestamp > cd.timestamp && n.timestamp - cd.timestamp < 1800);
      if (nav) {
        actions.push({
          kind: "waitForUrl",
          timestamp: nav.timestamp - 1,
          // ensure ordering between click and nav
          expectedUrl: nav.url,
          normalizedUrl: normalize(nav.url),
          navRefTimestamp: nav.timestamp
        });
      }
    }
    if (results.inputChanges) {
      for (const input of results.inputChanges) {
        if (input.action === "complete" || !input.action) {
          actions.push({
            kind: "input",
            timestamp: input.timestamp,
            locator: { primary: input.elementSelector, stableSelector: input.elementSelector, fallbacks: [] },
            value: input.value
          });
        }
      }
    }
    if (results.formElementChanges) {
      for (const fe of results.formElementChanges) {
        actions.push({
          kind: "form",
          timestamp: fe.timestamp,
          locator: { primary: fe.elementSelector, stableSelector: fe.elementSelector, fallbacks: [] },
          formType: fe.type,
          value: fe.value,
          checked: fe.checked
        });
      }
    }
    if (results.assertions) {
      for (const asrt of results.assertions) {
        actions.push({
          kind: "assertion",
          timestamp: asrt.timestamp,
          locator: { primary: asrt.selector, stableSelector: asrt.selector, fallbacks: [] },
          value: asrt.value
        });
      }
    }
    actions.sort((a, b) => a.timestamp - b.timestamp || weightOrder(a.kind) - weightOrder(b.kind));
    refineLocators(actions);
    return actions;
  }
  function weightOrder(kind) {
    switch (kind) {
      case "click":
        return 1;
      case "waitForUrl":
        return 2;
      case "nav":
        return 3;
      default:
        return 4;
    }
  }
  function refineLocators(actions) {
    if (typeof document === "undefined") return;
    const seen = /* @__PURE__ */ new Set();
    for (const a of actions) {
      if (!a.locator) continue;
      const { primary, fallbacks } = a.locator;
      const validated = [];
      if (isUnique(primary)) validated.push(primary);
      for (const fb of fallbacks) {
        if (validated.length >= 3) break;
        if (isUnique(fb)) validated.push(fb);
      }
      if (validated.length === 0) continue;
      a.locator.primary = validated[0];
      a.locator.fallbacks = validated.slice(1);
      const key = a.locator.primary + "::" + a.kind;
      if (seen.has(key) && a.locator.fallbacks.length > 0) {
        a.locator.primary = a.locator.fallbacks[0];
        a.locator.fallbacks = a.locator.fallbacks.slice(1);
      }
      seen.add(a.locator.primary + "::" + a.kind);
    }
  }
  function isUnique(sel) {
    if (!sel || /^(html|body|div|span|p|button|input)$/i.test(sel)) return false;
    try {
      const nodes = document.querySelectorAll(sel);
      return nodes.length === 1;
    } catch (e) {
      return false;
    }
  }

  // src/scenario.ts
  function buildScenario(results, actions) {
    var _a, _b, _c, _d, _e;
    const startedAt = ((_a = results == null ? void 0 : results.time) == null ? void 0 : _a.startedAt) || (((_b = actions[0]) == null ? void 0 : _b.timestamp) || Date.now());
    const completedAt = ((_c = results == null ? void 0 : results.time) == null ? void 0 : _c.completedAt) || (((_d = actions[actions.length - 1]) == null ? void 0 : _d.timestamp) || startedAt);
    return {
      version: 1,
      meta: {
        baseOrigin: typeof window !== "undefined" ? window.location.origin : "",
        startedAt,
        completedAt,
        durationMs: completedAt - startedAt,
        userAgent: typeof navigator !== "undefined" ? navigator.userAgent : void 0,
        viewport: typeof window !== "undefined" ? { width: window.innerWidth, height: window.innerHeight } : void 0,
        url: (_e = results == null ? void 0 : results.userInfo) == null ? void 0 : _e.url
      },
      actions
    };
  }
  function serializeScenario(s) {
    return JSON.stringify(s);
  }

  // src/playwright-generator.ts
  function escapeTextForAssertion(text) {
    if (!text) return "";
    return text.replace(/\\/g, "\\\\").replace(/'/g, "\\'").replace(/\n/g, "\\n").replace(/\r/g, "\\r").replace(/\t/g, "\\t").trim();
  }
  function normalizeText(t) {
    return (t || "").trim();
  }
  function locatorToSelector(l) {
    if (!l) return 'page.locator("body")';
    const primary = l.stableSelector || l.primary;
    if (/\[data-staktrak-id=/.test(primary)) {
      const m = primary.match(/\[data-staktrak-id=["']([^"']+)["']\]/);
      if (m) return `page.locator('[data-staktrak-id="${escapeTextForAssertion(m[1])}"]')`;
    }
    if (/\[data-testid=/.test(primary)) {
      const m = primary.match(/\[data-testid=["']([^"']+)["']\]/);
      if (m) return `page.getByTestId('${escapeTextForAssertion(m[1])}')`;
    }
    if (primary.startsWith("#") && /^[a-zA-Z][\w-]*$/.test(primary.slice(1)))
      return `page.locator('${primary}')`;
    if (/^[a-zA-Z]+\.[a-zA-Z0-9_-]+/.test(primary)) {
      return `page.locator('${primary}')`;
    }
    if (l.role && l.text) {
      const txt = normalizeText(l.text);
      if (txt && txt.length <= 50)
        return `page.getByRole('${l.role}', { name: '${escapeTextForAssertion(txt)}' })`;
    }
    if (l.text && l.text.length <= 30 && l.text.length > 1)
      return `page.getByText('${escapeTextForAssertion(normalizeText(l.text))}')`;
    if (primary && !primary.startsWith("page."))
      return `page.locator('${primary}')`;
    for (const fb of l.fallbacks) {
      if (fb && /\[data-staktrak-id=/.test(fb)) {
        const m = fb.match(/\[data-staktrak-id=["']([^"']+)["']\]/);
        if (m) return `page.locator('[data-staktrak-id="${escapeTextForAssertion(m[1])}"]')`;
      }
      if (fb && /\[data-testid=/.test(fb)) {
        const m = fb.match(/\[data-testid=["']([^"']+)["']\]/);
        if (m) return `page.getByTestId('${escapeTextForAssertion(m[1])}')`;
      }
      if (fb && !/^[a-zA-Z]+$/.test(fb)) return `page.locator('${fb}')`;
    }
    return 'page.locator("body")';
  }
  function generatePlaywrightTestFromActions(actions, options) {
    const name = options.testName || "Recorded flow";
    const viewport = options.viewport || { width: 1280, height: 720 };
    let body = "";
    let lastTs = null;
    const base = options.baseUrl ? options.baseUrl.replace(/\/$/, "") : "";
    function fullUrl(u) {
      if (!u) return "";
      if (/^https?:/i.test(u)) return u;
      if (u.startsWith("/")) return base + u;
      return base + "/" + u;
    }
    let i = 0;
    const collapsed = [];
    for (let k = 0; k < actions.length; k++) {
      const curr = actions[k];
      const prev = collapsed[collapsed.length - 1];
      if (curr.kind === "nav" && prev && prev.kind === "nav" && prev.url === curr.url) continue;
      collapsed.push(curr);
    }
    actions = collapsed;
    while (i < actions.length) {
      const a = actions[i];
      if (a.kind === "click" && i + 1 < actions.length) {
        const nxt = actions[i + 1];
        if (nxt.kind === "nav" && nxt.timestamp - a.timestamp < 1500) {
          if (lastTs != null) {
            const delta = Math.max(0, a.timestamp - lastTs);
            const wait = Math.min(3e3, Math.max(100, delta));
            if (wait > 400) body += `  await page.waitForTimeout(${wait});
`;
          }
          body += `  await Promise.all([
`;
          body += `    page.waitForURL('${fullUrl(nxt.url)}'),
`;
          body += `    ${locatorToSelector(a.locator)}.click()
`;
          body += `  ]);
`;
          lastTs = nxt.timestamp;
          i += 2;
          continue;
        }
      }
      if (lastTs != null) {
        const delta = Math.max(0, a.timestamp - lastTs);
        const wait = Math.min(3e3, Math.max(100, delta));
        if (wait > 500) body += `  await page.waitForTimeout(${wait});
`;
      }
      switch (a.kind) {
        case "nav": {
          const target = fullUrl(a.url);
          if (i === 0) {
            body += `  await page.goto('${target}');
`;
          } else {
            body += `  await page.waitForURL('${target}');
`;
          }
          break;
        }
        case "click":
          body += `  await ${locatorToSelector(a.locator)}.click();
`;
          break;
        case "input":
          body += `  await ${locatorToSelector(a.locator)}.fill('${escapeTextForAssertion(a.value || "")}');
`;
          break;
        case "form":
          if (a.formType === "checkbox" || a.formType === "radio") {
            body += a.checked ? `  await ${locatorToSelector(a.locator)}.check();
` : `  await ${locatorToSelector(a.locator)}.uncheck();
`;
          } else if (a.formType === "select") {
            body += `  await ${locatorToSelector(a.locator)}.selectOption('${escapeTextForAssertion(a.value || "")}');
`;
          }
          break;
        case "assertion":
          if (a.value && a.value.length > 0) {
            body += `  await expect(${locatorToSelector(a.locator)}).toContainText('${escapeTextForAssertion(a.value)}');
`;
          } else {
            body += `  await expect(${locatorToSelector(a.locator)}).toBeVisible();
`;
          }
          break;
      }
      lastTs = a.timestamp;
      i++;
    }
    return `import { test, expect } from '@playwright/test'

test('${name}', async ({ page }) => {
  await page.setViewportSize({ width: ${viewport.width}, height: ${viewport.height} })
${body.split("\n").filter((l) => l.trim()).map((l) => l).join("\n")}
})
`;
  }
  if (typeof window !== "undefined") {
    const existing = window.PlaywrightGenerator || {};
    existing.generatePlaywrightTestFromActions = generatePlaywrightTestFromActions;
    existing.locatorToSelector = locatorToSelector;
    existing.generatePlaywrightTest = (baseUrl, results) => {
      try {
        const actions = resultsToActions(results);
        return generatePlaywrightTestFromActions(actions, { baseUrl });
      } catch (e) {
        console.warn("PlaywrightGenerator.generatePlaywrightTest failed", e);
        return "";
      }
    };
    window.PlaywrightGenerator = existing;
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
      annotateInteractiveElements();
      startDynamicAnnotation();
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
      const hashHandler = () => {
        recordStateChange("hashchange");
      };
      window.addEventListener("hashchange", hashHandler);
      this.memory.alwaysListeners.push(
        () => window.removeEventListener("hashchange", hashHandler)
      );
      const anchorClickHandler = (e) => {
        const a = e.target.closest("a");
        if (!a) return;
        if (a.target && a.target !== "_self") return;
        const href = a.getAttribute("href");
        if (!href) return;
        try {
          const dest = new URL(href, window.location.href);
          if (dest.origin === window.location.origin) {
            this.results.pageNavigation.push({ type: "anchorClick", url: dest.href, timestamp: getTimeStamp() });
          }
        } catch (e2) {
        }
      };
      document.addEventListener("click", anchorClickHandler, true);
      this.memory.alwaysListeners.push(
        () => document.removeEventListener("click", anchorClickHandler, true)
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
      stopDynamicAnnotation();
      cleanupAnnotations();
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
    getAnnotationStats() {
      return getAnnotationStats();
    }
    annotateCurrentPage() {
      const result = annotateInteractiveElements();
      return result;
    }
    cleanupCurrentAnnotations() {
      const result = cleanupAnnotations();
      return result;
    }
    startDynamicAnnotation() {
      return startDynamicAnnotation();
    }
    stopDynamicAnnotation() {
      return stopDynamicAnnotation();
    }
    getDynamicAnnotationStatus() {
      return getDynamicAnnotationStatus();
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
  userBehaviour.getActions = () => resultsToActions(userBehaviour.result());
  userBehaviour.generatePlaywrightTest = (options) => {
    const actions = resultsToActions(userBehaviour.result());
    const code = generatePlaywrightTestFromActions(actions, options);
    userBehaviour._lastGeneratedUsingActions = true;
    return code;
  };
  userBehaviour.exportSession = (options) => {
    const actions = resultsToActions(userBehaviour.result());
    const test = generatePlaywrightTestFromActions(actions, options);
    userBehaviour._lastGeneratedUsingActions = true;
    return { actions, test };
  };
  userBehaviour.getAnnotationStats = () => userBehaviour.getAnnotationStats();
  userBehaviour.annotateCurrentPage = () => userBehaviour.annotateCurrentPage();
  userBehaviour.cleanupCurrentAnnotations = () => userBehaviour.cleanupCurrentAnnotations();
  userBehaviour.startDynamicAnnotation = () => userBehaviour.startDynamicAnnotation();
  userBehaviour.stopDynamicAnnotation = () => userBehaviour.stopDynamicAnnotation();
  userBehaviour.getDynamicAnnotationStatus = () => userBehaviour.getDynamicAnnotationStatus();
  userBehaviour.ensureElementId = ensureElementId;
  userBehaviour.getScenario = () => {
    const results = userBehaviour.result();
    const actions = resultsToActions(results);
    return buildScenario(results, actions);
  };
  userBehaviour.exportScenarioJSON = () => {
    const sc = userBehaviour.getScenario();
    return serializeScenario(sc);
  };
  userBehaviour.getSelectorScores = () => {
    const results = userBehaviour.result();
    if (!results.clicks || !results.clicks.clickDetails.length) return [];
    const last = results.clicks.clickDetails[results.clicks.clickDetails.length - 1];
    const sel = last.selectors;
    if (sel && sel.scores) return sel.scores;
    return [];
  };
  window.__stakTrakInitAnnotations = () => {
    try {
      annotateInteractiveElements();
      startDynamicAnnotation();
    } catch (e) {
      console.warn("[staktrak] Failed to initialize replay annotations:", e);
    }
  };
  var index_default = userBehaviour;
  return __toCommonJS(index_exports);
})();
