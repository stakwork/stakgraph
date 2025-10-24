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
  var __objRest = (source, exclude) => {
    var target = {};
    for (var prop in source)
      if (__hasOwnProp.call(source, prop) && exclude.indexOf(prop) < 0)
        target[prop] = source[prop];
    if (source != null && __getOwnPropSymbols)
      for (var prop of __getOwnPropSymbols(source)) {
        if (exclude.indexOf(prop) < 0 && __propIsEnum.call(source, prop))
          target[prop] = source[prop];
      }
    return target;
  };
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
  var src_exports = {};
  __export(src_exports, {
    default: () => src_default
  });

  // src/utils.ts
  var getTimeStamp = () => Date.now();
  var getElementRole = (el) => {
    const explicit = el.getAttribute("role");
    if (explicit)
      return explicit;
    const tag = el.tagName.toLowerCase();
    if (tag === "button")
      return "button";
    if (tag === "a" && el.hasAttribute("href"))
      return "link";
    if (tag === "input") {
      const type = (el.getAttribute("type") || "text").toLowerCase();
      if (["button", "submit", "reset"].includes(type))
        return "button";
      if (type === "checkbox")
        return "checkbox";
      if (type === "radio")
        return "radio";
      return "textbox";
    }
    if (tag === "select")
      return "combobox";
    if (tag === "textarea")
      return "textbox";
    if (tag === "nav")
      return "navigation";
    if (tag === "header")
      return "banner";
    if (tag === "footer")
      return "contentinfo";
    if (tag === "main")
      return "main";
    if (tag === "form")
      return "form";
    return null;
  };
  var getEnhancedElementText = (element) => {
    var _a2;
    const ariaLabel = element.getAttribute("aria-label");
    if (ariaLabel)
      return ariaLabel;
    const resolvedLabel = resolveAriaLabelledBy(element);
    if (resolvedLabel)
      return resolvedLabel;
    const tag = element.tagName.toLowerCase();
    if (tag === "button" || tag === "a" && element.hasAttribute("href")) {
      const text = (_a2 = element.textContent) == null ? void 0 : _a2.trim();
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
    var _a2;
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
    const text = (_a2 = element.textContent) == null ? void 0 : _a2.trim();
    if (text && text.length <= 2 && /[\u{1F300}-\u{1F9FF}]|[\u{2600}-\u{26FF}]|[\u{2700}-\u{27BF}]/u.test(text)) {
      return { type: "emoji", selector: `text="${text}"` };
    }
    return null;
  };
  var resolveAriaLabelledBy = (element) => {
    var _a2;
    const labelledBy = element.getAttribute("aria-labelledby");
    if (!labelledBy)
      return null;
    const ids = labelledBy.split(" ").filter((id) => id.trim());
    const texts = [];
    for (const id of ids) {
      const referencedEl = findElementById(element.ownerDocument || document, id);
      if (referencedEl) {
        const text = (_a2 = referencedEl.textContent) == null ? void 0 : _a2.trim();
        if (text)
          texts.push(text);
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
    var _a2;
    const htmlEl = element;
    const tagName = element.tagName.toLowerCase();
    const fallbacks = [];
    const reasonsMap = {};
    const scored = [];
    const pushCandidate = (sel, baseScore, reason) => {
      if (!sel)
        return;
      if (!reasonsMap[sel])
        reasonsMap[sel] = [];
      reasonsMap[sel].push(reason);
      let score = baseScore;
      if (sel.length > 60)
        score -= Math.min(20, Math.floor((sel.length - 60) / 5));
      const depth2 = sel.split(">").length - 1;
      if (depth2 > 3)
        score -= (depth2 - 3) * 2;
      if (/\.[a-zA-Z0-9_-]*[0-9a-f]{6,}\b/.test(sel))
        score -= 25;
      if (/^\w+$/.test(sel))
        score -= 30;
      if (sel.startsWith("text="))
        score -= 5;
      if (sel.startsWith("role:"))
        score -= 3;
      scored.push({ selector: sel, score, reasons: reasonsMap[sel] });
    };
    const finalizeReturn = (primary2, fallbacks2, extra) => {
      const dedup = {};
      for (const c of scored) {
        if (!dedup[c.selector] || dedup[c.selector].score < c.score)
          dedup[c.selector] = c;
      }
      const ordered = Object.values(dedup).sort((a, b) => b.score - a.score);
      return __spreadProps(__spreadValues({ primary: primary2, fallbacks: fallbacks2 }, extra), { scores: ordered });
    };
    const testId = (_a2 = htmlEl.dataset) == null ? void 0 : _a2.testid;
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
    if (cleanText.length === 0 || cleanText.length > 50)
      return null;
    if (tagName === "button" || tagName === "a" || getElementRole(element) === "button") {
      return `text=${cleanText}`;
    }
    return null;
  };
  var generateClassBasedSelector = (element) => {
    const tagName = element.tagName.toLowerCase();
    const classList = element.classList;
    if (!classList.length)
      return tagName;
    const safeClasses = Array.from(classList).filter((cls) => {
      if (cls.includes("_") && cls.match(/[0-9a-f]{6}/))
        return false;
      if (cls.includes("module__"))
        return false;
      if (cls.includes("emotion-"))
        return false;
      if (cls.includes("css-"))
        return false;
      if (cls.length > 30)
        return false;
      return /^[a-zA-Z][a-zA-Z0-9-]*$/.test(cls);
    });
    if (safeClasses.length === 0)
      return tagName;
    const limitedClasses = safeClasses.slice(0, 3);
    return `${tagName}.${limitedClasses.join(".")}`;
  };
  var generateContextualSelector = (element) => {
    const tagName = element.tagName.toLowerCase();
    const parent = element.parentElement;
    if (!parent)
      return null;
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
      if (parts.length > 10)
        break;
    }
    return "/" + parts.join("/");
  };
  var createClickDetail = (e) => {
    const target = e.target;
    const selectors = generateSelectorStrategies(target);
    const html = target;
    const testId = html.dataset && html.dataset["testid"] || void 0;
    const id = html.id || void 0;
    const accessibleName = getEnhancedElementText(html) || void 0;
    let nth;
    if (html.parentElement) {
      const same = Array.from(html.parentElement.children).filter((c) => c.tagName === html.tagName);
      if (same.length > 1)
        nth = same.indexOf(html) + 1;
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
    const selAny = selectors;
    selAny.id = id;
    selAny.testId = testId;
    selAny.accessibleName = accessibleName;
    if (nth)
      selAny.nth = nth;
    if (ancestors.length)
      selAny.ancestors = ancestors;
    const stabilized = chooseStablePrimary(html, selectors.primary, selectors.fallbacks, {
      testId,
      id,
      accessibleName,
      role: getElementRole(html) || void 0,
      nth
    });
    let uniqueStabilized = ensureStabilizedUnique(html, stabilized);
    try {
      if (typeof document !== "undefined" && !uniqueStabilized.startsWith("text=")) {
        const matches = document.querySelectorAll(uniqueStabilized);
        if (matches.length !== 1) {
          const ancestorOnly = buildAncestorNthSelector(html);
          if (ancestorOnly && ancestorOnly !== uniqueStabilized) {
            const mm = document.querySelectorAll(ancestorOnly);
            if (mm.length === 1)
              uniqueStabilized = ancestorOnly;
          }
        }
      }
    } catch (e2) {
    }
    selectors.stabilizedPrimary = uniqueStabilized;
    selectors.primary = uniqueStabilized;
    let visualSelector = null;
    const isCssResolvable = (s) => !s.startsWith("text=") && !s.startsWith("role:");
    if (isCssResolvable(uniqueStabilized))
      visualSelector = uniqueStabilized;
    else {
      const fbCss = (selectors.fallbacks || []).find(isCssResolvable);
      if (fbCss)
        visualSelector = fbCss;
      else {
        const anc = buildAncestorNthSelector(html);
        if (anc)
          visualSelector = anc;
      }
    }
    if (visualSelector)
      selectors.visualSelector = visualSelector;
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
      if (value)
        attrs[attr] = value;
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
    if (!clickDetails.length)
      return [];
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
      if (!clicksBySelector[key])
        clicksBySelector[key] = [];
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
  var isWeakSelector = (selector, el) => {
    if (!selector)
      return true;
    if (selector.startsWith("[data-testid="))
      return false;
    if (selector.startsWith("#"))
      return false;
    if (selector.startsWith("text="))
      return false;
    if (/^\w+$/.test(selector))
      return true;
    if (/^\w+\.[^.]+$/.test(selector)) {
      if (el && typeof document !== "undefined") {
        try {
          const count = document.querySelectorAll(selector).length;
          if (count === 1)
            return false;
        } catch (e) {
        }
      }
      return true;
    }
    return false;
  };
  var chooseStablePrimary = (el, current, fallbacks, meta) => {
    if (!isWeakSelector(current, el))
      return current;
    if (meta.testId)
      return `[data-testid="${meta.testId}"]`;
    if (meta.id && /^[a-zA-Z][\w-]*$/.test(meta.id))
      return `#${meta.id}`;
    if (typeof document !== "undefined") {
      const structural = [current, ...fallbacks].filter((s) => s && !s.startsWith("text=") && !s.startsWith("[") && !s.startsWith("#"));
      for (const s of structural) {
        try {
          if (document.querySelectorAll(s).length === 1) {
            return s;
          }
        } catch (e) {
        }
      }
    }
    if (meta.role && meta.accessibleName && meta.accessibleName.length < 60) {
      return `role:${meta.role}[name="${meta.accessibleName.replace(/"/g, '\\"')}"]`;
    }
    if (meta.accessibleName && meta.accessibleName.length < 40) {
      return `text=${meta.accessibleName.replace(/"/g, '\\"')}:exact`;
    }
    return current;
  };
  function isSelectorUnique(sel) {
    if (typeof document === "undefined")
      return false;
    try {
      const n = document.querySelectorAll(sel);
      return n.length === 1;
    } catch (e) {
      return false;
    }
  }
  function buildAncestorNthSelector(el) {
    if (!el.parentElement)
      return null;
    const path = [];
    let current = el;
    let depth2 = 0;
    while (current && depth2 < 6) {
      const tag = current.tagName.toLowerCase();
      let part = tag;
      const cur = current;
      if (cur && cur.parentElement) {
        const same = Array.from(cur.parentElement.children).filter((c) => c.tagName === cur.tagName);
        if (same.length > 1) {
          const idx = same.indexOf(cur) + 1;
          part += `:nth-of-type(${idx})`;
        }
      }
      path.unshift(part);
      const selector = path.join(" > ");
      if (isSelectorUnique(selector))
        return selector;
      current = current.parentElement;
      depth2++;
    }
    const withBody = "body > " + path.join(" > ");
    if (isSelectorUnique(withBody))
      return withBody;
    return null;
  }
  function ensureStabilizedUnique(html, stabilized) {
    if (stabilized.startsWith("#") || stabilized.startsWith("[data-testid="))
      return stabilized;
    if (stabilized.startsWith("role:")) {
      try {
        const el = findByRoleLike(stabilized);
        if (el)
          return stabilized;
      } catch (e) {
      }
    }
    if (stabilized.startsWith("text=") && stabilized.endsWith(":exact")) {
      const exactTxt = stabilized.slice("text=".length, -":exact".length);
      try {
        const candidates = Array.from(document.querySelectorAll("button, a, [role], input, textarea, select")).filter((e) => (e.textContent || "").trim() === exactTxt);
        if (candidates.length === 1)
          return stabilized;
      } catch (e) {
      }
    }
    if (isSelectorUnique(stabilized))
      return stabilized;
    const ancestor = buildAncestorNthSelector(html);
    if (ancestor && ancestor.length < 180)
      return ancestor;
    return stabilized;
  }
  function findByRoleLike(sel) {
    if (!sel.startsWith("role:"))
      return null;
    const m = sel.match(/^role:([^\[]+)(?:\[name="(.+?)"\])?/);
    if (!m)
      return null;
    const role = m[1];
    const name = m[2];
    const candidates = Array.from(document.querySelectorAll("*")).filter((el) => {
      const r = el.getAttribute("role") || inferRole(el);
      return r === role;
    });
    if (!name)
      return candidates[0] || null;
    return candidates.find((c) => (c.textContent || "").trim() === name) || null;
  }
  function inferRole(el) {
    const tag = el.tagName.toLowerCase();
    if (tag === "button")
      return "button";
    if (tag === "a" && el.hasAttribute("href"))
      return "link";
    if (tag === "input")
      return "textbox";
    return null;
  }
  function getRelativeUrl(url) {
    if (!url)
      return "/";
    let pathname;
    let search = "";
    let hash = "";
    try {
      const urlObj = new URL(url);
      pathname = urlObj.pathname;
      search = urlObj.search;
      hash = urlObj.hash;
    } catch (e) {
      if (!url.startsWith("/")) {
        return url;
      }
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
    const workspacePattern = /^\/w\/[a-zA-Z0-9_-]+/;
    pathname = pathname.replace(workspacePattern, "");
    if (!pathname) {
      pathname = "/";
    }
    return pathname + search + hash;
  }

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
    var _a2, _b, _c;
    try {
      const fiberKey = Object.keys(element).find(
        (key) => key.startsWith("__reactFiber$") || key.startsWith("__reactInternalInstance$")
      );
      if (!fiberKey) {
        return null;
      }
      let fiber = element[fiberKey];
      let level = 0;
      const maxTraversalDepth = Number((_a2 = window.STAKTRAK_CONFIG) == null ? void 0 : _a2.maxTraversalDepth) || 10;
      const extractSource = (source) => {
        if (!source)
          return null;
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
    var _a2, _b;
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
          if (!processedFiles.has(dataSource) || !((_a2 = processedFiles.get(dataSource)) == null ? void 0 : _a2.has(lineNum))) {
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
        if (components.length === 0)
          return void 0;
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
    var _a2, _b;
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
            if (peek.endsWith("]);"))
              break;
          }
          const block = blockLines.join(" ");
          const url = (_a2 = block.match(/page\.waitForURL\(['\"](.*?)['\"]\)/)) == null ? void 0 : _a2[1];
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
    var _a2, _b;
    const actualSelector = selector || `variable:${varName}`;
    switch (method) {
      case "click":
        return { type: "click", selector: actualSelector, comment, lineNumber };
      case "fill":
        const fillValue = ((_a2 = args.match(/['"](.*?)['"]/)) == null ? void 0 : _a2[1]) || "";
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
    if (labelMatch)
      return `getByLabel:${labelMatch[1]}`;
    const placeholderMatch = locatorCall.match(
      /getByPlaceholder\(['"](.*?)['"]\)/
    );
    if (placeholderMatch)
      return `getByPlaceholder:${placeholderMatch[1]}`;
    const testIdMatch = locatorCall.match(/getByTestId\(['"](.*?)['"]\)/);
    if (testIdMatch)
      return `getByTestId:${testIdMatch[1]}`;
    const titleMatch = locatorCall.match(/getByTitle\(['"](.*?)['"]\)/);
    if (titleMatch)
      return `getByTitle:${titleMatch[1]}`;
    const altMatch = locatorCall.match(/getByAltText\(['"](.*?)['"]\)/);
    if (altMatch)
      return `getByAltText:${altMatch[1]}`;
    const locatorMatch = locatorCall.match(/locator\(['"](.*?)['"]\)/);
    if (locatorMatch)
      return locatorMatch[1];
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
    if (chainCall.includes("first()"))
      return `${baseSelector}:first`;
    if (chainCall.includes("last()"))
      return `${baseSelector}:last`;
    const nthMatch = chainCall.match(/nth\((\d+)\)/);
    if (nthMatch)
      return `${baseSelector}:nth(${nthMatch[1]})`;
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
    if (locatorChainMatch)
      return `${baseSelector} >> ${locatorChainMatch[1]}`;
    return `${baseSelector}:${chainCall}`;
  }
  function extractSelectorFromArg(selectorArg) {
    return selectorArg.trim().replace(/^['"]|['"]$/g, "");
  }
  function parseDirectAction(method, args, comment, lineNumber, selector) {
    var _a2, _b;
    switch (method) {
      case "click":
        return { type: "click", selector, comment, lineNumber };
      case "fill":
        const fillValue = ((_a2 = args.match(/['"](.*?)['"]/)) == null ? void 0 : _a2[1]) || "";
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
      if (last.text)
        htmlElement.setAttribute("data-staktrak-matched-text", last.text);
    }
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
    var _a2;
    try {
      switch (action.type) {
        case "goto" /* GOTO */:
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
                lastPulse = Date.now();
              }
              await new Promise((r) => setTimeout(r, 120));
              if (tryMatch())
                break;
            }
            stopSignals.forEach((fn) => {
              try {
                fn();
              } catch (e) {
              }
            });
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
            if (((_a2 = action.options) == null ? void 0 : _a2.state) === "visible") {
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
              const rect = element.getBoundingClientRect();
              const isVisible = rect.top >= 0 && rect.left >= 0 && rect.bottom <= window.innerHeight && rect.right <= window.innerWidth;
              if (!isVisible) {
                element.scrollIntoView({
                  behavior: "smooth",
                  block: "nearest",
                  inline: "nearest"
                });
              }
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
    var _a2, _b;
    if (!selector || selector.trim() === "")
      return null;
    try {
      if ((selector.startsWith("text=") || selector.startsWith("role:")) && window.__stakTrakSelectorMap) {
        const map = window.__stakTrakSelectorMap;
        const entry = map[selector];
        if (entry == null ? void 0 : entry.visualSelector) {
          try {
            const cssEl = document.querySelector(entry.visualSelector);
            if (cssEl)
              return cssEl;
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
        if (text)
          el.__stakTrakMatchedText = text;
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
      if (exact.length === 1)
        return noteMatch(exact[0], selector, norm);
      if (exact.length > 1) {
        const deepest = exact.sort((a, b) => depth(b) - depth(a))[0];
        return noteMatch(deepest, selector, norm);
      }
      return noteMatch(null, selector);
    }
    if (selector.startsWith("getByTestId:")) {
      const val = selector.substring("getByTestId:".length);
      return noteMatch(document.querySelector(`[data-testid="${cssEscape(val)}"]`), `[data-testid="${val}"]`);
    }
    if (selector.startsWith("getByText-regex:")) {
      const body = selector.substring("getByText-regex:".length);
      const rx = body.match(/^\/(.*)\/(.*)$/);
      let r = null;
      if (rx)
        try {
          r = new RegExp(rx[1], rx[2]);
        } catch (e) {
        }
      const all = textSearchCandidates();
      for (const el of all) {
        const txt = ((_a2 = el.textContent) == null ? void 0 : _a2.trim()) || "";
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
        var _a3;
        return ((_a3 = l.textContent) == null ? void 0 : _a3.trim()) === label;
      });
      for (const lab of labels) {
        const forId = lab.getAttribute("for");
        if (forId) {
          const ctl = document.getElementById(forId);
          if (ctl)
            return noteMatch(ctl, selector);
        }
        const nested = lab.querySelector("input,select,textarea,button");
        if (nested)
          return noteMatch(nested, selector);
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
      if (element)
        return noteMatch(element, browserSelector);
    }
    const strategies = [
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
    var _a2;
    const aria = el.getAttribute("aria-label");
    if (aria)
      return aria.trim();
    const labelled = el.getAttribute("aria-labelledby");
    if (labelled) {
      const parts = labelled.split(/\s+/).map((id) => {
        var _a3, _b;
        return (_b = (_a3 = document.getElementById(id)) == null ? void 0 : _a3.textContent) == null ? void 0 : _b.trim();
      }).filter(Boolean);
      if (parts.length)
        return parts.join(" ");
    }
    const tag = el.tagName.toLowerCase();
    if (tag === "input" || tag === "textarea") {
      const val = el.value || el.getAttribute("placeholder");
      if (val)
        return val.trim();
    }
    const txt = (_a2 = el.textContent) == null ? void 0 : _a2.trim();
    if (txt)
      return txt.slice(0, 120);
    const title = el.getAttribute("title");
    if (title)
      return title.trim();
    return null;
  }
  function textSearchCandidates() {
    return Array.from(document.querySelectorAll("button, a, [role], input, textarea, select, label, div, span"));
  }
  function convertToBrowserSelector(selector) {
    var _a2;
    if (!selector)
      return selector;
    if (selector.includes(":has-text(")) {
      const textMatch = selector.match(/:has-text\("([^"]+)"\)/);
      if (textMatch) {
        const text = textMatch[1];
        const tagMatch = selector.match(/^([a-zA-Z]+)/);
        const tagName = tagMatch ? tagMatch[1] : "*";
        const elements = Array.from(document.querySelectorAll(tagName));
        for (const element of elements) {
          if (((_a2 = element.textContent) == null ? void 0 : _a2.trim()) === text) {
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
    if (!selector || selector.trim() === "")
      return false;
    try {
      document.querySelector(selector);
      return true;
    } catch (e) {
      return false;
    }
  }
  function findByDataTestId(selector) {
    var _a2;
    if (!selector.includes("data-testid"))
      return null;
    const testId = (_a2 = selector.match(/data-testid="([^"]+)"/)) == null ? void 0 : _a2[1];
    if (testId) {
      return document.querySelector(`[data-testid="${testId}"]`);
    }
    return null;
  }
  function findByClassUnique(selector) {
    if (!selector.includes("."))
      return null;
    if (selector.startsWith("text=") || selector.startsWith("role:"))
      return null;
    try {
      const els = document.querySelectorAll(selector);
      if (els.length === 1)
        return els[0];
    } catch (e) {
    }
    const classOnly = selector.match(/^\w+\.[^.]+$/);
    if (classOnly) {
      const els = Array.from(document.querySelectorAll(selector));
      const interactive = els.filter(isInteractive);
      if (interactive.length === 1)
        return interactive[0];
    }
    return null;
  }
  function findById(selector) {
    if (!selector.includes("#"))
      return null;
    const ids = selector.match(/#([^\s.#\[\]]+)/g);
    if (ids && ids.length > 0) {
      const id = ids[0].substring(1);
      return document.querySelector(`#${id}`);
    }
    return null;
  }
  function findByAriaLabel(selector) {
    const ariaMatch = selector.match(/\[aria-label="([^"]+)"\]/);
    if (!ariaMatch)
      return null;
    return document.querySelector(`[aria-label="${ariaMatch[1]}"]`);
  }
  function findByRole(selector) {
    const roleMatch = selector.match(/\[role="([^"]+)"\]/);
    if (!roleMatch)
      return null;
    return document.querySelector(`[role="${roleMatch[1]}"]`);
  }
  function findByTextContentTight(selector) {
    if (!selector.startsWith("text="))
      return null;
    const exact = selector.endsWith(":exact");
    const core = exact ? selector.slice("text=".length, -":exact".length) : selector.slice("text=".length);
    const norm = core.trim();
    const candidates = textSearchCandidates().filter(isInteractiveOrSmall);
    for (const el of candidates) {
      const txt = (el.textContent || "").trim();
      if (exact && txt === norm || !exact && txt.includes(norm))
        return el;
    }
    return null;
  }
  function isInteractive(el) {
    const tag = el.tagName.toLowerCase();
    if (["button", "a", "input", "textarea", "select", "option"].includes(tag))
      return true;
    const role = el.getAttribute("role");
    if (role && ["button", "link", "menuitem", "option", "tab"].includes(role))
      return true;
    return false;
  }
  function isInteractiveOrSmall(el) {
    if (isInteractive(el))
      return true;
    const rect = el.getBoundingClientRect();
    if (rect.width < 400 && rect.height < 200)
      return true;
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
    var _a2;
    if (element.id && /^[a-zA-Z][\w-]*$/.test(element.id)) {
      const idSelector = `#${element.id}`;
      if (document.querySelectorAll(idSelector).length === 1) {
        return idSelector;
      }
    }
    const testId = (_a2 = element.dataset) == null ? void 0 : _a2.testid;
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
    if (doc.querySelector("#staktrak-highlight-styles"))
      return;
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
    var _a2, _b;
    if (!action.selector)
      return;
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
        if (!textElement || !((_a2 = textElement.textContent) == null ? void 0 : _a2.includes(String(action.value || "")))) {
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
    var _a2, _b;
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
        return `Set viewport size to ${(_a2 = action.options) == null ? void 0 : _a2.width}x${(_b = action.options) == null ? void 0 : _b.height}`;
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

  // node_modules/modern-screenshot/dist/index.mjs
  function changeJpegDpi(uint8Array, dpi) {
    uint8Array[13] = 1;
    uint8Array[14] = dpi >> 8;
    uint8Array[15] = dpi & 255;
    uint8Array[16] = dpi >> 8;
    uint8Array[17] = dpi & 255;
    return uint8Array;
  }
  var _P = "p".charCodeAt(0);
  var _H = "H".charCodeAt(0);
  var _Y = "Y".charCodeAt(0);
  var _S = "s".charCodeAt(0);
  var pngDataTable;
  function createPngDataTable() {
    const crcTable = new Int32Array(256);
    for (let n = 0; n < 256; n++) {
      let c = n;
      for (let k = 0; k < 8; k++) {
        c = c & 1 ? 3988292384 ^ c >>> 1 : c >>> 1;
      }
      crcTable[n] = c;
    }
    return crcTable;
  }
  function calcCrc(uint8Array) {
    let c = -1;
    if (!pngDataTable)
      pngDataTable = createPngDataTable();
    for (let n = 0; n < uint8Array.length; n++) {
      c = pngDataTable[(c ^ uint8Array[n]) & 255] ^ c >>> 8;
    }
    return c ^ -1;
  }
  function searchStartOfPhys(uint8Array) {
    const length = uint8Array.length - 1;
    for (let i = length; i >= 4; i--) {
      if (uint8Array[i - 4] === 9 && uint8Array[i - 3] === _P && uint8Array[i - 2] === _H && uint8Array[i - 1] === _Y && uint8Array[i] === _S) {
        return i - 3;
      }
    }
    return 0;
  }
  function changePngDpi(uint8Array, dpi, overwritepHYs = false) {
    const physChunk = new Uint8Array(13);
    dpi *= 39.3701;
    physChunk[0] = _P;
    physChunk[1] = _H;
    physChunk[2] = _Y;
    physChunk[3] = _S;
    physChunk[4] = dpi >>> 24;
    physChunk[5] = dpi >>> 16;
    physChunk[6] = dpi >>> 8;
    physChunk[7] = dpi & 255;
    physChunk[8] = physChunk[4];
    physChunk[9] = physChunk[5];
    physChunk[10] = physChunk[6];
    physChunk[11] = physChunk[7];
    physChunk[12] = 1;
    const crc = calcCrc(physChunk);
    const crcChunk = new Uint8Array(4);
    crcChunk[0] = crc >>> 24;
    crcChunk[1] = crc >>> 16;
    crcChunk[2] = crc >>> 8;
    crcChunk[3] = crc & 255;
    if (overwritepHYs) {
      const startingIndex = searchStartOfPhys(uint8Array);
      uint8Array.set(physChunk, startingIndex);
      uint8Array.set(crcChunk, startingIndex + 13);
      return uint8Array;
    } else {
      const chunkLength = new Uint8Array(4);
      chunkLength[0] = 0;
      chunkLength[1] = 0;
      chunkLength[2] = 0;
      chunkLength[3] = 9;
      const finalHeader = new Uint8Array(54);
      finalHeader.set(uint8Array, 0);
      finalHeader.set(chunkLength, 33);
      finalHeader.set(physChunk, 37);
      finalHeader.set(crcChunk, 50);
      return finalHeader;
    }
  }
  var b64PhysSignature1 = "AAlwSFlz";
  var b64PhysSignature2 = "AAAJcEhZ";
  var b64PhysSignature3 = "AAAACXBI";
  function detectPhysChunkFromDataUrl(dataUrl) {
    let b64index = dataUrl.indexOf(b64PhysSignature1);
    if (b64index === -1) {
      b64index = dataUrl.indexOf(b64PhysSignature2);
    }
    if (b64index === -1) {
      b64index = dataUrl.indexOf(b64PhysSignature3);
    }
    return b64index;
  }
  var PREFIX = "[modern-screenshot]";
  var IN_BROWSER = typeof window !== "undefined";
  var SUPPORT_WEB_WORKER = IN_BROWSER && "Worker" in window;
  var SUPPORT_ATOB = IN_BROWSER && "atob" in window;
  var SUPPORT_BTOA = IN_BROWSER && "btoa" in window;
  var _a;
  var USER_AGENT = IN_BROWSER ? (_a = window.navigator) == null ? void 0 : _a.userAgent : "";
  var IN_CHROME = USER_AGENT.includes("Chrome");
  var IN_SAFARI = USER_AGENT.includes("AppleWebKit") && !IN_CHROME;
  var IN_FIREFOX = USER_AGENT.includes("Firefox");
  var isContext = (value) => value && "__CONTEXT__" in value;
  var isCssFontFaceRule = (rule) => rule.constructor.name === "CSSFontFaceRule";
  var isCSSImportRule = (rule) => rule.constructor.name === "CSSImportRule";
  var isElementNode = (node) => node.nodeType === 1;
  var isSVGElementNode = (node) => typeof node.className === "object";
  var isSVGImageElementNode = (node) => node.tagName === "image";
  var isSVGUseElementNode = (node) => node.tagName === "use";
  var isHTMLElementNode = (node) => isElementNode(node) && typeof node.style !== "undefined" && !isSVGElementNode(node);
  var isCommentNode = (node) => node.nodeType === 8;
  var isTextNode = (node) => node.nodeType === 3;
  var isImageElement = (node) => node.tagName === "IMG";
  var isVideoElement = (node) => node.tagName === "VIDEO";
  var isCanvasElement = (node) => node.tagName === "CANVAS";
  var isTextareaElement = (node) => node.tagName === "TEXTAREA";
  var isInputElement = (node) => node.tagName === "INPUT";
  var isStyleElement = (node) => node.tagName === "STYLE";
  var isScriptElement = (node) => node.tagName === "SCRIPT";
  var isSelectElement = (node) => node.tagName === "SELECT";
  var isSlotElement = (node) => node.tagName === "SLOT";
  var isIFrameElement = (node) => node.tagName === "IFRAME";
  var consoleWarn = (...args) => console.warn(PREFIX, ...args);
  function supportWebp(ownerDocument) {
    var _a2;
    const canvas = (_a2 = ownerDocument == null ? void 0 : ownerDocument.createElement) == null ? void 0 : _a2.call(ownerDocument, "canvas");
    if (canvas) {
      canvas.height = canvas.width = 1;
    }
    return Boolean(canvas) && "toDataURL" in canvas && Boolean(canvas.toDataURL("image/webp").includes("image/webp"));
  }
  var isDataUrl = (url) => url.startsWith("data:");
  function resolveUrl(url, baseUrl) {
    if (url.match(/^[a-z]+:\/\//i))
      return url;
    if (IN_BROWSER && url.match(/^\/\//))
      return window.location.protocol + url;
    if (url.match(/^[a-z]+:/i))
      return url;
    if (!IN_BROWSER)
      return url;
    const doc = getDocument().implementation.createHTMLDocument();
    const base = doc.createElement("base");
    const a = doc.createElement("a");
    doc.head.appendChild(base);
    doc.body.appendChild(a);
    if (baseUrl)
      base.href = baseUrl;
    a.href = url;
    return a.href;
  }
  function getDocument(target) {
    var _a2;
    return (_a2 = target && isElementNode(target) ? target == null ? void 0 : target.ownerDocument : target) != null ? _a2 : window.document;
  }
  var XMLNS = "http://www.w3.org/2000/svg";
  function createSvg(width, height, ownerDocument) {
    const svg = getDocument(ownerDocument).createElementNS(XMLNS, "svg");
    svg.setAttributeNS(null, "width", width.toString());
    svg.setAttributeNS(null, "height", height.toString());
    svg.setAttributeNS(null, "viewBox", `0 0 ${width} ${height}`);
    return svg;
  }
  function svgToDataUrl(svg, removeControlCharacter) {
    let xhtml = new XMLSerializer().serializeToString(svg);
    if (removeControlCharacter) {
      xhtml = xhtml.replace(/[\u0000-\u0008\v\f\u000E-\u001F\uD800-\uDFFF\uFFFE\uFFFF]/gu, "");
    }
    return `data:image/svg+xml;charset=utf-8,${encodeURIComponent(xhtml)}`;
  }
  function readBlob(blob, type) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(reader.result);
      reader.onerror = () => reject(reader.error);
      reader.onabort = () => reject(new Error(`Failed read blob to ${type}`));
      if (type === "dataUrl") {
        reader.readAsDataURL(blob);
      } else if (type === "arrayBuffer") {
        reader.readAsArrayBuffer(blob);
      }
    });
  }
  var blobToDataUrl = (blob) => readBlob(blob, "dataUrl");
  function createImage(url, ownerDocument) {
    const img = getDocument(ownerDocument).createElement("img");
    img.decoding = "sync";
    img.loading = "eager";
    img.src = url;
    return img;
  }
  function loadMedia(media, options) {
    return new Promise((resolve) => {
      const { timeout, ownerDocument, onError: userOnError, onWarn } = options != null ? options : {};
      const node = typeof media === "string" ? createImage(media, getDocument(ownerDocument)) : media;
      let timer = null;
      let removeEventListeners = null;
      function onResolve() {
        resolve(node);
        timer && clearTimeout(timer);
        removeEventListeners == null ? void 0 : removeEventListeners();
      }
      if (timeout) {
        timer = setTimeout(onResolve, timeout);
      }
      if (isVideoElement(node)) {
        const currentSrc = node.currentSrc || node.src;
        if (!currentSrc) {
          if (node.poster) {
            return loadMedia(node.poster, options).then(resolve);
          }
          return onResolve();
        }
        if (node.readyState >= 2) {
          return onResolve();
        }
        const onLoadeddata = onResolve;
        const onError = (error) => {
          onWarn == null ? void 0 : onWarn(
            "Failed video load",
            currentSrc,
            error
          );
          userOnError == null ? void 0 : userOnError(error);
          onResolve();
        };
        removeEventListeners = () => {
          node.removeEventListener("loadeddata", onLoadeddata);
          node.removeEventListener("error", onError);
        };
        node.addEventListener("loadeddata", onLoadeddata, { once: true });
        node.addEventListener("error", onError, { once: true });
      } else {
        const currentSrc = isSVGImageElementNode(node) ? node.href.baseVal : node.currentSrc || node.src;
        if (!currentSrc) {
          return onResolve();
        }
        const onLoad = async () => {
          if (isImageElement(node) && "decode" in node) {
            try {
              await node.decode();
            } catch (error) {
              onWarn == null ? void 0 : onWarn(
                "Failed to decode image, trying to render anyway",
                node.dataset.originalSrc || currentSrc,
                error
              );
            }
          }
          onResolve();
        };
        const onError = (error) => {
          onWarn == null ? void 0 : onWarn(
            "Failed image load",
            node.dataset.originalSrc || currentSrc,
            error
          );
          onResolve();
        };
        if (isImageElement(node) && node.complete) {
          return onLoad();
        }
        removeEventListeners = () => {
          node.removeEventListener("load", onLoad);
          node.removeEventListener("error", onError);
        };
        node.addEventListener("load", onLoad, { once: true });
        node.addEventListener("error", onError, { once: true });
      }
    });
  }
  async function waitUntilLoad(node, options) {
    if (isHTMLElementNode(node)) {
      if (isImageElement(node) || isVideoElement(node)) {
        await loadMedia(node, options);
      } else {
        await Promise.all(
          ["img", "video"].flatMap((selectors) => {
            return Array.from(node.querySelectorAll(selectors)).map((el) => loadMedia(el, options));
          })
        );
      }
    }
  }
  var uuid = /* @__PURE__ */ function uuid2() {
    let counter = 0;
    const random = () => `0000${(Math.random() * 36 ** 4 << 0).toString(36)}`.slice(-4);
    return () => {
      counter += 1;
      return `u${random()}${counter}`;
    };
  }();
  function splitFontFamily(fontFamily) {
    return fontFamily == null ? void 0 : fontFamily.split(",").map((val) => val.trim().replace(/"|'/g, "").toLowerCase()).filter(Boolean);
  }
  var uid = 0;
  function createLogger(debug) {
    const prefix = `${PREFIX}[#${uid}]`;
    uid++;
    return {
      // eslint-disable-next-line no-console
      time: (label) => debug && console.time(`${prefix} ${label}`),
      // eslint-disable-next-line no-console
      timeEnd: (label) => debug && console.timeEnd(`${prefix} ${label}`),
      warn: (...args) => debug && consoleWarn(...args)
    };
  }
  function getDefaultRequestInit(bypassingCache) {
    return {
      cache: bypassingCache ? "no-cache" : "force-cache"
    };
  }
  async function orCreateContext(node, options) {
    return isContext(node) ? node : createContext(node, __spreadProps(__spreadValues({}, options), { autoDestruct: true }));
  }
  async function createContext(node, options) {
    var _a2, _b, _c, _d, _e;
    const { scale = 1, workerUrl, workerNumber = 1 } = options || {};
    const debug = Boolean(options == null ? void 0 : options.debug);
    const features = (_a2 = options == null ? void 0 : options.features) != null ? _a2 : true;
    const ownerDocument = (_b = node.ownerDocument) != null ? _b : IN_BROWSER ? window.document : void 0;
    const ownerWindow = (_d = (_c = node.ownerDocument) == null ? void 0 : _c.defaultView) != null ? _d : IN_BROWSER ? window : void 0;
    const requests = /* @__PURE__ */ new Map();
    const context = __spreadProps(__spreadValues({
      // Options
      width: 0,
      height: 0,
      quality: 1,
      type: "image/png",
      scale,
      backgroundColor: null,
      style: null,
      filter: null,
      maximumCanvasSize: 0,
      timeout: 3e4,
      progress: null,
      debug,
      fetch: __spreadValues({
        requestInit: getDefaultRequestInit((_e = options == null ? void 0 : options.fetch) == null ? void 0 : _e.bypassingCache),
        placeholderImage: "data:image/png;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7",
        bypassingCache: false
      }, options == null ? void 0 : options.fetch),
      fetchFn: null,
      font: {},
      drawImageInterval: 100,
      workerUrl: null,
      workerNumber,
      onCloneEachNode: null,
      onCloneNode: null,
      onEmbedNode: null,
      onCreateForeignObjectSvg: null,
      includeStyleProperties: null,
      autoDestruct: false
    }, options), {
      // InternalContext
      __CONTEXT__: true,
      log: createLogger(debug),
      node,
      ownerDocument,
      ownerWindow,
      dpi: scale === 1 ? null : 96 * scale,
      svgStyleElement: createStyleElement(ownerDocument),
      svgDefsElement: ownerDocument == null ? void 0 : ownerDocument.createElementNS(XMLNS, "defs"),
      svgStyles: /* @__PURE__ */ new Map(),
      defaultComputedStyles: /* @__PURE__ */ new Map(),
      workers: [
        ...Array.from({
          length: SUPPORT_WEB_WORKER && workerUrl && workerNumber ? workerNumber : 0
        })
      ].map(() => {
        try {
          const worker = new Worker(workerUrl);
          worker.onmessage = async (event) => {
            var _a3, _b2, _c2, _d2;
            const { url, result } = event.data;
            if (result) {
              (_b2 = (_a3 = requests.get(url)) == null ? void 0 : _a3.resolve) == null ? void 0 : _b2.call(_a3, result);
            } else {
              (_d2 = (_c2 = requests.get(url)) == null ? void 0 : _c2.reject) == null ? void 0 : _d2.call(_c2, new Error(`Error receiving message from worker: ${url}`));
            }
          };
          worker.onmessageerror = (event) => {
            var _a3, _b2;
            const { url } = event.data;
            (_b2 = (_a3 = requests.get(url)) == null ? void 0 : _a3.reject) == null ? void 0 : _b2.call(_a3, new Error(`Error receiving message from worker: ${url}`));
          };
          return worker;
        } catch (error) {
          context.log.warn("Failed to new Worker", error);
          return null;
        }
      }).filter(Boolean),
      fontFamilies: /* @__PURE__ */ new Map(),
      fontCssTexts: /* @__PURE__ */ new Map(),
      acceptOfImage: `${[
        supportWebp(ownerDocument) && "image/webp",
        "image/svg+xml",
        "image/*",
        "*/*"
      ].filter(Boolean).join(",")};q=0.8`,
      requests,
      drawImageCount: 0,
      tasks: [],
      features,
      isEnable: (key) => {
        var _a3, _b2;
        if (key === "restoreScrollPosition") {
          return typeof features === "boolean" ? false : (_a3 = features[key]) != null ? _a3 : false;
        }
        if (typeof features === "boolean") {
          return features;
        }
        return (_b2 = features[key]) != null ? _b2 : true;
      },
      shadowRoots: []
    });
    context.log.time("wait until load");
    await waitUntilLoad(node, { timeout: context.timeout, onWarn: context.log.warn });
    context.log.timeEnd("wait until load");
    const { width, height } = resolveBoundingBox(node, context);
    context.width = width;
    context.height = height;
    return context;
  }
  function createStyleElement(ownerDocument) {
    if (!ownerDocument)
      return void 0;
    const style = ownerDocument.createElement("style");
    const cssText = style.ownerDocument.createTextNode(`
.______background-clip--text {
  background-clip: text;
  -webkit-background-clip: text;
}
`);
    style.appendChild(cssText);
    return style;
  }
  function resolveBoundingBox(node, context) {
    let { width, height } = context;
    if (isElementNode(node) && (!width || !height)) {
      const box = node.getBoundingClientRect();
      width = width || box.width || Number(node.getAttribute("width")) || 0;
      height = height || box.height || Number(node.getAttribute("height")) || 0;
    }
    return { width, height };
  }
  async function imageToCanvas(image, context) {
    const {
      log,
      timeout,
      drawImageCount,
      drawImageInterval
    } = context;
    log.time("image to canvas");
    const loaded = await loadMedia(image, { timeout, onWarn: context.log.warn });
    const { canvas, context2d } = createCanvas(image.ownerDocument, context);
    const drawImage = () => {
      try {
        context2d == null ? void 0 : context2d.drawImage(loaded, 0, 0, canvas.width, canvas.height);
      } catch (error) {
        context.log.warn("Failed to drawImage", error);
      }
    };
    drawImage();
    if (context.isEnable("fixSvgXmlDecode")) {
      for (let i = 0; i < drawImageCount; i++) {
        await new Promise((resolve) => {
          setTimeout(() => {
            context2d == null ? void 0 : context2d.clearRect(0, 0, canvas.width, canvas.height);
            drawImage();
            resolve();
          }, i + drawImageInterval);
        });
      }
    }
    context.drawImageCount = 0;
    log.timeEnd("image to canvas");
    return canvas;
  }
  function createCanvas(ownerDocument, context) {
    const { width, height, scale, backgroundColor, maximumCanvasSize: max } = context;
    const canvas = ownerDocument.createElement("canvas");
    canvas.width = Math.floor(width * scale);
    canvas.height = Math.floor(height * scale);
    canvas.style.width = `${width}px`;
    canvas.style.height = `${height}px`;
    if (max) {
      if (canvas.width > max || canvas.height > max) {
        if (canvas.width > max && canvas.height > max) {
          if (canvas.width > canvas.height) {
            canvas.height *= max / canvas.width;
            canvas.width = max;
          } else {
            canvas.width *= max / canvas.height;
            canvas.height = max;
          }
        } else if (canvas.width > max) {
          canvas.height *= max / canvas.width;
          canvas.width = max;
        } else {
          canvas.width *= max / canvas.height;
          canvas.height = max;
        }
      }
    }
    const context2d = canvas.getContext("2d");
    if (context2d && backgroundColor) {
      context2d.fillStyle = backgroundColor;
      context2d.fillRect(0, 0, canvas.width, canvas.height);
    }
    return { canvas, context2d };
  }
  function cloneCanvas(canvas, context) {
    if (canvas.ownerDocument) {
      try {
        const dataURL = canvas.toDataURL();
        if (dataURL !== "data:,") {
          return createImage(dataURL, canvas.ownerDocument);
        }
      } catch (error) {
        context.log.warn("Failed to clone canvas", error);
      }
    }
    const cloned = canvas.cloneNode(false);
    const ctx = canvas.getContext("2d");
    const clonedCtx = cloned.getContext("2d");
    try {
      if (ctx && clonedCtx) {
        clonedCtx.putImageData(
          ctx.getImageData(0, 0, canvas.width, canvas.height),
          0,
          0
        );
      }
      return cloned;
    } catch (error) {
      context.log.warn("Failed to clone canvas", error);
    }
    return cloned;
  }
  function cloneIframe(iframe, context) {
    var _a2;
    try {
      if ((_a2 = iframe == null ? void 0 : iframe.contentDocument) == null ? void 0 : _a2.body) {
        return cloneNode(iframe.contentDocument.body, context);
      }
    } catch (error) {
      context.log.warn("Failed to clone iframe", error);
    }
    return iframe.cloneNode(false);
  }
  function cloneImage(image) {
    const cloned = image.cloneNode(false);
    if (image.currentSrc && image.currentSrc !== image.src) {
      cloned.src = image.currentSrc;
      cloned.srcset = "";
    }
    if (cloned.loading === "lazy") {
      cloned.loading = "eager";
    }
    return cloned;
  }
  async function cloneVideo(video, context) {
    if (video.ownerDocument && !video.currentSrc && video.poster) {
      return createImage(video.poster, video.ownerDocument);
    }
    const cloned = video.cloneNode(false);
    cloned.crossOrigin = "anonymous";
    if (video.currentSrc && video.currentSrc !== video.src) {
      cloned.src = video.currentSrc;
    }
    const ownerDocument = cloned.ownerDocument;
    if (ownerDocument) {
      let canPlay = true;
      await loadMedia(cloned, { onError: () => canPlay = false, onWarn: context.log.warn });
      if (!canPlay) {
        if (video.poster) {
          return createImage(video.poster, video.ownerDocument);
        }
        return cloned;
      }
      cloned.currentTime = video.currentTime;
      await new Promise((resolve) => {
        cloned.addEventListener("seeked", resolve, { once: true });
      });
      const canvas = ownerDocument.createElement("canvas");
      canvas.width = video.offsetWidth;
      canvas.height = video.offsetHeight;
      try {
        const ctx = canvas.getContext("2d");
        if (ctx)
          ctx.drawImage(cloned, 0, 0, canvas.width, canvas.height);
      } catch (error) {
        context.log.warn("Failed to clone video", error);
        if (video.poster) {
          return createImage(video.poster, video.ownerDocument);
        }
        return cloned;
      }
      return cloneCanvas(canvas, context);
    }
    return cloned;
  }
  function cloneElement(node, context) {
    if (isCanvasElement(node)) {
      return cloneCanvas(node, context);
    }
    if (isIFrameElement(node)) {
      return cloneIframe(node, context);
    }
    if (isImageElement(node)) {
      return cloneImage(node);
    }
    if (isVideoElement(node)) {
      return cloneVideo(node, context);
    }
    return node.cloneNode(false);
  }
  function getSandBox(context) {
    let sandbox = context.sandbox;
    if (!sandbox) {
      const { ownerDocument } = context;
      try {
        if (ownerDocument) {
          sandbox = ownerDocument.createElement("iframe");
          sandbox.id = `__SANDBOX__${uuid()}`;
          sandbox.width = "0";
          sandbox.height = "0";
          sandbox.style.visibility = "hidden";
          sandbox.style.position = "fixed";
          ownerDocument.body.appendChild(sandbox);
          sandbox.srcdoc = '<!DOCTYPE html><meta charset="UTF-8"><title></title><body>';
          context.sandbox = sandbox;
        }
      } catch (error) {
        context.log.warn("Failed to getSandBox", error);
      }
    }
    return sandbox;
  }
  var ignoredStyles = [
    "width",
    "height",
    "-webkit-text-fill-color"
  ];
  var includedAttributes = [
    "stroke",
    "fill"
  ];
  function getDefaultStyle(node, pseudoElement, context) {
    const { defaultComputedStyles } = context;
    const nodeName = node.nodeName.toLowerCase();
    const isSvgNode = isSVGElementNode(node) && nodeName !== "svg";
    const attributes = isSvgNode ? includedAttributes.map((name) => [name, node.getAttribute(name)]).filter(([, value]) => value !== null) : [];
    const key = [
      isSvgNode && "svg",
      nodeName,
      attributes.map((name, value) => `${name}=${value}`).join(","),
      pseudoElement
    ].filter(Boolean).join(":");
    if (defaultComputedStyles.has(key))
      return defaultComputedStyles.get(key);
    const sandbox = getSandBox(context);
    const sandboxWindow = sandbox == null ? void 0 : sandbox.contentWindow;
    if (!sandboxWindow)
      return /* @__PURE__ */ new Map();
    const sandboxDocument = sandboxWindow == null ? void 0 : sandboxWindow.document;
    let root;
    let el;
    if (isSvgNode) {
      root = sandboxDocument.createElementNS(XMLNS, "svg");
      el = root.ownerDocument.createElementNS(root.namespaceURI, nodeName);
      attributes.forEach(([name, value]) => {
        el.setAttributeNS(null, name, value);
      });
      root.appendChild(el);
    } else {
      root = el = sandboxDocument.createElement(nodeName);
    }
    el.textContent = " ";
    sandboxDocument.body.appendChild(root);
    const computedStyle = sandboxWindow.getComputedStyle(el, pseudoElement);
    const styles = /* @__PURE__ */ new Map();
    for (let len = computedStyle.length, i = 0; i < len; i++) {
      const name = computedStyle.item(i);
      if (ignoredStyles.includes(name))
        continue;
      styles.set(name, computedStyle.getPropertyValue(name));
    }
    sandboxDocument.body.removeChild(root);
    defaultComputedStyles.set(key, styles);
    return styles;
  }
  function getDiffStyle(style, defaultStyle, includeStyleProperties) {
    var _a2;
    const diffStyle = /* @__PURE__ */ new Map();
    const prefixs = [];
    const prefixTree = /* @__PURE__ */ new Map();
    if (includeStyleProperties) {
      for (const name of includeStyleProperties) {
        applyTo(name);
      }
    } else {
      for (let len = style.length, i = 0; i < len; i++) {
        const name = style.item(i);
        applyTo(name);
      }
    }
    for (let len = prefixs.length, i = 0; i < len; i++) {
      (_a2 = prefixTree.get(prefixs[i])) == null ? void 0 : _a2.forEach((value, name) => diffStyle.set(name, value));
    }
    function applyTo(name) {
      const value = style.getPropertyValue(name);
      const priority = style.getPropertyPriority(name);
      const subIndex = name.lastIndexOf("-");
      const prefix = subIndex > -1 ? name.substring(0, subIndex) : void 0;
      if (prefix) {
        let map = prefixTree.get(prefix);
        if (!map) {
          map = /* @__PURE__ */ new Map();
          prefixTree.set(prefix, map);
        }
        map.set(name, [value, priority]);
      }
      if (defaultStyle.get(name) === value && !priority)
        return;
      if (prefix) {
        prefixs.push(prefix);
      } else {
        diffStyle.set(name, [value, priority]);
      }
    }
    return diffStyle;
  }
  function copyCssStyles(node, cloned, isRoot, context) {
    var _a2, _b, _c, _d;
    const { ownerWindow, includeStyleProperties, currentParentNodeStyle } = context;
    const clonedStyle = cloned.style;
    const computedStyle = ownerWindow.getComputedStyle(node);
    const defaultStyle = getDefaultStyle(node, null, context);
    currentParentNodeStyle == null ? void 0 : currentParentNodeStyle.forEach((_, key) => {
      defaultStyle.delete(key);
    });
    const style = getDiffStyle(computedStyle, defaultStyle, includeStyleProperties);
    style.delete("transition-property");
    style.delete("all");
    style.delete("d");
    style.delete("content");
    if (isRoot) {
      style.delete("margin-top");
      style.delete("margin-right");
      style.delete("margin-bottom");
      style.delete("margin-left");
      style.delete("margin-block-start");
      style.delete("margin-block-end");
      style.delete("margin-inline-start");
      style.delete("margin-inline-end");
      style.set("box-sizing", ["border-box", ""]);
    }
    if (((_a2 = style.get("background-clip")) == null ? void 0 : _a2[0]) === "text") {
      cloned.classList.add("______background-clip--text");
    }
    if (IN_CHROME) {
      if (!style.has("font-kerning"))
        style.set("font-kerning", ["normal", ""]);
      if ((((_b = style.get("overflow-x")) == null ? void 0 : _b[0]) === "hidden" || ((_c = style.get("overflow-y")) == null ? void 0 : _c[0]) === "hidden") && ((_d = style.get("text-overflow")) == null ? void 0 : _d[0]) === "ellipsis" && node.scrollWidth === node.clientWidth) {
        style.set("text-overflow", ["clip", ""]);
      }
    }
    for (let len = clonedStyle.length, i = 0; i < len; i++) {
      clonedStyle.removeProperty(clonedStyle.item(i));
    }
    style.forEach(([value, priority], name) => {
      clonedStyle.setProperty(name, value, priority);
    });
    return style;
  }
  function copyInputValue(node, cloned) {
    if (isTextareaElement(node) || isInputElement(node) || isSelectElement(node)) {
      cloned.setAttribute("value", node.value);
    }
  }
  var pseudoClasses = [
    ":before",
    ":after"
    // ':placeholder', TODO
  ];
  var scrollbarPseudoClasses = [
    ":-webkit-scrollbar",
    ":-webkit-scrollbar-button",
    // ':-webkit-scrollbar:horizontal', TODO
    ":-webkit-scrollbar-thumb",
    ":-webkit-scrollbar-track",
    ":-webkit-scrollbar-track-piece",
    // ':-webkit-scrollbar:vertical', TODO
    ":-webkit-scrollbar-corner",
    ":-webkit-resizer"
  ];
  function copyPseudoClass(node, cloned, copyScrollbar, context, addWordToFontFamilies) {
    const { ownerWindow, svgStyleElement, svgStyles, currentNodeStyle } = context;
    if (!svgStyleElement || !ownerWindow)
      return;
    function copyBy(pseudoClass) {
      var _a2;
      const computedStyle = ownerWindow.getComputedStyle(node, pseudoClass);
      let content = computedStyle.getPropertyValue("content");
      if (!content || content === "none")
        return;
      addWordToFontFamilies == null ? void 0 : addWordToFontFamilies(content);
      content = content.replace(/(')|(")|(counter\(.+\))/g, "");
      const klasses = [uuid()];
      const defaultStyle = getDefaultStyle(node, pseudoClass, context);
      currentNodeStyle == null ? void 0 : currentNodeStyle.forEach((_, key) => {
        defaultStyle.delete(key);
      });
      const style = getDiffStyle(computedStyle, defaultStyle, context.includeStyleProperties);
      style.delete("content");
      style.delete("-webkit-locale");
      if (((_a2 = style.get("background-clip")) == null ? void 0 : _a2[0]) === "text") {
        cloned.classList.add("______background-clip--text");
      }
      const cloneStyle = [
        `content: '${content}';`
      ];
      style.forEach(([value, priority], name) => {
        cloneStyle.push(`${name}: ${value}${priority ? " !important" : ""};`);
      });
      if (cloneStyle.length === 1)
        return;
      try {
        cloned.className = [cloned.className, ...klasses].join(" ");
      } catch (err) {
        context.log.warn("Failed to copyPseudoClass", err);
        return;
      }
      const cssText = cloneStyle.join("\n  ");
      let allClasses = svgStyles.get(cssText);
      if (!allClasses) {
        allClasses = [];
        svgStyles.set(cssText, allClasses);
      }
      allClasses.push(`.${klasses[0]}:${pseudoClass}`);
    }
    pseudoClasses.forEach(copyBy);
    if (copyScrollbar)
      scrollbarPseudoClasses.forEach(copyBy);
  }
  var excludeParentNodes = /* @__PURE__ */ new Set([
    "symbol"
    // test/fixtures/svg.symbol.html
  ]);
  async function appendChildNode(node, cloned, child, context, addWordToFontFamilies) {
    if (isElementNode(child) && (isStyleElement(child) || isScriptElement(child)))
      return;
    if (context.filter && !context.filter(child))
      return;
    if (excludeParentNodes.has(cloned.nodeName) || excludeParentNodes.has(child.nodeName)) {
      context.currentParentNodeStyle = void 0;
    } else {
      context.currentParentNodeStyle = context.currentNodeStyle;
    }
    const childCloned = await cloneNode(child, context, false, addWordToFontFamilies);
    if (context.isEnable("restoreScrollPosition")) {
      restoreScrollPosition(node, childCloned);
    }
    cloned.appendChild(childCloned);
  }
  async function cloneChildNodes(node, cloned, context, addWordToFontFamilies) {
    var _a2;
    let firstChild = node.firstChild;
    if (isElementNode(node)) {
      if (node.shadowRoot) {
        firstChild = (_a2 = node.shadowRoot) == null ? void 0 : _a2.firstChild;
        context.shadowRoots.push(node.shadowRoot);
      }
    }
    for (let child = firstChild; child; child = child.nextSibling) {
      if (isCommentNode(child))
        continue;
      if (isElementNode(child) && isSlotElement(child) && typeof child.assignedNodes === "function") {
        const nodes = child.assignedNodes();
        for (let i = 0; i < nodes.length; i++) {
          await appendChildNode(node, cloned, nodes[i], context, addWordToFontFamilies);
        }
      } else {
        await appendChildNode(node, cloned, child, context, addWordToFontFamilies);
      }
    }
  }
  function restoreScrollPosition(node, chlidCloned) {
    if (!isHTMLElementNode(node) || !isHTMLElementNode(chlidCloned))
      return;
    const { scrollTop, scrollLeft } = node;
    if (!scrollTop && !scrollLeft) {
      return;
    }
    const { transform } = chlidCloned.style;
    const matrix = new DOMMatrix(transform);
    const { a, b, c, d } = matrix;
    matrix.a = 1;
    matrix.b = 0;
    matrix.c = 0;
    matrix.d = 1;
    matrix.translateSelf(-scrollLeft, -scrollTop);
    matrix.a = a;
    matrix.b = b;
    matrix.c = c;
    matrix.d = d;
    chlidCloned.style.transform = matrix.toString();
  }
  function applyCssStyleWithOptions(cloned, context) {
    const { backgroundColor, width, height, style: styles } = context;
    const clonedStyle = cloned.style;
    if (backgroundColor)
      clonedStyle.setProperty("background-color", backgroundColor, "important");
    if (width)
      clonedStyle.setProperty("width", `${width}px`, "important");
    if (height)
      clonedStyle.setProperty("height", `${height}px`, "important");
    if (styles) {
      for (const name in styles)
        clonedStyle[name] = styles[name];
    }
  }
  var NORMAL_ATTRIBUTE_RE = /^[\w-:]+$/;
  async function cloneNode(node, context, isRoot = false, addWordToFontFamilies) {
    var _a2, _b, _c, _d;
    const { ownerDocument, ownerWindow, fontFamilies, onCloneEachNode } = context;
    if (ownerDocument && isTextNode(node)) {
      if (addWordToFontFamilies && /\S/.test(node.data)) {
        addWordToFontFamilies(node.data);
      }
      return ownerDocument.createTextNode(node.data);
    }
    if (ownerDocument && ownerWindow && isElementNode(node) && (isHTMLElementNode(node) || isSVGElementNode(node))) {
      const cloned2 = await cloneElement(node, context);
      if (context.isEnable("removeAbnormalAttributes")) {
        const names = cloned2.getAttributeNames();
        for (let len = names.length, i = 0; i < len; i++) {
          const name = names[i];
          if (!NORMAL_ATTRIBUTE_RE.test(name)) {
            cloned2.removeAttribute(name);
          }
        }
      }
      const style = context.currentNodeStyle = copyCssStyles(node, cloned2, isRoot, context);
      if (isRoot)
        applyCssStyleWithOptions(cloned2, context);
      let copyScrollbar = false;
      if (context.isEnable("copyScrollbar")) {
        const overflow = [
          (_a2 = style.get("overflow-x")) == null ? void 0 : _a2[0],
          (_b = style.get("overflow-y")) == null ? void 0 : _b[0]
        ];
        copyScrollbar = overflow.includes("scroll") || (overflow.includes("auto") || overflow.includes("overlay")) && (node.scrollHeight > node.clientHeight || node.scrollWidth > node.clientWidth);
      }
      const textTransform = (_c = style.get("text-transform")) == null ? void 0 : _c[0];
      const families = splitFontFamily((_d = style.get("font-family")) == null ? void 0 : _d[0]);
      const addWordToFontFamilies2 = families ? (word) => {
        if (textTransform === "uppercase") {
          word = word.toUpperCase();
        } else if (textTransform === "lowercase") {
          word = word.toLowerCase();
        } else if (textTransform === "capitalize") {
          word = word[0].toUpperCase() + word.substring(1);
        }
        families.forEach((family) => {
          let fontFamily = fontFamilies.get(family);
          if (!fontFamily) {
            fontFamilies.set(family, fontFamily = /* @__PURE__ */ new Set());
          }
          word.split("").forEach((text) => fontFamily.add(text));
        });
      } : void 0;
      copyPseudoClass(
        node,
        cloned2,
        copyScrollbar,
        context,
        addWordToFontFamilies2
      );
      copyInputValue(node, cloned2);
      if (!isVideoElement(node)) {
        await cloneChildNodes(
          node,
          cloned2,
          context,
          addWordToFontFamilies2
        );
      }
      await (onCloneEachNode == null ? void 0 : onCloneEachNode(cloned2));
      return cloned2;
    }
    const cloned = node.cloneNode(false);
    await cloneChildNodes(node, cloned, context);
    await (onCloneEachNode == null ? void 0 : onCloneEachNode(cloned));
    return cloned;
  }
  function destroyContext(context) {
    context.ownerDocument = void 0;
    context.ownerWindow = void 0;
    context.svgStyleElement = void 0;
    context.svgDefsElement = void 0;
    context.svgStyles.clear();
    context.defaultComputedStyles.clear();
    if (context.sandbox) {
      try {
        context.sandbox.remove();
      } catch (err) {
        context.log.warn("Failed to destroyContext", err);
      }
      context.sandbox = void 0;
    }
    context.workers = [];
    context.fontFamilies.clear();
    context.fontCssTexts.clear();
    context.requests.clear();
    context.tasks = [];
    context.shadowRoots = [];
  }
  function baseFetch(options) {
    const _a2 = options, { url, timeout, responseType } = _a2, requestInit = __objRest(_a2, ["url", "timeout", "responseType"]);
    const controller = new AbortController();
    const timer = timeout ? setTimeout(() => controller.abort(), timeout) : void 0;
    return fetch(url, __spreadValues({ signal: controller.signal }, requestInit)).then((response) => {
      if (!response.ok) {
        throw new Error("Failed fetch, not 2xx response", { cause: response });
      }
      switch (responseType) {
        case "arrayBuffer":
          return response.arrayBuffer();
        case "dataUrl":
          return response.blob().then(blobToDataUrl);
        case "text":
        default:
          return response.text();
      }
    }).finally(() => clearTimeout(timer));
  }
  function contextFetch(context, options) {
    const { url: rawUrl, requestType = "text", responseType = "text", imageDom } = options;
    let url = rawUrl;
    const {
      timeout,
      acceptOfImage,
      requests,
      fetchFn,
      fetch: {
        requestInit,
        bypassingCache,
        placeholderImage
      },
      font,
      workers,
      fontFamilies
    } = context;
    if (requestType === "image" && (IN_SAFARI || IN_FIREFOX)) {
      context.drawImageCount++;
    }
    let request = requests.get(rawUrl);
    if (!request) {
      if (bypassingCache) {
        if (bypassingCache instanceof RegExp && bypassingCache.test(url)) {
          url += (/\?/.test(url) ? "&" : "?") + (/* @__PURE__ */ new Date()).getTime();
        }
      }
      const canFontMinify = requestType.startsWith("font") && font && font.minify;
      const fontTexts = /* @__PURE__ */ new Set();
      if (canFontMinify) {
        const families = requestType.split(";")[1].split(",");
        families.forEach((family) => {
          if (!fontFamilies.has(family))
            return;
          fontFamilies.get(family).forEach((text) => fontTexts.add(text));
        });
      }
      const needFontMinify = canFontMinify && fontTexts.size;
      const baseFetchOptions = __spreadValues({
        url,
        timeout,
        responseType: needFontMinify ? "arrayBuffer" : responseType,
        headers: requestType === "image" ? { accept: acceptOfImage } : void 0
      }, requestInit);
      request = {
        type: requestType,
        resolve: void 0,
        reject: void 0,
        response: null
      };
      request.response = (async () => {
        if (fetchFn && requestType === "image") {
          const result = await fetchFn(rawUrl);
          if (result)
            return result;
        }
        if (!IN_SAFARI && rawUrl.startsWith("http") && workers.length) {
          return new Promise((resolve, reject) => {
            const worker = workers[requests.size & workers.length - 1];
            worker.postMessage(__spreadValues({ rawUrl }, baseFetchOptions));
            request.resolve = resolve;
            request.reject = reject;
          });
        }
        return baseFetch(baseFetchOptions);
      })().catch((error) => {
        requests.delete(rawUrl);
        if (requestType === "image" && placeholderImage) {
          context.log.warn("Failed to fetch image base64, trying to use placeholder image", url);
          return typeof placeholderImage === "string" ? placeholderImage : placeholderImage(imageDom);
        }
        throw error;
      });
      requests.set(rawUrl, request);
    }
    return request.response;
  }
  async function replaceCssUrlToDataUrl(cssText, baseUrl, context, isImage) {
    if (!hasCssUrl(cssText))
      return cssText;
    for (const [rawUrl, url] of parseCssUrls(cssText, baseUrl)) {
      try {
        const dataUrl = await contextFetch(
          context,
          {
            url,
            requestType: isImage ? "image" : "text",
            responseType: "dataUrl"
          }
        );
        cssText = cssText.replace(toRE(rawUrl), `$1${dataUrl}$3`);
      } catch (error) {
        context.log.warn("Failed to fetch css data url", rawUrl, error);
      }
    }
    return cssText;
  }
  function hasCssUrl(cssText) {
    return /url\((['"]?)([^'"]+?)\1\)/.test(cssText);
  }
  var URL_RE = /url\((['"]?)([^'"]+?)\1\)/g;
  function parseCssUrls(cssText, baseUrl) {
    const result = [];
    cssText.replace(URL_RE, (raw, quotation, url) => {
      result.push([url, resolveUrl(url, baseUrl)]);
      return raw;
    });
    return result.filter(([url]) => !isDataUrl(url));
  }
  function toRE(url) {
    const escaped = url.replace(/([.*+?^${}()|\[\]\/\\])/g, "\\$1");
    return new RegExp(`(url\\(['"]?)(${escaped})(['"]?\\))`, "g");
  }
  var properties = [
    "background-image",
    "border-image-source",
    "-webkit-border-image",
    "-webkit-mask-image",
    "list-style-image"
  ];
  function embedCssStyleImage(style, context) {
    return properties.map((property) => {
      const value = style.getPropertyValue(property);
      if (!value || value === "none") {
        return null;
      }
      if (IN_SAFARI || IN_FIREFOX) {
        context.drawImageCount++;
      }
      return replaceCssUrlToDataUrl(value, null, context, true).then((newValue) => {
        if (!newValue || value === newValue)
          return;
        style.setProperty(
          property,
          newValue,
          style.getPropertyPriority(property)
        );
      });
    }).filter(Boolean);
  }
  function embedImageElement(cloned, context) {
    if (isImageElement(cloned)) {
      const originalSrc = cloned.currentSrc || cloned.src;
      if (!isDataUrl(originalSrc)) {
        return [
          contextFetch(context, {
            url: originalSrc,
            imageDom: cloned,
            requestType: "image",
            responseType: "dataUrl"
          }).then((url) => {
            if (!url)
              return;
            cloned.srcset = "";
            cloned.dataset.originalSrc = originalSrc;
            cloned.src = url || "";
          })
        ];
      }
      if (IN_SAFARI || IN_FIREFOX) {
        context.drawImageCount++;
      }
    } else if (isSVGElementNode(cloned) && !isDataUrl(cloned.href.baseVal)) {
      const originalSrc = cloned.href.baseVal;
      return [
        contextFetch(context, {
          url: originalSrc,
          imageDom: cloned,
          requestType: "image",
          responseType: "dataUrl"
        }).then((url) => {
          if (!url)
            return;
          cloned.dataset.originalSrc = originalSrc;
          cloned.href.baseVal = url || "";
        })
      ];
    }
    return [];
  }
  function embedSvgUse(cloned, context) {
    var _a2;
    const { ownerDocument, svgDefsElement } = context;
    const href = (_a2 = cloned.getAttribute("href")) != null ? _a2 : cloned.getAttribute("xlink:href");
    if (!href)
      return [];
    const [svgUrl, id] = href.split("#");
    if (id) {
      const query = `#${id}`;
      const definition = context.shadowRoots.reduce(
        (res, root) => {
          return res != null ? res : root.querySelector(`svg ${query}`);
        },
        ownerDocument == null ? void 0 : ownerDocument.querySelector(`svg ${query}`)
      );
      if (svgUrl) {
        cloned.setAttribute("href", query);
      }
      if (svgDefsElement == null ? void 0 : svgDefsElement.querySelector(query))
        return [];
      if (definition) {
        svgDefsElement == null ? void 0 : svgDefsElement.appendChild(definition.cloneNode(true));
        return [];
      } else if (svgUrl) {
        return [
          contextFetch(context, {
            url: svgUrl,
            responseType: "text"
          }).then((svgData) => {
            svgDefsElement == null ? void 0 : svgDefsElement.insertAdjacentHTML("beforeend", svgData);
          })
        ];
      }
    }
    return [];
  }
  function embedNode(cloned, context) {
    const { tasks } = context;
    if (isElementNode(cloned)) {
      if (isImageElement(cloned) || isSVGImageElementNode(cloned)) {
        tasks.push(...embedImageElement(cloned, context));
      }
      if (isSVGUseElementNode(cloned)) {
        tasks.push(...embedSvgUse(cloned, context));
      }
    }
    if (isHTMLElementNode(cloned)) {
      tasks.push(...embedCssStyleImage(cloned.style, context));
    }
    cloned.childNodes.forEach((child) => {
      embedNode(child, context);
    });
  }
  async function embedWebFont(clone, context) {
    const {
      ownerDocument,
      svgStyleElement,
      fontFamilies,
      fontCssTexts,
      tasks,
      font
    } = context;
    if (!ownerDocument || !svgStyleElement || !fontFamilies.size) {
      return;
    }
    if (font && font.cssText) {
      const cssText = filterPreferredFormat(font.cssText, context);
      svgStyleElement.appendChild(ownerDocument.createTextNode(`${cssText}
`));
    } else {
      const styleSheets = Array.from(ownerDocument.styleSheets).filter((styleSheet) => {
        try {
          return "cssRules" in styleSheet && Boolean(styleSheet.cssRules.length);
        } catch (error) {
          context.log.warn(`Error while reading CSS rules from ${styleSheet.href}`, error);
          return false;
        }
      });
      await Promise.all(
        styleSheets.flatMap((styleSheet) => {
          return Array.from(styleSheet.cssRules).map(async (cssRule, index) => {
            if (isCSSImportRule(cssRule)) {
              let importIndex = index + 1;
              const baseUrl = cssRule.href;
              let cssText = "";
              try {
                cssText = await contextFetch(context, {
                  url: baseUrl,
                  requestType: "text",
                  responseType: "text"
                });
              } catch (error) {
                context.log.warn(`Error fetch remote css import from ${baseUrl}`, error);
              }
              const replacedCssText = cssText.replace(
                URL_RE,
                (raw, quotation, url) => raw.replace(url, resolveUrl(url, baseUrl))
              );
              for (const rule of parseCss(replacedCssText)) {
                try {
                  styleSheet.insertRule(
                    rule,
                    rule.startsWith("@import") ? importIndex += 1 : styleSheet.cssRules.length
                  );
                } catch (error) {
                  context.log.warn("Error inserting rule from remote css import", { rule, error });
                }
              }
            }
          });
        })
      );
      const cssRules = styleSheets.flatMap((styleSheet) => Array.from(styleSheet.cssRules));
      cssRules.filter((cssRule) => {
        var _a2;
        return isCssFontFaceRule(cssRule) && hasCssUrl(cssRule.style.getPropertyValue("src")) && ((_a2 = splitFontFamily(cssRule.style.getPropertyValue("font-family"))) == null ? void 0 : _a2.some((val) => fontFamilies.has(val)));
      }).forEach((value) => {
        const rule = value;
        const cssText = fontCssTexts.get(rule.cssText);
        if (cssText) {
          svgStyleElement.appendChild(ownerDocument.createTextNode(`${cssText}
`));
        } else {
          tasks.push(
            replaceCssUrlToDataUrl(
              rule.cssText,
              rule.parentStyleSheet ? rule.parentStyleSheet.href : null,
              context
            ).then((cssText2) => {
              cssText2 = filterPreferredFormat(cssText2, context);
              fontCssTexts.set(rule.cssText, cssText2);
              svgStyleElement.appendChild(ownerDocument.createTextNode(`${cssText2}
`));
            })
          );
        }
      });
    }
  }
  var COMMENTS_RE = /(\/\*[\s\S]*?\*\/)/g;
  var KEYFRAMES_RE = /((@.*?keyframes [\s\S]*?){([\s\S]*?}\s*?)})/gi;
  function parseCss(source) {
    if (source == null)
      return [];
    const result = [];
    let cssText = source.replace(COMMENTS_RE, "");
    while (true) {
      const matches = KEYFRAMES_RE.exec(cssText);
      if (!matches)
        break;
      result.push(matches[0]);
    }
    cssText = cssText.replace(KEYFRAMES_RE, "");
    const IMPORT_RE = /@import[\s\S]*?url\([^)]*\)[\s\S]*?;/gi;
    const UNIFIED_RE = new RegExp(
      // eslint-disable-next-line
      "((\\s*?(?:\\/\\*[\\s\\S]*?\\*\\/)?\\s*?@media[\\s\\S]*?){([\\s\\S]*?)}\\s*?})|(([\\s\\S]*?){([\\s\\S]*?)})",
      "gi"
    );
    while (true) {
      let matches = IMPORT_RE.exec(cssText);
      if (!matches) {
        matches = UNIFIED_RE.exec(cssText);
        if (!matches) {
          break;
        } else {
          IMPORT_RE.lastIndex = UNIFIED_RE.lastIndex;
        }
      } else {
        UNIFIED_RE.lastIndex = IMPORT_RE.lastIndex;
      }
      result.push(matches[0]);
    }
    return result;
  }
  var URL_WITH_FORMAT_RE = /url\([^)]+\)\s*format\((["']?)([^"']+)\1\)/g;
  var FONT_SRC_RE = /src:\s*(?:url\([^)]+\)\s*format\([^)]+\)[,;]\s*)+/g;
  function filterPreferredFormat(str, context) {
    const { font } = context;
    const preferredFormat = font ? font == null ? void 0 : font.preferredFormat : void 0;
    return preferredFormat ? str.replace(FONT_SRC_RE, (match) => {
      while (true) {
        const [src, , format] = URL_WITH_FORMAT_RE.exec(match) || [];
        if (!format)
          return "";
        if (format === preferredFormat)
          return `src: ${src};`;
      }
    }) : str;
  }
  async function domToForeignObjectSvg(node, options) {
    const context = await orCreateContext(node, options);
    if (isElementNode(context.node) && isSVGElementNode(context.node))
      return context.node;
    const {
      ownerDocument,
      log,
      tasks,
      svgStyleElement,
      svgDefsElement,
      svgStyles,
      font,
      progress,
      autoDestruct,
      onCloneNode,
      onEmbedNode,
      onCreateForeignObjectSvg
    } = context;
    log.time("clone node");
    const clone = await cloneNode(context.node, context, true);
    if (svgStyleElement && ownerDocument) {
      let allCssText = "";
      svgStyles.forEach((klasses, cssText) => {
        allCssText += `${klasses.join(",\n")} {
  ${cssText}
}
`;
      });
      svgStyleElement.appendChild(ownerDocument.createTextNode(allCssText));
    }
    log.timeEnd("clone node");
    await (onCloneNode == null ? void 0 : onCloneNode(clone));
    if (font !== false && isElementNode(clone)) {
      log.time("embed web font");
      await embedWebFont(clone, context);
      log.timeEnd("embed web font");
    }
    log.time("embed node");
    embedNode(clone, context);
    const count = tasks.length;
    let current = 0;
    const runTask = async () => {
      while (true) {
        const task = tasks.pop();
        if (!task)
          break;
        try {
          await task;
        } catch (error) {
          context.log.warn("Failed to run task", error);
        }
        progress == null ? void 0 : progress(++current, count);
      }
    };
    progress == null ? void 0 : progress(current, count);
    await Promise.all([...Array.from({ length: 4 })].map(runTask));
    log.timeEnd("embed node");
    await (onEmbedNode == null ? void 0 : onEmbedNode(clone));
    const svg = createForeignObjectSvg(clone, context);
    svgDefsElement && svg.insertBefore(svgDefsElement, svg.children[0]);
    svgStyleElement && svg.insertBefore(svgStyleElement, svg.children[0]);
    autoDestruct && destroyContext(context);
    await (onCreateForeignObjectSvg == null ? void 0 : onCreateForeignObjectSvg(svg));
    return svg;
  }
  function createForeignObjectSvg(clone, context) {
    const { width, height } = context;
    const svg = createSvg(width, height, clone.ownerDocument);
    const foreignObject = svg.ownerDocument.createElementNS(svg.namespaceURI, "foreignObject");
    foreignObject.setAttributeNS(null, "x", "0%");
    foreignObject.setAttributeNS(null, "y", "0%");
    foreignObject.setAttributeNS(null, "width", "100%");
    foreignObject.setAttributeNS(null, "height", "100%");
    foreignObject.append(clone);
    svg.appendChild(foreignObject);
    return svg;
  }
  async function domToCanvas(node, options) {
    var _a2;
    const context = await orCreateContext(node, options);
    const svg = await domToForeignObjectSvg(context);
    const dataUrl = svgToDataUrl(svg, context.isEnable("removeControlCharacter"));
    if (!context.autoDestruct) {
      context.svgStyleElement = createStyleElement(context.ownerDocument);
      context.svgDefsElement = (_a2 = context.ownerDocument) == null ? void 0 : _a2.createElementNS(XMLNS, "defs");
      context.svgStyles.clear();
    }
    const image = createImage(dataUrl, svg.ownerDocument);
    return await imageToCanvas(image, context);
  }
  async function domToDataUrl(node, options) {
    const context = await orCreateContext(node, options);
    const { log, quality, type, dpi } = context;
    const canvas = await domToCanvas(context);
    log.time("canvas to data url");
    let dataUrl = canvas.toDataURL(type, quality);
    if (["image/png", "image/jpeg"].includes(type) && dpi && SUPPORT_ATOB && SUPPORT_BTOA) {
      const [format, body] = dataUrl.split(",");
      let headerLength = 0;
      let overwritepHYs = false;
      if (type === "image/png") {
        const b64Index = detectPhysChunkFromDataUrl(body);
        if (b64Index >= 0) {
          headerLength = Math.ceil((b64Index + 28) / 3) * 4;
          overwritepHYs = true;
        } else {
          headerLength = 33 / 3 * 4;
        }
      } else if (type === "image/jpeg") {
        headerLength = 18 / 3 * 4;
      }
      const stringHeader = body.substring(0, headerLength);
      const restOfData = body.substring(headerLength);
      const headerBytes = window.atob(stringHeader);
      const uint8Array = new Uint8Array(headerBytes.length);
      for (let i = 0; i < uint8Array.length; i++) {
        uint8Array[i] = headerBytes.charCodeAt(i);
      }
      const finalArray = type === "image/png" ? changePngDpi(uint8Array, dpi, overwritepHYs) : changeJpegDpi(uint8Array, dpi);
      const base64Header = window.btoa(String.fromCharCode(...finalArray));
      dataUrl = [format, ",", base64Header, restOfData].join("");
    }
    log.timeEnd("canvas to data url");
    return dataUrl;
  }

  // src/playwright-replay/index.ts
  var playwrightReplayRef = {
    current: null
  };
  var parentOrigin = null;
  function getParentOrigin() {
    var _a2;
    if (parentOrigin) {
      return parentOrigin;
    }
    try {
      const configOrigin = (_a2 = window.STAKTRAK_CONFIG) == null ? void 0 : _a2.parentOrigin;
      if (configOrigin) {
        return configOrigin;
      }
    } catch (e) {
    }
    return "*";
  }
  async function captureScreenshot(actionIndex, url) {
    var _a2, _b, _c, _d, _e;
    try {
      let config = {};
      try {
        config = ((_a2 = window.STAKTRAK_CONFIG) == null ? void 0 : _a2.screenshot) || {};
      } catch (e) {
      }
      const screenshotOptions = {
        quality: (_b = config.quality) != null ? _b : 0.8,
        type: (_c = config.type) != null ? _c : "image/jpeg",
        scale: (_d = config.scale) != null ? _d : 1,
        backgroundColor: (_e = config.backgroundColor) != null ? _e : "#ffffff"
      };
      const dataUrl = await domToDataUrl(document.body, screenshotOptions);
      const timestamp = Date.now();
      const id = `${timestamp}-${actionIndex}`;
      window.parent.postMessage(
        {
          type: "staktrak-playwright-screenshot-captured",
          screenshot: dataUrl,
          actionIndex,
          url,
          timestamp,
          id
        },
        getParentOrigin()
      );
    } catch (error) {
      console.error(`[Screenshot] Error capturing for actionIndex=${actionIndex}:`, error);
    }
  }
  async function startPlaywrightReplay(testCode) {
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
        getParentOrigin()
      );
      executeNextPlaywrightAction();
    } catch (error) {
      window.parent.postMessage(
        {
          type: "staktrak-playwright-replay-error",
          error: error instanceof Error ? error.message : "Unknown error"
        },
        getParentOrigin()
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
        getParentOrigin()
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
        getParentOrigin()
      );
      await executePlaywrightAction(action);
      if (action.type === "waitForURL") {
        await captureScreenshot(state.currentActionIndex, window.location.href);
      }
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
        getParentOrigin()
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
        getParentOrigin()
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
        getParentOrigin()
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
        getParentOrigin()
      );
    }
  }
  function getPlaywrightReplayState() {
    const state = playwrightReplayRef.current;
    if (!state)
      return null;
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
      if (!data || !data.type)
        return;
      if (!parentOrigin && event.origin && event.origin !== "null") {
        parentOrigin = event.origin;
      }
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
            getParentOrigin()
          );
          break;
      }
    });
  }

  // src/messages.ts
  function isStakTrakMessage(event) {
    return event.data && typeof event.data.type === "string" && event.data.type.startsWith("staktrak-");
  }

  // src/actionModel.ts
  function resultsToActions(results) {
    var _a2, _b, _c;
    const actions = [];
    const navigations = (results.pageNavigation || []).slice().sort((a, b) => a.timestamp - b.timestamp);
    const normalize = (u) => {
      var _a3;
      try {
        const url = new URL(u, ((_a3 = results.userInfo) == null ? void 0 : _a3.url) || "http://localhost");
        return url.origin + url.pathname.replace(/\/$/, "");
      } catch (e) {
        return u.replace(/[?#].*$/, "").replace(/\/$/, "");
      }
    };
    const navTimestampsFromClicks = /* @__PURE__ */ new Set();
    const clicks = ((_a2 = results.clicks) == null ? void 0 : _a2.clickDetails) || [];
    for (let i = 0; i < clicks.length; i++) {
      const cd = clicks[i];
      actions.push({
        type: "click",
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
        navTimestampsFromClicks.add(nav.timestamp);
        actions.push({
          type: "waitForURL",
          timestamp: nav.timestamp - 1,
          // ensure ordering between click and nav
          expectedUrl: nav.url,
          normalizedUrl: normalize(nav.url),
          navRefTimestamp: nav.timestamp
        });
      }
    }
    for (const nav of navigations) {
      if (!navTimestampsFromClicks.has(nav.timestamp)) {
        actions.push({ type: "goto", timestamp: nav.timestamp, url: nav.url, normalizedUrl: normalize(nav.url) });
      }
    }
    if (results.inputChanges) {
      for (const input of results.inputChanges) {
        if (input.action === "complete" || !input.action) {
          actions.push({
            type: "input",
            timestamp: input.timestamp,
            locator: { primary: input.elementSelector, fallbacks: [] },
            value: input.value
          });
        }
      }
    }
    if (results.formElementChanges) {
      for (const fe of results.formElementChanges) {
        actions.push({
          type: "form",
          timestamp: fe.timestamp,
          locator: { primary: fe.elementSelector, fallbacks: [] },
          formType: fe.type,
          value: fe.value,
          checked: fe.checked
        });
      }
    }
    if (results.assertions) {
      for (const asrt of results.assertions) {
        actions.push({
          type: "assertion",
          timestamp: asrt.timestamp,
          locator: { primary: asrt.selector, fallbacks: [] },
          value: asrt.value
        });
      }
    }
    actions.sort((a, b) => a.timestamp - b.timestamp || weightOrder(a.type) - weightOrder(b.type));
    refineLocators(actions);
    for (let i = actions.length - 1; i > 0; i--) {
      const current = actions[i];
      const previous = actions[i - 1];
      if (current.type === "waitForURL" && previous.type === "waitForURL" && current.normalizedUrl === previous.normalizedUrl) {
        actions.splice(i, 1);
      }
    }
    for (let i = actions.length - 1; i > 0; i--) {
      const current = actions[i];
      const previous = actions[i - 1];
      if (current.type === "input" && previous.type === "input" && ((_b = current.locator) == null ? void 0 : _b.primary) === ((_c = previous.locator) == null ? void 0 : _c.primary) && current.value === previous.value) {
        actions.splice(i, 1);
      }
    }
    return actions;
  }
  function weightOrder(type) {
    switch (type) {
      case "click":
        return 1;
      case "waitForURL":
        return 2;
      case "goto":
        return 3;
      default:
        return 4;
    }
  }
  function refineLocators(actions) {
    if (typeof document === "undefined")
      return;
    const seen = /* @__PURE__ */ new Set();
    for (const a of actions) {
      if (!a.locator)
        continue;
      const { primary, fallbacks } = a.locator;
      const validated = [];
      if (isUnique(primary))
        validated.push(primary);
      for (const fb of fallbacks) {
        if (validated.length >= 3)
          break;
        if (isUnique(fb))
          validated.push(fb);
      }
      if (validated.length === 0)
        continue;
      a.locator.primary = validated[0];
      a.locator.fallbacks = validated.slice(1);
      const key = a.locator.primary + "::" + a.type;
      if (seen.has(key) && a.locator.fallbacks.length > 0) {
        a.locator.primary = a.locator.fallbacks[0];
        a.locator.fallbacks = a.locator.fallbacks.slice(1);
      }
      seen.add(a.locator.primary + "::" + a.type);
    }
  }
  function isUnique(sel) {
    if (!sel || /^(html|body|div|span|p|button|input)$/i.test(sel))
      return false;
    try {
      const nodes = document.querySelectorAll(sel);
      return nodes.length === 1;
    } catch (e) {
      return false;
    }
  }

  // src/scenario.ts
  function buildScenario(results, actions) {
    var _a2, _b, _c, _d, _e;
    const startedAt = ((_a2 = results == null ? void 0 : results.time) == null ? void 0 : _a2.startedAt) || (((_b = actions[0]) == null ? void 0 : _b.timestamp) || Date.now());
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
  var RecordingManager = class {
    constructor() {
      this.trackingData = {
        pageNavigation: [],
        clicks: { clickCount: 0, clickDetails: [] },
        inputChanges: [],
        formElementChanges: [],
        assertions: [],
        keyboardActivities: [],
        mouseMovement: [],
        mouseScroll: [],
        focusChanges: [],
        visibilitychanges: [],
        windowSizes: [],
        touchEvents: [],
        audioVideoInteractions: []
      };
      this.capturedActions = [];
      this.actionIdCounter = 0;
    }
    /**
     * Handle an event from the iframe and store it
     */
    handleEvent(eventType, eventData) {
      switch (eventType) {
        case "click":
          this.trackingData.clicks.clickDetails.push(eventData);
          this.trackingData.clicks.clickCount++;
          break;
        case "nav":
        case "navigation":
          this.trackingData.pageNavigation.push({
            type: "navigation",
            url: eventData.url,
            timestamp: eventData.timestamp
          });
          break;
        case "input":
          this.trackingData.inputChanges.push({
            elementSelector: eventData.selector || "",
            value: eventData.value,
            timestamp: eventData.timestamp,
            action: "complete"
          });
          break;
        case "form":
          this.trackingData.formElementChanges.push({
            elementSelector: eventData.selector || "",
            type: eventData.formType || "input",
            checked: eventData.checked,
            value: eventData.value || "",
            text: eventData.text,
            timestamp: eventData.timestamp
          });
          break;
        case "assertion":
          this.trackingData.assertions.push({
            id: eventData.id,
            type: eventData.type || "hasText",
            selector: eventData.selector,
            value: eventData.value || "",
            timestamp: eventData.timestamp
          });
          break;
        default:
          return null;
      }
      const action = this.createAction(eventType, eventData);
      if (action) {
        this.capturedActions.push(action);
      }
      return action;
    }
    createAction(eventType, eventData) {
      const id = `${Date.now()}_${this.actionIdCounter++}`;
      const baseAction = {
        id,
        timestamp: eventData.timestamp || Date.now()
      };
      switch (eventType) {
        case "click":
          return __spreadProps(__spreadValues({}, baseAction), {
            type: "click",
            locator: eventData.selectors || eventData.locator,
            elementInfo: eventData.elementInfo
          });
        case "nav":
        case "navigation":
          return __spreadProps(__spreadValues({}, baseAction), {
            type: "goto",
            url: getRelativeUrl(eventData.url)
          });
        case "input":
          return __spreadProps(__spreadValues({}, baseAction), {
            type: "input",
            value: eventData.value,
            locator: eventData.locator || { primary: eventData.selector }
          });
        case "form":
          return __spreadProps(__spreadValues({}, baseAction), {
            type: "form",
            formType: eventData.formType,
            checked: eventData.checked,
            value: eventData.value,
            locator: eventData.locator || { primary: eventData.selector }
          });
        case "assertion":
          return __spreadProps(__spreadValues({}, baseAction), {
            type: "assertion",
            value: eventData.value,
            locator: { primary: eventData.selector, fallbacks: [] }
          });
        default:
          return __spreadProps(__spreadValues({}, baseAction), {
            type: eventType
          });
      }
    }
    /**
     * Remove an action by ID
     */
    removeAction(actionId) {
      const action = this.capturedActions.find((a) => a.id === actionId);
      if (!action)
        return false;
      this.capturedActions = this.capturedActions.filter((a) => a.id !== actionId);
      this.removeFromTrackingData(action);
      return true;
    }
    removeFromTrackingData(action) {
      const timestamp = action.timestamp;
      switch (action.type) {
        case "click":
          this.trackingData.clicks.clickDetails = this.trackingData.clicks.clickDetails.filter(
            (c) => c.timestamp !== timestamp
          );
          this.trackingData.clicks.clickCount = this.trackingData.clicks.clickDetails.length;
          break;
        case "goto":
          this.trackingData.pageNavigation = this.trackingData.pageNavigation.filter(
            (n) => n.timestamp !== timestamp
          );
          break;
        case "input":
          this.trackingData.inputChanges = this.trackingData.inputChanges.filter(
            (i) => i.timestamp !== timestamp
          );
          break;
        case "form":
          this.trackingData.formElementChanges = this.trackingData.formElementChanges.filter(
            (f) => f.timestamp !== timestamp
          );
          break;
        case "assertion":
          this.trackingData.assertions = this.trackingData.assertions.filter(
            (a) => a.timestamp !== timestamp
          );
          const clickBeforeAssertion = this.trackingData.clicks.clickDetails.filter((c) => c.timestamp < timestamp).sort((a, b) => b.timestamp - a.timestamp)[0];
          if (clickBeforeAssertion && timestamp - clickBeforeAssertion.timestamp < 1e3) {
            this.trackingData.clicks.clickDetails = this.trackingData.clicks.clickDetails.filter(
              (c) => c.timestamp !== clickBeforeAssertion.timestamp
            );
            this.trackingData.clicks.clickCount = this.trackingData.clicks.clickDetails.length;
          }
          break;
      }
    }
    /**
     * Generate Playwright test from current data
     */
    generateTest(url, options) {
      const actions = resultsToActions(this.trackingData);
      return generatePlaywrightTestFromActions(actions, __spreadValues({
        baseUrl: url
      }, options));
    }
    /**
     * Get current actions for UI display
     */
    getActions() {
      return [...this.capturedActions];
    }
    /**
     * Get tracking data (for compatibility)
     */
    getTrackingData() {
      return this.trackingData;
    }
    /**
     * Clear all recorded data
     */
    clear() {
      this.trackingData = {
        pageNavigation: [],
        clicks: { clickCount: 0, clickDetails: [] },
        inputChanges: [],
        formElementChanges: [],
        assertions: [],
        keyboardActivities: [],
        mouseMovement: [],
        mouseScroll: [],
        focusChanges: [],
        visibilitychanges: [],
        windowSizes: [],
        touchEvents: [],
        audioVideoInteractions: []
      };
      this.capturedActions = [];
      this.actionIdCounter = 0;
    }
    /**
     * Clear all actions (but keep recording)
     */
    clearAllActions() {
      this.clear();
    }
  };
  function escapeTextForAssertion(text) {
    if (!text)
      return "";
    return text.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  }
  function generatePlaywrightTestFromActions(actions, options = {}) {
    const { baseUrl = "" } = options;
    const needsInitialGoto = baseUrl && (actions.length === 0 || actions[0].type !== "goto");
    const initialGoto = needsInitialGoto ? `  await page.goto('${baseUrl}');
` : "";
    const body = actions.map((action) => {
      var _a2, _b, _c, _d, _e;
      switch (action.type) {
        case "goto":
          return `  await page.goto('${getRelativeUrl(action.url || baseUrl)}');`;
        case "waitForURL":
          if (action.normalizedUrl) {
            return `  await page.waitForURL('${action.normalizedUrl}');`;
          }
          return "";
        case "click": {
          const selector = ((_a2 = action.locator) == null ? void 0 : _a2.stableSelector) || ((_b = action.locator) == null ? void 0 : _b.primary);
          if (!selector)
            return "";
          return `  await page.click('${selector}');`;
        }
        case "input": {
          const selector = (_c = action.locator) == null ? void 0 : _c.primary;
          if (!selector || action.value === void 0)
            return "";
          const value = action.value.replace(/'/g, "\\'");
          return `  await page.fill('${selector}', '${value}');`;
        }
        case "form": {
          const selector = (_d = action.locator) == null ? void 0 : _d.primary;
          if (!selector)
            return "";
          if (action.formType === "checkbox" || action.formType === "radio") {
            if (action.checked) {
              return `  await page.check('${selector}');`;
            } else {
              return `  await page.uncheck('${selector}');`;
            }
          } else if (action.formType === "select" && action.value) {
            return `  await page.selectOption('${selector}', '${action.value}');`;
          }
          return "";
        }
        case "assertion": {
          const selector = (_e = action.locator) == null ? void 0 : _e.primary;
          if (!selector || action.value === void 0)
            return "";
          const escapedValue = escapeTextForAssertion(action.value);
          return `  await expect(page.locator('${selector}')).toContainText('${escapedValue}');`;
        }
        default:
          return "";
      }
    }).filter((line) => line !== "").join("\n");
    if (!initialGoto && !body)
      return "";
    return `import { test, expect } from '@playwright/test';

test('Recorded test', async ({ page }) => {
${initialGoto}${body.split("\n").filter((l) => l.trim()).map((l) => l).join("\n")}
});`;
  }
  function generatePlaywrightTest(url, trackingData) {
    try {
      const actions = resultsToActions(trackingData);
      return generatePlaywrightTestFromActions(actions, { baseUrl: url });
    } catch (error) {
      console.error("Error generating Playwright test:", error);
      return "";
    }
  }
  if (typeof window !== "undefined") {
    const existing = window.PlaywrightGenerator || {};
    existing.RecordingManager = RecordingManager;
    existing.generatePlaywrightTestFromActions = generatePlaywrightTestFromActions;
    existing.generatePlaywrightTest = generatePlaywrightTest;
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
    /**
     * Send event data to parent for recording
     */
    sendEventToParent(eventType, data) {
      window.parent.postMessage(
        {
          type: "staktrak-event",
          eventType,
          data
        },
        "*"
      );
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
      Object.values(this.memory.inputDebounceTimers).forEach((timer) => clearTimeout(timer));
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
          if (this.memory.selectionMode) {
            return;
          }
          const target = e.target;
          const isLabelForFormInput = (element) => {
            if (element.tagName !== "LABEL")
              return false;
            const label = element;
            if (label.control) {
              const control = label.control;
              return control.tagName === "INPUT" && (control.type === "radio" || control.type === "checkbox");
            }
            if (label.htmlFor) {
              const control = document.getElementById(label.htmlFor);
              return control && control.tagName === "INPUT" && (control.type === "radio" || control.type === "checkbox");
            }
            return false;
          };
          const isFormElement = target.tagName === "INPUT" && (target.type === "checkbox" || target.type === "radio") || isLabelForFormInput(target);
          if (!isFormElement) {
            this.results.clicks.clickCount++;
            const clickDetail = createClickDetail(e);
            this.results.clicks.clickDetails.push(clickDetail);
            this.sendEventToParent("click", clickDetail);
            window.parent.postMessage(
              {
                type: "staktrak-action-added",
                action: {
                  id: clickDetail.timestamp + "_click",
                  kind: "click",
                  timestamp: clickDetail.timestamp,
                  locator: {
                    primary: clickDetail.selectors.primary,
                    text: clickDetail.selectors.text
                  }
                }
              },
              "*"
            );
          }
          this.saveSessionState();
        };
        document.addEventListener("click", clickHandler);
        this.memory.listeners.push(() => document.removeEventListener("click", clickHandler));
      }
      if (this.config.mouseScroll) {
        const scrollHandler = () => {
          this.results.mouseScroll.push([window.scrollX, window.scrollY, getTimeStamp()]);
        };
        window.addEventListener("scroll", scrollHandler);
        this.memory.listeners.push(() => window.removeEventListener("scroll", scrollHandler));
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
          this.results.windowSizes.push([window.innerWidth, window.innerHeight, getTimeStamp()]);
        };
        window.addEventListener("resize", resizeHandler);
        this.memory.listeners.push(() => window.removeEventListener("resize", resizeHandler));
      }
      if (this.config.visibilitychange) {
        const visibilityHandler = () => {
          this.results.visibilitychanges.push([document.visibilityState, getTimeStamp()]);
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
        this.memory.listeners.push(() => document.removeEventListener("keypress", keyHandler));
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
        this.memory.listeners.push(() => document.removeEventListener("touchstart", touchHandler));
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
                const formChange = {
                  elementSelector: selector,
                  type: "select",
                  value: selectEl.value,
                  text: (selectedOption == null ? void 0 : selectedOption.text) || "",
                  timestamp: getTimeStamp()
                };
                this.results.formElementChanges.push(formChange);
                this.sendEventToParent("form", {
                  selector,
                  formType: "select",
                  value: selectEl.value,
                  text: (selectedOption == null ? void 0 : selectedOption.text) || "",
                  timestamp: formChange.timestamp
                });
                window.parent.postMessage(
                  {
                    type: "staktrak-action-added",
                    action: {
                      id: formChange.timestamp + "_form",
                      kind: "form",
                      timestamp: formChange.timestamp,
                      formType: formChange.type,
                      value: formChange.text
                    }
                  },
                  "*"
                );
              } else {
                const formChange = {
                  elementSelector: selector,
                  type: inputEl.type,
                  checked: inputEl.checked,
                  value: inputEl.value,
                  timestamp: getTimeStamp()
                };
                this.results.formElementChanges.push(formChange);
                this.sendEventToParent("form", {
                  selector,
                  formType: inputEl.type,
                  checked: inputEl.checked,
                  value: inputEl.value,
                  timestamp: formChange.timestamp
                });
                window.parent.postMessage(
                  {
                    type: "staktrak-action-added",
                    action: {
                      id: formChange.timestamp + "_form",
                      kind: "form",
                      timestamp: formChange.timestamp,
                      formType: formChange.type,
                      checked: formChange.checked,
                      value: formChange.value
                    }
                  },
                  "*"
                );
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
                const inputAction2 = {
                  elementSelector: selector,
                  value: inputEl.value,
                  timestamp: getTimeStamp(),
                  action: "complete"
                };
                this.results.inputChanges.push(inputAction2);
                this.sendEventToParent("input", {
                  selector,
                  value: inputEl.value,
                  timestamp: inputAction2.timestamp
                });
                window.parent.postMessage(
                  {
                    type: "staktrak-action-added",
                    action: {
                      id: inputAction2.timestamp + "_input",
                      kind: "input",
                      timestamp: inputAction2.timestamp,
                      value: inputAction2.value,
                      locator: { primary: selector, fallbacks: [] }
                    }
                  },
                  "*"
                );
                delete this.memory.inputDebounceTimers[elementId];
                this.saveSessionState();
              }, this.config.inputDebounceDelay);
              const inputAction = {
                elementSelector: selector,
                value: inputEl.value,
                timestamp: getTimeStamp(),
                action: "intermediate"
              };
              this.results.inputChanges.push(inputAction);
              window.parent.postMessage(
                {
                  type: "staktrak-action-added",
                  action: {
                    id: inputAction.timestamp + "_input",
                    kind: "input",
                    timestamp: inputAction.timestamp,
                    value: inputAction.value
                  }
                },
                "*"
              );
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
                const hadTimer = !!this.memory.inputDebounceTimers[elementId];
                if (hadTimer) {
                  clearTimeout(this.memory.inputDebounceTimers[elementId]);
                  delete this.memory.inputDebounceTimers[elementId];
                  const inputAction = {
                    elementSelector: selector,
                    value: inputEl.value,
                    timestamp: getTimeStamp(),
                    action: "complete"
                  };
                  this.results.inputChanges.push(inputAction);
                  this.sendEventToParent("input", {
                    selector,
                    value: inputEl.value,
                    timestamp: inputAction.timestamp
                  });
                  window.parent.postMessage(
                    {
                      type: "staktrak-action-added",
                      action: {
                        id: inputAction.timestamp + "_input",
                        kind: "input",
                        timestamp: inputAction.timestamp,
                        value: inputAction.value
                      }
                    },
                    "*"
                  );
                }
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
        const navAction = {
          type,
          url: document.URL,
          timestamp: getTimeStamp()
        };
        if (this.isRunning) {
          this.results.pageNavigation.push(navAction);
          this.sendEventToParent("navigation", navAction);
          window.parent.postMessage(
            {
              type: "staktrak-action-added",
              action: {
                id: navAction.timestamp + "_nav",
                kind: "nav",
                timestamp: navAction.timestamp,
                url: navAction.url
              }
            },
            "*"
          );
        }
        window.parent.postMessage({ type: "staktrak-page-navigation", data: document.URL }, "*");
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
      this.memory.alwaysListeners.push(() => window.removeEventListener("popstate", popstateHandler));
      const hashHandler = () => {
        recordStateChange("hashchange");
      };
      window.addEventListener("hashchange", hashHandler);
      this.memory.alwaysListeners.push(() => window.removeEventListener("hashchange", hashHandler));
      const anchorClickHandler = (e) => {
        const a = e.target.closest("a");
        if (!a)
          return;
        if (a.target && a.target !== "_self")
          return;
        const href = a.getAttribute("href");
        if (!href)
          return;
        try {
          const dest = new URL(href, window.location.href);
          if (dest.origin === window.location.origin) {
            const navAction = { type: "anchorClick", url: getRelativeUrl(dest.href), timestamp: getTimeStamp() };
            if (this.isRunning) {
              this.results.pageNavigation.push(navAction);
              window.parent.postMessage(
                {
                  type: "staktrak-action-added",
                  action: {
                    id: navAction.timestamp + "_nav",
                    kind: "nav",
                    timestamp: navAction.timestamp,
                    url: navAction.url
                  }
                },
                "*"
              );
            }
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
      if (this.memory.alwaysListeners.length > 0)
        return;
      const actionRemovalHandlers = {
        "staktrak-remove-navigation": (data) => {
          try {
            if (!data.timestamp) {
              console.warn("Missing timestamp for navigation removal");
              return false;
            }
            const initialLength = this.results.pageNavigation.length;
            this.results.pageNavigation = this.results.pageNavigation.filter(
              (nav) => nav.timestamp !== data.timestamp
            );
            return this.results.pageNavigation.length < initialLength;
          } catch (error) {
            console.error("Failed to remove navigation:", error);
            return false;
          }
        },
        "staktrak-remove-click": (data) => {
          try {
            if (!data.timestamp) {
              console.warn("Missing timestamp for click removal");
              return false;
            }
            const initialLength = this.results.clicks.clickDetails.length;
            this.results.clicks.clickDetails = this.results.clicks.clickDetails.filter(
              (click) => click.timestamp !== data.timestamp
            );
            return this.results.clicks.clickDetails.length < initialLength;
          } catch (error) {
            console.error("Failed to remove click:", error);
            return false;
          }
        },
        "staktrak-remove-input": (data) => {
          try {
            if (!data.timestamp) {
              console.warn("Missing timestamp for input removal");
              return false;
            }
            const initialLength = this.results.inputChanges.length;
            this.results.inputChanges = this.results.inputChanges.filter(
              (input) => input.timestamp !== data.timestamp
            );
            return this.results.inputChanges.length < initialLength;
          } catch (error) {
            console.error("Failed to remove input:", error);
            return false;
          }
        },
        "staktrak-remove-form": (data) => {
          try {
            if (!data.timestamp) {
              console.warn("Missing timestamp for form removal");
              return false;
            }
            const initialLength = this.results.formElementChanges.length;
            this.results.formElementChanges = this.results.formElementChanges.filter(
              (form) => form.timestamp !== data.timestamp
            );
            return this.results.formElementChanges.length < initialLength;
          } catch (error) {
            console.error("Failed to remove form change:", error);
            return false;
          }
        }
      };
      const messageHandler = (event) => {
        if (!isStakTrakMessage(event))
          return;
        const message = event.data;
        if (actionRemovalHandlers[message.type]) {
          const success = actionRemovalHandlers[message.type](message);
          if (!success) {
            console.warn(`Failed to process ${message.type}`);
          }
          return;
        }
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
                id: event.data.assertion.id,
                type: event.data.assertion.type || "hasText",
                selector: event.data.assertion.selector,
                value: event.data.assertion.value || "",
                timestamp: event.data.assertion.timestamp || getTimeStamp()
              });
            }
            break;
          case "staktrak-remove-assertion":
            if (event.data.assertionId) {
              const assertionToRemove = this.memory.assertions.find(
                (assertion) => assertion.id === event.data.assertionId
              );
              this.memory.assertions = this.memory.assertions.filter(
                (assertion) => assertion.id !== event.data.assertionId
              );
              if (assertionToRemove) {
                const assertionTime = assertionToRemove.timestamp;
                const clicksBefore = this.results.clicks.clickDetails.filter(
                  (click) => click.timestamp < assertionTime
                );
                if (clicksBefore.length > 0) {
                  const mostRecentClick = clicksBefore.reduce(
                    (latest, current) => current.timestamp > latest.timestamp ? current : latest
                  );
                  this.results.clicks.clickDetails = this.results.clicks.clickDetails.filter(
                    (click) => click.timestamp !== mostRecentClick.timestamp
                  );
                }
              }
            }
            break;
          case "staktrak-clear-assertions":
          case "staktrak-clear-all-actions":
            this.clearAllActions();
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
      this.memory.alwaysListeners.push(() => window.removeEventListener("message", messageHandler));
    }
    checkDebugInfo() {
      setTimeout(() => {
        if (isReactDevModeActive()) {
          window.parent.postMessage({ type: "staktrak-debug-init" }, "*");
        }
      }, 1500);
    }
    setSelectionMode(isActive) {
      var _a2;
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
              const assertionId = Date.now() + Math.random();
              const assertion = {
                id: assertionId,
                type: "hasText",
                selector,
                value: text,
                timestamp: getTimeStamp()
              };
              this.memory.assertions.push(assertion);
              this.sendEventToParent("assertion", assertion);
            }, 300);
          }
        };
        document.addEventListener("mouseup", mouseUpHandler);
        this.memory.listeners.push(() => document.removeEventListener("mouseup", mouseUpHandler));
      } else {
        document.body.classList.remove("staktrak-selection-active");
        (_a2 = window.getSelection()) == null ? void 0 : _a2.removeAllRanges();
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
      window.parent.postMessage({ type: "staktrak-results", data: this.results }, "*");
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
    clearAllActions() {
      this.results.pageNavigation = [];
      this.results.clicks.clickDetails = [];
      this.results.clicks.clickCount = 0;
      this.results.inputChanges = [];
      this.results.formElementChanges = [];
      this.memory.assertions = [];
    }
    attemptSessionRestoration() {
      try {
        const activeRecording = sessionStorage.getItem("stakTrakActiveRecording");
        if (!activeRecording) {
          return;
        }
        const recordingData = JSON.parse(activeRecording);
        if (recordingData && recordingData.isRecording && recordingData.version === "1.0") {
          const timeSinceLastSave = Date.now() - (recordingData.lastSaved || 0);
          const isLikelyIframeReload = timeSinceLastSave < 1e4;
          if (isLikelyIframeReload) {
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
            this.verifyEventListeners();
            window.parent.postMessage({ type: "staktrak-replay-ready" }, "*");
          } else {
            sessionStorage.removeItem("stakTrakActiveRecording");
          }
        } else {
          sessionStorage.removeItem("stakTrakActiveRecording");
        }
      } catch (error) {
        sessionStorage.removeItem("stakTrakActiveRecording");
      }
    }
    verifyEventListeners() {
      if (this.isRunning && this.memory.listeners.length === 0) {
        this.setupEventListeners();
      }
    }
    recoverRecording() {
      if (!this.isRunning) {
        return;
      }
      this.verifyEventListeners();
      this.saveSessionState();
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
    }
  };
  var userBehaviour = new UserBehaviorTracker();
  var initializeStakTrak = () => {
    userBehaviour.makeConfig({
      processData: (results) => {
      }
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
    if (!results.clicks || !results.clicks.clickDetails.length)
      return [];
    const last = results.clicks.clickDetails[results.clicks.clickDetails.length - 1];
    const sel = last.selectors;
    if (sel && sel.scores)
      return sel.scores;
    return [];
  };
  var src_default = userBehaviour;
  return __toCommonJS(src_exports);
})();
