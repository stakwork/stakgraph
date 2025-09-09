"use strict";
var stakReplay = (() => {
  var __defProp = Object.defineProperty;
  var __getOwnPropDesc = Object.getOwnPropertyDescriptor;
  var __getOwnPropNames = Object.getOwnPropertyNames;
  var __hasOwnProp = Object.prototype.hasOwnProperty;
  var __export = (target, all) => {
      for (var name in all)
        __defProp(target, name, { get: all[name], enumerable: !0 });
    },
    __copyProps = (to, from, except, desc) => {
      if ((from && typeof from == "object") || typeof from == "function")
        for (let key of __getOwnPropNames(from))
          !__hasOwnProp.call(to, key) &&
            key !== except &&
            __defProp(to, key, {
              get: () => from[key],
              enumerable:
                !(desc = __getOwnPropDesc(from, key)) || desc.enumerable,
            });
      return to;
    };
  var __toCommonJS = (mod) =>
    __copyProps(__defProp({}, "__esModule", { value: !0 }), mod);
  var replay_exports = {};
  __export(replay_exports, {
    DEFAULT_SPEED: () => DEFAULT_SPEED,
    INITIAL_DELAY: () => INITIAL_DELAY,
    MAX_DELAY: () => MAX_DELAY,
    MIN_DELAY: () => MIN_DELAY,
    convertToReplayActions: () => convertToReplayActions,
    createCursor: () => createCursor,
    createReplayStyles: () => createReplayStyles,
    executeAction: () => executeAction,
    findElement: () => findElement,
    highlightElement: () => highlightElement,
    initReplay: () => initReplay,
    moveCursorToElement: () => moveCursorToElement,
    pauseReplay: () => pauseReplay,
    resumeReplay: () => resumeReplay,
    selectOption: () => selectOption,
    showClickEffect: () => showClickEffect,
    stopReplay: () => stopReplay,
    typeText: () => typeText,
  });
  var DEFAULT_SPEED = 1,
    MIN_DELAY = 0,
    MAX_DELAY = 1e4,
    INITIAL_DELAY = 500,
    cursorRef = { current: null },
    statusRef = { current: "idle" },
    speedRef = { current: DEFAULT_SPEED },
    actionsRef = { current: [] },
    currentActionIndexRef = { current: 0 },
    isTypingRef = { current: !1 },
    timeoutIdsRef = { current: [] },
    timeoutRef = { current: null };
  function registerTimeout(timeoutId) {
    return (timeoutIdsRef.current.push(timeoutId), timeoutId);
  }
  function clearAllTimeouts() {
    (timeoutIdsRef.current.forEach(clearTimeout), (timeoutIdsRef.current = []));
  }
  function convertToBrowserSelector(selector) {
    var _a;
    if (!selector) return selector;
    if (selector.includes(":has-text(")) {
      let textMatch = selector.match(/:has-text\("([^"]+)"\)/);
      if (textMatch) {
        let text = textMatch[1],
          tagMatch = selector.match(/^([a-zA-Z]+)/),
          tagName = tagMatch ? tagMatch[1] : "*",
          elements = Array.from(document.querySelectorAll(tagName));
        for (let element of elements)
          if (
            ((_a = element.textContent) == null ? void 0 : _a.trim()) === text
          ) {
            let uniqueSelector = createUniqueSelector(element);
            if (uniqueSelector && isValidSelector(uniqueSelector))
              return uniqueSelector;
          }
        return tagName;
      }
    }
    return (
      (selector = selector.replace(/:visible/g, "")),
      (selector = selector.replace(/:enabled/g, "")),
      (selector = selector.replace(/>>.*$/g, "")),
      selector.trim()
    );
  }
  function findBestSelector(clickDetail) {
    let { selectors } = clickDetail;
    if (selectors.primary) {
      let convertedPrimary = convertToBrowserSelector(selectors.primary);
      if (convertedPrimary && isValidSelector(convertedPrimary))
        return convertedPrimary;
    }
    if (
      selectors.text &&
      (selectors.tagName === "button" ||
        selectors.tagName === "a" ||
        selectors.role === "button" ||
        selectors.role === "link")
    ) {
      let textBasedSelector = findElementByText(
        selectors.tagName,
        selectors.text,
      );
      if (textBasedSelector) return textBasedSelector;
    }
    if (selectors.ariaLabel) {
      let ariaSelector = `[aria-label="${selectors.ariaLabel}"]`;
      if (isValidSelector(ariaSelector)) return ariaSelector;
    }
    for (let fallback of selectors.fallbacks) {
      let convertedFallback = convertToBrowserSelector(fallback);
      if (convertedFallback && isValidSelector(convertedFallback))
        return convertedFallback;
    }
    if (selectors.role) {
      let roleSelector = `[role="${selectors.role}"]`;
      if (isValidSelector(roleSelector)) return roleSelector;
    }
    if (clickDetail.elementInfo) {
      let fromElementInfo = createSelectorFromElementInfo(
        clickDetail.elementInfo,
      );
      if (fromElementInfo && isValidSelector(fromElementInfo))
        return fromElementInfo;
    }
    return selectors.tagName && isValidSelector(selectors.tagName)
      ? selectors.tagName
      : null;
  }
  function createSelectorFromElementInfo(elementInfo) {
    let { tagName, id, className, attributes } = elementInfo;
    if (id) {
      let idSelector = `#${id}`;
      if (isValidSelector(idSelector)) return idSelector;
    }
    if (attributes && attributes["data-testid"]) {
      let testIdSelector = `[data-testid="${attributes["data-testid"]}"]`;
      if (isValidSelector(testIdSelector)) return testIdSelector;
    }
    if (className) {
      let classes = className.split(" ").filter((cls) => cls && cls.length > 0);
      if (classes.length > 0) {
        for (let cls of classes)
          if (!cls.match(/^[a-zA-Z0-9_-]*[0-9a-f]{6,}/) && cls.length < 30) {
            let classSelector = `${tagName}.${cls}`;
            if (
              isValidSelector(classSelector) &&
              document.querySelectorAll(classSelector).length === 1
            )
              return classSelector;
          }
      }
    }
    if (attributes) {
      let priorityAttrs = ["name", "type", "role", "aria-label"];
      for (let attr of priorityAttrs)
        if (attributes[attr]) {
          let attrSelector = `${tagName}[${attr}="${attributes[attr]}"]`;
          if (
            isValidSelector(attrSelector) &&
            document.querySelectorAll(attrSelector).length <= 3
          )
            return attrSelector;
        }
    }
    return null;
  }
  function findElementByText(tagName, text) {
    var _a, _b, _c;
    if (!text || text.length === 0) return null;
    let elements = Array.from(document.querySelectorAll(tagName)),
      matchingElement = null;
    for (let element of elements)
      if (((_a = element.textContent) == null ? void 0 : _a.trim()) === text) {
        matchingElement = element;
        break;
      }
    if (!matchingElement)
      for (let element of elements) {
        let elementText =
          (_b = element.textContent) == null ? void 0 : _b.trim();
        if (elementText && elementText.includes(text)) {
          matchingElement = element;
          break;
        }
      }
    if (!matchingElement) return null;
    let uniqueSelector = createUniqueSelector(matchingElement);
    if (uniqueSelector && isValidSelector(uniqueSelector)) {
      let foundElement = document.querySelector(uniqueSelector);
      if (
        foundElement &&
        (_c = foundElement.textContent) != null &&
        _c.trim().includes(text)
      )
        return uniqueSelector;
    }
    return null;
  }
  function findElement(selector) {
    if (!selector || selector.trim() === "") return null;
    let browserSelector = convertToBrowserSelector(selector);
    if (browserSelector && isValidSelector(browserSelector)) {
      let element = document.querySelector(browserSelector);
      if (element) return element;
    }
    let strategies = [
      () => findByDataTestId(selector),
      () => findByClass(selector),
      () => findById(selector),
      () => findByAriaLabel(selector),
      () => findByRole(selector),
      () => findByTextContent(selector),
      () => findByCoordinates(selector),
    ];
    for (let strategy of strategies)
      try {
        let element = strategy();
        if (element)
          return (
            console.log(
              `Found element using fallback strategy for selector: ${selector}`,
            ),
            element
          );
      } catch (error) {
        console.warn(`Strategy failed for selector ${selector}:`, error);
      }
    return (
      console.warn(`Could not find element for selector: ${selector}`),
      null
    );
  }
  function findByCoordinates(selector) {
    let clickableElements = document.querySelectorAll(
      'button, a, input, select, [role="button"], [onclick]',
    );
    return clickableElements.length > 0 ? clickableElements[0] : null;
  }
  function findByTextContent(selector) {
    var _a;
    let text = null,
      tagName = "*";
    if (selector.includes('text="')) {
      let textMatch = selector.match(/text="([^"]+)"/);
      text = textMatch ? textMatch[1] : null;
    } else if (selector.includes('textContent="')) {
      let textMatch = selector.match(/textContent="([^"]+)"/);
      text = textMatch ? textMatch[1] : null;
    } else if (selector.includes(":has-text(")) {
      let textMatch = selector.match(/:has-text\("([^"]+)"\)/);
      text = textMatch ? textMatch[1] : null;
    }
    let tagMatch = selector.match(/^([a-zA-Z]+)/);
    if ((tagMatch && (tagName = tagMatch[1]), !text)) return null;
    let elements = Array.from(document.querySelectorAll(tagName));
    for (let element of elements) {
      let elementText = (_a = element.textContent) == null ? void 0 : _a.trim();
      if (
        elementText === text ||
        (elementText != null && elementText.includes(text))
      )
        return element;
    }
    return null;
  }
  function createUniqueSelector(element) {
    var _a;
    if (element.id && /^[a-zA-Z][\w-]*$/.test(element.id)) {
      let idSelector = `#${element.id}`;
      if (document.querySelectorAll(idSelector).length === 1) return idSelector;
    }
    let testId = (_a = element.dataset) == null ? void 0 : _a.testid;
    if (testId) {
      let testIdSelector = `[data-testid="${testId}"]`;
      if (document.querySelectorAll(testIdSelector).length === 1)
        return testIdSelector;
    }
    let ariaLabel = element.getAttribute("aria-label");
    if (ariaLabel) {
      let ariaSelector = `[aria-label="${ariaLabel}"]`;
      if (document.querySelectorAll(ariaSelector).length === 1)
        return ariaSelector;
    }
    let tagName = element.tagName.toLowerCase(),
      classes = Array.from(element.classList).filter(
        (cls) =>
          !cls.match(/^[a-zA-Z0-9_-]*[0-9a-f]{6,}/) &&
          !cls.includes("emotion-") &&
          !cls.includes("css-") &&
          !cls.includes("module__") &&
          cls.length < 30,
      );
    if (classes.length > 0)
      for (let i = 1; i <= Math.min(classes.length, 3); i++) {
        let classSelector = `${tagName}.${classes.slice(0, i).join(".")}`;
        if (
          isValidSelector(classSelector) &&
          document.querySelectorAll(classSelector).length === 1
        )
          return classSelector;
      }
    let attributes = ["type", "name", "role", "title"];
    for (let attr of attributes) {
      let value = element.getAttribute(attr);
      if (value) {
        let attrSelector = `${tagName}[${attr}="${value}"]`;
        if (
          isValidSelector(attrSelector) &&
          document.querySelectorAll(attrSelector).length === 1
        )
          return attrSelector;
      }
    }
    let parent = element.parentElement;
    if (parent) {
      let index = Array.from(parent.children).indexOf(element);
      if (index >= 0) {
        let nthSelector = `${tagName}:nth-child(${index + 1})`;
        if (isValidSelector(nthSelector)) return nthSelector;
      }
      let typeIndex = Array.from(parent.children)
        .filter((child) => child.tagName === element.tagName)
        .indexOf(element);
      if (typeIndex >= 0) {
        let nthTypeSelector = `${tagName}:nth-of-type(${typeIndex + 1})`;
        if (isValidSelector(nthTypeSelector)) return nthTypeSelector;
      }
    }
    return tagName;
  }
  function convertToReplayActions(trackingData) {
    var _a;
    if (!trackingData)
      return (
        console.error("No tracking data provided to convertToReplayActions"),
        []
      );
    console.log("Converting tracking data to replay actions:", trackingData);
    let actions = [];
    try {
      let { clicks, inputChanges, formElementChanges } = trackingData;
      ((_a = clicks == null ? void 0 : clicks.clickDetails) != null &&
        _a.length &&
        clicks.clickDetails.forEach((clickDetail, index) => {
          let detail = clickDetail,
            bestSelector = findBestSelector(detail);
          if (!bestSelector) {
            if (
              (console.warn(
                "Could not find valid selector for click detail:",
                detail,
              ),
              detail.selectors.text)
            ) {
              let textElement = findElementByText(
                detail.selectors.tagName,
                detail.selectors.text,
              );
              textElement && (bestSelector = textElement);
            }
            bestSelector || (bestSelector = detail.selectors.tagName || "div");
          }
          bestSelector &&
            actions.push({
              type: "click",
              selector: bestSelector,
              timestamp: detail.timestamp,
              x: detail.x,
              y: detail.y,
            });
        }),
        inputChanges != null &&
          inputChanges.length &&
          inputChanges
            .filter((change) => change.action === "complete" || !change.action)
            .forEach((change) => {
              if (
                !change.elementSelector.includes('type="checkbox"') &&
                !change.elementSelector.includes('type="radio"')
              ) {
                let validSelector = validateAndFixSelector(
                  change.elementSelector,
                );
                validSelector &&
                  actions.push({
                    type: "input",
                    selector: validSelector,
                    value: change.value,
                    timestamp: change.timestamp,
                  });
              }
            }),
        formElementChanges != null &&
          formElementChanges.length &&
          formElementChanges.forEach((change) => {
            if (!change.elementSelector) return;
            let validSelector = validateAndFixSelector(change.elementSelector);
            validSelector &&
              (change.type === "checkbox" || change.type === "radio"
                ? actions.push({
                    type: change.checked ? "check" : "uncheck",
                    selector: validSelector,
                    value: change.value,
                    timestamp: change.timestamp,
                  })
                : change.type === "select" &&
                  actions.push({
                    type: "select",
                    selector: validSelector,
                    value: change.value,
                    timestamp: change.timestamp,
                  }));
          }));
    } catch (e) {
      console.error("Error processing tracking data", e);
    }
    (actions.length === 0 &&
      console.warn("No actions extracted from tracking data"),
      actions.sort((a, b) => a.timestamp - b.timestamp));
    for (let i = 1; i < actions.length; i++)
      actions[i].timestamp - actions[i - 1].timestamp < 250 &&
        (actions[i].timestamp = actions[i - 1].timestamp + 250);
    return (
      console.log("Converted replay actions:", actions),
      actions.map((action) => ({
        type: action.type || "click",
        selector: action.selector || "[data-testid]",
        timestamp: action.timestamp || Date.now(),
        x: action.x || 100,
        y: action.y || 100,
        value: action.value || "",
      }))
    );
  }
  function validateAndFixSelector(selector) {
    if (!selector || selector === "undefined" || selector === "null")
      return null;
    let cleanSelector = selector.trim(),
      browserSelector = convertToBrowserSelector(cleanSelector);
    if (browserSelector && isValidSelector(browserSelector))
      return browserSelector;
    console.warn(`Invalid selector: ${cleanSelector}. Attempting to fix.`);
    let fixStrategies = [
      () => fixDataTestIdSelector(cleanSelector),
      () => fixClassSelector(cleanSelector),
      () => fixIdSelector(cleanSelector),
      () => fixTagSelector(cleanSelector),
      () => fixTextSelector(cleanSelector),
    ];
    for (let strategy of fixStrategies) {
      let fixedSelector = strategy();
      if (fixedSelector && isValidSelector(fixedSelector))
        return (
          console.log(`Fixed selector: ${cleanSelector} -> ${fixedSelector}`),
          fixedSelector
        );
    }
    return (console.error(`Could not fix selector: ${cleanSelector}`), null);
  }
  function fixTextSelector(selector) {
    var _a;
    if (selector.includes(":has-text(")) {
      let textMatch = selector.match(/:has-text\("([^"]+)"\)/),
        tagMatch = selector.match(/^([a-zA-Z]+)/);
      if (textMatch && tagMatch) {
        let text = textMatch[1],
          tagName = tagMatch[1],
          elements = Array.from(document.querySelectorAll(tagName));
        for (let element of elements)
          if (
            ((_a = element.textContent) == null ? void 0 : _a.trim()) === text
          ) {
            let uniqueSelector = createUniqueSelector(element);
            if (uniqueSelector && isValidSelector(uniqueSelector))
              return uniqueSelector;
          }
        return tagName;
      }
    }
    return null;
  }
  function fixDataTestIdSelector(selector) {
    if (!selector.includes("data-testid=")) return null;
    let testIdMatch = selector.match(/data-testid="([^"]+)"/);
    return testIdMatch && testIdMatch[1]
      ? `[data-testid="${testIdMatch[1]}"]`
      : null;
  }
  function fixClassSelector(selector) {
    if (!selector.includes("class=")) return null;
    let classMatch = selector.match(/class="([^"]+)"/);
    if (classMatch && classMatch[1]) {
      let classNames = classMatch[1].split(" ").filter((cls) => cls.length > 0);
      if (classNames.length > 0) return `.${classNames[0]}`;
    }
    return null;
  }
  function fixIdSelector(selector) {
    if (!selector.includes("id=")) return null;
    let idMatch = selector.match(/id="([^"]+)"/);
    return idMatch && idMatch[1] ? `#${idMatch[1]}` : null;
  }
  function fixTagSelector(selector) {
    let tagMatch = selector.match(/^([a-zA-Z]+)/);
    return tagMatch && tagMatch[1] ? tagMatch[1] : null;
  }
  function isValidSelector(selector) {
    if (!selector || selector.trim() === "") return !1;
    try {
      return (document.querySelector(selector), !0);
    } catch (e) {
      return !1;
    }
  }
  function findByDataTestId(selector) {
    var _a;
    if (!selector.includes("data-testid")) return null;
    let testId =
      (_a = selector.match(/data-testid="([^"]+)"/)) == null ? void 0 : _a[1];
    return testId ? document.querySelector(`[data-testid="${testId}"]`) : null;
  }
  function findByClass(selector) {
    if (!selector.includes(".")) return null;
    let classes = selector.match(/\.([^\s.#\[\]]+)/g);
    if (classes && classes.length > 0) {
      let className = classes[0].substring(1);
      return document.querySelector(`.${className}`);
    }
    return null;
  }
  function findById(selector) {
    if (!selector.includes("#")) return null;
    let ids = selector.match(/#([^\s.#\[\]]+)/g);
    if (ids && ids.length > 0) {
      let id = ids[0].substring(1);
      return document.querySelector(`#${id}`);
    }
    return null;
  }
  function findByAriaLabel(selector) {
    let ariaMatch = selector.match(/\[aria-label="([^"]+)"\]/);
    return ariaMatch
      ? document.querySelector(`[aria-label="${ariaMatch[1]}"]`)
      : null;
  }
  function findByRole(selector) {
    let roleMatch = selector.match(/\[role="([^"]+)"\]/);
    return roleMatch
      ? document.querySelector(`[role="${roleMatch[1]}"]`)
      : null;
  }
  function createCursor() {
    let cursor = document.createElement("div");
    return (
      (cursor.className = "replay-cursor"),
      (cursor.style.position = "fixed"),
      (cursor.style.width = "24px"),
      (cursor.style.height = "24px"),
      (cursor.style.borderRadius = "50%"),
      (cursor.style.backgroundColor = "rgba(255, 0, 0, 0.6)"),
      (cursor.style.border = "2px solid white"),
      (cursor.style.boxShadow = "0 0 10px rgba(255, 0, 0, 0.8)"),
      (cursor.style.zIndex = "9999"),
      (cursor.style.pointerEvents = "none"),
      (cursor.style.display = "none"),
      (cursor.style.transform = "translate(-50%, -50%)"),
      (cursor.style.transition =
        "transform 0.1s ease-in-out, left 0.1s ease-in-out, top 0.1s ease-in-out"),
      document.body.appendChild(cursor),
      cursor
    );
  }
  function createReplayStyles() {
    let style = document.createElement("style");
    return (
      (style.textContent = `
    @keyframes staktrak-click-ripple {
      0% {
        transform: translate(-50%, -50%) scale(1);
        opacity: 1;
      }
      100% {
        transform: translate(-50%, -50%) scale(8);
        opacity: 0;
      }
    }
    
    @keyframes staktrak-pulse {
      0% { transform: scale(1); }
      50% { transform: scale(1.03); }
      100% { transform: scale(1); }
    }
    
    .staktrak-replay-pulse {
      animation: staktrak-pulse 0.5s ease-in-out infinite;
    }
  `),
      document.head.appendChild(style),
      style
    );
  }
  function showClickEffect(cursorRef2) {
    if (!cursorRef2.current) return;
    let ripple = document.createElement("div");
    ((ripple.className = "staktrak-click-ripple"),
      (ripple.style.position = "fixed"),
      (ripple.style.left = cursorRef2.current.style.left),
      (ripple.style.top = cursorRef2.current.style.top),
      (ripple.style.transform = "translate(-50%, -50%)"),
      (ripple.style.width = "20px"),
      (ripple.style.height = "20px"),
      (ripple.style.background = "rgba(255, 0, 0, 0.5)"),
      (ripple.style.borderRadius = "50%"),
      (ripple.style.zIndex = "9998"),
      (ripple.style.pointerEvents = "none"),
      (ripple.style.animation = "staktrak-click-ripple 1s ease-out forwards"),
      document.body.appendChild(ripple),
      (cursorRef2.current.style.transform = "translate(-50%, -50%) scale(0.8)"),
      setTimeout(() => {
        cursorRef2.current &&
          (cursorRef2.current.style.transform =
            "translate(-50%, -50%) scale(1)");
      }, 200),
      setTimeout(() => {
        ripple.parentNode && ripple.parentNode.removeChild(ripple);
      }, 1e3));
  }
  function highlightElement(element, speedRef2) {
    (element.classList.add("staktrak-replay-pulse"),
      setTimeout(() => {
        element && element.classList.remove("staktrak-replay-pulse");
      }, 200 / speedRef2.current));
  }
  function moveCursorToElement(element, cursorRef2, statusRef2) {
    return new Promise((resolve) => {
      if (!element || !cursorRef2.current || statusRef2.current !== "playing") {
        resolve();
        return;
      }
      let rect = element.getBoundingClientRect(),
        targetX = rect.left + rect.width / 2,
        targetY = rect.top + rect.height / 2;
      ((cursorRef2.current.style.display = "block"),
        element.scrollIntoView({ behavior: "smooth", block: "center" }),
        setTimeout(() => {
          (cursorRef2.current &&
            ((cursorRef2.current.style.left = `${targetX}px`),
            (cursorRef2.current.style.top = `${targetY}px`)),
            setTimeout(resolve, 300));
        }, 150));
    });
  }
  function typeText(
    element,
    value,
    speedRef2,
    statusRef2,
    isTypingRef2,
    registerTimeout2,
  ) {
    return new Promise((resolve) => {
      if (statusRef2.current !== "playing") {
        resolve();
        return;
      }
      ((isTypingRef2.current = !0), element.focus(), (element.value = ""));
      let index = 0,
        typeChar = () => {
          if (statusRef2.current !== "playing") {
            ((isTypingRef2.current = !1), resolve());
            return;
          }
          index < value.length
            ? ((element.value += value[index]),
              element.dispatchEvent(new Event("input", { bubbles: !0 })),
              index++,
              registerTimeout2(setTimeout(typeChar, 70 / speedRef2.current)))
            : (element.dispatchEvent(new Event("change", { bubbles: !0 })),
              (isTypingRef2.current = !1),
              resolve());
        };
      typeChar();
    });
  }
  function selectOption(
    element,
    value,
    speedRef2,
    statusRef2,
    registerTimeout2,
  ) {
    return new Promise((resolve) => {
      if (statusRef2.current !== "playing") {
        resolve();
        return;
      }
      (element.focus(),
        registerTimeout2(
          setTimeout(() => {
            (statusRef2.current === "playing" &&
              ((element.value = value),
              element.dispatchEvent(new Event("change", { bubbles: !0 }))),
              resolve());
          }, 50 / speedRef2.current),
        ));
    });
  }
  async function executeAction(
    index,
    actionsRef2,
    statusRef2,
    cursorRef2,
    speedRef2,
    isTypingRef2,
    registerTimeout2,
    setCurrentActionIndex,
    setStatus,
    timeoutRef2,
  ) {
    if (statusRef2.current !== "playing") return;
    if (index >= actionsRef2.current.length) {
      (setCurrentActionIndex(actionsRef2.current.length - 1),
        setStatus("completed"),
        window.parent.postMessage(
          {
            type: "staktrak-replay-completed",
            totalActions: actionsRef2.current.length,
          },
          "*",
        ),
        setTimeout(() => {
          window.parent.postMessage({ type: "staktrak-replay-fadeout" }, "*");
        }, 100),
        cursorRef2.current && (cursorRef2.current.style.display = "none"));
      return;
    }
    let action = actionsRef2.current[index];
    try {
      let element = findElement(action.selector),
        attempts = 0,
        maxAttempts = 5;
      for (; !element && attempts < maxAttempts; ) {
        attempts++;
        let delay2 = Math.min(500 * attempts, 2e3);
        if (
          (await new Promise((resolve) => setTimeout(resolve, delay2)),
          (element = findElement(action.selector)),
          statusRef2.current !== "playing")
        )
          return;
      }
      if (!element) {
        (console.warn(
          `Could not find element for action ${index}: ${action.type} on ${action.selector} after ${maxAttempts} attempts`,
        ),
          setCurrentActionIndex(index + 1),
          executeAction(
            index + 1,
            actionsRef2,
            statusRef2,
            cursorRef2,
            speedRef2,
            isTypingRef2,
            registerTimeout2,
            setCurrentActionIndex,
            setStatus,
            timeoutRef2,
          ));
        return;
      }
      if (
        (setCurrentActionIndex(index),
        window.parent.postMessage(
          {
            type: "staktrak-replay-progress",
            currentAction: index,
            totalActions: actionsRef2.current.length,
            action,
          },
          "*",
        ),
        await moveCursorToElement(element, cursorRef2, statusRef2),
        statusRef2.current !== "playing")
      )
        return;
      switch ((highlightElement(element, speedRef2), action.type)) {
        case "click":
          (showClickEffect(cursorRef2),
            element.scrollIntoView({ behavior: "smooth", block: "center" }),
            await new Promise((resolve) => setTimeout(resolve, 50)));
          try {
            element.focus();
          } catch (e) {
            console.warn("Could not focus element:", e);
          }
          try {
            (element.dispatchEvent(
              new MouseEvent("mousedown", {
                bubbles: !0,
                cancelable: !0,
                view: window,
              }),
            ),
              await new Promise((resolve) => setTimeout(resolve, 10)),
              element.dispatchEvent(
                new MouseEvent("mouseup", {
                  bubbles: !0,
                  cancelable: !0,
                  view: window,
                }),
              ),
              await new Promise((resolve) => setTimeout(resolve, 10)),
              element.click(),
              element.dispatchEvent(
                new MouseEvent("click", {
                  bubbles: !0,
                  cancelable: !0,
                  view: window,
                }),
              ));
          } catch (clickError) {
            console.error("Error during click operation:", clickError);
          }
          await new Promise((resolve) => setTimeout(resolve, 50));
          break;
        case "input":
          await typeText(
            element,
            action.value || "",
            speedRef2,
            statusRef2,
            isTypingRef2,
            registerTimeout2,
          );
          break;
        case "select":
          await selectOption(
            element,
            action.value || "",
            speedRef2,
            statusRef2,
            registerTimeout2,
          );
          break;
        case "check":
          ((element.checked = !0),
            element.dispatchEvent(new Event("change", { bubbles: !0 })));
          break;
        case "uncheck":
          ((element.checked = !1),
            element.dispatchEvent(new Event("change", { bubbles: !0 })));
          break;
      }
      if (statusRef2.current !== "playing") return;
      let nextAction = actionsRef2.current[index + 1],
        delay = 500;
      if (nextAction && action.timestamp && nextAction.timestamp) {
        let timeDiff = nextAction.timestamp - action.timestamp;
        delay = Math.min(
          MAX_DELAY,
          Math.max(MIN_DELAY, timeDiff / speedRef2.current),
        );
      }
      timeoutRef2.current = registerTimeout2(
        setTimeout(() => {
          statusRef2.current === "playing"
            ? (setCurrentActionIndex(index + 1),
              executeAction(
                index + 1,
                actionsRef2,
                statusRef2,
                cursorRef2,
                speedRef2,
                isTypingRef2,
                registerTimeout2,
                setCurrentActionIndex,
                setStatus,
                timeoutRef2,
              ))
            : console.warn(
                `Not moving to next action because status is ${statusRef2.current}`,
              );
        }, delay),
      );
    } catch (error) {
      (console.error("Error executing action:", error),
        window.parent.postMessage(
          { type: "staktrak-replay-error", error: error.message, action },
          "*",
        ),
        statusRef2.current === "playing" &&
          (timeoutRef2.current = registerTimeout2(
            setTimeout(() => {
              (setCurrentActionIndex(index + 1),
                executeAction(
                  index + 1,
                  actionsRef2,
                  statusRef2,
                  cursorRef2,
                  speedRef2,
                  isTypingRef2,
                  registerTimeout2,
                  setCurrentActionIndex,
                  setStatus,
                  timeoutRef2,
                ));
            }, 2e3),
          )));
    }
  }
  function pauseReplay() {
    statusRef.current === "playing" &&
      ((statusRef.current = "paused"),
      window.parent.postMessage({ type: "staktrak-replay-paused" }, "*"));
  }
  function resumeReplay() {
    statusRef.current === "paused" &&
      ((statusRef.current = "playing"),
      executeAction(
        currentActionIndexRef.current,
        actionsRef,
        statusRef,
        cursorRef,
        speedRef,
        isTypingRef,
        registerTimeout,
        (index) => {
          currentActionIndexRef.current = index;
        },
        (status) => {
          statusRef.current = status;
        },
        timeoutRef,
      ),
      window.parent.postMessage({ type: "staktrak-replay-resumed" }, "*"));
  }
  function stopReplay() {
    (clearAllTimeouts(),
      (statusRef.current = "idle"),
      cursorRef.current && (cursorRef.current.style.display = "none"),
      document.querySelectorAll(".staktrak-click-ripple").forEach((ripple) => {
        ripple.parentNode && ripple.parentNode.removeChild(ripple);
      }),
      document.querySelectorAll(".staktrak-replay-pulse").forEach((element) => {
        (element.classList.remove("staktrak-replay-pulse"),
          (element.style.outline = ""),
          (element.style.boxShadow = ""),
          (element.style.zIndex = ""),
          (element.style.transition = ""));
      }),
      window.parent.postMessage({ type: "staktrak-replay-stopped" }, "*"));
  }
  function initReplay() {
    let replayStyles = createReplayStyles(),
      cursor = createCursor();
    ((cursorRef.current = cursor),
      (statusRef.current = "idle"),
      (speedRef.current = DEFAULT_SPEED),
      (actionsRef.current = []),
      (currentActionIndexRef.current = 0),
      (isTypingRef.current = !1),
      (timeoutIdsRef.current = []),
      (timeoutRef.current = null),
      window.addEventListener("message", (event) => {
        let { data } = event;
        if (!(!data || !data.type))
          switch (data.type) {
            case "staktrak-replay-actions":
              let actions = convertToReplayActions(data.actions);
              actionsRef.current = actions || [];
              break;
            case "staktrak-replay-start":
              (clearAllTimeouts(),
                (statusRef.current = "playing"),
                (currentActionIndexRef.current = 0),
                (speedRef.current = data.speed || DEFAULT_SPEED),
                cursorRef.current &&
                  (cursorRef.current.style.display = "block"),
                setTimeout(() => {
                  executeAction(
                    0,
                    actionsRef,
                    statusRef,
                    cursorRef,
                    speedRef,
                    isTypingRef,
                    registerTimeout,
                    (index) => {
                      currentActionIndexRef.current = index;
                    },
                    (status) => {
                      statusRef.current = status;
                    },
                    timeoutRef,
                  );
                }, INITIAL_DELAY / speedRef.current));
              break;
            case "staktrak-replay-pause":
              pauseReplay();
              break;
            case "staktrak-replay-resume":
              resumeReplay();
              break;
            case "staktrak-replay-stop":
              stopReplay();
              break;
            case "staktrak-replay-speed":
              speedRef.current = data.speed || DEFAULT_SPEED;
              break;
            case "staktrak-replay-ping":
              window.parent.postMessage({ type: "staktrak-replay-ready" }, "*");
              break;
          }
      }),
      window.parent.postMessage({ type: "staktrak-replay-ready" }, "*"));
  }
  document.addEventListener("DOMContentLoaded", () => {
    initReplay();
  });
  document.readyState === "loading"
    ? document.addEventListener("DOMContentLoaded", initReplay)
    : initReplay();
  return __toCommonJS(replay_exports);
})();
