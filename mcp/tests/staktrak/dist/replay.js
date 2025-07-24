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
    MIN_DELAY = 1e3,
    MAX_DELAY = 5e3,
    INITIAL_DELAY = 2e3,
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
        clicks.clickDetails.forEach(([x, y, selector, timestamp]) => {
          if (!selector || selector === "undefined" || selector === "null") {
            console.warn("Skipping click with invalid selector:", selector);
            return;
          }
          let cleanSelector = selector.trim();
          try {
            (document.querySelector(cleanSelector),
              actions.push({
                type: "click",
                selector: cleanSelector,
                timestamp,
                x,
                y,
              }));
          } catch (e) {
            if (
              (console.warn(
                `Invalid selector in click event: ${cleanSelector}. Attempting to fix.`,
              ),
              cleanSelector.includes("data-testid="))
            )
              try {
                let testIdMatch = cleanSelector.match(/data-testid="([^"]+)"/);
                if (testIdMatch && testIdMatch[1]) {
                  let simpleSelector = `[data-testid="${testIdMatch[1]}"]`;
                  try {
                    (document.querySelector(simpleSelector),
                      actions.push({
                        type: "click",
                        selector: simpleSelector,
                        timestamp,
                        x,
                        y,
                      }));
                  } catch (err) {
                    actions.push({
                      type: "click",
                      selector: "[data-testid]",
                      timestamp,
                      x,
                      y,
                    });
                  }
                }
              } catch (err) {
                console.error(
                  "Failed to create valid selector from data-testid",
                  err,
                );
              }
            else if (cleanSelector.includes("class="))
              try {
                let classMatch = cleanSelector.match(/class="([^"]+)"/);
                if (classMatch && classMatch[1]) {
                  let classNames = classMatch[1].split(" ");
                  if (classNames.length > 0) {
                    let simpleSelector = `.${classNames[0]}`;
                    (document.querySelector(simpleSelector),
                      actions.push({
                        type: "click",
                        selector: simpleSelector,
                        timestamp,
                        x,
                        y,
                      }));
                  }
                }
              } catch (err) {
                console.error(
                  "Failed to create valid selector from class",
                  err,
                );
              }
            else if (cleanSelector.includes("id="))
              try {
                let idMatch = cleanSelector.match(/id="([^"]+)"/);
                if (idMatch && idMatch[1]) {
                  let simpleSelector = `#${idMatch[1]}`;
                  (document.querySelector(simpleSelector),
                    actions.push({
                      type: "click",
                      selector: simpleSelector,
                      timestamp,
                      x,
                      y,
                    }));
                }
              } catch (err) {
                console.error("Failed to create valid selector from id", err);
              }
            else {
              let tagMatch = cleanSelector.match(/^([a-zA-Z]+)/);
              if (tagMatch && tagMatch[1])
                try {
                  let simpleSelector = tagMatch[1];
                  (document.querySelector(simpleSelector),
                    actions.push({
                      type: "click",
                      selector: simpleSelector,
                      timestamp,
                      x,
                      y,
                    }));
                } catch (err) {
                  console.error(
                    "Failed to create valid selector from tag name",
                    err,
                  );
                }
            }
          }
        }),
        inputChanges != null &&
          inputChanges.length &&
          inputChanges
            .filter((change) => change.action === "complete" || !change.action)
            .forEach((change) => {
              if (
                !change.elementSelector.includes('type="checkbox"') &&
                !change.elementSelector.includes('type="radio"')
              )
                try {
                  (document.querySelector(change.elementSelector),
                    actions.push({
                      type: "input",
                      selector: change.elementSelector,
                      value: change.value,
                      timestamp: change.timestamp,
                    }));
                } catch (e) {
                  if (
                    (console.warn(
                      `Invalid selector in input: ${change.elementSelector}, attempting to fix`,
                    ),
                    change.elementSelector.includes("data-testid="))
                  ) {
                    let testIdMatch = change.elementSelector.match(
                      /data-testid="([^"]+)"/,
                    );
                    testIdMatch &&
                      testIdMatch[1] &&
                      actions.push({
                        type: "input",
                        selector: `[data-testid="${testIdMatch[1]}"]`,
                        value: change.value,
                        timestamp: change.timestamp,
                      });
                  }
                }
            }),
        formElementChanges != null &&
          formElementChanges.length &&
          formElementChanges.forEach((change) => {
            if (change.elementSelector)
              try {
                (document.querySelector(change.elementSelector),
                  change.type === "checkbox" || change.type === "radio"
                    ? actions.push({
                        type: change.checked ? "check" : "uncheck",
                        selector: change.elementSelector,
                        value: change.value,
                        timestamp: change.timestamp,
                      })
                    : change.type === "select" &&
                      actions.push({
                        type: "select",
                        selector: change.elementSelector,
                        value: change.value,
                        timestamp: change.timestamp,
                      }));
              } catch (e) {
                if (
                  (console.warn(
                    `Invalid selector in form element: ${change.elementSelector}, attempting to fix`,
                  ),
                  change.elementSelector.includes("data-testid="))
                ) {
                  let testIdMatch = change.elementSelector.match(
                    /data-testid="([^"]+)"/,
                  );
                  if (testIdMatch && testIdMatch[1]) {
                    let selector = `[data-testid="${testIdMatch[1]}"]`;
                    change.type === "checkbox" || change.type === "radio"
                      ? actions.push({
                          type: change.checked ? "check" : "uncheck",
                          selector,
                          value: change.value,
                          timestamp: change.timestamp,
                        })
                      : change.type === "select" &&
                        actions.push({
                          type: "select",
                          selector,
                          value: change.value,
                          timestamp: change.timestamp,
                        });
                  }
                }
              }
          }));
    } catch (e) {
      console.error("Error processing tracking data", e);
    }
    (actions.length === 0 &&
      console.warn("No actions extracted from tracking data"),
      actions.sort((a, b) => a.timestamp - b.timestamp));
    for (let i = 1; i < actions.length; i++)
      actions[i].timestamp - actions[i - 1].timestamp < 600 &&
        (actions[i].timestamp = actions[i - 1].timestamp + 600);
    return (console.log("Converted replay actions:", actions), actions);
  }
  function findElement(selector) {
    var _a;
    let element = document.querySelector(selector);
    if (!element) {
      if (selector.includes("data-testid=")) {
        let testId =
          (_a = selector.match(/data-testid="([^"]+)"/)) == null
            ? void 0
            : _a[1];
        testId &&
          (element = document.querySelector(`[data-testid="${testId}"]`));
      }
      if (!element && selector.includes(".")) {
        let classes = selector.match(/\.([^\s.#\[\]]+)/g);
        if (classes && classes.length > 0) {
          let className = classes[0].substring(1);
          element = document.querySelector(`.${className}`);
        }
      }
      if (!element && selector.includes("#")) {
        let ids = selector.match(/#([^\s.#\[\]]+)/g);
        if (ids && ids.length > 0) {
          let id = ids[0].substring(1);
          element = document.querySelector(`#${id}`);
        }
      }
    }
    return element;
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
        "transform 0.5s ease-in-out, left 0.5s ease-in-out, top 0.5s ease-in-out"),
      document.body.appendChild(cursor),
      cursor
    );
  }
  function createReplayStyles() {
    let style = document.createElement("style");
    return (
      (style.textContent = `
    @keyframes click-ripple {
      0% {
        transform: translate(-50%, -50%) scale(1);
        opacity: 1;
      }
      100% {
        transform: translate(-50%, -50%) scale(8);
        opacity: 0;
      }
    }
    
    @keyframes pulse {
      0% { transform: scale(1); }
      50% { transform: scale(1.03); }
      100% { transform: scale(1); }
    }
    
    .replay-pulse {
      animation: pulse 0.5s ease-in-out infinite;
    }
  `),
      document.head.appendChild(style),
      style
    );
  }
  function showClickEffect(cursorRef2) {
    if (!cursorRef2.current) return;
    let ripple = document.createElement("div");
    ((ripple.className = "click-ripple"),
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
      (ripple.style.animation = "click-ripple 1s ease-out forwards"),
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
    let originalOutline = element.style.outline,
      originalBoxShadow = element.style.boxShadow,
      originalZIndex = element.style.zIndex,
      originalTransition = element.style.transition;
    ((element.style.transition = "all 0.3s ease-in-out"),
      (element.style.outline = "3px solid #ff3333"),
      (element.style.boxShadow = "0 0 15px rgba(255, 51, 51, 0.7)"),
      (element.style.zIndex = "1000"),
      element.classList.add("replay-pulse"),
      setTimeout(() => {
        element &&
          ((element.style.outline = originalOutline),
          (element.style.boxShadow = originalBoxShadow),
          (element.style.zIndex = originalZIndex),
          (element.style.transition = originalTransition),
          element.classList.remove("replay-pulse"));
      }, 2e3 / speedRef2.current));
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
            setTimeout(resolve, 800));
        }, 400));
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
              registerTimeout2(setTimeout(typeChar, 100 / speedRef2.current)))
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
          }, 500 / speedRef2.current),
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
        }, 1e3),
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
            await new Promise((resolve) => setTimeout(resolve, 300)));
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
              await new Promise((resolve) => setTimeout(resolve, 50)),
              element.dispatchEvent(
                new MouseEvent("mouseup", {
                  bubbles: !0,
                  cancelable: !0,
                  view: window,
                }),
              ),
              await new Promise((resolve) => setTimeout(resolve, 50)),
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
          await new Promise((resolve) => setTimeout(resolve, 300));
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
        delay = MIN_DELAY / speedRef2.current;
      if (nextAction && action.timestamp && nextAction.timestamp) {
        let timeDiff = nextAction.timestamp - action.timestamp;
        delay =
          Math.max(MIN_DELAY, Math.min(MAX_DELAY, timeDiff)) /
          speedRef2.current;
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
      document.querySelectorAll(".click-ripple").forEach((ripple) => {
        ripple.parentNode && ripple.parentNode.removeChild(ripple);
      }),
      document.querySelectorAll(".replay-pulse").forEach((element) => {
        (element.classList.remove("replay-pulse"),
          (element.style.outline = ""),
          (element.style.boxShadow = ""),
          (element.style.zIndex = ""),
          (element.style.transition = ""));
      }),
      window.parent.postMessage({ type: "staktrak-replay-stopped" }, "*"));
  }
  function initReplay(options) {
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
              actionsRef.current = data.actions || [];
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
    ((window.convertToReplayActions = convertToReplayActions),
      (window.findElement = findElement),
      (window.pauseReplay = pauseReplay),
      (window.resumeReplay = resumeReplay),
      (window.stopReplay = stopReplay),
      initReplay());
  });
  return __toCommonJS(replay_exports);
})();
