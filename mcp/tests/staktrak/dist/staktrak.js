"use strict";
var userBehaviour = (() => {
  var __defProp = Object.defineProperty;
  var __getOwnPropDesc = Object.getOwnPropertyDescriptor;
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
  var isInputOrTextarea = (element) => element.tagName === "INPUT" || element.tagName === "TEXTAREA" || element.isContentEditable;
  var getElementSelector = (element) => {
    if (!element || element.nodeType !== 1)
      return "";
    const dataset = element.dataset;
    if (dataset == null ? void 0 : dataset.testid)
      return `[data-testid="${dataset.testid}"]`;
    const id = element.id;
    if (id)
      return `#${id}`;
    let selector = element.tagName.toLowerCase();
    const className = element.className;
    if (className) {
      const classes = Array.from(element.classList).filter((cls) => cls !== "staktrak-selection-active").join(".");
      if (classes)
        selector += `.${classes}`;
    }
    if (element.tagName === "INPUT") {
      const type = element.type;
      if (type)
        selector += `[type="${type}"]`;
    }
    return selector;
  };
  var createClickPath = (e) => {
    const path = [];
    e.composedPath().forEach((el, i) => {
      const composedPath = e.composedPath();
      if (i < composedPath.length - 2) {
        let node = el.localName;
        const dataset = el.dataset;
        if (dataset == null ? void 0 : dataset.testid) {
          node += `[data-testid="${dataset.testid}"]`;
        } else {
          const className = el.className;
          if (className) {
            el.classList.forEach((cls) => {
              if (cls !== "staktrak-selection-active")
                node += `.${cls}`;
            });
          }
          const id = el.id;
          if (id)
            node += `#${id}`;
        }
        path.push(node);
      }
    });
    return path.reverse().join(">");
  };
  var filterClickDetails = (clickDetails, assertions, config) => {
    if (!clickDetails.length)
      return [];
    let filtered = config.filterAssertionClicks ? clickDetails.filter(
      (click) => !assertions.some(
        (assertion) => Math.abs(click[3] - assertion.timestamp) < 1e3 && (click[2].includes(assertion.selector) || assertion.selector.includes(click[2]))
      )
    ) : clickDetails;
    const clicksBySelector = {};
    filtered.forEach((click) => {
      const selector = click[2];
      if (!clicksBySelector[selector])
        clicksBySelector[selector] = [];
      clicksBySelector[selector].push({ detail: click, timestamp: click[3] });
    });
    const result = [];
    Object.values(clicksBySelector).forEach((clicks) => {
      clicks.sort((a, b) => a.timestamp - b.timestamp);
      let lastClick = null;
      clicks.forEach((click) => {
        if (!lastClick || click.timestamp - lastClick.timestamp > config.multiClickInterval) {
          result.push(click.detail);
        }
        lastClick = click;
      });
    });
    return result.sort((a, b) => a[3] - b[3]);
  };

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
    pageNavigation: true,
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
        postMessageListeners: []
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
    }
    start() {
      this.cleanup();
      this.resetResults();
      this.setupEventListeners();
      this.isRunning = true;
      window.parent.postMessage({ type: "staktrak-setup" }, "*");
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
          const path = createClickPath(e);
          this.results.clicks.clickDetails.push([
            e.clientX,
            e.clientY,
            path,
            getTimeStamp()
          ]);
          const target = e.target;
          if (target.tagName === "INPUT" && (target.type === "checkbox" || target.type === "radio")) {
            this.results.formElementChanges.push({
              elementSelector: getElementSelector(target),
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
          this.processResults();
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
      if (this.config.pageNavigation) {
        this.setupPageNavigation();
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
      history.pushState = (...args) => {
        originalPushState.apply(history, args);
        this.results.pageNavigation.push({
          type: "pushState",
          url: document.URL,
          timestamp: getTimeStamp()
        });
      };
      history.replaceState = (...args) => {
        originalReplaceState.apply(history, args);
        this.results.pageNavigation.push({
          type: "replaceState",
          url: document.URL,
          timestamp: getTimeStamp()
        });
      };
      const popstateHandler = () => {
        this.results.pageNavigation.push({
          type: "popstate",
          url: document.URL,
          timestamp: getTimeStamp()
        });
      };
      window.addEventListener("popstate", popstateHandler);
      this.memory.listeners.push(
        () => window.removeEventListener("popstate", popstateHandler)
      );
    }
    setupMessageHandling() {
      if (this.memory.postMessageListeners.length > 0)
        return;
      const messageHandler = (event) => {
        var _a;
        if (!((_a = event.data) == null ? void 0 : _a.type))
          return;
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
        }
      };
      window.addEventListener("message", messageHandler);
      this.memory.postMessageListeners.push(
        () => window.removeEventListener("message", messageHandler)
      );
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
      if (!this.isRunning)
        return this;
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
  document.addEventListener("DOMContentLoaded", () => {
    userBehaviour.makeConfig({
      processData: (results) => console.log("StakTrak recording processed:", results)
    }).listen();
  });
  var src_default = userBehaviour;
  return __toCommonJS(src_exports);
})();
