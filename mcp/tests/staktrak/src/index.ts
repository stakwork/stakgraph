import { Config, Results, Memory, Assertion } from "./types";
import {
  getTimeStamp,
  isInputOrTextarea,
  getElementSelector,
  createClickPath,
  filterClickDetails,
} from "./utils";

const defaultConfig: Config = {
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
  inputDebounceDelay: 2000,
  multiClickInterval: 300,
  filterAssertionClicks: true,
  processData: (results: Results) => console.log(results),
};

class UserBehaviorTracker {
  private config: Config = defaultConfig;
  private results: Results = this.createEmptyResults();
  private memory: Memory = {
    mousePosition: [0, 0, 0],
    inputDebounceTimers: {},
    selectionMode: false,
    assertionDebounceTimer: null,
    assertions: [],
    mutationObserver: null,
    mouseInterval: null,
    listeners: [],
    postMessageListeners: [],
  };
  private isRunning = false;

  private createEmptyResults(): Results {
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
      assertions: [],
    };
  }

  makeConfig(newConfig: Partial<Config>) {
    this.config = { ...this.config, ...newConfig };
    return this;
  }

  listen() {
    this.setupMessageHandling();
  }

  start() {
    // Clean up any existing listeners first
    this.cleanup();

    this.resetResults();
    this.setupEventListeners();
    this.isRunning = true;

    window.parent.postMessage({ type: "staktrak-setup" }, "*");
    return this;
  }

  private resetResults() {
    this.memory.assertions = [];
    this.results = this.createEmptyResults();

    if (this.config.userInfo) {
      this.results.userInfo = {
        url: document.URL,
        userAgent: navigator.userAgent,
        platform: navigator.platform,
        windowSize: [window.innerWidth, window.innerHeight],
      };
    }

    if (this.config.timeCount) {
      this.results.time = {
        startedAt: getTimeStamp(),
        completedAt: 0,
        totalSeconds: 0,
      };
    }
  }

  private cleanup() {
    // Clean up existing listeners
    this.memory.listeners.forEach((cleanup) => cleanup());
    this.memory.listeners = [];

    // Clean up mutation observer
    if (this.memory.mutationObserver) {
      this.memory.mutationObserver.disconnect();
      this.memory.mutationObserver = null;
    }

    // Clean up intervals
    if (this.memory.mouseInterval) {
      clearInterval(this.memory.mouseInterval);
      this.memory.mouseInterval = null;
    }

    // Clean up debounce timers
    Object.values(this.memory.inputDebounceTimers).forEach((timer) =>
      clearTimeout(timer)
    );
    this.memory.inputDebounceTimers = {};

    // Clean up assertion timer
    if (this.memory.assertionDebounceTimer) {
      clearTimeout(this.memory.assertionDebounceTimer);
      this.memory.assertionDebounceTimer = null;
    }

    // Exit selection mode
    if (this.memory.selectionMode) {
      this.setSelectionMode(false);
    }
  }

  private setupEventListeners() {
    if (this.config.clicks) {
      const clickHandler = (e: MouseEvent) => {
        this.results.clicks.clickCount++;
        const path = createClickPath(e);
        this.results.clicks.clickDetails.push([
          e.clientX,
          e.clientY,
          path,
          getTimeStamp(),
        ]);

        const target = e.target as HTMLInputElement;
        if (
          target.tagName === "INPUT" &&
          (target.type === "checkbox" || target.type === "radio")
        ) {
          this.results.formElementChanges.push({
            elementSelector: getElementSelector(target),
            type: target.type,
            checked: target.checked,
            value: target.value,
            timestamp: getTimeStamp(),
          });
        }
      };
      document.addEventListener("click", clickHandler);
      this.memory.listeners.push(() =>
        document.removeEventListener("click", clickHandler)
      );
    }

    if (this.config.mouseScroll) {
      const scrollHandler = () => {
        this.results.mouseScroll.push([
          window.scrollX,
          window.scrollY,
          getTimeStamp(),
        ]);
      };
      window.addEventListener("scroll", scrollHandler);
      this.memory.listeners.push(() =>
        window.removeEventListener("scroll", scrollHandler)
      );
    }

    if (this.config.mouseMovement) {
      const mouseMoveHandler = (e: MouseEvent) => {
        this.memory.mousePosition = [e.clientX, e.clientY, getTimeStamp()];
      };
      document.addEventListener("mousemove", mouseMoveHandler);
      this.memory.mouseInterval = setInterval(() => {
        if (this.memory.mousePosition[2] + 500 > getTimeStamp()) {
          this.results.mouseMovement.push(this.memory.mousePosition);
        }
      }, this.config.mouseMovementInterval * 1000);

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
          getTimeStamp(),
        ]);
      };
      window.addEventListener("resize", resizeHandler);
      this.memory.listeners.push(() =>
        window.removeEventListener("resize", resizeHandler)
      );
    }

    if (this.config.visibilitychange) {
      const visibilityHandler = () => {
        this.results.visibilitychanges.push([
          document.visibilityState,
          getTimeStamp(),
        ]);
        this.processResults();
      };
      document.addEventListener("visibilitychange", visibilityHandler);
      this.memory.listeners.push(() =>
        document.removeEventListener("visibilitychange", visibilityHandler)
      );
    }

    if (this.config.keyboardActivity) {
      const keyHandler = (e: KeyboardEvent) => {
        if (!isInputOrTextarea(e.target as Element)) {
          this.results.keyboardActivities.push([e.key, getTimeStamp()]);
        }
      };
      document.addEventListener("keypress", keyHandler);
      this.memory.listeners.push(() =>
        document.removeEventListener("keypress", keyHandler)
      );
    }

    if (this.config.formInteractions) {
      this.setupFormInteractions();
    }

    if (this.config.touchEvents) {
      const touchHandler = (e: TouchEvent) => {
        if (e.touches.length > 0) {
          const touch = e.touches[0];
          this.results.touchEvents.push({
            type: "touchstart",
            x: touch.clientX,
            y: touch.clientY,
            timestamp: getTimeStamp(),
          });
        }
      };
      document.addEventListener("touchstart", touchHandler);
      this.memory.listeners.push(() =>
        document.removeEventListener("touchstart", touchHandler)
      );
    }

    if (this.config.pageNavigation) {
      this.setupPageNavigation();
    }
  }

  private setupFormInteractions() {
    const attachFormListeners = (element: Element) => {
      const htmlEl = element as HTMLElement;
      if (
        htmlEl.tagName === "INPUT" ||
        htmlEl.tagName === "SELECT" ||
        htmlEl.tagName === "TEXTAREA"
      ) {
        const inputEl = htmlEl as HTMLInputElement;

        if (
          inputEl.type === "checkbox" ||
          inputEl.type === "radio" ||
          htmlEl.tagName === "SELECT"
        ) {
          const changeHandler = () => {
            const selector = getElementSelector(htmlEl);
            if (htmlEl.tagName === "SELECT") {
              const selectEl = htmlEl as HTMLSelectElement;
              const selectedOption = selectEl.options[selectEl.selectedIndex];
              this.results.formElementChanges.push({
                elementSelector: selector,
                type: "select",
                value: selectEl.value,
                text: selectedOption?.text || "",
                timestamp: getTimeStamp(),
              });
            } else {
              this.results.formElementChanges.push({
                elementSelector: selector,
                type: inputEl.type,
                checked: inputEl.checked,
                value: inputEl.value,
                timestamp: getTimeStamp(),
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
                action: "complete",
              });
              delete this.memory.inputDebounceTimers[elementId];
            }, this.config.inputDebounceDelay);

            this.results.inputChanges.push({
              elementSelector: selector,
              value: inputEl.value,
              timestamp: getTimeStamp(),
              action: "intermediate",
            });
          };

          const focusHandler = (e: FocusEvent) => {
            const selector = getElementSelector(htmlEl);
            this.results.focusChanges.push({
              elementSelector: selector,
              type: e.type,
              timestamp: getTimeStamp(),
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
                action: "complete",
              });
            }
          };

          htmlEl.addEventListener("input", inputHandler);
          htmlEl.addEventListener("focus", focusHandler);
          htmlEl.addEventListener("blur", focusHandler);
        }
      }
    };

    document
      .querySelectorAll("input, select, textarea")
      .forEach(attachFormListeners);

    this.memory.mutationObserver = new MutationObserver((mutations) => {
      mutations.forEach((mutation) => {
        mutation.addedNodes.forEach((node) => {
          if (node.nodeType === 1) {
            attachFormListeners(node as Element);
            (node as Element)
              .querySelectorAll("input, select, textarea")
              .forEach(attachFormListeners);
          }
        });
      });
    });

    this.memory.mutationObserver.observe(document.body, {
      childList: true,
      subtree: true,
    });
    this.memory.listeners.push(() => {
      if (this.memory.mutationObserver) {
        this.memory.mutationObserver.disconnect();
        this.memory.mutationObserver = null;
      }
    });
  }

  private setupPageNavigation() {
    const originalPushState = history.pushState;
    const originalReplaceState = history.replaceState;

    history.pushState = (...args) => {
      originalPushState.apply(history, args);
      this.results.pageNavigation.push({
        type: "pushState",
        url: document.URL,
        timestamp: getTimeStamp(),
      });
    };

    history.replaceState = (...args) => {
      originalReplaceState.apply(history, args);
      this.results.pageNavigation.push({
        type: "replaceState",
        url: document.URL,
        timestamp: getTimeStamp(),
      });
    };

    const popstateHandler = () => {
      this.results.pageNavigation.push({
        type: "popstate",
        url: document.URL,
        timestamp: getTimeStamp(),
      });
    };
    window.addEventListener("popstate", popstateHandler);
    this.memory.listeners.push(() =>
      window.removeEventListener("popstate", popstateHandler)
    );

    // Note: We don't restore original pushState/replaceState since they're global
    // and would break if multiple instances exist
  }

  private setupMessageHandling() {
    // this listener only needs to be setup once
    if (this.memory.postMessageListeners.length > 0) return;

    const messageHandler = (event: MessageEvent) => {
      if (!event.data?.type) return;

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
              timestamp: getTimeStamp(),
            });
          }
          break;
      }
    };
    window.addEventListener("message", messageHandler);
    this.memory.postMessageListeners.push(() =>
      window.removeEventListener("message", messageHandler)
    );
  }

  private setSelectionMode(isActive: boolean) {
    this.memory.selectionMode = isActive;

    if (isActive) {
      document.body.classList.add("staktrak-selection-active");
      const mouseUpHandler = () => {
        const selection = window.getSelection();
        if (selection?.toString().trim()) {
          const text = selection.toString();
          let container = selection.getRangeAt(0).commonAncestorContainer;
          if (container.nodeType === 3)
            container = container.parentNode as Node;

          if (this.memory.assertionDebounceTimer)
            clearTimeout(this.memory.assertionDebounceTimer);

          this.memory.assertionDebounceTimer = setTimeout(() => {
            const selector = getElementSelector(container as Element);
            const assertion = {
              type: "hasText",
              selector,
              value: text,
              timestamp: getTimeStamp(),
            };
            this.memory.assertions.push(assertion);

            window.parent.postMessage(
              { type: "staktrak-selection", text, selector },
              "*"
            );
            // window.parent.postMessage(
            //   {
            //     type: "staktrak-popup",
            //     message: `Assertion added: hasText "${text.slice(0, 30)}${
            //       text.length > 30 ? "..." : ""
            //     }"`,
            //     // type: "success",
            //   },
            //   "*"
            // );
          }, 300);
        }
      };
      document.addEventListener("mouseup", mouseUpHandler);
      this.memory.listeners.push(() =>
        document.removeEventListener("mouseup", mouseUpHandler)
      );
    } else {
      document.body.classList.remove("staktrak-selection-active");
      window.getSelection()?.removeAllRanges();
    }

    window.parent.postMessage(
      {
        type: `staktrak-selection-mode-${isActive ? "started" : "ended"}`,
      },
      "*"
    );
  }

  private processResults() {
    if (this.config.timeCount && this.results.time) {
      this.results.time.completedAt = getTimeStamp();
      this.results.time.totalSeconds =
        (this.results.time.completedAt - this.results.time.startedAt) / 1000;
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
    if (!this.isRunning) return this;

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

  addAssertion(type: string, selector: string, value: string = "") {
    this.memory.assertions.push({
      type,
      selector,
      value,
      timestamp: getTimeStamp(),
    });
  }
}

// Create global instance
const userBehaviour = new UserBehaviorTracker();

// Auto-start when DOM is ready
document.addEventListener("DOMContentLoaded", () => {
  userBehaviour
    .makeConfig({
      processData: (results: Results) =>
        console.log("StakTrak recording processed:", results),
    })
    .listen();
});

export default userBehaviour;
