import { Config, Results, Memory } from "./types";
import {
  getTimeStamp,
  isInputOrTextarea,
  getElementSelector,
  createClickDetail,
  filterClickDetails,
  annotateInteractiveElements,
  getAnnotationStats,
  cleanupAnnotations,
  startDynamicAnnotation,
  stopDynamicAnnotation,
  getDynamicAnnotationStatus,
  ensureElementId,
} from "./utils";
import { debugMsg, isReactDevModeActive } from "./debug";
import { initPlaywrightReplay } from "./playwright-replay/index";
import { resultsToActions } from "./actionModel";
import { buildScenario, serializeScenario } from './scenario';
import {
  generatePlaywrightTestFromActions,
  GenerateOptions,
} from "./playwright-generator";


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
  public results: Results = this.createEmptyResults();
  public memory: Memory = {
    mousePosition: [0, 0, 0],
    inputDebounceTimers: {},
    selectionMode: false,
    assertionDebounceTimer: null,
    assertions: [],
    mutationObserver: null,
    mouseInterval: null,
    listeners: [],
    alwaysListeners: [],
    healthCheckInterval: null,
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
    console.log("🔍 STAKTRAK: Recording state saved to sessionStorage");

    return this;
  }

  private saveSessionState() {
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
      sessionStorage.setItem('stakTrakActiveRecording', JSON.stringify(sessionData));
    } catch (error) {
      
    }
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

    // Clean up health check
    if (this.memory.healthCheckInterval) {
      clearInterval(this.memory.healthCheckInterval);
      this.memory.healthCheckInterval = null;
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

  public setupEventListeners() {
    console.log("🔍 STAKTRAK: Setting up event listeners", { isRunning: this.isRunning });
    
    if (this.config.clicks) {
      const clickHandler = (e: MouseEvent) => {
        this.results.clicks.clickCount++;
        const clickDetail = createClickDetail(e);
        this.results.clicks.clickDetails.push(clickDetail);

        // Handle form elements
        const target = e.target as HTMLInputElement;
        if (
          target.tagName === "INPUT" &&
          (target.type === "checkbox" || target.type === "radio")
        ) {
          this.results.formElementChanges.push({
            elementSelector: clickDetail.selectors.primary,
            type: target.type,
            checked: target.checked,
            value: target.value,
            timestamp: getTimeStamp(),
          });
        }

        // Save state after each click for iframe reload persistence
        this.saveSessionState();
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
            // Save state after form element changes
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
                action: "complete",
              });
              delete this.memory.inputDebounceTimers[elementId];
              // Save state after input completion
              this.saveSessionState();
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

    const recordStateChange = (type: string) => {
      this.results.pageNavigation.push({
        type,
        url: document.URL,
        timestamp: getTimeStamp(),
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
    this.memory.alwaysListeners.push(() =>
      window.removeEventListener("popstate", popstateHandler)
    );

    const hashHandler = () => {
      recordStateChange("hashchange");
    };
    window.addEventListener('hashchange', hashHandler);
    this.memory.alwaysListeners.push(() =>
      window.removeEventListener('hashchange', hashHandler)
    );

    const anchorClickHandler = (e: Event) => {
      const a = (e.target as HTMLElement).closest('a');
      if (!a) return;
      if (a.target && a.target !== '_self') return;
      const href = a.getAttribute('href');
      if (!href) return;
      try {
        const dest = new URL(href, window.location.href);
        if (dest.origin === window.location.origin) {
          this.results.pageNavigation.push({ type: 'anchorClick', url: dest.href, timestamp: getTimeStamp() });
        }
      } catch {}
    };
    document.addEventListener('click', anchorClickHandler, true);
    this.memory.alwaysListeners.push(() =>
      document.removeEventListener('click', anchorClickHandler, true)
    );

    // Note: We don't restore original pushState/replaceState since they're global
    // and would break if multiple instances exist
  }

  private setupMessageHandling() {
    // this listener only needs to be setup once
    if (this.memory.alwaysListeners.length > 0) return;

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
        case "staktrak-debug-request":
          debugMsg({
            messageId: event.data.messageId,
            coordinates: event.data.coordinates,
          });
          break;
        case "staktrak-recover":
          this.recoverRecording();
      }
    };
    window.addEventListener("message", messageHandler);
    this.memory.alwaysListeners.push(() =>
      window.removeEventListener("message", messageHandler)
    );
  }

  private checkDebugInfo() {
    setTimeout(() => {
      if (isReactDevModeActive()) {
        window.parent.postMessage({ type: "staktrak-debug-init" }, "*");
      }
    }, 1500);
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
    if (!this.isRunning) {
      console.log("StakTrak is not running");
      return this;
    }

    this.cleanup();
    
    stopDynamicAnnotation();
    cleanupAnnotations();
    
    this.processResults();
    this.isRunning = false;

    sessionStorage.removeItem('stakTrakActiveRecording');
    console.log("🔍 STAKTRAK: Recording state cleared from sessionStorage");

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

  public attemptSessionRestoration() {
    try {
      const activeRecording = sessionStorage.getItem('stakTrakActiveRecording');
      if (!activeRecording) {
        console.log("🔍 STAKTRAK: No previous session to restore");
        return;
      }

      const recordingData = JSON.parse(activeRecording);
      console.log("🔍 STAKTRAK: Found previous session data in sessionStorage");

      // Simple validation: if session data exists and claims to be recording, restore it
      if (recordingData && recordingData.isRecording && recordingData.version === "1.0") {
        console.log("🔍 STAKTRAK: Attempting session restoration...");

        // Detect if this is an iframe reload (page loaded recently after session was saved)
        const timeSinceLastSave = Date.now() - (recordingData.lastSaved || 0);
        const isLikelyIframeReload = timeSinceLastSave < 10000; // Within 10 seconds

        if (isLikelyIframeReload) {
          console.log("🔍 STAKTRAK: Detected iframe reload, restoring recording state");
          
          // Restore state
          if (recordingData.results) {
            this.results = { ...this.createEmptyResults(), ...recordingData.results };
          }
          if (recordingData.memory) {
            this.memory.assertions = recordingData.memory.assertions || [];
            this.memory.selectionMode = recordingData.memory.selectionMode || false;
          }

          // Reactivate recording
          this.isRunning = true;
          this.setupEventListeners();
          
          // Start health check for restored session
          this.startHealthCheck();
          
          console.log("🔍 STAKTRAK: Session restored successfully", {
            clicks: this.results.clicks.clickCount,
            inputs: this.results.inputChanges.length,
            assertions: this.memory.assertions.length
          });

          // Verify event listeners are working
          this.verifyEventListeners();

          // Notify parent that recording is active again
          window.parent.postMessage({ type: "staktrak-replay-ready" }, "*");
        } else {
          console.log("🔍 STAKTRAK: Session data is too old, starting fresh");
          sessionStorage.removeItem('stakTrakActiveRecording');
        }
      } else {
        console.log("🔍 STAKTRAK: Invalid session data, starting fresh");
        sessionStorage.removeItem('stakTrakActiveRecording');
      }
    } catch (error) {
      
      sessionStorage.removeItem('stakTrakActiveRecording');
    }
  }

  private verifyEventListeners() {
    console.log("🔍 STAKTRAK: Verifying event listeners", {
      isRunning: this.isRunning,
      listenersCount: this.memory.listeners.length,
      mutationObserver: !!this.memory.mutationObserver
    });
    
    // If we have fewer listeners than expected, re-setup
    if (this.isRunning && this.memory.listeners.length === 0) {
      
      this.setupEventListeners();
    }
  }

  public recoverRecording() {
    console.log("🔍 STAKTRAK: Attempting recording recovery");
    if (!this.isRunning) {
      console.log("🔍 STAKTRAK: Recording was not active, starting fresh");
      return;
    }
    
    // Ensure event listeners are active
    this.verifyEventListeners();
    
    // Save current state
    this.saveSessionState();
    
    console.log("🔍 STAKTRAK: Recording recovery completed");
  }

  private startHealthCheck() {
    // Health check every 5 seconds to ensure recording stays active
    this.memory.healthCheckInterval = setInterval(() => {
      if (this.isRunning) {
        // Verify listeners are still active
        if (this.memory.listeners.length === 0) {
          
          this.recoverRecording();
        }
        
        // Save state periodically in case of unexpected iframe reloads
        this.saveSessionState();
      }
    }, 5000);
    
    console.log("🔍 STAKTRAK: Health check started");
  }
}

// Create global instance (simple, always works)
const userBehaviour = new UserBehaviorTracker();

// Auto-start when DOM is ready
const initializeStakTrak = () => {
  userBehaviour
    .makeConfig({
      processData: (results) =>
        console.log("StakTrak recording processed:", results),
    })
    .listen();
  
  // Enhanced session restoration with iframe reload detection
  userBehaviour.attemptSessionRestoration();
  
  initPlaywrightReplay();
};

document.readyState === "loading"
  ? document.addEventListener("DOMContentLoaded", initializeStakTrak)
  : initializeStakTrak();

// Add utility functions to the userBehaviour object for testing
(userBehaviour as any).createClickDetail = createClickDetail;
(userBehaviour as any).getActions = () =>
  resultsToActions(userBehaviour.result());
(userBehaviour as any).generatePlaywrightTest = (options: GenerateOptions) => {
  const actions = resultsToActions(userBehaviour.result());
  const code = generatePlaywrightTestFromActions(actions, options);
  (userBehaviour as any)._lastGeneratedUsingActions = true;
  return code;
};
(userBehaviour as any).exportSession = (options: GenerateOptions) => {
  const actions = resultsToActions(userBehaviour.result());
  const test = generatePlaywrightTestFromActions(actions, options);
  (userBehaviour as any)._lastGeneratedUsingActions = true;
  return { actions, test };
};
(userBehaviour as any).getAnnotationStats = () => userBehaviour.getAnnotationStats();
(userBehaviour as any).annotateCurrentPage = () => userBehaviour.annotateCurrentPage();
(userBehaviour as any).cleanupCurrentAnnotations = () => userBehaviour.cleanupCurrentAnnotations();
(userBehaviour as any).startDynamicAnnotation = () => userBehaviour.startDynamicAnnotation();
(userBehaviour as any).stopDynamicAnnotation = () => userBehaviour.stopDynamicAnnotation();
(userBehaviour as any).getDynamicAnnotationStatus = () => userBehaviour.getDynamicAnnotationStatus();
(userBehaviour as any).ensureElementId = ensureElementId;

(userBehaviour as any).getScenario = () => {
  const results = userBehaviour.result();
  const actions = resultsToActions(results);
  return buildScenario(results, actions);
};
(userBehaviour as any).exportScenarioJSON = () => {
  const sc = (userBehaviour as any).getScenario();
  return serializeScenario(sc);
};

(userBehaviour as any).getSelectorScores = () => {
  const results = userBehaviour.result();
  if (!results.clicks || !results.clicks.clickDetails.length) return [];
  const last = results.clicks.clickDetails[results.clicks.clickDetails.length - 1];
  const sel = last.selectors as any;
  if (sel && sel.scores) return sel.scores;
  return [];
};

export default userBehaviour;
