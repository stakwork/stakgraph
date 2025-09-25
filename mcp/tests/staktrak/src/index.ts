import { Config, Results, Memory } from "./types";
import {
  getTimeStamp,
  isInputOrTextarea,
  getElementSelector,
  createClickDetail,
  filterClickDetails,
} from "./utils";
import { debugMsg, isReactDevModeActive } from "./debug";
import { initPlaywrightReplay } from "./playwright-replay/index";
import { StakTrakMessage, isStakTrakMessage, isRemoveActionMessage, isAddAssertionMessage } from "./messages";
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

  /**
   * Send event data to parent for recording
   */
  private sendEventToParent(eventType: string, data: any) {
    window.parent.postMessage({
      type: "staktrak-event",
      eventType,
      data
    }, "*");
  }

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
    // Clean up any existing listeners first
    this.cleanup();

    this.resetResults();
    this.setupEventListeners();
    this.isRunning = true;

    // Start health check
    this.startHealthCheck();

    // Persist recording state to survive script reloads
    this.saveSessionState();

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
    
    if (this.config.clicks) {
      const clickHandler = (e: MouseEvent) => {
        // Skip click recording when in selection mode (creating assertions)
        if (this.memory.selectionMode) {
          return;
        }

        const target = e.target as HTMLInputElement;

        // Helper function to detect labels associated with radio/checkbox inputs
        const isLabelForFormInput = (element: HTMLElement): boolean => {
          if (element.tagName !== "LABEL") return false;

          const label = element as HTMLLabelElement;
          // Check if label controls a radio/checkbox using modern browser property
          if (label.control) {
            const control = label.control as HTMLInputElement;
            return control.tagName === "INPUT" &&
              (control.type === "radio" || control.type === "checkbox");
          }

          // Fallback: check htmlFor attribute
          if (label.htmlFor) {
            const control = document.getElementById(label.htmlFor) as HTMLInputElement;
            return control && control.tagName === "INPUT" &&
              (control.type === "radio" || control.type === "checkbox");
          }

          return false;
        };

        const isFormElement = (target.tagName === "INPUT" &&
          (target.type === "checkbox" || target.type === "radio")) ||
          isLabelForFormInput(target);

        // Debug logging to identify duplicate action sources
        console.log('🖱️ Click detected:', {
          tagName: target.tagName,
          type: target.type || 'none',
          isFormElement,
          className: target.className,
          id: target.id,
          textContent: target.textContent?.substring(0, 20)
        });

        // Skip click recording for form elements when form interactions are enabled
        // This prevents duplicate actions since form changes are handled separately
        if (!isFormElement) {
          console.log('✅ Recording click action for:', target.tagName);
          this.results.clicks.clickCount++;
          const clickDetail = createClickDetail(e);
          this.results.clicks.clickDetails.push(clickDetail);

          // Send complete click data to parent for recording
          this.sendEventToParent("click", clickDetail);

          // Keep backward compatibility with action-added message
          window.parent.postMessage({
            type: "staktrak-action-added",
            action: {
              id: clickDetail.timestamp + '_click',
              kind: 'click',
              timestamp: clickDetail.timestamp,
              locator: {
                primary: clickDetail.selectors.primary,
                text: clickDetail.elementInfo?.text
              }
            }
          }, "*");
        }

        // Form changes are handled by the dedicated change event handler

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

            // Debug logging for form changes
            console.log('📝 Form change detected:', {
              tagName: htmlEl.tagName,
              type: (htmlEl as HTMLInputElement).type,
              selector: selector
            });

            if (htmlEl.tagName === "SELECT") {
              const selectEl = htmlEl as HTMLSelectElement;
              const selectedOption = selectEl.options[selectEl.selectedIndex];
              const formChange = {
                elementSelector: selector,
                type: "select",
                value: selectEl.value,
                text: selectedOption?.text || "",
                timestamp: getTimeStamp(),
              };
              this.results.formElementChanges.push(formChange);
              // Send complete form data to parent
              this.sendEventToParent("form", {
                selector: selector,
                formType: "select",
                value: selectEl.value,
                text: selectedOption?.text || "",
                timestamp: formChange.timestamp
              });

              // Broadcast form action in real-time
              window.parent.postMessage({
                type: "staktrak-action-added",
                action: {
                  id: formChange.timestamp + '_form',
                  kind: 'form',
                  timestamp: formChange.timestamp,
                  formType: formChange.type,
                  value: formChange.text
                }
              }, "*");
            } else {
              const formChange = {
                elementSelector: selector,
                type: inputEl.type,
                checked: inputEl.checked,
                value: inputEl.value,
                timestamp: getTimeStamp(),
              };
              this.results.formElementChanges.push(formChange);
              // Send complete form data to parent
              this.sendEventToParent("form", {
                selector: selector,
                formType: inputEl.type,
                checked: inputEl.checked,
                value: inputEl.value,
                timestamp: formChange.timestamp
              });

              // Broadcast form action in real-time
              window.parent.postMessage({
                type: "staktrak-action-added",
                action: {
                  id: formChange.timestamp + '_form',
                  kind: 'form',
                  timestamp: formChange.timestamp,
                  formType: formChange.type,
                  checked: formChange.checked,
                  value: formChange.value
                }
              }, "*");
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
              const inputAction = {
                elementSelector: selector,
                value: inputEl.value,
                timestamp: getTimeStamp(),
                action: "complete",
              };
              this.results.inputChanges.push(inputAction);
              // Send complete input data to parent
              this.sendEventToParent("input", {
                selector: selector,
                value: inputEl.value,
                timestamp: inputAction.timestamp
              });

              // Broadcast input action in real-time
              window.parent.postMessage({
                type: "staktrak-action-added",
                action: {
                  id: inputAction.timestamp + '_input',
                  kind: 'input',
                  timestamp: inputAction.timestamp,
                  value: inputAction.value
                }
              }, "*");

              delete this.memory.inputDebounceTimers[elementId];
              // Save state after input completion
              this.saveSessionState();
            }, this.config.inputDebounceDelay);

            const inputAction = {
              elementSelector: selector,
              value: inputEl.value,
              timestamp: getTimeStamp(),
              action: "intermediate",
            };
            this.results.inputChanges.push(inputAction);

            // Broadcast intermediate input action in real-time
            window.parent.postMessage({
              type: "staktrak-action-added",
              action: {
                id: inputAction.timestamp + '_input',
                kind: 'input',
                timestamp: inputAction.timestamp,
                value: inputAction.value
              }
            }, "*");
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
              const inputAction = {
                elementSelector: selector,
                value: inputEl.value,
                timestamp: getTimeStamp(),
                action: "complete",
              };
              this.results.inputChanges.push(inputAction);

              // Broadcast final input action in real-time
              window.parent.postMessage({
                type: "staktrak-action-added",
                action: {
                  id: inputAction.timestamp + '_input',
                  kind: 'input',
                  timestamp: inputAction.timestamp,
                  value: inputAction.value
                }
              }, "*");
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
      const navAction = {
        type,
        url: document.URL,
        timestamp: getTimeStamp(),
      };
      this.results.pageNavigation.push(navAction);
      // Send complete navigation data to parent
      this.sendEventToParent("navigation", navAction);

      // Broadcast navigation action in real-time
      window.parent.postMessage({
        type: "staktrak-action-added",
        action: {
          id: navAction.timestamp + '_nav',
          kind: 'nav',
          timestamp: navAction.timestamp,
          url: navAction.url
        }
      }, "*");
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
          const navAction = { type: 'anchorClick', url: dest.href, timestamp: getTimeStamp() };
          this.results.pageNavigation.push(navAction);

          // Broadcast navigation action in real-time
          window.parent.postMessage({
            type: "staktrak-action-added",
            action: {
              id: navAction.timestamp + '_nav',
              kind: 'nav',
              timestamp: navAction.timestamp,
              url: navAction.url
            }
          }, "*");
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

    // Define action removal handlers with error handling
    const actionRemovalHandlers: Record<string, (data: any) => boolean> = {
      'staktrak-remove-navigation': (data) => {
        try {
          if (!data.timestamp) {
            console.warn('Missing timestamp for navigation removal');
            return false;
          }
          const initialLength = this.results.pageNavigation.length;
          this.results.pageNavigation = this.results.pageNavigation.filter(
            nav => nav.timestamp !== data.timestamp
          );
          return this.results.pageNavigation.length < initialLength;
        } catch (error) {
          console.error('Failed to remove navigation:', error);
          return false;
        }
      },
      'staktrak-remove-click': (data) => {
        try {
          if (!data.timestamp) {
            console.warn('Missing timestamp for click removal');
            return false;
          }
          const initialLength = this.results.clicks.clickDetails.length;
          this.results.clicks.clickDetails = this.results.clicks.clickDetails.filter(
            click => click.timestamp !== data.timestamp
          );
          return this.results.clicks.clickDetails.length < initialLength;
        } catch (error) {
          console.error('Failed to remove click:', error);
          return false;
        }
      },
      'staktrak-remove-input': (data) => {
        try {
          if (!data.timestamp) {
            console.warn('Missing timestamp for input removal');
            return false;
          }
          const initialLength = this.results.inputChanges.length;
          this.results.inputChanges = this.results.inputChanges.filter(
            input => input.timestamp !== data.timestamp
          );
          return this.results.inputChanges.length < initialLength;
        } catch (error) {
          console.error('Failed to remove input:', error);
          return false;
        }
      },
      'staktrak-remove-form': (data) => {
        try {
          if (!data.timestamp) {
            console.warn('Missing timestamp for form removal');
            return false;
          }
          const initialLength = this.results.formElementChanges.length;
          this.results.formElementChanges = this.results.formElementChanges.filter(
            form => form.timestamp !== data.timestamp
          );
          return this.results.formElementChanges.length < initialLength;
        } catch (error) {
          console.error('Failed to remove form change:', error);
          return false;
        }
      }
    };

    const messageHandler = (event: MessageEvent) => {
      if (!isStakTrakMessage(event)) return;

      const message = event.data as StakTrakMessage;

      // Check if this is an action removal message
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
              timestamp: event.data.assertion.timestamp || getTimeStamp(),
            });
          }
          break;
        case "staktrak-remove-assertion":
          if (event.data.assertionId) {
            // Find the assertion being removed to get its timestamp
            const assertionToRemove = this.memory.assertions.find(
              assertion => assertion.id === event.data.assertionId
            );

            // Remove the assertion
            this.memory.assertions = this.memory.assertions.filter(
              assertion => assertion.id !== event.data.assertionId
            );

            // Also remove the click that created this assertion
            // Find the most recent click before the assertion timestamp
            if (assertionToRemove) {
              const assertionTime = assertionToRemove.timestamp;
              // Find clicks that happened before this assertion
              const clicksBefore = this.results.clicks.clickDetails.filter(
                click => click.timestamp < assertionTime
              );
              if (clicksBefore.length > 0) {
                // Find the most recent click before the assertion
                const mostRecentClick = clicksBefore.reduce((latest, current) =>
                  current.timestamp > latest.timestamp ? current : latest
                );
                // Remove that click
                this.results.clicks.clickDetails = this.results.clicks.clickDetails.filter(
                  click => click.timestamp !== mostRecentClick.timestamp
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
            const assertionId = Date.now() + Math.random();
            const assertion = {
              id: assertionId,
              type: "hasText",
              selector,
              value: text,
              timestamp: getTimeStamp(),
            };
            this.memory.assertions.push(assertion);
            // Send complete assertion data to parent
            this.sendEventToParent("assertion", assertion);
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
    this.processResults();
    this.isRunning = false;

    // Clear persisted state after successful stop
    sessionStorage.removeItem('stakTrakActiveRecording');

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

  private clearAllActions() {
    // Clear all tracking data
    this.results.pageNavigation = [];
    this.results.clicks.clickDetails = [];
    this.results.clicks.clickCount = 0;
    this.results.inputChanges = [];
    this.results.formElementChanges = [];
    this.memory.assertions = [];
  }

  public attemptSessionRestoration() {
    try {
      const activeRecording = sessionStorage.getItem('stakTrakActiveRecording');
      if (!activeRecording) {
        return;
      }

      const recordingData = JSON.parse(activeRecording);

      // Simple validation: if session data exists and claims to be recording, restore it
      if (recordingData && recordingData.isRecording && recordingData.version === "1.0") {

        // Detect if this is an iframe reload (page loaded recently after session was saved)
        const timeSinceLastSave = Date.now() - (recordingData.lastSaved || 0);
        const isLikelyIframeReload = timeSinceLastSave < 10000; // Within 10 seconds

        if (isLikelyIframeReload) {
          
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
          

          // Verify event listeners are working
          this.verifyEventListeners();

          // Notify parent that recording is active again
          window.parent.postMessage({ type: "staktrak-replay-ready" }, "*");
        } else {
          sessionStorage.removeItem('stakTrakActiveRecording');
        }
      } else {
        // Invalid session data, starting fresh
        sessionStorage.removeItem('stakTrakActiveRecording');
      }
    } catch (error) {
      
      sessionStorage.removeItem('stakTrakActiveRecording');
    }
  }

  private verifyEventListeners() {
    
    // If we have fewer listeners than expected, re-setup
    if (this.isRunning && this.memory.listeners.length === 0) {
      
      this.setupEventListeners();
    }
  }

  public recoverRecording() {
    if (!this.isRunning) {
      return;
    }
    
    // Ensure event listeners are active
    this.verifyEventListeners();
    
    // Save current state
    this.saveSessionState();
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
    
  }
}

// Create global instance (simple, always works)
const userBehaviour = new UserBehaviorTracker();

// Auto-start when DOM is ready
const initializeStakTrak = () => {
  userBehaviour
    .makeConfig({
      processData: (results) => {
        // Recording completed - results are available
      },
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
