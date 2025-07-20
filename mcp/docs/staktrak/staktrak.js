var userBehaviour = (function () {
  var defaults = {
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
    processData: function (results) {
      console.log(results);
    },
  };

  var user_config = {};
  var mem = {
    processInterval: null,
    mouseInterval: null,
    mousePosition: [],
    inputDebounceTimers: {},
    selectionMode: false,
    assertionDebounceTimer: null,
    assertions: [],
    eventListeners: {
      scroll: null,
      click: null,
      mouseMovement: null,
      windowResize: null,
      visibilitychange: null,
      keyboardActivity: null,
      inputChange: null,
      focusChange: null,
      touchStart: null,
      documentFocus: null,
      documentBlur: null,
      documentInput: null,
      mouseUp: null,
      keyDown: null,
      formElementChange: null,
    },
    mutationObserver: null,
    eventsFunctions: {
      scroll: () => {
        results.mouseScroll.push([
          window.scrollX,
          window.scrollY,
          getTimeStamp(),
        ]);
      },
      click: (e) => {
        results.clicks.clickCount++;
        var path = [];
        var node = "";
        e.composedPath().forEach((el, i) => {
          if (
            i !== e.composedPath().length - 1 &&
            i !== e.composedPath().length - 2
          ) {
            node = el.localName;

            if (el.dataset && el.dataset.testid) {
              node += `[data-testid="${el.dataset.testid}"]`;
            } else {
              if (el.className !== "") {
                el.classList.forEach((clE) => {
                  if (clE !== "staktrak-selection-active") {
                    node += "." + clE;
                  }
                });
              }
              if (el.id !== "") {
                node += "#" + el.id;
              }
            }

            path.push(node);
          }
        });
        path = path.reverse().join(">");
        results.clicks.clickDetails.push([
          e.clientX,
          e.clientY,
          path,
          getTimeStamp(),
        ]);

        const target = e.target;
        if (
          target.tagName === "INPUT" &&
          (target.type === "checkbox" || target.type === "radio")
        ) {
          results.formElementChanges.push({
            elementSelector: getElementSelector(target),
            type: target.type,
            checked: target.checked,
            value: target.value,
            timestamp: getTimeStamp(),
          });
        }
      },
      mouseMovement: (e) => {
        mem.mousePosition = [e.clientX, e.clientY, getTimeStamp()];
      },
      windowResize: (e) => {
        results.windowSizes.push([
          window.innerWidth,
          window.innerHeight,
          getTimeStamp(),
        ]);
      },
      visibilitychange: (e) => {
        results.visibilitychanges.push([
          document.visibilityState,
          getTimeStamp(),
        ]);
        processResults();
      },
      keyboardActivity: (e) => {
        if (!isInputOrTextarea(e.target)) {
          results.keyboardActivities.push([e.key, getTimeStamp()]);
        }
      },
      inputChange: (e) => {
        const target = e.target;
        const selector = getElementSelector(target);
        const elementId = target.id || selector;

        if (mem.inputDebounceTimers[elementId]) {
          clearTimeout(mem.inputDebounceTimers[elementId]);
        }

        mem.inputDebounceTimers[elementId] = setTimeout(() => {
          results.inputChanges.push({
            elementSelector: selector,
            value: target.value,
            timestamp: getTimeStamp(),
            action: "complete",
          });

          delete mem.inputDebounceTimers[elementId];
        }, user_config.inputDebounceDelay);

        results.inputChanges.push({
          elementSelector: selector,
          value: target.value,
          timestamp: getTimeStamp(),
          action: "intermediate",
        });
      },
      formElementChange: (e) => {
        const target = e.target;
        const selector = getElementSelector(target);

        if (target.tagName === "SELECT") {
          const selectedOption = target.options[target.selectedIndex];
          results.formElementChanges.push({
            elementSelector: selector,
            type: "select",
            value: target.value,
            text: selectedOption ? selectedOption.text : "",
            timestamp: getTimeStamp(),
          });
        } else if (
          target.tagName === "INPUT" &&
          (target.type === "checkbox" || target.type === "radio")
        ) {
          results.formElementChanges.push({
            elementSelector: selector,
            type: target.type,
            checked: target.checked,
            value: target.value,
            timestamp: getTimeStamp(),
          });
        }
      },
      focusChange: (e) => {
        const target = e.target;
        if (isInputOrTextarea(target)) {
          const selector = getElementSelector(target);
          results.focusChanges.push({
            elementSelector: selector,
            type: e.type,
            timestamp: getTimeStamp(),
          });

          if (e.type === "blur") {
            const elementId = target.id || selector;
            if (mem.inputDebounceTimers[elementId]) {
              clearTimeout(mem.inputDebounceTimers[elementId]);
              delete mem.inputDebounceTimers[elementId];
            }

            results.inputChanges.push({
              elementSelector: selector,
              value: target.value,
              timestamp: getTimeStamp(),
              action: "complete",
            });
          }
        }
      },
      touchStart: (e) => {
        if (e.touches && e.touches.length > 0) {
          const touch = e.touches[0];
          results.touchEvents.push({
            type: "touchstart",
            x: touch.clientX,
            y: touch.clientY,
            timestamp: getTimeStamp(),
          });
        }
      },
      mouseUp: (e) => {
        if (mem.selectionMode) {
          const selection = window.getSelection();
          if (selection && selection.toString().trim() !== "") {
            const text = selection.toString();

            let container = selection.getRangeAt(0).commonAncestorContainer;

            if (container.nodeType === 3) {
              container = container.parentNode;
            }

            const selector = getElementSelector(container);

            if (mem.assertionDebounceTimer) {
              clearTimeout(mem.assertionDebounceTimer);
            }

            mem.assertionDebounceTimer = setTimeout(() => {
              const assertion = {
                type: "hasText",
                selector: selector,
                value: text,
                timestamp: getTimeStamp(),
              };

              mem.assertions.push(assertion);

              window.parent.postMessage(
                {
                  type: "staktrak-selection",
                  text: text,
                  selector: selector,
                },
                "*"
              );

              window.parent.postMessage(
                {
                  type: "staktrak-popup",
                  message: `Assertion added: hasText "${text.slice(0, 30)}${
                    text.length > 30 ? "..." : ""
                  }"`,
                  type: "success",
                },
                "*"
              );
            }, 300);
          }
        }
      },
    },
  };

  var results = {};

  // All processing logic stays in the iframe
  function processRecordedResults() {
    if (
      results.clicks &&
      results.clicks.clickDetails &&
      results.clicks.clickDetails.length > 0
    ) {
      const filteredClicks = filterClickDetails(
        results.clicks.clickDetails,
        mem.assertions
      );

      results.clicks.clickDetails = filteredClicks;
    }

    results.assertions = mem.assertions;

    return results;
  }

  /**
   * Filter and process click details to remove duplicates and assertion-related clicks
   * @param {Array} clickDetails - Raw click data
   * @param {Array} assertions - User assertions
   * @returns {Array} - Filtered click data
   */
  function filterClickDetails(clickDetails, assertions) {
    if (!clickDetails || !clickDetails.length) return [];

    let filteredClicks = clickDetails;

    // Filter out clicks that are related to assertions
    if (user_config.filterAssertionClicks && assertions.length > 0) {
      filteredClicks = filteredClicks.filter((clickDetail) => {
        const clickSelector = clickDetail[2];
        const clickTime = clickDetail[3];

        return !assertions.some((assertion) => {
          const assertionTime = assertion.timestamp;
          const assertionSelector = assertion.selector;

          const isCloseInTime = Math.abs(clickTime - assertionTime) < 1000;
          const isSameElement =
            clickSelector.includes(assertionSelector) ||
            assertionSelector.includes(clickSelector) ||
            (clickSelector.match(/\w+(?=[.#\[]|$)/) &&
              assertionSelector.match(/\w+(?=[.#\[]|$)/) &&
              clickSelector.match(/\w+(?=[.#\[]|$)/)[0] ===
                assertionSelector.match(/\w+(?=[.#\[]|$)/)[0]);

          return isCloseInTime && isSameElement;
        });
      });
    }

    // Group clicks by selector and remove rapid multi-clicks
    const clicksBySelector = {};
    filteredClicks.forEach((clickDetail) => {
      const selector = clickDetail[2];
      const timestamp = clickDetail[3];

      if (!clicksBySelector[selector]) {
        clicksBySelector[selector] = [];
      }
      clicksBySelector[selector].push({
        detail: clickDetail,
        timestamp,
      });
    });

    const finalFilteredClicks = [];
    Object.values(clicksBySelector).forEach((clicks) => {
      clicks.sort((a, b) => a.timestamp - b.timestamp);

      const resultClicks = [];
      let lastClick = null;

      clicks.forEach((click) => {
        if (
          !lastClick ||
          click.timestamp - lastClick.timestamp > user_config.multiClickInterval
        ) {
          resultClicks.push(click);
        }
        lastClick = click;
      });

      resultClicks.forEach((click) => finalFilteredClicks.push(click.detail));
    });

    finalFilteredClicks.sort((a, b) => a[3] - b[3]);

    return finalFilteredClicks;
  }

  function setupMutationObserver() {
    const observerCallback = (mutationsList) => {
      for (const mutation of mutationsList) {
        if (mutation.type === "childList") {
          mutation.addedNodes.forEach((node) => {
            if (node.nodeType === 1) {
              if (
                node.tagName === "INPUT" ||
                node.tagName === "SELECT" ||
                node.tagName === "TEXTAREA"
              ) {
                attachFormElementListeners(node);
              }

              const formElements = node.querySelectorAll(
                "input, select, textarea"
              );
              formElements.forEach((formElement) => {
                attachFormElementListeners(formElement);
              });
            }
          });
        }
      }
    };

    mem.mutationObserver = new MutationObserver(observerCallback);

    mem.mutationObserver.observe(document.body, {
      childList: true,
      subtree: true,
    });
  }

  function attachFormElementListeners(element) {
    if (
      element.tagName === "INPUT" &&
      (element.type === "checkbox" || element.type === "radio")
    ) {
      element.addEventListener("change", mem.eventsFunctions.formElementChange);
    } else if (element.tagName === "SELECT") {
      element.addEventListener("change", mem.eventsFunctions.formElementChange);
    } else if (element.tagName === "INPUT" || element.tagName === "TEXTAREA") {
      element.addEventListener("input", mem.eventsFunctions.inputChange);
      element.addEventListener("focus", mem.eventsFunctions.focusChange);
      element.addEventListener("blur", mem.eventsFunctions.focusChange);
    }
  }

  function setSelectionMode(isActive) {
    mem.selectionMode = isActive;

    if (isActive) {
      document.body.classList.add("staktrak-selection-active");
      document.addEventListener("mouseup", mem.eventsFunctions.mouseUp);
    } else {
      document.body.classList.remove("staktrak-selection-active");
      document.removeEventListener("mouseup", mem.eventsFunctions.mouseUp);
      window.getSelection().removeAllRanges();
    }

    window.parent.postMessage(
      {
        type: "staktrak-selection-mode-" + (isActive ? "started" : "ended"),
      },
      "*"
    );
  }

  function isInputOrTextarea(element) {
    return (
      element.tagName === "INPUT" ||
      element.tagName === "TEXTAREA" ||
      element.isContentEditable
    );
  }

  function getElementSelector(element) {
    if (!element || element.nodeType !== 1) return "";

    let selector = "";

    if (element.dataset && element.dataset.testid) {
      return `[data-testid="${element.dataset.testid}"]`;
    }

    if (element.id) {
      return `#${element.id}`;
    }

    selector = element.tagName.toLowerCase();

    if (element.className) {
      const classNames = Array.from(element.classList).filter(
        (cls) => cls !== "staktrak-selection-active"
      );

      if (classNames.length > 0) {
        selector += `.${classNames.join(".")}`;
      }
    }

    if (element.tagName === "INPUT" && element.type) {
      selector += `[type="${element.type}"]`;
    }

    return selector;
  }

  function resetResults() {
    mem.assertions = [];

    results = {
      userInfo: user_config.userInfo
        ? {
            url: document.URL,
            userAgent: navigator.userAgent,
            platform: navigator.platform,
            windowSize: [window.innerWidth, window.innerHeight],
          }
        : {},
      pageNavigation: [],
      clicks: {
        clickCount: 0,
        clickDetails: [],
      },
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

    if (user_config.timeCount) {
      results.time = {
        startedAt: getTimeStamp(),
        completedAt: 0,
        totalSeconds: 0,
      };
    }
  }

  function getTimeStamp() {
    return new Date().getTime();
  }

  function config(ob) {
    user_config = { ...defaults, ...ob };
    return userBehaviour;
  }

  function start() {
    resetResults();

    if (user_config.clicks) {
      mem.eventListeners.click = mem.eventsFunctions.click;
      document.addEventListener("click", mem.eventListeners.click);
    }

    if (user_config.mouseScroll) {
      mem.eventListeners.scroll = mem.eventsFunctions.scroll;
      window.addEventListener("scroll", mem.eventListeners.scroll);
    }

    if (user_config.mouseMovement) {
      mem.eventListeners.mouseMovement = mem.eventsFunctions.mouseMovement;
      document.addEventListener("mousemove", mem.eventListeners.mouseMovement);
      mem.mouseInterval = setInterval(function () {
        if (
          mem.mousePosition.length > 0 &&
          mem.mousePosition[2] + 500 > getTimeStamp()
        ) {
          results.mouseMovement.push(mem.mousePosition);
        }
      }, user_config.mouseMovementInterval * 1000);
    }

    if (user_config.windowResize) {
      mem.eventListeners.windowResize = mem.eventsFunctions.windowResize;
      window.addEventListener("resize", mem.eventListeners.windowResize);
    }

    if (user_config.visibilitychange) {
      mem.eventListeners.visibilitychange =
        mem.eventsFunctions.visibilitychange;
      document.addEventListener(
        "visibilitychange",
        mem.eventListeners.visibilitychange
      );
    }

    if (user_config.keyboardActivity) {
      mem.eventListeners.keyboardActivity =
        mem.eventsFunctions.keyboardActivity;
      document.addEventListener(
        "keypress",
        mem.eventListeners.keyboardActivity
      );
    }

    if (user_config.formInteractions) {
      const formElements = document.querySelectorAll("input, select, textarea");
      formElements.forEach((element) => {
        attachFormElementListeners(element);
      });

      setupMutationObserver();
    }

    if (user_config.touchEvents) {
      mem.eventListeners.touchStart = mem.eventsFunctions.touchStart;
      document.addEventListener("touchstart", mem.eventListeners.touchStart);
    }

    if (user_config.pageNavigation) {
      const pushState = history.pushState;
      history.pushState = function () {
        pushState.apply(history, arguments);
        results.pageNavigation.push({
          type: "pushState",
          url: document.URL,
          timestamp: getTimeStamp(),
        });
      };

      const replaceState = history.replaceState;
      history.replaceState = function () {
        replaceState.apply(history, arguments);
        results.pageNavigation.push({
          type: "replaceState",
          url: document.URL,
          timestamp: getTimeStamp(),
        });
      };

      window.addEventListener("popstate", function () {
        results.pageNavigation.push({
          type: "popstate",
          url: document.URL,
          timestamp: getTimeStamp(),
        });
      });
    }

    window.parent.postMessage({ type: "staktrak-setup" }, "*");

    // Handle messages from parent
    window.addEventListener("message", function (event) {
      if (event.data && event.data.type) {
        switch (event.data.type) {
          case "staktrak-start":
            resetResults();
            break;

          case "staktrak-stop":
            userBehaviour.stop();
            break;

          case "staktrak-enable-selection":
            setSelectionMode(true);
            break;

          case "staktrak-disable-selection":
            setSelectionMode(false);
            break;

          case "staktrak-add-assertion":
            if (event.data.assertion) {
              mem.assertions.push({
                type: event.data.assertion.type || "hasText",
                selector: event.data.assertion.selector,
                value: event.data.assertion.value || "",
                timestamp: getTimeStamp(),
              });
            }
            break;
        }
      }
    });

    return userBehaviour;
  }

  function processResults() {
    if (user_config.timeCount) {
      results.time.completedAt = getTimeStamp();
      results.time.totalSeconds =
        (results.time.completedAt - results.time.startedAt) / 1000;
    }

    // Process all data in the iframe before sending to parent
    const processedResults = processRecordedResults();

    window.parent.postMessage(
      {
        type: "staktrak-results",
        data: processedResults,
      },
      "*"
    );

    user_config.processData(processedResults);

    if (user_config.clearAfterProcess) {
      resetResults();
    }
  }

  function stop() {
    if (user_config.clicks) {
      document.removeEventListener("click", mem.eventListeners.click);
    }

    if (user_config.mouseScroll) {
      window.removeEventListener("scroll", mem.eventListeners.scroll);
    }

    if (user_config.mouseMovement) {
      document.removeEventListener(
        "mousemove",
        mem.eventListeners.mouseMovement
      );
      clearInterval(mem.mouseInterval);
    }

    if (user_config.windowResize) {
      window.removeEventListener("resize", mem.eventListeners.windowResize);
    }

    if (user_config.visibilitychange) {
      document.removeEventListener(
        "visibilitychange",
        mem.eventListeners.visibilitychange
      );
    }

    if (user_config.keyboardActivity) {
      document.removeEventListener(
        "keypress",
        mem.eventListeners.keyboardActivity
      );
    }

    if (user_config.formInteractions) {
      const formElements = document.querySelectorAll("input, select, textarea");
      formElements.forEach((element) => {
        if (
          element.tagName === "INPUT" &&
          (element.type === "checkbox" || element.type === "radio")
        ) {
          element.removeEventListener(
            "change",
            mem.eventsFunctions.formElementChange
          );
        } else if (element.tagName === "SELECT") {
          element.removeEventListener(
            "change",
            mem.eventsFunctions.formElementChange
          );
        } else if (
          element.tagName === "INPUT" ||
          element.tagName === "TEXTAREA"
        ) {
          element.removeEventListener("input", mem.eventsFunctions.inputChange);
          element.removeEventListener("focus", mem.eventsFunctions.focusChange);
          element.removeEventListener("blur", mem.eventsFunctions.focusChange);
        }
      });

      if (mem.mutationObserver) {
        mem.mutationObserver.disconnect();
      }
    }

    if (user_config.touchEvents) {
      document.removeEventListener("touchstart", mem.eventListeners.touchStart);
    }

    if (mem.selectionMode) {
      setSelectionMode(false);
    }

    for (const elementId in mem.inputDebounceTimers) {
      clearTimeout(mem.inputDebounceTimers[elementId]);
    }
    mem.inputDebounceTimers = {};

    processResults();
    return userBehaviour;
  }

  function result() {
    return results;
  }

  function showConfig() {
    return user_config;
  }

  return {
    config,
    start,
    stop,
    result,
    showConfig,
    addAssertion: function (type, selector, value) {
      mem.assertions.push({
        type: type || "hasText",
        selector: selector,
        value: value || "",
        timestamp: getTimeStamp(),
      });
    },
  };
})();

// Initialize when DOM is ready
document.addEventListener("DOMContentLoaded", function () {
  userBehaviour
    .config({
      processData: function (results) {
        console.log("StakTrak recording processed:", results);
      },
    })
    .start();
});
