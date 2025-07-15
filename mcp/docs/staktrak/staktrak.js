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
    processData: function (results) {
      console.log(results);
    },
  };
  var user_config = {};
  var mem = {
    processInterval: null,
    mouseInterval: null,
    mousePosition: [], //x,y,timestamp
    inputDebounceTimers: {},
    selectionMode: false,
    assertionDebounceTimer: null,
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

            // Use data-testid if available, otherwise fall back to class/id
            if (el.dataset && el.dataset.testid) {
              node += `[data-testid="${el.dataset.testid}"]`;
            } else {
              // Only add classes and IDs if no data-testid
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
      documentFocus: (e) => {
        if (isInputOrTextarea(e.target)) {
          mem.eventsFunctions.focusChange(e);
        }
      },
      documentBlur: (e) => {
        if (isInputOrTextarea(e.target)) {
          mem.eventsFunctions.focusChange(e);
        }
      },
      documentInput: (e) => {
        if (isInputOrTextarea(e.target)) {
          mem.eventsFunctions.inputChange(e);
        }
      },
      pageNavigation: () => {
        results.navigationHistory.push([location.href, getTimeStamp()]);
      },
      formInteraction: (e) => {
        e.preventDefault(); // Prevent the form from submitting normally
        results.formInteractions.push([e.target.name, getTimeStamp()]);
        // Optionally, submit the form programmatically after tracking
      },
      touchStart: (e) => {
        results.touchEvents.push([
          "touchstart",
          e.touches[0].clientX,
          e.touches[0].clientY,
          getTimeStamp(),
        ]);
      },
      mediaInteraction: (e) => {
        results.mediaInteractions.push([
          "play",
          e.target.currentSrc,
          getTimeStamp(),
        ]);
      },
      mouseUp: (e) => {
        if (!mem.selectionMode) return;

        const selection = window.getSelection();
        if (selection && selection.toString().trim() !== "") {
          const selectedText = selection.toString().trim();
          let selectedElement = null;

          if (selection.rangeCount > 0) {
            const range = selection.getRangeAt(0);
            selectedElement = range.commonAncestorContainer;

            if (selectedElement.nodeType === 3) {
              selectedElement = selectedElement.parentElement;
            }
          }

          const selector = selectedElement
            ? getElementSelector(selectedElement)
            : "";

          if (mem.assertionDebounceTimer) {
            clearTimeout(mem.assertionDebounceTimer);
          }

          mem.assertionDebounceTimer = setTimeout(() => {
            results.assertions = results.assertions || [];

            const hasSimilarAssertion = results.assertions.some(
              (assertion) =>
                assertion.selector === selector &&
                Math.abs(assertion.timestamp - getTimeStamp()) < 1000
            );

            if (!hasSimilarAssertion) {
              results.assertions.push({
                type: "hasText",
                selector: selector,
                value: selectedText,
                timestamp: getTimeStamp(),
              });

              window.parent.postMessage(
                {
                  type: "staktrak-selection",
                  text: selectedText,
                  selector: selector,
                },
                "*"
              );

              window.parent.postMessage(
                {
                  type: "staktrak-show-popup",
                  text: selectedText,
                  selector: selector,
                },
                "*"
              );
            }

            setTimeout(() => {
              if (window.getSelection) {
                if (window.getSelection().empty) {
                  window.getSelection().empty();
                } else if (window.getSelection().removeAllRanges) {
                  window.getSelection().removeAllRanges();
                }
              }
            }, 500);

            mem.assertionDebounceTimer = null;
          }, 500);
        }
      },
      keyDown: (e) => {
        if (mem.selectionMode && e.key === "Escape") {
          setSelectionMode(false);
          window.parent.postMessage(
            { type: "staktrak-selection-mode-ended" },
            "*"
          );
        }
      },
    },
  };
  var results = {};

  function setSelectionMode(isActive) {
    mem.selectionMode = isActive;
    if (isActive) {
      document.body.classList.add("staktrak-selection-active");
    } else {
      document.body.classList.remove("staktrak-selection-active");
      if (mem.assertionDebounceTimer) {
        clearTimeout(mem.assertionDebounceTimer);
        mem.assertionDebounceTimer = null;
      }
    }
  }

  function isInputOrTextarea(element) {
    return (
      element &&
      (element.tagName === "INPUT" ||
        element.tagName === "TEXTAREA" ||
        element.tagName.toLowerCase() === "input" ||
        element.tagName.toLowerCase() === "textarea")
    );
  }

  function getElementSelector(element) {
    if (element.dataset && element.dataset.testid) {
      return `[data-testid="${element.dataset.testid}"]`;
    }

    if (element.id) {
      return `#${element.id}`;
    }

    let selector = element.tagName.toLowerCase();

    if (element.className) {
      const classes = Array.from(element.classList)
        .filter((cls) => cls !== "staktrak-selection-active")
        .join(".");
      if (classes) {
        selector += `.${classes}`;
      }
    }

    if (selector !== element.tagName.toLowerCase()) {
      return selector;
    }

    let path = "";
    let currentElement = element;
    const maxDepth = 3;
    let depth = 0;

    while (
      currentElement &&
      currentElement !== document.body &&
      depth < maxDepth
    ) {
      let elementSelector = currentElement.tagName.toLowerCase();

      if (currentElement.id) {
        elementSelector = `#${currentElement.id}`;
      } else if (currentElement.className) {
        const classes = Array.from(currentElement.classList)
          .filter((cls) => cls !== "staktrak-selection-active")
          .join(".");
        if (classes) {
          elementSelector += `.${classes}`;
        }
      } else if (currentElement.parentElement) {
        const siblings = Array.from(
          currentElement.parentElement.children
        ).filter((el) => el.tagName === currentElement.tagName);

        if (siblings.length > 1) {
          const index = siblings.indexOf(currentElement) + 1;
          elementSelector += `:nth-child(${index})`;
        }
      }

      path = path ? `${elementSelector} > ${path}` : elementSelector;
      currentElement = currentElement.parentElement;
      depth++;
    }

    return path || selector;
  }

  function resetResults() {
    results = {
      userInfo: {
        windowSize: [window.innerWidth, window.innerHeight],
        appCodeName: navigator.appCodeName || "",
        appName: navigator.appName || "",
        vendor: navigator.vendor || "",
        platform: navigator.platform || "",
        userAgent: navigator.userAgent || "",
      },
      time: {
        startTime: 0,
        currentTime: 0,
        stopTime: 0,
      },
      clicks: {
        clickCount: 0,
        clickDetails: [],
      },
      mouseMovements: [],
      mouseScroll: [],
      keyboardActivities: [],
      inputChanges: [],
      focusChanges: [],
      navigationHistory: [],
      formInteractions: [],
      formElementChanges: [],
      touchEvents: [],
      mediaInteractions: [],
      windowSizes: [],
      visibilitychanges: [],
      assertions: [],
    };
  }
  resetResults();

  function getTimeStamp() {
    return Date.now();
  }

  function config(ob) {
    user_config = {};
    Object.keys(defaults).forEach((i) => {
      i in ob ? (user_config[i] = ob[i]) : (user_config[i] = defaults[i]);
    });
  }

  function setupMutationObserver() {
    const observerCallback = (mutationsList) => {
      for (const mutation of mutationsList) {
        if (mutation.type === "childList") {
          mutation.addedNodes.forEach((node) => {
            if (user_config.formInteractions) {
              if (node.nodeName === "FORM") {
                node.addEventListener(
                  "submit",
                  mem.eventsFunctions.formInteraction
                );
              } else if (node.querySelectorAll) {
                const forms = node.querySelectorAll("form");
                forms.forEach((form) => {
                  form.addEventListener(
                    "submit",
                    mem.eventsFunctions.formInteraction
                  );
                });
              }
            }

            if (user_config.audioVideoInteraction) {
              if (node.nodeName === "VIDEO" || node.nodeName === "AUDIO") {
                node.addEventListener(
                  "play",
                  mem.eventsFunctions.mediaInteraction
                );
              } else if (node.querySelectorAll) {
                const media = node.querySelectorAll("video, audio");
                media.forEach((mediaElement) => {
                  mediaElement.addEventListener(
                    "play",
                    mem.eventsFunctions.mediaInteraction
                  );
                });
              }
            }

            if (
              node.nodeName === "SELECT" ||
              (node.nodeName === "INPUT" &&
                (node.type === "checkbox" || node.type === "radio"))
            ) {
              node.addEventListener(
                "change",
                mem.eventsFunctions.formElementChange
              );
            } else if (node.querySelectorAll) {
              const formElements = node.querySelectorAll(
                'select, input[type="checkbox"], input[type="radio"]'
              );
              formElements.forEach((element) => {
                element.addEventListener(
                  "change",
                  mem.eventsFunctions.formElementChange
                );
              });
            }
          });
        }
      }
    };

    const observer = new MutationObserver(observerCallback);
    observer.observe(document.body, { childList: true, subtree: true });

    return observer;
  }

  function start() {
    if (Object.keys(user_config).length !== Object.keys(defaults).length) {
      console.log("no config provided. using default..");
      user_config = defaults;
    }
    // TIME SET
    if (user_config.timeCount !== undefined && user_config.timeCount) {
      results.time.startTime = getTimeStamp();
    }
    // MOUSE MOVEMENTS
    if (user_config.mouseMovement) {
      mem.eventListeners.mouseMovement = window.addEventListener(
        "mousemove",
        mem.eventsFunctions.mouseMovement
      );
      mem.mouseInterval = setInterval(() => {
        if (mem.mousePosition && mem.mousePosition.length) {
          if (
            !results.mouseMovements.length ||
            (mem.mousePosition[0] !==
              results.mouseMovements[results.mouseMovements.length - 1][0] &&
              mem.mousePosition[1] !==
                results.mouseMovements[results.mouseMovements.length - 1][1])
          ) {
            results.mouseMovements.push(mem.mousePosition);
          }
        }
      }, defaults.mouseMovementInterval * 1000);
    }
    //CLICKS
    if (user_config.clicks) {
      mem.eventListeners.click = window.addEventListener(
        "click",
        mem.eventsFunctions.click
      );
    }
    //SCROLL
    if (user_config.mouseScroll) {
      mem.eventListeners.scroll = window.addEventListener(
        "scroll",
        mem.eventsFunctions.scroll
      );
    }
    //Window sizes
    if (user_config.windowResize !== false) {
      mem.eventListeners.windowResize = window.addEventListener(
        "resize",
        mem.eventsFunctions.windowResize
      );
    }
    //Before unload / visibilitychange
    if (user_config.visibilitychange !== false) {
      mem.eventListeners.visibilitychange = window.addEventListener(
        "visibilitychange",
        mem.eventsFunctions.visibilitychange
      );
    }
    //Keyboard Activity
    if (user_config.keyboardActivity) {
      mem.eventListeners.keyboardActivity = window.addEventListener(
        "keydown",
        mem.eventsFunctions.keyboardActivity
      );

      document.addEventListener(
        "focus",
        mem.eventsFunctions.documentFocus,
        true
      );
      document.addEventListener("blur", mem.eventsFunctions.documentBlur, true);
      document.addEventListener(
        "input",
        mem.eventsFunctions.documentInput,
        true
      );

      mem.eventListeners.documentFocus = true;
      mem.eventListeners.documentBlur = true;
      mem.eventListeners.documentInput = true;
    }
    //Page Navigation
    if (user_config.pageNavigation) {
      window.history.pushState = ((f) =>
        function pushState() {
          var ret = f.apply(this, arguments);
          window.dispatchEvent(new Event("pushstate"));
          window.dispatchEvent(new Event("locationchange"));
          return ret;
        })(window.history.pushState);

      window.addEventListener("popstate", mem.eventsFunctions.pageNavigation);
      window.addEventListener("pushstate", mem.eventsFunctions.pageNavigation);
      window.addEventListener(
        "locationchange",
        mem.eventsFunctions.pageNavigation
      );
    }
    //Form Interactions
    if (user_config.formInteractions) {
      document
        .querySelectorAll("form")
        .forEach((form) =>
          form.addEventListener("submit", mem.eventsFunctions.formInteraction)
        );

      document
        .querySelectorAll('select, input[type="checkbox"], input[type="radio"]')
        .forEach((element) => {
          element.addEventListener(
            "change",
            mem.eventsFunctions.formElementChange
          );
        });

      mem.eventListeners.formElementChange = true;
    }
    //Touch Events
    if (user_config.touchEvents) {
      mem.eventListeners.touchStart = window.addEventListener(
        "touchstart",
        mem.eventsFunctions.touchStart
      );
    }
    //Audio & Video Interaction
    if (user_config.audioVideoInteraction) {
      document.querySelectorAll("video").forEach((video) => {
        video.addEventListener("play", mem.eventsFunctions.mediaInteraction);
        // Add other media events as needed
      });
    }

    document.addEventListener("mouseup", mem.eventsFunctions.mouseUp);
    document.addEventListener("keydown", mem.eventsFunctions.keyDown);
    mem.eventListeners.mouseUp = true;
    mem.eventListeners.keyDown = true;

    mem.mutationObserver = setupMutationObserver();
  }

  function processResults() {
    for (const elementId in mem.inputDebounceTimers) {
      clearTimeout(mem.inputDebounceTimers[elementId]);
      delete mem.inputDebounceTimers[elementId];
    }

    user_config.processData(result());
    if (user_config.clearAfterProcess) {
      resetResults();
    }
  }

  function stop() {
    clearInterval(mem.mouseInterval);
    window.removeEventListener("scroll", mem.eventsFunctions.scroll);
    window.removeEventListener("click", mem.eventsFunctions.click);
    window.removeEventListener("mousemove", mem.eventsFunctions.mouseMovement);
    window.removeEventListener("resize", mem.eventsFunctions.windowResize);
    window.removeEventListener(
      "visibilitychange",
      mem.eventsFunctions.visibilitychange
    );
    window.removeEventListener("keydown", mem.eventsFunctions.keyboardActivity);

    document.removeEventListener(
      "focus",
      mem.eventsFunctions.documentFocus,
      true
    );
    document.removeEventListener(
      "blur",
      mem.eventsFunctions.documentBlur,
      true
    );
    document.removeEventListener(
      "input",
      mem.eventsFunctions.documentInput,
      true
    );

    window.removeEventListener("touchstart", mem.eventsFunctions.touchStart);

    document.removeEventListener("mouseup", mem.eventsFunctions.mouseUp);
    document.removeEventListener("keydown", mem.eventsFunctions.keyDown);

    if (mem.eventListeners.formElementChange) {
      document
        .querySelectorAll('select, input[type="checkbox"], input[type="radio"]')
        .forEach((element) => {
          element.removeEventListener(
            "change",
            mem.eventsFunctions.formElementChange
          );
        });
    }

    if (mem.mutationObserver) {
      mem.mutationObserver.disconnect();
      mem.mutationObserver = null;
    }

    if (mem.assertionDebounceTimer) {
      clearTimeout(mem.assertionDebounceTimer);
      mem.assertionDebounceTimer = null;
    }

    for (const elementId in mem.inputDebounceTimers) {
      clearTimeout(mem.inputDebounceTimers[elementId]);
      delete mem.inputDebounceTimers[elementId];
    }

    setSelectionMode(false);

    results.time.stopTime = getTimeStamp();
    processResults();
  }

  function result() {
    if (
      user_config.userInfo === false &&
      userBehaviour.showResult().userInfo !== undefined
    ) {
      delete userBehaviour.showResult().userInfo;
    }
    if (user_config.timeCount !== undefined && user_config.timeCount) {
      results.time.currentTime = getTimeStamp();
    }
    return results;
  }

  function showConfig() {
    if (Object.keys(user_config).length !== Object.keys(defaults).length) {
      return defaults;
    } else {
      return user_config;
    }
  }

  return {
    showConfig: showConfig,
    config: config,
    start: start,
    stop: stop,
    showResult: result,
    processResults: processResults,
    registerCustomEvent: (eventName, callback) => {
      window.addEventListener(eventName, callback);
    },
    enableSelectionMode: () => {
      setSelectionMode(true);
    },
    disableSelectionMode: () => {
      setSelectionMode(false);
    },
  };
})();

window.addEventListener("DOMContentLoaded", () => {
  setTimeout(() => {
    window.parent.postMessage({ type: "staktrak-setup", status: "ready" }, "*");
  }, 500);
});

window.addEventListener("message", (event) => {
  if (event.data && event.data.type) {
    switch (event.data.type) {
      case "staktrak-start":
        userBehaviour.start();
        break;
      case "staktrak-stop":
        const results = userBehaviour.showResult();
        window.parent.postMessage(
          {
            type: "staktrak-results",
            data: results,
          },
          "*"
        );
        userBehaviour.stop();
        break;
      case "staktrak-enable-selection":
        userBehaviour.enableSelectionMode();
        break;
      case "staktrak-disable-selection":
        userBehaviour.disableSelectionMode();
        break;
    }
  }
});
