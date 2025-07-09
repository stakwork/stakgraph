var userBehaviour = (function () {
  var defaults = {
    userInfo: true,
    clicks: true,
    mouseMovement: false,
    mouseMovementInterval: 1,
    mouseScroll: true,
    timeCount: true,
    clearAfterProcess: true,
    processTime: 15,
    windowResize: true,
    visibilitychange: true,
    keyboardActivity: true,
    pageNavigation: true,
    formInteractions: true,
    touchEvents: true,
    audioVideoInteraction: true,
    customEventRegistration: true,
    processData: function (results) {
      console.log(results);
    },
  };
  var user_config = {};
  var mem = {
    processInterval: null,
    mouseInterval: null,
    mousePosition: [], //x,y,timestamp
    eventListeners: {
      scroll: null,
      click: null,
      mouseMovement: null,
      windowResize: null,
      visibilitychange: null,
      keyboardActivity: null,
      touchStart: null,
    },
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
            el.className !== ""
              ? el.classList.forEach((clE) => {
                  node += "." + clE;
                })
              : 0;
            el.id !== "" ? (node += "#" + el.id) : 0;
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
        results.keyboardActivities.push([e.key, getTimeStamp()]);
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
    },
  };
  var results = {};

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
      navigationHistory: [],
      formInteractions: [],
      touchEvents: [],
      mediaInteractions: [],
      windowSizes: [],
      visibilitychanges: [],
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

    //PROCESS INTERVAL
    if (user_config.processTime !== false) {
      mem.processInterval = setInterval(() => {
        processResults();
      }, user_config.processTime * 1000);
    }
  }

  function processResults() {
    user_config.processData(result());
    if (user_config.clearAfterProcess) {
      resetResults();
    }
  }

  function stop() {
    if (user_config.processTime !== false) {
      clearInterval(mem.processInterval);
    }
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
    window.removeEventListener("touchstart", mem.eventsFunctions.touchStart);
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
  };
})();

userBehaviour.start();
