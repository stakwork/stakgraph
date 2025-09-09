import { useState, useEffect, useRef } from "https://esm.sh/preact/hooks";

export function useIframeMessaging(iframeRef, initialURL) {
  const popupHook = usePopup();
  const [isRecording, setIsRecording] = useState(false);
  const [isAssertionMode, setIsAssertionMode] = useState(false);
  const [canGenerate, setCanGenerate] = useState(false);
  const [trackingData, setTrackingData] = useState(null);
  const [selectedText, setSelectedText] = useState(null);
  const [url, setUrl] = useState(initialURL);
  const [displayUrl, setDisplayUrl] = useState(initialURL);

  const { showPopup } = popupHook;
  const selectedDisplayTimeout = useRef(null);

  const handleUrlChange = (e) => {
    setUrl(e.target.value);
    setDisplayUrl(e.target.value);
  };

  const navigateToUrl = () => {
    if (iframeRef.current) {
      iframeRef.current.src = url;
    }
  };

  const displaySelectedText = (text) => {
    setSelectedText(text);

    if (selectedDisplayTimeout.current) {
      clearTimeout(selectedDisplayTimeout.current);
    }

    selectedDisplayTimeout.current = setTimeout(() => {
      clearSelectedTextDisplay();
    }, 2000);
  };

  const clearSelectedTextDisplay = () => {
    const appDisplayEl = document.getElementById("app-selection-display");
    if (appDisplayEl) {
      appDisplayEl.classList.add("slide-out");
      setTimeout(() => {
        setSelectedText(null);
      }, 300);
    } else {
      setSelectedText(null);
    }

    if (selectedDisplayTimeout.current) {
      clearTimeout(selectedDisplayTimeout.current);
      selectedDisplayTimeout.current = null;
    }
  };

  useEffect(() => {
    const handleMessage = (event) => {
      if (event.data && event.data.type) {
        if (event.data && event.data.type.startsWith("staktrak-")) {
          console.log("Staktrak message received:", event.data);
        }
        switch (event.data.type) {
          case "staktrak-setup":
            console.log("Staktrak setup message received");
            break;
          case "staktrak-results":
            console.log("Staktrak results received", event.data.data);
            setTrackingData(event.data.data);
            setCanGenerate(true);
            break;
          case "staktrak-selection":
            displaySelectedText(event.data.text);
            break;
          case "staktrak-popup":
            if (event.data.message) {
              showPopup(event.data.message, event.data.type || "info");
            }
            break;
          case "staktrak-page-navigation":
            console.log("Staktrak page navigation:", event.data.data);
            setDisplayUrl(event.data.data);
            break;
        }
      }
    };

    window.addEventListener("message", handleMessage);
    return () => {
      window.removeEventListener("message", handleMessage);
      clearSelectedTextDisplay();
    };
  }, []);

  const startRecording = () => {
    if (iframeRef.current && iframeRef.current.contentWindow) {
      iframeRef.current.contentWindow.postMessage(
        { type: "staktrak-start" },
        "*"
      );
      setIsRecording(true);
      setIsAssertionMode(false);
      setCanGenerate(false);
    }
  };

  const stopRecording = () => {
    if (iframeRef.current && iframeRef.current.contentWindow) {
      iframeRef.current.contentWindow.postMessage(
        { type: "staktrak-stop" },
        "*"
      );
      setIsRecording(false);
      setIsAssertionMode(false);
      clearSelectedTextDisplay();
    }
  };

  const enableAssertionMode = () => {
    setIsAssertionMode(true);
    if (iframeRef.current && iframeRef.current.contentWindow) {
      iframeRef.current.contentWindow.postMessage(
        { type: "staktrak-enable-selection" },
        "*"
      );
    }
  };

  const disableAssertionMode = () => {
    setIsAssertionMode(false);
    clearSelectedTextDisplay();
    if (iframeRef.current && iframeRef.current.contentWindow) {
      iframeRef.current.contentWindow.postMessage(
        { type: "staktrak-disable-selection" },
        "*"
      );
    }
  };

  return {
    isRecording,
    isAssertionMode,
    canGenerate,
    trackingData,
    selectedText,
    startRecording,
    stopRecording,
    enableAssertionMode,
    disableAssertionMode,
    url,
    setUrl,
    handleUrlChange,
    navigateToUrl,
    displayUrl,
    setDisplayUrl,
  };
}

export function useTestGenerator() {
  const [generatedTest, setGeneratedTest] = useState(null);
  const [isGenerating, setIsGenerating] = useState(false);
  const [error, setError] = useState(null);

  const generateTest = async (url, trackingData) => {
    setIsGenerating(true);
    setError(null);

    try {
      if (!window.PlaywrightGenerator) {
        setError("PlaywrightGenerator not available");
        setIsGenerating(false);
        return null;
      }

      const testCode = window.PlaywrightGenerator.generatePlaywrightTest(
        url,
        trackingData
      );

      setGeneratedTest(testCode);
      setIsGenerating(false);
      return testCode;
    } catch (err) {
      setError(err.message || "Error generating test");
      setIsGenerating(false);
      return null;
    }
  };

  return {
    generatedTest,
    isGenerating,
    error,
    generateTest,
    setGeneratedTest,
  };
}

export function useTestFiles() {
  const [testFiles, setTestFiles] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [testResults, setTestResults] = useState({});
  const [expandedTests, setExpandedTests] = useState({});
  const [loadingTests, setLoadingTests] = useState({});

  const fetchTestFiles = async () => {
    setIsLoading(true);
    try {
      const response = await fetch("/test/list");
      const data = await response.json();
      if (data.success) {
        setTestFiles((data.tests || []).map(normalizeTestFile));
      } else {
        console.error("Unexpected response format:", data);
      }
    } catch (error) {
      console.error("Error fetching test files:", error);
    } finally {
      setIsLoading(false);
    }
  };

  const normalizeTestFile = (file) => {
    if (file.filename) {
      return file;
    }

    if (file.name) {
      return {
        filename: file.name,
        created: file.created || new Date().toISOString(),
        modified: file.modified || new Date().toISOString(),
        size: file.size || 0,
      };
    }

    return {
      filename: file.name || file.filename || "unknown.spec.js",
      created: file.created || new Date().toISOString(),
      modified: file.modified || new Date().toISOString(),
      size: file.size || 0,
    };
  };

  const runTest = async (testName) => {
    setLoadingTests((prev) => ({
      ...prev,
      [testName]: true,
    }));

    try {
      const getResponse = await fetch(
        `/test/get?name=${encodeURIComponent(testName)}`
      );
      const testData = await getResponse.json();

      if (!testData || !testData.success) {
        throw new Error(testData.error || "Failed to retrieve test content");
      }

      const runResponse = await fetch(
        `/test?test=${encodeURIComponent(testName)}`
      );
      const runResult = await runResponse.json();

      const testResult = {
        success: runResult.success,
        output: runResult.output || "",
        errors: runResult.errors || "",
      };

      setTestResults((prevResults) => ({
        ...prevResults,
        [testName]: testResult,
      }));

      setExpandedTests((prev) => ({
        ...prev,
        [testName]: true,
      }));

      return testResult;
    } catch (error) {
      console.error("Error running test:", error);
      return { success: false, error: error.message };
    } finally {
      setLoadingTests((prev) => ({
        ...prev,
        [testName]: false,
      }));
    }
  };

  const deleteTest = async (testName) => {
    if (!confirm(`Are you sure you want to delete ${testName}?`)) return false;

    try {
      const response = await fetch(
        `/test/delete?name=${encodeURIComponent(testName)}`
      );
      const data = await response.json();

      if (data.success) {
        setTestResults((prevResults) => {
          const newResults = { ...prevResults };
          delete newResults[testName];
          return newResults;
        });

        await fetchTestFiles();
        return true;
      }
      return false;
    } catch (error) {
      console.error("Error deleting test:", error);
      return false;
    }
  };

  const saveTest = async (filename, testCode) => {
    if (!testCode) return { success: false, error: "No test code to save" };

    if (!filename.trim()) {
      return { success: false, error: "Filename is required" };
    }

    try {
      let formattedFilename = filename;
      if (!formattedFilename.endsWith(".spec.js")) {
        formattedFilename = formattedFilename.endsWith(".js")
          ? formattedFilename.replace(".js", ".spec.js")
          : `${formattedFilename}.spec.js`;
      }

      const response = await fetch("/test/save", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          name: formattedFilename,
          text: testCode,
        }),
      });

      const result = await response.json();

      if (result.success) {
        await fetchTestFiles();
      }

      return result;
    } catch (error) {
      console.error("Error saving test:", error);
      return { success: false, error: error.message };
    }
  };

  const toggleTestExpansion = (testName) => {
    setExpandedTests((prev) => ({
      ...prev,
      [testName]: !prev[testName],
    }));
  };

  useEffect(() => {
    fetchTestFiles();
  }, []);

  return {
    testFiles,
    isLoading,
    testResults,
    expandedTests,
    loadingTests,
    fetchTestFiles,
    runTest,
    deleteTest,
    saveTest,
    toggleTestExpansion,
  };
}

export function usePopup() {
  const showPopup = (message, type = "info") => {
    const existingPopup = document.querySelector(".popup");
    if (existingPopup) {
      existingPopup.remove();
    }

    const popup = document.createElement("div");
    popup.className = `popup popup-${type}`;
    popup.textContent = message;

    const popupContainer = document.getElementById("popupContainer");
    if (popupContainer) {
      popupContainer.appendChild(popup);

      setTimeout(() => {
        popup.classList.add("show");
      }, 10);

      setTimeout(() => {
        popup.classList.remove("show");
        setTimeout(() => {
          if (popup.parentNode) {
            popup.parentNode.removeChild(popup);
          }
        }, 300);
      }, 3000);
    }
  };

  return { showPopup };
}

export function useIframeReplay(iframeRef) {
  const { showPopup } = usePopup();
  const [isReplaying, setIsReplaying] = useState(false);
  const [isPaused, setIsPaused] = useState(false);
  const [progress, setProgress] = useState({ current: 0, total: 0 });
  const [replaySpeed, setReplaySpeed] = useState(1);
  const [replayStatus, setReplayStatus] = useState("idle");
  const replayInitializedRef = useRef(false);

  const startReplay = (trackingData) => {
    if (!iframeRef?.current?.contentWindow) {
      showPopup("Iframe not available for replay", "error");
      return false;
    }

    const actions = trackingData;
    if (!actions) {
      showPopup("No actions to replay", "warning");
      return false;
    }

    setIsReplaying(true);
    setIsPaused(false);
    setReplayStatus("playing");
    const lens =
      actions.clicks.clickDetails.length +
      actions.inputChanges.length +
      actions.formElementChanges.length;
    setProgress({ current: 0, total: lens });

    try {
      // Clean and validate actions

      const container = document.querySelector(".iframe-container");
      if (container) {
        container.classList.add("replaying");
      }

      // Send actions to iframe
      iframeRef.current.contentWindow.postMessage(
        {
          type: "staktrak-replay-actions",
          actions,
        },
        "*"
      );

      // Start replay with delay
      setTimeout(() => {
        if (iframeRef.current && iframeRef.current.contentWindow) {
          iframeRef.current.contentWindow.postMessage(
            {
              type: "staktrak-replay-start",
              speed: replaySpeed,
            },
            "*"
          );
          showPopup(`Test replay started with ${lens}} actions`, "info");
        } else {
          showPopup("Iframe not available for replay", "error");
          setIsReplaying(false);

          if (container) {
            container.classList.remove("replaying");
          }
        }
      }, 500);

      return true;
    } catch (error) {
      console.error("Error starting replay:", error);
      showPopup(`Error starting replay: ${error.message}`, "error");
      setIsReplaying(false);

      const container = document.querySelector(".iframe-container");
      if (container) {
        container.classList.remove("replaying");
      }

      return false;
    }
  };

  const pauseReplay = () => {
    if (!isReplaying || !iframeRef?.current?.contentWindow) return;

    try {
      iframeRef.current.contentWindow.postMessage(
        { type: "staktrak-replay-pause" },
        "*"
      );
      setIsPaused(true);
      setReplayStatus("paused");
      showPopup("Test replay paused", "info");
    } catch (error) {
      console.error("Error pausing replay:", error);
      showPopup(`Error pausing replay: ${error.message}`, "error");
    }
  };

  const resumeReplay = () => {
    if (!isReplaying || !isPaused || !iframeRef?.current?.contentWindow) return;

    try {
      iframeRef.current.contentWindow.postMessage(
        { type: "staktrak-replay-resume" },
        "*"
      );
      setIsPaused(false);
      setReplayStatus("playing");
      showPopup("Test replay resumed", "info");
    } catch (error) {
      console.error("Error resuming replay:", error);
      showPopup(`Error resuming replay: ${error.message}`, "error");
    }
  };

  const stopReplay = () => {
    if (!iframeRef?.current?.contentWindow) return;

    try {
      iframeRef.current.contentWindow.postMessage(
        {
          type: "staktrak-replay-stop",
        },
        "*"
      );

      setIsReplaying(false);
      setIsPaused(false);
      setReplayStatus("idle");
      showPopup("Test replay stopped", "warning");
    } catch (error) {
      console.error("Error stopping replay:", error);
      showPopup(`Error stopping replay: ${error.message}`, "error");
      setIsReplaying(false);
      setIsPaused(false);
      setReplayStatus("idle");

      const container = document.querySelector(".iframe-container");
      if (container) {
        container.classList.remove("replaying");
        container.classList.remove("replaying-fadeout");
      }
    }
  };

  const changeReplaySpeed = (speed) => {
    if (!iframeRef?.current?.contentWindow) return;

    const newSpeed = parseFloat(speed);
    if (isNaN(newSpeed) || newSpeed <= 0) return;

    try {
      iframeRef.current.contentWindow.postMessage(
        {
          type: "staktrak-replay-speed",
          speed: newSpeed,
        },
        "*"
      );

      setReplaySpeed(newSpeed);
      showPopup(`Replay speed set to ${newSpeed}x`, "info");
    } catch (error) {
      console.error("Error changing replay speed:", error);
      showPopup(`Error changing speed: ${error.message}`, "error");
    }
  };

  useEffect(() => {
    const checkReplayReady = () => {
      if (!iframeRef?.current?.contentWindow) return;

      try {
        iframeRef.current.contentWindow.postMessage(
          { type: "staktrak-replay-ping" },
          "*"
        );
      } catch (error) {
        console.error("Error checking replay ready state:", error);
      }
    };

    const interval = setInterval(checkReplayReady, 2000);
    checkReplayReady();

    return () => {
      clearInterval(interval);
    };
  }, [iframeRef]);

  useEffect(() => {
    const handleMessage = (event) => {
      if (!event.data?.type) return;

      switch (event.data.type) {
        case "staktrak-replay-ready":
          replayInitializedRef.current = true;
          break;

        case "staktrak-replay-progress":
          setProgress({
            current: event.data.currentAction,
            total: event.data.totalActions,
          });
          break;

        case "staktrak-replay-completed":
          if (event.data.totalActions) {
            setProgress({
              current: event.data.totalActions - 1,
              total: event.data.totalActions,
            });
          }
          setReplayStatus("completed");
          showPopup("Test replay completed", "success");
          break;

        case "staktrak-replay-fadeout":
          const container = document.querySelector(".iframe-container");
          if (container) {
            container.classList.add("replaying-fadeout");
          }

          setTimeout(() => {
            if (container) {
              container.classList.remove("replaying");
              container.classList.remove("replaying-fadeout");
              setIsReplaying(false);
              setIsPaused(false);
            }
          }, 1000);
          break;

        case "staktrak-replay-paused":
          setIsPaused(true);
          setReplayStatus("paused");
          break;

        case "staktrak-replay-resumed":
          setIsPaused(false);
          setReplayStatus("playing");
          break;

        case "staktrak-replay-stopped":
          setIsReplaying(false);
          setIsPaused(false);
          setReplayStatus("idle");

          const containerElement = document.querySelector(".iframe-container");
          if (containerElement) {
            containerElement.classList.remove("replaying");
            containerElement.classList.remove("replaying-fadeout");
          }
          break;

        case "staktrak-replay-error":
          showPopup(`Error during replay: ${event.data.error}`, "error");
          break;
      }
    };

    window.addEventListener("message", handleMessage);
    return () => {
      window.removeEventListener("message", handleMessage);
    };
  }, []);

  return {
    isReplaying,
    isPaused,
    replayStatus,
    progress,
    replaySpeed,
    startReplay,
    pauseReplay,
    resumeReplay,
    stopReplay,
    changeReplaySpeed,
    isReplayReady: replayInitializedRef.current,
  };
}

export function usePlaywrightReplay(iframeRef) {
  const { showPopup } = usePopup();
  const [isPlaywrightReplaying, setIsPlaywrightReplaying] = useState(false);
  const [isPlaywrightPaused, setIsPlaywrightPaused] = useState(false);
  const [playwrightProgress, setPlaywrightProgress] = useState({ current: 0, total: 0 });

  const [playwrightStatus, setPlaywrightStatus] = useState("idle");
  const [currentAction, setCurrentAction] = useState(null);
  const [replayErrors, setReplayErrors] = useState([]);

  const startPlaywrightReplay = (testCode) => {
    if (!iframeRef?.current?.contentWindow) {
      showPopup("Iframe not available for Playwright replay", "error");
      return false;
    }

    if (!testCode || typeof testCode !== 'string') {
      showPopup("No test code provided for replay", "warning");
      return false;
    }

    if (!testCode.includes('page.') || !testCode.includes('test(')) {
      showPopup("Invalid Playwright test format", "error");
      return false;
    }

    setIsPlaywrightReplaying(true);
    setIsPlaywrightPaused(false);
    setPlaywrightStatus("playing");
    setReplayErrors([]);
    setCurrentAction(null);

    try {
      const container = document.querySelector(".iframe-container");
      if (container) {
        container.classList.add("playwright-replaying");
      }

      iframeRef.current.contentWindow.postMessage(
        {
          type: "staktrak-playwright-replay-start",
          testCode,
        },
        "*"
      );

      showPopup("Playwright replay started", "info");
      return true;
    } catch (error) {
      console.error("Error starting Playwright replay:", error);
      showPopup(`Error starting Playwright replay: ${error.message}`, "error");
      setIsPlaywrightReplaying(false);

      const container = document.querySelector(".iframe-container");
      if (container) {
        container.classList.remove("playwright-replaying");
      }

      return false;
    }
  };

  const pausePlaywrightReplay = () => {
    if (!isPlaywrightReplaying || !iframeRef?.current?.contentWindow) return;

    try {
      iframeRef.current.contentWindow.postMessage(
        { type: "staktrak-playwright-replay-pause" },
        "*"
      );
      setIsPlaywrightPaused(true);
      setPlaywrightStatus("paused");
      showPopup("Playwright replay paused", "info");
    } catch (error) {
      console.error("Error pausing Playwright replay:", error);
      showPopup(`Error pausing Playwright replay: ${error.message}`, "error");
    }
  };

  const resumePlaywrightReplay = () => {
    if (!isPlaywrightReplaying || !isPlaywrightPaused || !iframeRef?.current?.contentWindow) return;

    try {
      iframeRef.current.contentWindow.postMessage(
        { type: "staktrak-playwright-replay-resume" },
        "*"
      );
      setIsPlaywrightPaused(false);
      setPlaywrightStatus("playing");
      showPopup("Playwright replay resumed", "info");
    } catch (error) {
      console.error("Error resuming Playwright replay:", error);
      showPopup(`Error resuming Playwright replay: ${error.message}`, "error");
    }
  };

  const stopPlaywrightReplay = () => {
    if (!isPlaywrightReplaying || !iframeRef?.current?.contentWindow) return;

    try {
      iframeRef.current.contentWindow.postMessage(
        { type: "staktrak-playwright-replay-stop" },
        "*"
      );
      setIsPlaywrightReplaying(false);
      setIsPlaywrightPaused(false);
      setPlaywrightStatus("idle");
      setCurrentAction(null);
      setPlaywrightProgress({ current: 0, total: 0 });

      const container = document.querySelector(".iframe-container");
      if (container) {
        container.classList.remove("playwright-replaying");
      }

      showPopup("Playwright replay stopped", "warning");
    } catch (error) {
      console.error("Error stopping Playwright replay:", error);
      showPopup(`Error stopping Playwright replay: ${error.message}`, "error");
    }
  };


  useEffect(() => {
    const handleMessage = (event) => {
      const { data } = event;
      if (!data || !data.type) return;

      switch (data.type) {
        case "staktrak-playwright-replay-started":
          setPlaywrightProgress({ current: 0, total: data.totalActions || 0 });
          break;

        case "staktrak-playwright-replay-progress":
          setPlaywrightProgress({ current: data.current, total: data.total });
          setCurrentAction(data.action);
          break;

        case "staktrak-playwright-replay-completed":
          setIsPlaywrightReplaying(false);
          setIsPlaywrightPaused(false);
          setPlaywrightStatus("completed");
          setCurrentAction(null);
          
          const container = document.querySelector(".iframe-container");
          if (container) {
            container.classList.remove("playwright-replaying");
          }
          
          showPopup("Playwright replay completed successfully", "success");
          break;

        case "staktrak-playwright-replay-error":
          const errorMsg = data.error || "Unknown error";
          setReplayErrors(prev => [...prev, {
            message: errorMsg,
            actionIndex: data.actionIndex,
            action: data.action,
            timestamp: new Date().toISOString()
          }]);
          
          // Don't stop replay on error, just log it
          console.warn("Playwright replay error:", errorMsg);
          if (data.actionIndex !== undefined) {
            showPopup(`Error at action ${data.actionIndex + 1}: ${errorMsg}`, "warning");
          }
          break;

        case "staktrak-playwright-replay-paused":
          setIsPlaywrightPaused(true);
          setPlaywrightStatus("paused");
          break;

        case "staktrak-playwright-replay-resumed":
          setIsPlaywrightPaused(false);
          setPlaywrightStatus("playing");
          break;

        case "staktrak-playwright-replay-stopped":
          setIsPlaywrightReplaying(false);
          setIsPlaywrightPaused(false);
          setPlaywrightStatus("idle");
          setCurrentAction(null);
          setPlaywrightProgress({ current: 0, total: 0 });
          
          const stopContainer = document.querySelector(".iframe-container");
          if (stopContainer) {
            stopContainer.classList.remove("playwright-replaying");
          }
          break;

        default:
          break;
      }
    };

    window.addEventListener("message", handleMessage);
    return () => window.removeEventListener("message", handleMessage);
  }, [showPopup]);

  return {
    isPlaywrightReplaying,
    isPlaywrightPaused,
    playwrightStatus,
    playwrightProgress,
    currentAction,
    replayErrors,
    startPlaywrightReplay,
    pausePlaywrightReplay,
    resumePlaywrightReplay,
    stopPlaywrightReplay,
  };
}