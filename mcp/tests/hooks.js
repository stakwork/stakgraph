import { useState, useEffect, useRef } from "https://esm.sh/preact/hooks";

export function useIframeMessaging(iframeRef) {
  const popupHook = usePopup();
  const [isRecording, setIsRecording] = useState(false);
  const [isAssertionMode, setIsAssertionMode] = useState(false);
  const [canGenerate, setCanGenerate] = useState(false);
  const [trackingData, setTrackingData] = useState(null);
  const [selectedText, setSelectedText] = useState(null);

  const { showPopup } = popupHook;
  const selectedDisplayTimeout = useRef(null);

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

export function useURL(initialURL) {
  const [url, setUrl] = useState(initialURL);
  const iframeRef = useRef(null);

  const handleUrlChange = (e) => {
    setUrl(e.target.value);
  };

  const navigateToUrl = () => {
    if (iframeRef.current) {
      iframeRef.current.src = url;
    }
  };

  return { url, setUrl, handleUrlChange, navigateToUrl, iframeRef };
}
