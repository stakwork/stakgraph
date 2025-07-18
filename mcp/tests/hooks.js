import { useState, useEffect } from "https://esm.sh/preact/hooks";

export function useMessageListener() {
  const [showControls, setShowControls] = useState(false);
  const [trackingData, setTrackingData] = useState(null);
  const [selectedElement, setSelectedElement] = useState({
    text: "",
    selector: "",
  });

  useEffect(() => {
    const handleMessage = (event) => {
      if (event.data && event.data.type) {
        switch (event.data.type) {
          case "staktrak-setup":
            setShowControls(true);
            break;
          case "staktrak-results":
            setTrackingData(event.data.data);
            break;
          case "staktrak-selection":
            setSelectedElement({
              text: event.data.text || "",
              selector: event.data.selector || "",
            });
            break;
          default:
            break;
        }
      }
    };

    window.addEventListener("message", handleMessage);

    // Cleanup function to remove event listener
    return () => {
      window.removeEventListener("message", handleMessage);
    };
  }, []);

  return { showControls, trackingData, selectedElement };
}

export function useTestFiles() {
  const [testFiles, setTestFiles] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  const fetchTestFiles = async () => {
    setIsLoading(true);
    setError(null);
    try {
      const response = await fetch("/test/list");
      const data = await response.json();
      if (data.success) {
        setTestFiles(data.tests || []);
      } else {
        setError(data.error || "Failed to load test files");
      }
    } catch (error) {
      console.error("Error fetching test files:", error);
      setError("Failed to load test files");
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchTestFiles();
  }, []);

  return { testFiles, isLoading, error, refreshTestFiles: fetchTestFiles };
}

export function useTestExecution() {
  const [testResults, setTestResults] = useState({});
  const [expandedTests, setExpandedTests] = useState({});
  const [isRunning, setIsRunning] = useState(false);

  const runTest = async (testName) => {
    setIsRunning(true);
    try {
      const response = await fetch(
        `/test?test=${encodeURIComponent(testName)}`
      );
      const data = await response.json();

      setTestResults((prevResults) => ({
        ...prevResults,
        [testName]: data,
      }));

      setExpandedTests((prev) => ({
        ...prev,
        [testName]: true,
      }));

      return data;
    } catch (error) {
      console.error("Error running test:", error);
      return { success: false, error: error.message };
    } finally {
      setIsRunning(false);
    }
  };

  const deleteTest = async (testName) => {
    try {
      const response = await fetch(
        `/test/delete?name=${encodeURIComponent(testName)}`
      );
      const data = await response.json();
      return data;
    } catch (error) {
      console.error("Error deleting test:", error);
      return { success: false, error: error.message };
    }
  };

  const toggleExpansion = (testName) => {
    setExpandedTests((prev) => ({
      ...prev,
      [testName]: !prev[testName],
    }));
  };

  return {
    testResults,
    expandedTests,
    isRunning,
    runTest,
    deleteTest,
    toggleExpansion,
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
