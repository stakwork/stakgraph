// app.js
import htm from "https://esm.sh/htm";
import { h, render } from "https://esm.sh/preact";
import { useState, useEffect, useRef } from "https://esm.sh/preact/hooks";
import { useMessageListener } from "./hooks.js";

export const html = htm.bind(h);

let playGenLoaded = false;

function waitForPlaywrightGenerator() {
  return new Promise((resolve) => {
    if (window.PlaywrightGenerator) {
      playGenLoaded = true;
      resolve(window.PlaywrightGenerator);
      return;
    }

    const maxAttempts = 300;
    let attempts = 0;

    const interval = setInterval(() => {
      attempts++;
      if (window.PlaywrightGenerator) {
        clearInterval(interval);
        playGenLoaded = true;
        console.log("PlaywrightGenerator found after waiting");
        resolve(window.PlaywrightGenerator);
      } else if (attempts >= maxAttempts) {
        clearInterval(interval);
        console.error("PlaywrightGenerator not found after waiting");
        resolve(null);
      }
    }, 100);
  });
}

const Staktrak = () => {
  const [isRecording, setIsRecording] = useState(false);
  const [isAssertionMode, setIsAssertionMode] = useState(false);
  const [canGenerate, setCanGenerate] = useState(false);
  const [url, setUrl] = useState("http://localhost:3001/preact/frame.html");
  const [trackingData, setTrackingData] = useState(null);
  const [generatedTest, setGeneratedTest] = useState(null);
  const [testFiles, setTestFiles] = useState([]);
  const [testResults, setTestResults] = useState({});
  const [expandedTests, setExpandedTests] = useState({});
  const [filenameInput, setFilenameInput] = useState("");
  const [loadingTests, setLoadingTests] = useState({});

  const iframeRef = useRef(null);
  const assertions = useRef([]);
  const selectedTextRef = useRef("");
  const selectedSelectorRef = useRef("");
  const isWaitingForSelectionRef = useRef(false);
  const assertionCountRef = useRef(0);

  useEffect(() => {
    waitForPlaywrightGenerator().then((generator) => {
      if (generator) {
        console.log("PlaywrightGenerator is ready to use");
      }
    });

    const handleMessage = (event) => {
      if (event.data && event.data.type) {
        switch (event.data.type) {
          case "staktrak-setup":
            console.log("Staktrak setup message received");
            break;
          case "staktrak-results":
            console.log("Staktrak results received:", event.data.data);
            setTrackingData(event.data.data);
            setCanGenerate(true);
            break;
          case "staktrak-selection":
            if (isWaitingForSelectionRef.current) {
              selectedTextRef.current = event.data.text || "";
              selectedSelectorRef.current = event.data.selector || "";

              const isCheckbox =
                selectedSelectorRef.current.includes(
                  'input[type="checkbox"]'
                ) || selectedSelectorRef.current.includes("checkbox");
              const isRadio =
                selectedSelectorRef.current.includes('input[type="radio"]') ||
                selectedSelectorRef.current.includes("radio");

              if (selectedTextRef.current && selectedSelectorRef.current) {
                if (isCheckbox || isRadio) {
                  const assertionType = confirm(
                    "Is this a checked state assertion? Click OK for 'isChecked', Cancel for 'isNotChecked'"
                  )
                    ? "isChecked"
                    : "isNotChecked";

                  addAssertion(assertionType);
                } else {
                  addAssertion();
                }
                assertionCountRef.current++;
              }
            }
            break;
        }
      }
    };

    window.addEventListener("message", handleMessage);

    fetchTestFiles();

    return () => {
      window.removeEventListener("message", handleMessage);
    };
  }, []);

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

  const fetchTestFiles = async () => {
    try {
      const response = await fetch("/test/list");
      const data = await response.json();
      if (data.success) {
        setTestFiles((data.tests || []).map(normalizeTestFile));
      } else {
        console.error("Unexpected response format:", data);
        showPopup(
          "Failed to load test files: Unexpected response format",
          "error"
        );
      }
    } catch (error) {
      console.error("Error fetching test files:", error);
      showPopup("Failed to load test files", "error");
    }
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

      showPopup(
        `Test ${testName} ${testResult.success ? "completed" : "failed"}`,
        testResult.success ? "success" : "error"
      );
    } catch (error) {
      console.error("Error running test:", error);
      showPopup("Failed to run test: " + error.message, "error");
    } finally {
      setLoadingTests((prev) => ({
        ...prev,
        [testName]: false,
      }));
    }
  };

  const deleteTest = async (testName) => {
    if (!confirm(`Are you sure you want to delete ${testName}?`)) return;

    try {
      const response = await fetch(
        `/test/delete?name=${encodeURIComponent(testName)}`
      );
      const data = await response.json();

      if (data.success) {
        showPopup(`Test ${testName} deleted successfully`, "success");
        fetchTestFiles();

        setTestResults((prevResults) => {
          const newResults = { ...prevResults };
          delete newResults[testName];
          return newResults;
        });
      } else {
        showPopup(`Failed to delete test: ${data.error}`, "error");
      }
    } catch (error) {
      console.error("Error deleting test:", error);
      showPopup("Failed to delete test: " + error.message, "error");
    }
  };

  const toggleTestExpansion = (testName) => {
    setExpandedTests((prev) => ({
      ...prev,
      [testName]: !prev[testName],
    }));
  };

  const handleUrlChange = (e) => {
    setUrl(e.target.value);
  };

  const navigateToUrl = () => {
    if (iframeRef.current) {
      iframeRef.current.src = url;
    }
  };

  const handleRecord = () => {
    if (!isRecording) {
      if (iframeRef.current && iframeRef.current.contentWindow) {
        iframeRef.current.contentWindow.postMessage(
          { type: "staktrak-start" },
          "*"
        );
        setIsRecording(true);
        setIsAssertionMode(false);
        setCanGenerate(false);
        assertions.current = [];
        assertionCountRef.current = 0;
        setGeneratedTest(null);
        showPopup("Recording started", "info");
      }
    } else {
      if (iframeRef.current && iframeRef.current.contentWindow) {
        iframeRef.current.contentWindow.postMessage(
          { type: "staktrak-stop" },
          "*"
        );
        setIsRecording(false);

        if (isWaitingForSelectionRef.current) {
          isWaitingForSelectionRef.current = false;
          iframeRef.current.contentWindow.postMessage(
            { type: "staktrak-disable-selection" },
            "*"
          );
        }
        setIsAssertionMode(false);
        showPopup("Recording stopped", "warning");
      }
    }
  };

  const handleMode = () => {
    if (!isAssertionMode) {
      isWaitingForSelectionRef.current = true;
      setIsAssertionMode(true);
      showPopup("Select text in the iframe to add assertions", "info");

      if (iframeRef.current && iframeRef.current.contentWindow) {
        iframeRef.current.contentWindow.postMessage(
          { type: "staktrak-enable-selection" },
          "*"
        );
      }
    } else {
      isWaitingForSelectionRef.current = false;
      setIsAssertionMode(false);
      showPopup("Returned to interaction mode", "info");

      if (iframeRef.current && iframeRef.current.contentWindow) {
        iframeRef.current.contentWindow.postMessage(
          { type: "staktrak-disable-selection" },
          "*"
        );
      }
    }
  };

  const addAssertion = (assertionType = null) => {
    if (!selectedSelectorRef.current) {
      showPopup("No element selected", "error");
      return;
    }

    let type = "isVisible";
    let value = "";

    if (assertionType) {
      type = assertionType;
      value = "";
    } else if (
      selectedTextRef.current &&
      selectedTextRef.current.trim() !== ""
    ) {
      type = "hasText";
      value = selectedTextRef.current;
    }

    assertions.current.push({
      type,
      selector: selectedSelectorRef.current,
      value,
      timestamp: Date.now(),
    });

    showPopup(`Assertion added: ${type} "${value || ""}"`, "success");

    selectedTextRef.current = "";
    selectedSelectorRef.current = "";
  };

  const handleGenerate = async () => {
    console.log("Generate button clicked");
    console.log("trackingData:", trackingData);

    if (!window.PlaywrightGenerator) {
      console.error("PlaywrightGenerator not available");
      showPopup(
        "Error: PlaywrightGenerator not available. Try refreshing the page.",
        "error"
      );

      try {
        const script = document.createElement("script");
        script.type = "module";
        script.src = "/tests/playwright-generator.js";
        document.head.appendChild(script);

        await new Promise((resolve) => {
          script.onload = () => {
            console.log("PlaywrightGenerator script loaded dynamically");
            resolve();
          };
          script.onerror = (err) => {
            console.error("Error loading PlaywrightGenerator script:", err);
            resolve();
          };
        });
      } catch (err) {
        console.error("Failed to load PlaywrightGenerator:", err);
      }
    }

    if (trackingData) {
      if (isWaitingForSelectionRef.current) {
        isWaitingForSelectionRef.current = false;
        if (iframeRef.current && iframeRef.current.contentWindow) {
          iframeRef.current.contentWindow.postMessage(
            { type: "staktrak-disable-selection" },
            "*"
          );
        }
      }

      if (trackingData.clicks && trackingData.clicks.clickDetails) {
        let filteredClicks = trackingData.clicks.clickDetails.filter(
          (clickDetail) => {
            const clickSelector = clickDetail[2];
            const clickTime = clickDetail[3];

            return !assertions.current.some((assertion) => {
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
          }
        );

        const MAX_MULTICLICK_INTERVAL = 300;

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
              click.timestamp - lastClick.timestamp > MAX_MULTICLICK_INTERVAL
            ) {
              resultClicks.push(click);
            }
            lastClick = click;
          });

          resultClicks.forEach((click) =>
            finalFilteredClicks.push(click.detail)
          );
        });

        finalFilteredClicks.sort((a, b) => a[3] - b[3]);

        trackingData.clicks.clickDetails = finalFilteredClicks;
      }

      const modifiedTrackingData = {
        ...trackingData,
        assertions: assertions.current,
      };

      console.log("Modified tracking data:", modifiedTrackingData);
      console.log(
        "window.PlaywrightGenerator available:",
        !!window.PlaywrightGenerator
      );

      try {
        let testCode;

        if (window.PlaywrightGenerator) {
          console.log("Using window.PlaywrightGenerator");
          testCode = window.PlaywrightGenerator.generatePlaywrightTest(
            url,
            modifiedTrackingData
          );
        } else {
          console.log("Using fallback generator");
          testCode = generateFallbackTest(url, modifiedTrackingData);
        }

        console.log(
          "Generated test code:",
          testCode ? testCode.substring(0, 100) + "..." : "undefined"
        );

        if (testCode) {
          setGeneratedTest(testCode);
          showPopup("Playwright test generated successfully", "success");
        } else {
          throw new Error("Failed to generate test code");
        }
      } catch (error) {
        console.error("Error generating test:", error);
        showPopup("Error generating test: " + error.message, "error");
      }
    } else {
      console.log("No tracking data available");
      showPopup("No tracking data available", "error");
    }
  };

  const generateFallbackTest = (url, trackingData) => {
    const { userInfo = { windowSize: [1280, 720] } } = trackingData;

    let interactions = "";

    if (trackingData.clicks && trackingData.clicks.clickDetails) {
      trackingData.clicks.clickDetails.forEach((click) => {
        interactions += `\n  // Click on ${click[2]}\n  await page.click('${click[2]}');\n`;
      });
    }

    if (trackingData.inputChanges) {
      trackingData.inputChanges
        .filter((change) => change.action === "complete" || !change.action)
        .forEach((change) => {
          interactions += `\n  // Fill input\n  await page.fill('${change.elementSelector}', '${change.value}');\n`;
        });
    }

    if (trackingData.assertions) {
      trackingData.assertions.forEach((assertion) => {
        if (assertion.type === "hasText") {
          interactions += `\n  // Assert text\n  await expect(page.locator('${assertion.selector}')).toHaveText('${assertion.value}');\n`;
        } else if (assertion.type === "isVisible") {
          interactions += `\n  // Assert visibility\n  await expect(page.locator('${assertion.selector}')).toBeVisible();\n`;
        } else if (assertion.type === "isChecked") {
          interactions += `\n  // Assert checked state\n  await expect(page.locator('${assertion.selector}')).toBeChecked();\n`;
        } else if (assertion.type === "isNotChecked") {
          interactions += `\n  // Assert not checked state\n  await expect(page.locator('${assertion.selector}')).not.toBeChecked();\n`;
        }
      });
    }

    return `import { test, expect } from '@playwright/test';
  
test('User interaction replay', async ({ page }) => {
  // Navigate to the page
  await page.goto('${url}');
  
  // Wait for page to load
  await page.waitForLoadState('networkidle');
  
  // Set viewport size to match recorded session
  await page.setViewportSize({ 
    width: ${userInfo.windowSize[0]}, 
    height: ${userInfo.windowSize[1]} 
  });
  ${interactions}
  await page.waitForTimeout(2500);
});`;
  };

  const copyTestToClipboard = () => {
    if (generatedTest) {
      navigator.clipboard.writeText(generatedTest).then(() => {
        showPopup("Test code copied to clipboard!", "success");
      });
    }
  };

  const saveTestToDisk = async () => {
    if (!generatedTest) return;

    if (!filenameInput.trim()) {
      showPopup("Please enter a filename", "error");
      return;
    }

    try {
      let filename = filenameInput;
      if (!filename.endsWith(".spec.js")) {
        filename = filename.endsWith(".js")
          ? filename.replace(".js", ".spec.js")
          : `${filename}.spec.js`;
      }

      const response = await fetch("/test/save", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          name: filename,
          text: generatedTest,
        }),
      });

      const result = await response.json();

      if (result.success) {
        showPopup(`Test saved as ${result.filename}`, "success");
        setFilenameInput("");
        fetchTestFiles();
      } else {
        showPopup(result.error || "Failed to save test", "error");
      }
    } catch (error) {
      console.error("Error saving test:", error);
      showPopup("Error saving test: " + error.message, "error");
    }
  };

  const showPopup = (message, type = "info") => {
    const existingPopup = document.querySelector(".popup");
    if (existingPopup) {
      existingPopup.remove();
    }

    const popup = document.createElement("div");
    popup.className = `popup popup-${type}`;
    popup.textContent = message;

    document.getElementById("popupContainer").appendChild(popup);

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
  };

  return html`
    <div>
      <div id="popupContainer"></div>

      <header class="header">
        <div class="logo-container">
          <svg
            class="logo-icon"
            xmlns="http://www.w3.org/2000/svg"
            viewBox="0 0 24 24"
            width="24"
            height="24"
            fill="#38bdf8"
          >
            <path
              d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z"
            />
          </svg>
          <h3>StakTrak</h3>
        </div>
        <div class="url-bar">
          <input
            type="text"
            value=${url}
            onChange=${handleUrlChange}
            placeholder="Enter URL to test..."
          />
          <button onClick=${navigateToUrl}>Go</button>
        </div>
        <div class="controls">
          <button
            class=${isRecording ? "stop" : "record"}
            onClick=${handleRecord}
          >
            ${isRecording
              ? html`<span class="btn-icon">⏹</span> Stop Recording`
              : html`<span class="btn-icon">⏺</span> Start Recording`}
          </button>
          <button
            class=${isAssertionMode ? "interact" : "assert"}
            onClick=${handleMode}
            disabled=${!isRecording}
          >
            ${isAssertionMode
              ? html`<span class="btn-icon">🖱️</span> Interaction Mode`
              : html`<span class="btn-icon">✓</span> Assertion Mode`}
          </button>
          <button
            class="generate"
            onClick=${handleGenerate}
            disabled=${!canGenerate}
          >
            <span class="btn-icon">⚙️</span> Generate Playwright Test
          </button>
        </div>
      </header>

      <div class="main-content">
        <div class="iframe-container">
          <iframe ref=${iframeRef} src=${url} id="trackingFrame"></iframe>
        </div>

        ${generatedTest !== null
          ? html`
              <div id="playwrightTest">
                <h3>
                  <span class="section-icon">📝</span> Generated Playwright Test
                </h3>
                <div class="save-controls">
                  <button class="copy-btn" onClick=${copyTestToClipboard}>
                    <span class="btn-icon">📋</span> Copy to Clipboard
                  </button>
                  <input
                    type="text"
                    id="filenameInput"
                    placeholder="test-filename.spec.js"
                    value=${filenameInput}
                    onChange=${(e) => setFilenameInput(e.target.value)}
                  />
                  <button class="save" onClick=${saveTestToDisk}>
                    <span class="btn-icon">💾</span> Save to Disk
                  </button>
                </div>
                <pre>${generatedTest}</pre>
              </div>
            `
          : html`<div class="no-test-message">
              <div class="empty-state">
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  width="48"
                  height="48"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  stroke-width="1"
                  stroke-linecap="round"
                  stroke-linejoin="round"
                >
                  <path
                    d="M14.5 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7.5L14.5 2z"
                  ></path>
                  <polyline points="14 2 14 8 20 8"></polyline>
                  <path d="M12 18v-6"></path>
                  <path d="M8 15h8"></path>
                </svg>
                <p>
                  Click "Generate Playwright Test" after recording to generate a
                  test
                </p>
              </div>
            </div>`}

        <div class="test-files">
          <h3><span class="section-icon">📁</span> Available Tests</h3>
          ${testFiles.length === 0
            ? html`<div class="empty-state">
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  width="48"
                  height="48"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  stroke-width="1"
                  stroke-linecap="round"
                  stroke-linejoin="round"
                >
                  <path
                    d="M13 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9z"
                  ></path>
                  <polyline points="13 2 13 9 20 9"></polyline>
                </svg>
                <p>No test files available</p>
              </div>`
            : null}
          <ul>
            ${testFiles.map(
              (file) => html`
                <li class="test-file">
                  <div class="test-file-header">
                    <span class="test-name">
                      <svg
                        class="file-icon"
                        xmlns="http://www.w3.org/2000/svg"
                        width="16"
                        height="16"
                        viewBox="0 0 24 24"
                        fill="none"
                        stroke="currentColor"
                        stroke-width="2"
                        stroke-linecap="round"
                        stroke-linejoin="round"
                      >
                        <path
                          d="M14.5 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7.5L14.5 2z"
                        ></path>
                        <polyline points="14 2 14 8 20 8"></polyline>
                      </svg>
                      ${file.filename}
                    </span>
                    <div class="test-actions">
                      <button
                        class="run-test"
                        onClick=${() => runTest(file.filename)}
                        disabled=${loadingTests[file.filename]}
                      >
                        <span class="btn-icon">▶️</span> ${loadingTests[
                          file.filename
                        ]
                          ? "Running..."
                          : "Run"}
                      </button>
                      <button
                        class="delete-test"
                        onClick=${() => deleteTest(file.filename)}
                      >
                        <span class="btn-icon">🗑️</span> Delete
                      </button>
                      <button
                        class="toggle-result"
                        onClick=${() => toggleTestExpansion(file.filename)}
                      >
                        ${expandedTests[file.filename]
                          ? html`<span class="btn-icon">▲</span> Hide`
                          : html`<span class="btn-icon">▼</span> Show`}
                      </button>
                    </div>
                  </div>
                  ${expandedTests[file.filename] && testResults[file.filename]
                    ? html`
                        <div
                          class="test-result ${testResults[file.filename]
                            .success
                            ? "success"
                            : "error"}"
                        >
                          <h4>
                            <span class="status-icon"
                              >${testResults[file.filename].success
                                ? "✅"
                                : "❌"}</span
                            >
                            Test Result:
                            ${testResults[file.filename].success
                              ? "Success"
                              : "Failed"}
                          </h4>
                          <pre>
${testResults[file.filename].output || ""}${testResults[file.filename].errors
                              ? "\n\nErrors:\n" +
                                testResults[file.filename].errors
                              : ""}</pre
                          >
                        </div>
                      `
                    : null}
                </li>
              `
            )}
          </ul>
        </div>
      </div>
    </div>
  `;
};

// Initialize the app when DOM is loaded
document.addEventListener("DOMContentLoaded", () => {
  render(h(Staktrak, null), document.getElementById("app"));
});
