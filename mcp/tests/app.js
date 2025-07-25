// app.js
import htm from "https://esm.sh/htm";
import { h, render } from "https://esm.sh/preact";
import { useState } from "https://esm.sh/preact/hooks";
import {
  useIframeMessaging,
  useTestGenerator,
  useTestFiles,
  usePopup,
  useURL,
} from "./hooks.js";

export const html = htm.bind(h);

const Staktrak = () => {
  const { showPopup } = usePopup();
  const initUrl = window.location.href + "/frame/frame.html";
  const { url, handleUrlChange, navigateToUrl, iframeRef } = useURL(initUrl);
  const {
    isRecording,
    isAssertionMode,
    canGenerate,
    trackingData,
    selectedText,
    startRecording,
    stopRecording,
    enableAssertionMode,
    disableAssertionMode,
  } = useIframeMessaging(iframeRef);

  const { generatedTest, generateTest } = useTestGenerator();

  const {
    testFiles,
    testResults,
    expandedTests,
    loadingTests,
    runTest,
    deleteTest,
    saveTest,
    toggleTestExpansion,
  } = useTestFiles();

  const [filenameInput, setFilenameInput] = useState("");

  const handleRecord = () => {
    if (!isRecording) {
      startRecording();
      showPopup("Recording started", "info");
    } else {
      stopRecording();
      showPopup("Recording stopped", "warning");
    }
  };

  const handleMode = () => {
    if (!isAssertionMode) {
      enableAssertionMode();
      showPopup("Select text in the iframe to add assertions", "info");
    } else {
      disableAssertionMode();
      showPopup("Returned to interaction mode", "info");
    }
  };

  const handleGenerate = async () => {
    if (!trackingData) {
      showPopup("No tracking data available", "error");
      return;
    }

    const testCode = await generateTest(url, trackingData);

    if (testCode) {
      showPopup("Playwright test generated successfully", "success");
    } else {
      showPopup("Failed to generate test", "error");
    }
  };

  const copyTestToClipboard = () => {
    if (generatedTest) {
      navigator.clipboard.writeText(generatedTest).then(() => {
        showPopup("Test code copied to clipboard!", "success");
      });
    }
  };

  const handleSaveTest = async () => {
    if (!generatedTest) return;

    if (!filenameInput.trim()) {
      showPopup("Please enter a filename", "error");
      return;
    }

    const result = await saveTest(filenameInput, generatedTest);

    if (result.success) {
      showPopup(`Test saved as ${result.filename}`, "success");
      setFilenameInput("");
    } else {
      showPopup(result.error || "Failed to save test", "error");
    }
  };

  const handleDeleteTest = async (testName) => {
    const success = await deleteTest(testName);
    if (success) {
      showPopup(`Test ${testName} deleted successfully`, "success");
    } else {
      showPopup(`Failed to delete test: ${testName}`, "error");
    }
  };

  const handleRunTest = async (testName) => {
    const result = await runTest(testName);
    showPopup(
      `Test ${testName} ${result.success ? "completed" : "failed"}`,
      result.success ? "success" : "error"
    );
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
        ${selectedText &&
        html`<div class="selected-text" id="app-selection-display">
          Selected: "${selectedText}"
        </div>`}
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
                  <button class="save" onClick=${handleSaveTest}>
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
                        onClick=${() => handleRunTest(file.filename)}
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
                        onClick=${() => handleDeleteTest(file.filename)}
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
