// app.js
import htm from "https://esm.sh/htm";
import { h, render } from "https://esm.sh/preact";
import { useState, useRef } from "https://esm.sh/preact/hooks";
import {
  useIframeMessaging,
  useTestGenerator,
  useTestFiles,
  usePopup,
  useIframeReplay,
  usePlaywrightReplay,
} from "./hooks.js";

export const html = htm.bind(h);

const Staktrak = () => {
  const { showPopup } = usePopup();
  const initUrl = window.location.href + "/frame/frame.html";
  const iframeRef = useRef(null);
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
    url,
    handleUrlChange,
    navigateToUrl,
    displayUrl,
  } = useIframeMessaging(iframeRef, initUrl);

  console.log("Staktrak URL:", url, displayUrl);

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

  const {
    isReplaying,
    isPaused,
    replayStatus,
    progress,
    startReplay,
    pauseReplay,
    resumeReplay,
    stopReplay,
  } = useIframeReplay(iframeRef);

  const {
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
  } = usePlaywrightReplay(iframeRef);

  const [filenameInput, setFilenameInput] = useState("");
  const [testCodeInput, setTestCodeInput] = useState("");
  const [showPlaywrightReplay, setShowPlaywrightReplay] = useState(false);

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

  const handleReplay = () => {
    if (isReplaying) {
      if (isPaused) {
        resumeReplay();
      } else {
        pauseReplay();
      }
    } else {
      if (trackingData) {
        const success = startReplay(trackingData);

        if (!success) {
          const simplifiedTrackingData = {
            clicks: { clickDetails: trackingData.clicks.clickDetails },
          };
          startReplay(simplifiedTrackingData);
        }
      } else {
        showPopup("No tracking data available for replay", "error");
      }
    }
  };

  const handleStopReplay = () => {
    stopReplay();
  };

  const handlePlaywrightReplay = () => {
    if (isPlaywrightReplaying) {
      if (isPlaywrightPaused) {
        resumePlaywrightReplay();
      } else {
        pausePlaywrightReplay();
      }
    } else {
      if (testCodeInput.trim()) {
        const currentTestCode = testCodeInput;
        
        if (iframeRef.current) {
          iframeRef.current.src = iframeRef.current.src;
        }
        
        setTimeout(() => {
          startPlaywrightReplay(currentTestCode);
        }, 100);
      } else {
        showPopup("Please enter Playwright test code", "error");
      }
    }
  };

  const handleStopPlaywrightReplay = () => {
    stopPlaywrightReplay();
  };

  const loadTestForReplay = async (testName) => {
    try {
      const response = await fetch(`/test/get?name=${encodeURIComponent(testName)}`);
      const result = await response.json();
      
      if (result.success) {
        setTestCodeInput(result.content);
        setShowPlaywrightReplay(true);
        showPopup(`Test "${testName}" loaded for replay`, "success");
      } else {
        showPopup(`Failed to load test: ${result.error}`, "error");
      }
    } catch (error) {
      showPopup(`Error loading test: ${error.message}`, "error");
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
            value=${displayUrl}
            onChange=${handleUrlChange}
            placeholder="Enter URL to test..."
          />
          <button onClick=${navigateToUrl}>Go</button>
        </div>
        <div class="controls">
          <button
            class=${isRecording ? "stop" : "record"}
            onClick=${handleRecord}
            disabled=${isReplaying || isPlaywrightReplaying}
          >
            ${isRecording
              ? html`<span class="btn-icon">⏹</span> Stop Recording`
              : html`<span class="btn-icon">⏺</span> Start Recording`}
          </button>
          <button
            class=${isAssertionMode ? "interact" : "assert"}
            onClick=${handleMode}
            disabled=${!isRecording || isReplaying || isPlaywrightReplaying}
          >
            ${isAssertionMode
              ? html`<span class="btn-icon">🖱️</span> Interaction Mode`
              : html`<span class="btn-icon">✓</span> Assertion Mode`}
          </button>
          <button
            class="generate"
            onClick=${handleGenerate}
            disabled=${!canGenerate || isReplaying || isPlaywrightReplaying}
          >
            <span class="btn-icon">⚙️</span> Generate Playwright Test
          </button>
          <div class="replay-controls">
            <button
              class=${isReplaying ? (isPaused ? "resume" : "pause") : "replay"}
              onClick=${handleReplay}
              disabled=${!canGenerate || isRecording || isPlaywrightReplaying}
            >
              ${isReplaying
                ? isPaused
                  ? html`<span class="btn-icon">▶️</span> Resume Replay`
                  : html`<span class="btn-icon">⏸️</span> Pause Replay`
                : html`<span class="btn-icon">🔄</span> Replay in Iframe`}
            </button>
            ${isReplaying &&
            html`
              <button class="stop-replay" onClick=${handleStopReplay}>
                <span class="btn-icon">⏹</span> Stop Replay
              </button>
            `}
            <button
              class="playwright-toggle"
              onClick=${() => setShowPlaywrightReplay(!showPlaywrightReplay)}
            >
              <span class="btn-icon">🎬</span> ${showPlaywrightReplay ? "Hide" : "Show"} Code Replay
            </button>
          </div>
        </div>
      </header>

      ${isReplaying &&
      html`
        <div class="replay-progress-bar">
          <div
            class="replay-progress-inner"
            style="width: ${(progress.current /
              Math.max(1, progress.total - 1)) *
            100}%"
          ></div>
          <div class="replay-progress-text">
            Step ${progress.current + 1} of ${progress.total} (${replayStatus})
          </div>
        </div>
      `}

      ${isPlaywrightReplaying &&
      html`
        <div class="playwright-replay-progress-bar">
          <div
            class="playwright-replay-progress-inner"
            style="width: ${(playwrightProgress.current /
              Math.max(1, playwrightProgress.total)) *
            100}%"
          ></div>
          <div class="playwright-replay-progress-text">
            Playwright: Step ${playwrightProgress.current} of ${playwrightProgress.total} (${playwrightStatus})
          </div>
        </div>
      `}

      ${showPlaywrightReplay &&
      html`
        <div class="playwright-replay-section">
          <h3>🎬 Playwright Code Replay</h3>
          <div class="playwright-controls">
            <textarea
              class="test-code-input"
              placeholder="Paste your Playwright test code here..."
              value=${testCodeInput}
              onInput=${(e) => setTestCodeInput(e.target.value)}
              rows="10"
            ></textarea>
            <div class="playwright-buttons">
              <button
                class=${`playwright-replay-btn ${isPlaywrightReplaying ? "active" : ""}`}
                onClick=${handlePlaywrightReplay}
                disabled=${!testCodeInput.trim() && !isPlaywrightReplaying}
              >
                ${isPlaywrightReplaying
                  ? isPlaywrightPaused
                    ? "▶️ Resume Playwright"
                    : "⏸️ Pause Playwright"
                  : "🔄 Start Playwright Replay"}
              </button>
              ${isPlaywrightReplaying
                ? html`<button class="stop-btn" onClick=${handleStopPlaywrightReplay}>
                    ⏹️ Stop Playwright
                  </button>`
                : null}
              
            </div>
            ${replayErrors.length > 0
              ? html`<div class="replay-errors">
                  <h4>⚠️ Replay Errors (${replayErrors.length}):</h4>
                  ${replayErrors.slice(-3).map(error => html`
                    <div class="error-item">
                      Action ${error.actionIndex + 1}: ${error.message}
                    </div>
                  `)}
                </div>`
              : null}
          </div>
        </div>
      `}

      <div class="main-content">
        ${selectedText &&
        html`<div class="selected-text" id="app-selection-display">
          Selected: "${selectedText}"
        </div>`}
        <div class="iframe-container ${isReplaying ? "replaying" : ""} ${isPlaywrightReplaying ? "playwright-replaying" : ""}">
          <iframe ref=${iframeRef} src=${url} id="trackingFrame"></iframe>
        </div>

        ${generatedTest !== null
          ? html`
              <div id="playwrightTest">
                <h3>
                  <span class="section-icon">📝</span> Generated Playwright Test
                </h3>
                <div class="save-controls">
                  <button
                    class="copy-btn"
                    onClick=${copyTestToClipboard}
                    disabled=${isReplaying}
                  >
                    <span class="btn-icon">📋</span> Copy to Clipboard
                  </button>
                  <input
                    type="text"
                    id="filenameInput"
                    placeholder="test-filename.spec.js"
                    value=${filenameInput}
                    onChange=${(e) => setFilenameInput(e.target.value)}
                    disabled=${isReplaying}
                  />
                  <button
                    class="save"
                    onClick=${handleSaveTest}
                    disabled=${isReplaying}
                  >
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
                        disabled=${loadingTests[file.filename] || isReplaying}
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
                        disabled=${isReplaying}
                      >
                        <span class="btn-icon">🗑️</span> Delete
                      </button>
                      <button
                        class="replay-test-btn"
                        onClick=${() => loadTestForReplay(file.filename)}
                        title="Load for Playwright replay"
                        disabled=${isReplaying || isPlaywrightReplaying}
                      >
                        <span class="btn-icon">🎬</span> Replay
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
