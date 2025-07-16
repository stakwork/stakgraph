// frame.js
import htm from "https://esm.sh/htm";
import { h, render } from "https://esm.sh/preact";
import { useState, useEffect } from "https://esm.sh/preact/hooks";
import { useMessageListener } from "./hooks.js";

export const html = htm.bind(h);

const Staktrak = () => {
  const [isRecording, setIsRecording] = useState(false);
  const [isAssertionMode, setIsAssertionMode] = useState(false);
  const [canGenerate, setCanGenerate] = useState(false);
  const [url, setUrl] = useState("http://localhost:3001/preact/frame.html");

  const { showControls } = useMessageListener();

  const handleRecord = () => {
    setIsRecording(!isRecording);
  };

  const handleMode = () => {
    setIsAssertionMode(!isAssertionMode);
  };

  return html`<div>
    <div id="popupContainer"></div>
    <h3>staktrak</h3>
    <iframe src=${url} id="trackingFrame"></iframe>

    <div
      class="controls"
      id="trackingControls"
      style=${showControls ? "display: block;" : "display: none;"}
    >
      <button id="recordBtn" class="record" onClick=${handleRecord}>
        ${isRecording ? "Stop Recording" : "Start Recording"}
      </button>
      <button id="modeBtn" class="mode" onClick=${handleMode}>
        ${isAssertionMode ? "Assertion Mode" : "Interaction Mode"}
      </button>
      <button id="generateBtn" class="generate" disabled=${!canGenerate}>
        Generate Playwright Test
      </button>
    </div>

    <div id="playwrightTest"></div>
  </div>`;
};

// Initialize the app when DOM is loaded
document.addEventListener("DOMContentLoaded", () => {
  render(h(Staktrak, null), document.getElementById("app"));
});
