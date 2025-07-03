import { h, render } from "https://esm.sh/preact";
import { useEffect, useState } from "https://esm.sh/preact/hooks";
import { html, getRepoNameFromUrl, LoadingSvg, POST, useSSE } from "./utils.js";

const App = () => {
  const [repoUrl, setRepoUrl] = useState("");
  const [username, setUsername] = useState("");
  const [pat, setPat] = useState("");

  const [currentRepoName, setCurrentRepoName] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [repoExists, setRepoExists] = useState(false);
  const [progressInfo, setProgressInfo] = useState(null);

  useEffect(() => {
    const fetchRepo = async () => {
      try {
        const response = await POST("/fetch-repo", {
          repo_name: currentRepoName,
        });
        if (response.status === "success") {
          setRepoExists(true);
        }
      } catch (e) {}
    };
    if (currentRepoName) {
      fetchRepo();
    }
  }, [currentRepoName]);

  const { closeConnection } = useSSE("/events", {
    onMessage: (data, event) => {
      console.log("=>", data);
      if (data && data.step !== undefined) {
        setProgressInfo(data);
      }
    },
  });

  const handleRepoUrlChange = (event) => {
    console.log("handleRepoUrlChange", event.target.value);
    setRepoUrl(event.target.value);
    const repoName = getRepoNameFromUrl(event.target.value);
    if (repoName) {
      console.log("repoName", repoName);
      setCurrentRepoName(repoName);
    }
  };
  const handleUsernameChange = (event) => {
    setUsername(event.target.value);
  };
  const handlePatChange = (event) => {
    setPat(event.target.value);
  };

  const handleSubmit = async () => {
    setIsLoading(true);
    setProgressInfo(null);
    await POST("/ingest", { repo_url: repoUrl, username, pat });
    setIsLoading(false);
  };

  const handleSync = async () => {
    setIsLoading(true);
    setProgressInfo(null);
    await POST("/process", { repo_url: repoUrl, username, pat });
    setIsLoading(false);
  };

  const renderProgressBar = () => {
    if (!progressInfo) return null;

    const stepPercentage = Math.min(
      100,
      Math.round((progressInfo.step / progressInfo.total_steps) * 100)
    );
    const stepWidth = `${stepPercentage}%`;

    const progressPercentage = Math.min(
      100,
      Math.round(progressInfo.progress * 100)
    );
    const progressWidth = `${progressPercentage}%`;

    return html`
      <div class="progress-container">
        <div class="progress-info">
          <div>
            Step${progressInfo.step}/${progressInfo.total_steps}:${progressInfo.message}
          </div>
          <div>${stepPercentage}%</div>
        </div>
        <div class="progress-bar steps-bar">
          <div class="steps-fill" style="width: ${stepWidth}"></div>
        </div>

        <div class="progress-info sub-progress">
          <div>Progress within step:</div>
          <div>${progressPercentage}%</div>
        </div>
        <div class="progress-bar progress-bar-small">
          <div
            class="progress-fill-small"
            style="width: ${progressWidth}"
          ></div>
        </div>
      </div>
    `;
  };

  return html`
    <div class="app-container">
      <div class="app-header">
        <h1>${currentRepoName ? currentRepoName : "Stakgraph"}</h1>
      </div>
      <div class="app-body">
        <input
          type="text"
          placeholder="Repo URL"
          value=${repoUrl}
          onInput=${handleRepoUrlChange}
        />
        <input
          type="text"
          placeholder="Username (optional)"
          value=${username}
          onInput=${handleUsernameChange}
        />
        <input
          type="text"
          placeholder="PAT (optional)"
          value=${pat}
          onInput=${handlePatChange}
        />
        <button onClick=${handleSubmit} disabled=${!repoUrl || isLoading}>
          ${isLoading
            ? html`<div class="loading"><${LoadingSvg} /></div>`
            : "Ingest Repo"}
        </button>
        ${repoExists &&
        html`<button onClick=${handleSync} disabled=${isLoading}>
          ${isLoading
            ? html`<div class="loading"><${LoadingSvg} /></div>`
            : "Sync Repo to Latest"}
        </button>`}
        ${progressInfo && renderProgressBar()}
      </div>
    </div>
  `;
};

// Initialize the app when DOM is loaded
document.addEventListener("DOMContentLoaded", () => {
  render(h(App, null), document.getElementById("app"));
});
