import { h, render } from "https://esm.sh/preact";
import { useEffect, useState } from "https://esm.sh/preact/hooks";
import * as utils from "./utils.js";

const html = utils.html;
const LoadingSvg = utils.LoadingSvg;

const App = () => {
  const [repoUrl, setRepoUrl] = useState("");
  const [username, setUsername] = useState("");
  const [pat, setPat] = useState("");
  const [repoUrlError, setRepoUrlError] = useState("");
  const [currentRepoName, setCurrentRepoName] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [repoExists, setRepoExists] = useState(false);
  const [status, setStatus] = useState(null);
  const [progress, setProgress] = useState(0);
  const [token, setToken] = useState("");

  useEffect(() => {
    const fetchToken = async () => {
      const data = await utils.GET("/token");
      console.log("fetchToken", data);
      if (data.token) {
        setToken(data.token);
        utils.setApiToken(data.token);
      }
    };
    fetchToken();
  }, []);

  async function checkRepoExists() {
    if (!currentRepoName) {
      return;
    }
    try {
      const data = await utils.POST("/fetch-repo", {
        repo_name: currentRepoName,
      });
      if (data.status === "success") {
        setRepoExists(true);
      }
    } catch (e) {
      setRepoExists(false);
    }
  }

  useEffect(() => {
    checkRepoExists();
  }, [currentRepoName]);

  utils.useSSE("/events", {
    onMessage: (data, event) => {
      console.log("=>", data);
      if (data && data.message !== "") {
        setStatus(data);
      }
      if (data && data.progress !== undefined) {
        setProgress(data.progress);
      }
    },
  });

  const handleRepoUrlChange = (event) => {
    const value = event.target.value;
    console.log("handleRepoUrlChange", value);
    const repoName = utils.getRepoNameFromUrl(value);
    if (repoName) {
      console.log("repoName", repoName);
      setCurrentRepoName(repoName);
    }
    if (value && !isValidGithubUrl(value)) {
      setRepoUrlError("Please enter a valid GitHub repository URL.");
    } else {
      setRepoUrlError("");
    }
  };
  const handleUsernameChange = (event) => {
    setUsername(event.target.value);
  };
  const handlePatChange = (event) => {
    setPat(event.target.value);
  };
  function isValidGithubUrl(url) {
    // Accepts https://github.com/user/repo or git@github.com:user/repo.git
    return (
      /^https:\/\/github\.com\/[^\/]+\/[^\/]+(\.git)?\/?$/.test(url.trim()) ||
      /^git@github\.com:[^\/]+\/[^\/]+(\.git)?$/.test(url.trim())
    );
  }

  const btnDisabled = !repoUrl || isLoading || !isValidGithubUrl(repoUrl);

  const handleSubmit = async () => {
    console.log("handleSubmit", repoUrl);
    if (btnDisabled) {
      return;
    }
    setIsLoading(true);
    setStatus({ message: "Cloning repo to /tmp/..." });
    await utils.POST("/ingest", { repo_url: repoUrl, username, pat });
    setIsLoading(false);
  };

  const handleSync = async () => {
    if (btnDisabled) {
      return;
    }
    setIsLoading(true);
    setStatus(null);
    await utils.POST("/process", { repo_url: repoUrl, username, pat });
    setIsLoading(false);
  };

  const renderProgressBar = () => {
    if (!status) return null;
    console.log("status", status);
    const showText = status.total_steps ? true : false;
    const stepPercentage = showText
      ? Math.min(100, Math.round((status.step / status.total_steps) * 100))
      : 0;
    const stepWidth = `${stepPercentage}%`;
    const progressWidth = `${progress}%`;

    return html`
      <div class="progress-container">
        <div class="progress-info">
          ${showText &&
          html`<div>Step ${status.step}/${status.total_steps}:</div>`}
          <div class="progress-message">${status.message}</div>
        </div>
        <div class="progress-bar steps-bar">
          <div class="steps-fill" style="width: ${stepWidth}"></div>
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

  const buttonText = repoExists ? "Sync Repo to Latest" : "Ingest Repo";
  const handleBtnClick = repoExists ? handleSync : handleSubmit;
  return html`
    <div class="app-container">
      <div class="app-header">
        <h2>${currentRepoName ? currentRepoName : "Stakgraph"}</h2>
      </div>
      <div class="app-body">
        <input
          type="text"
          placeholder="Repo URL"
          value=${repoUrl}
          onInput=${handleRepoUrlChange}
        />
        ${repoUrlError && html`<div class="input-error">${repoUrlError}</div>`}
        <div class="input-horizontal-container">
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
        </div>
        <button onClick=${handleBtnClick} disabled=${btnDisabled}>
          ${isLoading
            ? html`<div class="loading"><${LoadingSvg} /></div>`
            : buttonText}
        </button>
        ${status && renderProgressBar()}
      </div>
    </div>
  `;
};

// Initialize the app when DOM is loaded
document.addEventListener("DOMContentLoaded", () => {
  render(h(App, null), document.getElementById("app"));
});
