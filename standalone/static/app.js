import { h, render } from "https://esm.sh/preact";
import { useEffect, useState } from "https://esm.sh/preact/hooks";
import * as utils from "./utils.js";

const html = utils.html;
const LoadingSvg = utils.LoadingSvg;

const App = () => {
  const [repoUrl, setRepoUrl] = useState("");
  const [username, setUsername] = useState("");
  const [pat, setPat] = useState("");
  const [commit, setCommit] = useState("");
  const [useLsp, setUseLsp] = useState(true);

  const [currentRepoName, setCurrentRepoName] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [repoExists, setRepoExists] = useState(false);
  const [status, setStatus] = useState(null);
  const [progress, setProgress] = useState(0);
  const [token, setToken] = useState("");
  const [stats, setStats] = useState({});

  useEffect(() => {
    const fetchToken = async () => {
      const data = await utils.GET("/token");
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
        if (data.progress === 100 || data.status === "Complete") {
          setIsLoading(false);
        }
      }
      if (data && data.stats) {
        setStats((prevStats) => ({ ...prevStats, ...data.stats }));
      }
    },
  });

  const handleRepoUrlChange = (event) => {
    console.log("handleRepoUrlChange", event.target.value);
    setRepoUrl(event.target.value);
    const repoName = utils.getRepoNameFromUrl(event.target.value);
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

  const handleCommitChange = (event) => {
    setCommit(event.target.value);
  };

  const btnDisabled = !repoUrl || isLoading;

  const handleSubmit = async () => {
    console.log("handleSubmit", repoUrl);
    if (btnDisabled) {
      return;
    }
    setIsLoading(true);
    setStatus({ message: "Cloning repo to /tmp/..." });
    setStats({});
    try {
      await utils.POST("/ingest", {
        repo_url: repoUrl,
        username,
        pat,
        commit: commit || undefined,
        use_lsp: useLsp,
      });
    } catch (e) {
      let message = e.message
        .split("\n")
        .map((line) => `<p>${line}</p>`)
        .join("");
      setStatus({ status: "Error", message });
      setIsLoading(false);
      return;
    }
    setIsLoading(false);
  };

  const handleSync = async () => {
    if (btnDisabled) {
      return;
    }
    setIsLoading(true);
    setStatus(null);
    setStats({});
    try {
      await utils.POST("/process", { repo_url: repoUrl, username, pat });
    } catch (e) {
      setStatus({ status: "Error", message: e.message });
      setIsLoading(false);
      return;
    }
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

    const isCloning =
      status.message && status.message.toLowerCase().includes("cloning");

    return html`
      <div class="progress-container">
        <div class="progress-info">
          ${isCloning
            ? html`<div class="clone-label">Clone Repository</div>
                <div
                  class="progress-message"
                  dangerouslySetInnerHTML=${{ __html: status.message }}
                ></div>`
            : html`
                ${showText &&
                html`<div>Step ${status.step}/${status.total_steps}</div>`}
                <div class="progress-message">${status.message}</div>
                <div class="progress-percentage">${stepPercentage}%</div>
              `}
        </div>
        <div
          class="progress-bar steps-bar"
          style="${isCloning ? "display: none;" : ""}"
        >
          <div class="steps-fill" style="width: ${stepWidth}"></div>
        </div>

        <div
          class="progress-bar progress-bar-small"
          style="${isCloning ? "display: none;" : ""}"
        >
          <div
            class="progress-fill-small"
            style="width: ${progressWidth}"
          ></div>
        </div>

        ${Object.keys(stats).length > 0 &&
        !isCloning &&
        html`
          <div class="sub-progress">Current Step Progress: ${progress}%</div>
          <div class="stats-container">
            ${Object.entries(stats).map(
              ([key, value]) => html`
                <div class="stat-item">
                  <div class="stat-label">${formatStatLabel(key)}</div>
                  <div class="stat-value">${formatStatValue(value)}</div>
                </div>
              `
            )}
          </div>
        `}
      </div>
    `;
  };

  const renderActionButtons = () => {
    if (
      !isLoading &&
      status &&
      (status.status === "Complete" || progress === 100)
    ) {
      return html`
        <div
          style="margin-top: 24px; display: flex; gap: 12px; justify-content: space-around;"
        >
          <button onClick=${handleSync}>Sync Repo to Latest</button>
          <button onClick=${handleReset}>Ingest Another Repo</button>
        </div>
      `;
    }
    return null;
  };

  const handleReset = () => {
    setRepoUrl("");
    setUsername("");
    setPat("");
    setCommit("");
    setUseLsp(true);
    setCurrentRepoName("");
    setStatus(null);
    setProgress(0);
    setStats({});
    setRepoExists(false);
  };

  const formatStatLabel = (key) => {
    return key
      .replace(/_/g, " ")
      .split(" ")
      .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
      .join(" ");
  };

  const formatStatValue = (value) => {
    if (typeof value === "number" && value > 999) {
      return value.toLocaleString();
    }
    return value;
  };

  const buttonText = repoExists ? "Sync Repo to Latest" : "Ingest Repo";
  const handleBtnClick = repoExists ? handleSync : handleSubmit;
  return html`
    <div class="app-container">
      <h1 class="app-header-title">StakGraph</h1>
      <div class="app-header">
        <h2>${currentRepoName ? currentRepoName : " "}</h2>
      </div>
      <div class="app-body">
        <input
          type="text"
          placeholder="Repo URL"
          value=${repoUrl}
          onInput=${handleRepoUrlChange}
        />
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
        <input
          type="text"
          placeholder="Commit hash (optional)"
          value=${commit}
          onInput=${handleCommitChange}
        />
        <div class="checkbox-wrapper-14">
          <label for="lsp-switch" class="lsp-label"> LSP </label>
          <input
            id="lsp-switch"
            type="checkbox"
            class="switch"
            checked=${useLsp}
            onChange=${(e) => setUseLsp(e.target.checked)}
          />
        </div>
        <button onClick=${handleBtnClick} disabled=${btnDisabled}>
          ${isLoading
            ? html`<div class="loading"><${LoadingSvg} /></div>`
            : buttonText}
        </button>
        ${status && renderProgressBar()} ${renderActionButtons()}
      </div>
    </div>
  `;
};

// Initialize the app when DOM is loaded
document.addEventListener("DOMContentLoaded", () => {
  render(h(App, null), document.getElementById("app"));
});
