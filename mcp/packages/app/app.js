import { h, render } from "https://esm.sh/preact";
import { useState, useEffect } from "https://esm.sh/preact/hooks";
import { html } from "./utils.js";
import { Prompt } from "./prompt.js";
import { Messages } from "./messages.js";
import { BotSelector } from "./bots.js"; // Import the new component

let vscode;
if (window.acquireVsCodeApi) {
  vscode = window.acquireVsCodeApi();
  console.log("VSCode API is available");
} else {
  console.log("VSCode API is not available");
}

function postMessage(message) {
  if (vscode) {
    vscode.postMessage(message);
  } else {
    window.parent.postMessage(message, "*");
  }
}

const App = () => {
  const [darkMode, setDarkMode] = useState(
    window.matchMedia &&
      window.matchMedia("(prefers-color-scheme: dark)").matches
  );

  const [messages, setMessages] = useState([
    {
      role: "assistant",
      content: "Hello! How can I help you today?",
    },
  ]);
  const [baseUrl, setBaseUrl] = useState("");
  const [selectedModel, setSelectedModel] = useState(null);
  const [apiKeys, setApiKeys] = useState({});
  const [apiToken, setApiToken] = useState(null);
  const handleModelSelect = (model) => {
    setSelectedModel(model);
    console.log(`Selected model: ${model}`);
    postMessage({
      type: "set-model",
      model: model,
    });
  };

  // Request API key
  const requestApiKey = (model) => {
    console.log("=>> requestApiKey", model);
    postMessage({
      type: "request-api-key",
      model: model,
    });
  };

  useEffect(() => {
    postMessage({
      type: "request-base-url",
    });
    postMessage({
      type: "init-api-keys",
    });
    window.addEventListener("message", (event) => {
      if (event.data.type === "chat-res") {
        const message = event.data.message;
        setMessages((prevMessages) => {
          // Check if the last message is a loading message
          const lastMessage = prevMessages[prevMessages.length - 1];
          if (lastMessage && lastMessage.loading === true) {
            // Insert the new message before the loading message
            return [
              ...prevMessages.slice(0, prevMessages.length - 1),
              message,
              lastMessage,
            ];
          } else {
            // Otherwise just append the new message
            return [...prevMessages, message];
          }
        });
      } else if (event.data.type === "set-base-url") {
        console.log("=>> setBaseUrl", event.data.url);
        setBaseUrl(event.data.url);
      } else if (event.data.type === "set-api-token") {
        console.log("=>> setApiToken", event.data.token);
        setApiToken(event.data.token);
      } else if (event.data.type === "done") {
        // Remove all messages with loading: true
        setMessages((prevMessages) =>
          prevMessages.filter((message) => !message.loading)
        );
      } else if (event.data.type === "set-api-key") {
        console.log("=>> setApiKey", event.data.model);
        setApiKeys((prevApiKeys) => ({
          ...prevApiKeys,
          [event.data.model]: true,
        }));
      } else if (event.data.type === "init-model-choice") {
        console.log("=>> initModelChoice", event.data.model);
        setSelectedModel(event.data.model);
      }
    });
  }, []);

  useEffect(() => {
    const darkModeMediaQuery = window.matchMedia(
      "(prefers-color-scheme: dark)"
    );
    const handleChange = (e) => setDarkMode(e.matches);

    if (darkMode) {
      document.body.classList.add("dark-mode");
    } else {
      document.body.classList.remove("dark-mode");
    }

    darkModeMediaQuery.addEventListener("change", handleChange);
    return () => darkModeMediaQuery.removeEventListener("change", handleChange);
  }, [darkMode]);

  const handleSend = (message) => {
    console.log("Message sent:", message);
    const userMessage = {
      role: "user",
      content: message.content,
      taggedWords: message.taggedWords,
    };
    const loadingMessage = {
      role: "assistant",
      content: "Loading...",
      loading: true,
    };
    console.log("=>> loadingMessage", loadingMessage);
    setMessages((prevMessages) => [
      ...prevMessages,
      userMessage,
      loadingMessage,
    ]);

    postMessage({
      type: "chat-msg",
      message: userMessage,
    });
  };

  const handleNewChat = () => {
    setMessages([
      {
        role: "assistant",
        content: "Hello! How can I help you today?",
      },
    ]);
    postMessage({ type: "new-chat" });
  };

  const IconButton = ({ onClick, title, children }) => {
    return html`
      <button class="icon-button" onClick=${onClick} title=${title}>
        ${children}
      </button>
    `;
  };

  // Define a PlusIcon component for the new chat button
  const PlusIcon = () => {
    return html`
      <svg
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
        <line x1="12" y1="5" x2="12" y2="19"></line>
        <line x1="5" y1="12" x2="19" y2="12"></line>
      </svg>
    `;
  };

  return html`
    <div class="app-container">
      <div class="app-header">
        <div class="header-buttons">
          <${IconButton} onClick=${handleNewChat} title="New Chat">
            <${PlusIcon} />
          </${IconButton}>
          <${BotSelector}
            onModelSelect=${handleModelSelect}
            requestApiKey=${requestApiKey}
            selectedModel=${selectedModel}
            apiKeys=${apiKeys}
          />
        </div>
      </div>
      <${Messages} messages=${messages} />
      <${Prompt} onSend=${handleSend} baseUrl=${baseUrl} apiToken=${apiToken} />
    </div>
  `;
};

// Initialize the app when DOM is loaded
document.addEventListener("DOMContentLoaded", () => {
  render(h(App, null), document.getElementById("app"));
});
