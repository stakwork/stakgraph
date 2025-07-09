import htm from "https://esm.sh/htm";
import { h } from "https://esm.sh/preact";
import { useEffect, useRef } from "https://esm.sh/preact/hooks";

export const html = htm.bind(h);

export function getRepoNameFromUrl(url) {
  // Extract organization/repo format: stakwork/sphinx-tribes-frontend
  const matches = url.match(/github\.com\/([^\/]+\/[^\/\.]+)(\.git)?$/);
  if (matches && matches[1]) {
    return matches[1];
  }

  // As a fallback, extract just the repo name
  const repoName = url.split("/").pop()?.replace(".git", "") || "";
  return repoName;
}

export function isValidGithubUrl(url) {
  const trimmed = url.trim();

  const httpsPattern = /^https:\/\/github\.com\/[^\/\s]+\/[^\/\s]+/;
  const sshPattern = /^git@github\.com:[^\/\s]+\/[^\/\s]+/;
  return httpsPattern.test(trimmed) || sshPattern.test(trimmed);
}

export const LoadingSvg = () => html`
  <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 300 150">
    <path
      fill="none"
      stroke="#878787"
      stroke-width="15"
      stroke-linecap="round"
      stroke-dasharray="300 385"
      stroke-dashoffset="0"
      d="M275 75c0 31-27 50-50 50-58 0-92-100-150-100-28 0-50 22-50 50s23 50 50 50c58 0 92-100 150-100 24 0 50 19 50 50Z"
    >
      <animate
        attributeName="stroke-dashoffset"
        calcMode="spline"
        dur="2"
        values="685;-685"
        keySplines="0 0 1 1"
        repeatCount="indefinite"
      ></animate>
    </path>
  </svg>
`;

let apiToken = null;

// Function to set the token (called from your main app)
export const setApiToken = (token) => {
  apiToken = token;
};

// Function to get current headers with token if available
const getFetchHeaders = () => {
  const headers = {
    "Content-Type": "application/json",
  };
  if (apiToken) {
    headers["Authorization"] = `Bearer ${apiToken}`;
  }
  return headers;
};

export const GET = async (url) => {
  const r = await fetch(url, {
    method: "GET",
    headers: getFetchHeaders(),
  });
  if (!r.ok) {
    throw new Error(`GET ${url} failed: ${r.statusText}`);
  }
  const data = await r.json();
  return data;
};

export const POST = async (url, body) => {
  const r = await fetch(url, {
    method: "POST",
    body: JSON.stringify(body),
    headers: getFetchHeaders(),
  });
  if (!r.ok) {
    throw new Error(`POST ${url} failed: ${r.statusText}`);
  }
  const data = await r.json();
  return data;
};

export const useSSE = (url, options = {}) => {
  const eventSourceRef = useRef(null);
  const reconnectTimeoutRef = useRef(null);

  const {
    onMessage,
    onError,
    onOpen,
    onClose,
    autoReconnect = true,
    customEvents = [],
  } = options;

  useEffect(() => {
    const connect = () => {
      // Clear any existing connection
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
      }

      console.log("Connecting to SSE:", url);
      eventSourceRef.current = new EventSource(url);
      const eventSource = eventSourceRef.current;

      eventSource.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          onMessage?.(data, event);
        } catch (error) {
          console.log("SSE data (not JSON):", event.data);
          onMessage?.(event.data, event);
        }
      };

      eventSource.onopen = (event) => {
        console.log("SSE Connection opened", event);
        onOpen?.(event);
      };

      eventSource.onerror = (error) => {
        if (eventSource.readyState === EventSource.CLOSED) {
          console.log("SSE connection closed");
          if (autoReconnect) {
            console.log("Attempting to reconnect in 3 seconds...");
            reconnectTimeoutRef.current = setTimeout(connect, 3000);
          }
        }
        onError?.(error);
      };

      // Handle custom event types
      customEvents.forEach((eventType) => {
        eventSource.addEventListener(eventType, (event) => {
          try {
            const data = JSON.parse(event.data);
            console.log(`SSE Custom Event (${eventType}):`, data);
          } catch (error) {
            console.log(`SSE Custom Event (${eventType}):`, event.data);
          }
        });
      });
    };

    connect();

    return () => {
      console.log("Cleaning up SSE connection");
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
      }
      onClose?.();
    };
  }, [url]);

  const closeConnection = () => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
    }
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
    }
  };

  return { closeConnection };
};
