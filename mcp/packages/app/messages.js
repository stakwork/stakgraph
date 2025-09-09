import { useEffect, useRef } from "https://esm.sh/preact/hooks";
import { html } from "./utils.js";

const LoadingSvg = () => html`
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

export const Messages = ({ messages }) => {
  const messagesEndRef = useRef(null);

  // Scroll to bottom whenever messages change
  useEffect(() => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [messages]);

  return html`
    <div class="messages-container">
      ${messages.map(
        (message, index) => html`
          <div key=${index} class="message-wrapper ${message.role}">
            <div class="message-bubble ${message.role}">
              ${message.loading
                ? html`<div class="loading-indicator"><${LoadingSvg} /></div>`
                : html`<pre class="message-content">${message.content}</pre>`}
            </div>
          </div>
        `
      )}
      <div ref=${messagesEndRef}></div>
    </div>
  `;
};
