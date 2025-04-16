import { h, render } from "https://esm.sh/preact";
import { useState, useEffect } from "https://esm.sh/preact/hooks";
import { html } from "./utils.js";
import { Prompt } from "./prompt.js";

const App = () => {
  const [darkMode, setDarkMode] = useState(
    window.matchMedia &&
      window.matchMedia("(prefers-color-scheme: dark)").matches
  );

  // Check for dark mode
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

    // Log the message text and tagged words
    console.log("Message text:", message.text);
    console.log("Tagged words:", message.taggedWords);
  };

  return html`
    <div>
      <${Prompt} onSend=${handleSend} />
    </div>
  `;
};

// Initialize the app when DOM is loaded
document.addEventListener("DOMContentLoaded", () => {
  render(h(App, null), document.getElementById("app"));
});
