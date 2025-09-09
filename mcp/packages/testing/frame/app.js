// frame.js
import htm from "https://esm.sh/htm";
import { h, render } from "https://esm.sh/preact";
import { useState, useEffect } from "https://esm.sh/preact/hooks";

export const html = htm.bind(h);

const Frame = () => {
  const [popups, setPopups] = useState([]);
  const [inputValue, setInputValue] = useState("");
  const [showInput, setShowInput] = useState(false);
  const [checkboxValue, setCheckboxValue] = useState(false);
  const [radioValue, setRadioValue] = useState("option1");
  const [selectValue, setSelectValue] = useState("apple");
  const [showFormElements, setShowFormElements] = useState(false);

  // Function to create and show popup
  const showPopup = (message, popupClass) => {
    const newPopup = {
      id: Date.now() + Math.random(),
      message,
      popupClass,
      show: false,
    };

    setPopups((prev) => [...prev, newPopup]);

    // Trigger show animation
    setTimeout(() => {
      setPopups((prev) =>
        prev.map((popup) =>
          popup.id === newPopup.id ? { ...popup, show: true } : popup
        )
      );
    }, 10);

    // Remove popup after 3 seconds
    setTimeout(() => {
      setPopups((prev) =>
        prev.map((popup) =>
          popup.id === newPopup.id ? { ...popup, show: false } : popup
        )
      );

      // Actually remove from array after animation
      setTimeout(() => {
        setPopups((prev) => prev.filter((popup) => popup.id !== newPopup.id));
      }, 300);
    }, 3000);
  };

  // Button click handlers
  const handleTestIdClick = () => {
    showPopup("Button 1 clicked", "popup-testid");
  };

  const handleClassClick = () => {
    showPopup("Button 2 clicked", "popup-class");
  };

  const handleIdClick = () => {
    showPopup("Button 3 clicked", "popup-id");
    const url = new URL(location);
    url.searchParams.set("foo", "bar");
    history.pushState({}, "", url);
  };

  // Input change handler
  const handleInputChange = (event) => {
    const value = event.target.value;
    setInputValue(value);
    showPopup(`Input: ${value}`, "popup-input");
  };

  // Toggle input visibility
  const toggleInput = () => {
    setShowInput(!showInput);
  };

  const toggleFormElements = () => {
    setShowFormElements(!showFormElements);
  };

  const handleCheckboxChange = (event) => {
    const checked = event.target.checked;
    setCheckboxValue(checked);
    showPopup(`Checkbox: ${checked ? "Checked" : "Unchecked"}`, "popup-form");
  };

  const handleRadioChange = (event) => {
    const value = event.target.value;
    setRadioValue(value);
    showPopup(`Radio: Selected ${value}`, "popup-form");
  };

  const handleSelectChange = (event) => {
    const value = event.target.value;
    setSelectValue(value);
    showPopup(`Select: Chose ${value}`, "popup-form");
  };

  useEffect(() => {
    const messageHandler = (event) => {
      if (event.data && event.data.type === "staktrak-show-popup") {
        showPopup(`Selected: "${event.data.text}"`, "popup-selection");
      }
    };

    window.addEventListener("message", messageHandler);

    return () => {
      window.removeEventListener("message", messageHandler);
    };
  }, []);

  return html`
    <div>
      <h2>Preact Iframe Content</h2>
      <p>This is running inside the iframe.</p>

      <button data-testid="staktrak-div" onClick=${handleTestIdClick}>
        data-testid
      </button>

      <button class="staktrak-div" onClick=${handleClassClick}>class</button>

      <button id="staktrak-div" onClick=${handleIdClick}>id</button>

      <div>
        <button onClick=${toggleInput} data-testid="staktrak-show-input">
          ${showInput ? "Hide Input" : "Show Input"}
        </button>
        <button
          onClick=${toggleFormElements}
          data-testid="staktrak-form-elements"
        >
          ${showFormElements ? "Hide Form Elements" : "Show Form Elements"}
        </button>
      </div>

      ${showInput &&
      html`
        <div>
          <label for="test-input">Test Input:</label>
          <input
            type="text"
            id="test-input"
            data-testid="staktrak-input"
            placeholder="Type here to record keystrokes"
            value=${inputValue}
            onInput=${handleInputChange}
          />
        </div>
      `}
      ${showFormElements &&
      html`
        <div class="form-elements">
          <div class="form-group">
            <label>
              <input
                type="checkbox"
                data-testid="staktrak-checkbox"
                checked=${checkboxValue}
                onChange=${handleCheckboxChange}
              />
              Test Checkbox
            </label>
          </div>

          <div class="form-group">
            <fieldset>
              <legend>Test Radio Buttons</legend>
              <label>
                <input
                  type="radio"
                  name="test-radio"
                  value="option1"
                  data-testid="staktrak-radio-1"
                  checked=${radioValue === "option1"}
                  onChange=${handleRadioChange}
                />
                Option 1
              </label>
              <label>
                <input
                  type="radio"
                  name="test-radio"
                  value="option2"
                  data-testid="staktrak-radio-2"
                  checked=${radioValue === "option2"}
                  onChange=${handleRadioChange}
                />
                Option 2
              </label>
              <label>
                <input
                  type="radio"
                  name="test-radio"
                  value="option3"
                  data-testid="staktrak-radio-3"
                  checked=${radioValue === "option3"}
                  onChange=${handleRadioChange}
                />
                Option 3
              </label>
            </fieldset>
          </div>

          <div class="form-group select-group">
            <label for="test-select">Test Select:</label>
            <select
              id="test-select"
              data-testid="staktrak-select"
              value=${selectValue}
              onChange=${handleSelectChange}
            >
              <option value="apple">Apple</option>
              <option value="banana">Banana</option>
              <option value="cherry">Cherry</option>
              <option value="durian">Durian</option>
            </select>
          </div>
        </div>
      `}
      ${popups.map(
        (popup) => html`
          <div
            key=${popup.id}
            class=${`popup ${popup.popupClass} ${popup.show ? "show" : ""}`}
          >
            ${popup.message}
          </div>
        `
      )}
    </div>
  `;
};

// Initialize the app when DOM is loaded
document.addEventListener("DOMContentLoaded", () => {
  render(h(Frame, null), document.getElementById("app"));
});
