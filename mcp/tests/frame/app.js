// frame/app.js — StakTrak interaction playground
// A rich target app rendered inside the iframe, used to exercise every
// capture / selector / replay path the recorder needs to handle:
//   • buttons keyed by every selector strategy (testid / id / class / role+text / aria)
//   • non-unique repeated items (selector-collision tester)
//   • all input types, textarea, range, date/color, contenteditable
//   • checkbox / radio / single + multi select / form submit
//   • SPA navigation (pushState views + hash + query links) → waitForURL targets
//   • dynamic / delayed content + modal (wait-for-element / disappearing nodes)
//   • a live event log so capture ORDER is visible while recording
import htm from "https://esm.sh/htm";
import { h, render } from "https://esm.sh/preact";
import { useState, useEffect, useRef } from "https://esm.sh/preact/hooks";

export const html = htm.bind(h);

const VIEWS = [
  { id: "buttons", label: "Buttons", icon: "◉" },
  { id: "inputs", label: "Inputs", icon: "✎" },
  { id: "forms", label: "Forms", icon: "☑" },
  { id: "nav", label: "Navigation", icon: "⇄" },
  { id: "dynamic", label: "Dynamic", icon: "◷" },
];

const App = () => {
  // ---- routing ----
  const initialView = new URLSearchParams(location.search).get("view") || "buttons";
  const [view, setView] = useState(VIEWS.some((v) => v.id === initialView) ? initialView : "buttons");

  // ---- shared feedback ----
  const [toasts, setToasts] = useState([]);
  const [log, setLog] = useState([]);
  const seq = useRef(0);

  const pushToast = (message, kind = "info") => {
    const id = Date.now() + Math.random();
    setToasts((p) => [...p, { id, message, kind, show: false }]);
    setTimeout(() => setToasts((p) => p.map((t) => (t.id === id ? { ...t, show: true } : t))), 10);
    setTimeout(() => {
      setToasts((p) => p.map((t) => (t.id === id ? { ...t, show: false } : t)));
      setTimeout(() => setToasts((p) => p.filter((t) => t.id !== id)), 250);
    }, 2600);
  };

  const record = (label, kind = "info") => {
    seq.current += 1;
    setLog((p) => [{ n: seq.current, label, t: new Date().toLocaleTimeString() }, ...p].slice(0, 40));
    pushToast(label, kind);
  };

  const navTo = (id) => {
    const url = new URL(location);
    url.searchParams.set("view", id);
    history.pushState({}, "", url);
    setView(id);
    record(`Navigated → ${id}`, "nav");
  };

  useEffect(() => {
    const onPop = () => {
      const v = new URLSearchParams(location.search).get("view") || "buttons";
      setView(v);
    };
    const onMsg = (e) => {
      if (e.data && e.data.type === "staktrak-show-popup") record(`Assertion on: "${e.data.text}"`, "assert");
    };
    window.addEventListener("popstate", onPop);
    window.addEventListener("message", onMsg);
    return () => {
      window.removeEventListener("popstate", onPop);
      window.removeEventListener("message", onMsg);
    };
  }, []);

  return html`
    <div class="stk-shell">
      <header class="stk-header">
        <div class="stk-brand"><span class="stk-logo">▶</span> StakTrak <em>Playground</em></div>
        <div class="stk-hint">Record actions here — every control is a capture target.</div>
      </header>

      <div class="stk-body">
        <nav class="stk-nav">
          ${VIEWS.map(
            (v) => html`
              <button
                key=${v.id}
                class=${`stk-nav-item ${view === v.id ? "active" : ""}`}
                data-testid=${`nav-${v.id}`}
                onClick=${() => navTo(v.id)}
              >
                <span class="stk-nav-icon">${v.icon}</span> ${v.label}
              </button>
            `
          )}
        </nav>

        <main class="stk-main">
          ${view === "buttons" && html`<${ButtonsView} record=${record} />`}
          ${view === "inputs" && html`<${InputsView} record=${record} />`}
          ${view === "forms" && html`<${FormsView} record=${record} />`}
          ${view === "nav" && html`<${NavView} record=${record} navTo=${navTo} />`}
          ${view === "dynamic" && html`<${DynamicView} record=${record} />`}
        </main>

        <aside class="stk-log">
          <div class="stk-log-head">
            Event Log <span class="stk-badge" data-testid="log-count">${log.length}</span>
          </div>
          <ol class="stk-log-list" data-testid="event-log">
            ${log.length === 0
              ? html`<li class="stk-log-empty">Interact with the app to see events…</li>`
              : log.map((e) => html`<li key=${e.n}><b>#${e.n}</b> ${e.label} <time>${e.t}</time></li>`)}
          </ol>
        </aside>
      </div>

      ${toasts.map(
        (t) => html`<div key=${t.id} class=${`stk-toast ${t.kind} ${t.show ? "show" : ""}`}>${t.message}</div>`
      )}
    </div>
  `;
};

// ---------------------------------------------------------------------------
// Buttons — one target per selector strategy + a non-unique collision tester
// ---------------------------------------------------------------------------
const ButtonsView = ({ record }) => {
  const [count, setCount] = useState(0);
  const [lastItem, setLastItem] = useState("—");

  return html`
    <${Section} title="Buttons by selector strategy"
      desc="Each button is reachable by a different selector kind. Good for testing selector generation + resolution.">
      <div class="stk-grid">
        <button class="stk-btn primary" data-testid="staktrak-div" onClick=${() => record("Clicked: data-testid button")}>
          By data-testid
        </button>
        <button class="stk-btn" id="staktrak-div" onClick=${() => record("Clicked: id button")}>
          By id
        </button>
        <button class="stk-btn staktrak-div" onClick=${() => record("Clicked: class button")}>
          By class
        </button>
        <button class="stk-btn" onClick=${() => record("Clicked: role+text button")}>
          Submit Order
        </button>
        <button class="stk-btn icon" aria-label="Add to favorites" onClick=${() => record("Clicked: aria-label button")}>
          ★
        </button>
        <button class="stk-btn" onClick=${() => record("Clicked: nested-text button")}>
          <span class="stk-btn-sub">Go to</span> <strong>Checkout</strong>
        </button>
      </div>
    <//>

    <${Section} title="Counter" desc="Assertion target — the count text updates on each click.">
      <div class="stk-counter">
        <button class="stk-btn" data-testid="counter-dec" onClick=${() => { setCount((c) => c - 1); record("Counter −1"); }}>−</button>
        <span class="stk-counter-val" data-testid="counter-value">${count}</span>
        <button class="stk-btn primary" data-testid="counter-inc" onClick=${() => { setCount((c) => c + 1); record("Counter +1"); }}>+</button>
      </div>
    <//>

    <${Section} title="Non-unique items (collision tester)"
      desc="Three buttons with IDENTICAL text + class. Only data-testid disambiguates. This is the case that made replay 'click the wrong thing'.">
      <div class="stk-row">
        ${[1, 2, 3].map(
          (i) => html`
            <button
              key=${i}
              class="stk-btn item-action"
              data-testid=${`item-${i}`}
              onClick=${() => { setLastItem(`Item ${i}`); record(`Selected Item ${i}`); }}
            >
              Select
            </button>
          `
        )}
      </div>
      <p class="stk-out">Last selected: <b data-testid="last-item">${lastItem}</b></p>
    <//>
  `;
};

// ---------------------------------------------------------------------------
// Inputs — every text-ish input type + range/date/color + contenteditable
// ---------------------------------------------------------------------------
const InputsView = ({ record }) => {
  const [text, setText] = useState("");
  const [range, setRange] = useState(50);

  return html`
    <${Section} title="Text inputs" desc="Typed values feed page.fill(). The echo is an assertion target.">
      <div class="stk-field">
        <label for="in-text">Text</label>
        <input id="in-text" data-testid="staktrak-input" type="text" placeholder="Type here…"
          value=${text} onInput=${(e) => setText(e.target.value)} onBlur=${() => record(`Typed text: "${text}"`)} />
      </div>
      <p class="stk-out">Echo: <span data-testid="text-echo">${text || "—"}</span></p>

      <div class="stk-two">
        <div class="stk-field"><label for="in-email">Email</label>
          <input id="in-email" data-testid="input-email" type="email" placeholder="you@example.com" onBlur=${(e) => record(`Email: ${e.target.value}`)} /></div>
        <div class="stk-field"><label for="in-pass">Password</label>
          <input id="in-pass" data-testid="input-password" type="password" placeholder="••••••" onBlur=${() => record("Password entered")} /></div>
        <div class="stk-field"><label for="in-num">Number</label>
          <input id="in-num" data-testid="input-number" type="number" min="0" max="100" onBlur=${(e) => record(`Number: ${e.target.value}`)} /></div>
        <div class="stk-field"><label for="in-search">Search</label>
          <input id="in-search" data-testid="input-search" type="search" placeholder="Search…" onBlur=${(e) => record(`Search: ${e.target.value}`)} /></div>
      </div>

      <div class="stk-field"><label for="in-area">Textarea</label>
        <textarea id="in-area" data-testid="input-textarea" rows="3" placeholder="Multi-line…" onBlur=${(e) => record(`Textarea: ${e.target.value.length} chars`)}></textarea></div>
    <//>

    <${Section} title="Widgets" desc="Non-text inputs: range, date, color, and a contenteditable region.">
      <div class="stk-field"><label for="in-range">Range: <b data-testid="range-value">${range}</b></label>
        <input id="in-range" data-testid="input-range" type="range" min="0" max="100" value=${range}
          onInput=${(e) => setRange(+e.target.value)} onChange=${() => record(`Range → ${range}`)} /></div>
      <div class="stk-two">
        <div class="stk-field"><label for="in-date">Date</label>
          <input id="in-date" data-testid="input-date" type="date" onChange=${(e) => record(`Date: ${e.target.value}`)} /></div>
        <div class="stk-field"><label for="in-color">Color</label>
          <input id="in-color" data-testid="input-color" type="color" onChange=${(e) => record(`Color: ${e.target.value}`)} /></div>
      </div>
      <div class="stk-field"><label>Contenteditable</label>
        <div class="stk-editable" data-testid="input-editable" contenteditable="true"
          onBlur=${(e) => record(`Editable: ${e.target.textContent.slice(0, 24)}`)}>Edit me…</div></div>
    <//>
  `;
};

// ---------------------------------------------------------------------------
// Forms — checkbox / radio / single + multi select / submit
// ---------------------------------------------------------------------------
const FormsView = ({ record }) => {
  const [checked, setChecked] = useState(false);
  const [radio, setRadio] = useState("standard");
  const [fruit, setFruit] = useState("apple");
  const [submitted, setSubmitted] = useState(false);

  const submit = (e) => {
    e.preventDefault();
    setSubmitted(true);
    record("Form submitted", "assert");
  };

  return html`
    <${Section} title="Choices" desc="Checkbox, radio group and selects — page.check / page.selectOption targets.">
      <form class="stk-form" data-testid="staktrak-form-elements" onSubmit=${submit}>
        <label class="stk-check">
          <input type="checkbox" data-testid="staktrak-checkbox" checked=${checked}
            onChange=${(e) => { setChecked(e.target.checked); record(`Checkbox ${e.target.checked ? "on" : "off"}`); }} />
          Subscribe to updates
        </label>

        <fieldset class="stk-fieldset">
          <legend>Shipping</legend>
          ${[["standard", "Standard"], ["express", "Express"], ["overnight", "Overnight"]].map(
            ([val, lbl], i) => html`
              <label key=${val} class="stk-check">
                <input type="radio" name="shipping" value=${val} data-testid=${`staktrak-radio-${i + 1}`}
                  checked=${radio === val} onChange=${() => { setRadio(val); record(`Shipping: ${lbl}`); }} />
                ${lbl}
              </label>
            `
          )}
        </fieldset>

        <div class="stk-field"><label for="sel-fruit">Fruit</label>
          <select id="sel-fruit" data-testid="staktrak-select" value=${fruit}
            onChange=${(e) => { setFruit(e.target.value); record(`Fruit: ${e.target.value}`); }}>
            <option value="apple">Apple</option>
            <option value="banana">Banana</option>
            <option value="cherry">Cherry</option>
            <option value="durian">Durian</option>
          </select></div>

        <div class="stk-field"><label for="sel-tags">Tags (multi)</label>
          <select id="sel-tags" data-testid="select-multi" multiple size="4"
            onChange=${(e) => record(`Tags: ${[...e.target.selectedOptions].map((o) => o.value).join(",")}`)}>
            <option value="red">Red</option>
            <option value="green">Green</option>
            <option value="blue">Blue</option>
            <option value="gold">Gold</option>
          </select></div>

        <button class="stk-btn primary" type="submit" data-testid="form-submit">Place Order</button>
        ${submitted && html`<p class="stk-success" data-testid="form-result">✓ Order placed</p>`}
      </form>
    <//>
  `;
};

// ---------------------------------------------------------------------------
// Navigation — pushState views, hash + query links, anchors
// ---------------------------------------------------------------------------
const NavView = ({ record, navTo }) => {
  const setQuery = (k, v) => {
    const url = new URL(location);
    url.searchParams.set(k, v);
    history.pushState({}, "", url);
    record(`URL → ?${k}=${v}`, "nav");
  };

  return html`
    <${Section} title="SPA navigation" desc="pushState route changes — these become waitForURL steps.">
      <div class="stk-row">
        <button class="stk-btn" data-testid="go-inputs" onClick=${() => navTo("inputs")}>Go to Inputs</button>
        <button class="stk-btn" data-testid="go-forms" onClick=${() => navTo("forms")}>Go to Forms</button>
        <button class="stk-btn" data-testid="set-step-2" onClick=${() => setQuery("step", "2")}>Set step=2</button>
        <button class="stk-btn" data-testid="set-page-checkout" onClick=${() => setQuery("page", "checkout")}>Go to checkout</button>
      </div>
    <//>
    <${Section} title="Anchor links" desc="Real <a> elements — hash + in-page navigation.">
      <div class="stk-row">
        <a class="stk-link" href="#section-details" data-testid="link-details" onClick=${() => record("Anchor: #details")}>Jump to details</a>
        <a class="stk-link" href="?view=nav&ref=promo" data-testid="link-promo" onClick=${() => record("Anchor: promo")}>Open promo</a>
      </div>
      <div id="section-details" class="stk-note" data-testid="details-anchor">Details section (anchor target).</div>
    <//>
  `;
};

// ---------------------------------------------------------------------------
// Dynamic — delayed content, appending list, modal (appear/disappear)
// ---------------------------------------------------------------------------
const DynamicView = ({ record }) => {
  const [loading, setLoading] = useState(false);
  const [loaded, setLoaded] = useState(false);
  const [items, setItems] = useState([]);
  const [modal, setModal] = useState(false);

  const loadAsync = () => {
    setLoading(true);
    setLoaded(false);
    record("Async load started");
    setTimeout(() => {
      setLoading(false);
      setLoaded(true);
      record("Async content ready", "assert");
    }, 1500);
  };

  return html`
    <${Section} title="Delayed content" desc="Reveals after 1.5s — tests wait-for-element before clicking.">
      <button class="stk-btn primary" data-testid="load-async" onClick=${loadAsync}>Load async content</button>
      ${loading && html`<div class="stk-spinner" data-testid="spinner">Loading…</div>`}
      ${loaded && html`
        <div class="stk-panel" data-testid="async-panel">
          <p>Loaded! You can act on this now.</p>
          <button class="stk-btn" data-testid="async-action" onClick=${() => record("Clicked async-revealed button", "assert")}>Confirm</button>
        </div>`}
    <//>

    <${Section} title="Growing list" desc="Appends nodes — dynamic repeated content the recorder must keep binding.">
      <button class="stk-btn" data-testid="add-item" onClick=${() => { setItems((p) => [...p, p.length + 1]); record(`Added row ${items.length + 1}`); }}>Add row</button>
      <ul class="stk-list" data-testid="dynamic-list">
        ${items.map((n) => html`<li key=${n} data-testid=${`row-${n}`}>Row ${n} <button class="stk-btn tiny" data-testid=${`row-${n}-del`} onClick=${() => { setItems((p) => p.filter((x) => x !== n)); record(`Removed row ${n}`); }}>✕</button></li>`)}
      </ul>
    <//>

    <${Section} title="Modal" desc="Opens an overlay — tests elements appearing then disappearing.">
      <button class="stk-btn" data-testid="open-modal" onClick=${() => { setModal(true); record("Opened modal"); }}>Open dialog</button>
    <//>

    ${modal && html`
      <div class="stk-modal-backdrop" onClick=${() => { setModal(false); record("Closed modal (backdrop)"); }}>
        <div class="stk-modal" role="dialog" data-testid="modal" onClick=${(e) => e.stopPropagation()}>
          <h3>Confirm action</h3>
          <p>This dialog appeared dynamically.</p>
          <div class="stk-row">
            <button class="stk-btn" data-testid="modal-cancel" onClick=${() => { setModal(false); record("Modal: cancel"); }}>Cancel</button>
            <button class="stk-btn primary" data-testid="modal-confirm" onClick=${() => { setModal(false); record("Modal: confirm", "assert"); }}>Confirm</button>
          </div>
        </div>
      </div>`}
  `;
};

// ---------------------------------------------------------------------------
const Section = ({ title, desc, children }) => html`
  <section class="stk-section">
    <div class="stk-section-head">
      <h2>${title}</h2>
      ${desc && html`<p>${desc}</p>`}
    </div>
    <div class="stk-section-body">${children}</div>
  </section>
`;

document.addEventListener("DOMContentLoaded", () => {
  render(h(App, null), document.getElementById("app"));
});
