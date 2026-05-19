// WindowPicker — a four-option segmented control. Every page that
// pulls windowed analytics from the backend uses this so the option
// set stays consistent (and stays in lockstep with the Go-side
// parseWindow whitelist).

import type { Window } from "../../api/types";

const OPTIONS: Window[] = ["1h", "24h", "7d", "30d"];

interface Props {
  value: Window;
  onChange: (w: Window) => void;
}

export function WindowPicker({ value, onChange }: Props) {
  return (
    <div class="btn-group" role="group" aria-label="time window">
      {OPTIONS.map((w) => (
        <button
          key={w}
          type="button"
          class={"btn" + (w === value ? " is-active" : "")}
          onClick={() => onChange(w)}
        >
          {w}
        </button>
      ))}
    </div>
  );
}
