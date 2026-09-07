
// ── Category chip + editor ──────────────────────────────────────────────────
//
// Small chip in the topbar showing the selected workflow's category. Click to
// open a popover with a free-text input (datalist-suggested from existing
// categories) — save sets the category, empty clears it. Metadata-only on the
// server: no new workflow version is published.

import { useState } from "preact/hooks";
import * as api from "../api";

export function CategoryEditor(props: {
  workflow: string;
  category?: string;
  categories: string[];
  onSaved: () => void | Promise<void>;
}) {
  const [open, setOpen] = useState(false);
  const [value, setValue] = useState(props.category ?? "");
  const [saving, setSaving] = useState(false);

  const toggle = () => {
    setValue(props.category ?? "");
    setOpen((o) => !o);
  };

  const save = async () => {
    const next = value.trim();
    if (next === (props.category ?? "")) { setOpen(false); return; }
    setSaving(true);
    try {
      await api.setWorkflowCategory(props.workflow, next || null);
      await props.onSaved();
      setOpen(false);
    } catch (e) {
      console.error(e);
    } finally {
      setSaving(false);
    }
  };

  return (
    <span class="cat-anchor">
      <button
        class={`cat-chip${props.category ? "" : " is-empty"}`}
        title={props.category ? "Change category" : "Set category"}
        onClick={toggle}
      >
        {props.category ?? "+ category"}
      </button>
      {open && (
        <div class="cat-popover">
          <input
            type="text"
            list="cat-suggestions"
            value={value}
            placeholder="Category (empty to clear)"
            autoFocus
            onInput={(e) => setValue((e.target as HTMLInputElement).value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") save();
              if (e.key === "Escape") setOpen(false);
            }}
          />
          <datalist id="cat-suggestions">
            {props.categories.map((c) => <option key={c} value={c} />)}
          </datalist>
          <div class="cat-popover-actions">
            <button class="btn" onClick={() => setOpen(false)}>Cancel</button>
            <button class="btn btn-primary" disabled={saving} onClick={save}>Save</button>
          </div>
        </div>
      )}
    </span>
  );
}
