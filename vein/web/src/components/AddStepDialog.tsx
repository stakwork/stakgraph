import { useState, useEffect, useRef } from "preact/hooks";

// ── Add Step Dialog (searchable) ────────────────────────────────────────────

export interface StepTypeEntry {
  type: string;
  source: "core" | "lib" | "custom";
  description?: string;
}

export function AddStepDialog(props: {
  stepTypes: StepTypeEntry[];
  onSelect: (type: string) => void;
  onClose: () => void;
}) {
  const [query, setQuery] = useState("");
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => { inputRef.current?.focus(); }, []);

  const q = query.toLowerCase().trim();
  const filtered = q
    ? props.stepTypes.filter((s) => s.type.toLowerCase().includes(q) || s.description?.toLowerCase().includes(q))
    : props.stepTypes;

  // Group by source
  const core = filtered.filter((s) => s.source === "core");
  const lib = filtered.filter((s) => s.source === "lib");
  const custom = filtered.filter((s) => s.source === "custom");

  const handleKeyDown = (e: KeyboardEvent) => {
    if (e.key === "Escape") props.onClose();
    if (e.key === "Enter" && filtered.length === 1) {
      props.onSelect(filtered[0]!.type);
    }
  };

  return (
    <div class="dialog-backdrop" onClick={(e) => { if (e.target === e.currentTarget) props.onClose(); }}>
      <div class="dialog add-step-dialog">
        <div class="dialog-title">Add Step</div>
        <div class="add-step-search">
          <input
            ref={inputRef}
            type="text"
            value={query}
            onInput={(e) => setQuery((e.target as HTMLInputElement).value)}
            onKeyDown={handleKeyDown}
            placeholder="Search step types..."
          />
        </div>
        <div class="add-step-list">
          {core.length > 0 && (
            <StepGroup label="Core" items={core} onSelect={props.onSelect} />
          )}
          {lib.length > 0 && (
            <StepGroup label="Library" items={lib} onSelect={props.onSelect} />
          )}
          {custom.length > 0 && (
            <StepGroup label="Custom" items={custom} onSelect={props.onSelect} />
          )}
          {filtered.length === 0 && (
            <div class="add-step-empty">No matching step types</div>
          )}
        </div>
      </div>
    </div>
  );
}

function StepGroup(props: {
  label: string;
  items: StepTypeEntry[];
  onSelect: (type: string) => void;
}) {
  return (
    <div class="add-step-group">
      <div class="add-step-group-label">{props.label}</div>
      {props.items.map((s) => (
        <button
          key={s.type}
          class="add-step-item"
          onClick={() => props.onSelect(s.type)}
        >
          <span class="add-step-item-type">{s.type}</span>
          {s.description && <span class="add-step-item-desc">{s.description}</span>}
        </button>
      ))}
    </div>
  );
}
