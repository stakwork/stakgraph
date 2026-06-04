import * as api from "../api";
import { humanize } from "../helpers";
// ── Config Field Renderer ──────────────────────────────────────────────────

export function ConfigField(props: {
  field: api.FieldDesc;
  value: unknown;
  onChange: (v: unknown) => void;
}) {
  const { field, value, onChange } = props;
  const label = `${humanize(field.name)}${field.required ? "" : " (optional)"}`;

  if (field.kind === "enum" && field.enumValues) {
    return (
      <div class="flyout-field">
        <label>{label}</label>
        <select
          value={value != null ? String(value) : (field.default != null ? String(field.default) : "")}
          onChange={(e) => onChange((e.target as HTMLSelectElement).value)}
        >
          {!field.required && <option value="">--</option>}
          {field.enumValues.map((v) => (
            <option key={v} value={v}>{v}</option>
          ))}
        </select>
      </div>
    );
  }

  if (field.kind === "boolean") {
    const checked = value != null ? Boolean(value) : (field.default != null ? Boolean(field.default) : false);
    return (
      <div class="flyout-field">
        <label class="flyout-checkbox-label">
          <input
            type="checkbox"
            checked={checked}
            onChange={(e) => onChange((e.target as HTMLInputElement).checked)}
          />
          {label}
        </label>
      </div>
    );
  }

  if (field.kind === "number") {
    return (
      <div class="flyout-field">
        <label>{label}</label>
        <input
          type="number"
          value={value != null ? String(value) : (field.default != null ? String(field.default) : "")}
          placeholder={field.default != null ? `default: ${field.default}` : undefined}
          onInput={(e) => {
            const raw = (e.target as HTMLInputElement).value;
            onChange(raw === "" ? undefined : Number(raw));
          }}
        />
      </div>
    );
  }

  if (field.kind === "json") {
    const display = value != null
      ? (typeof value === "string" ? value : JSON.stringify(value, null, 2))
      : "";
    return (
      <div class="flyout-field">
        <label>{label}</label>
        <textarea
          value={display}
          rows={4}
          placeholder="JSON or template expression"
          onInput={(e) => {
            const raw = (e.target as HTMLTextAreaElement).value;
            if (raw === "") { onChange(undefined); return; }
            try { onChange(JSON.parse(raw)); } catch { onChange(raw); }
          }}
        />
      </div>
    );
  }

  // Default: string
  return (
    <div class="flyout-field">
      <label>{label}</label>
      <input
        type="text"
        value={value != null ? String(value) : ""}
        placeholder={field.default != null ? `default: ${field.default}` : undefined}
        onInput={(e) => onChange((e.target as HTMLInputElement).value)}
      />
    </div>
  );
}
