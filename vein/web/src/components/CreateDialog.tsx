
// ── Create Workflow Dialog ──────────────────────────────────────────────────

import { useState } from "preact/hooks";
import yaml from "js-yaml";

const EXAMPLE_YAML = `name: my-workflow
steps:
  - id: greet
    type: log
    config:
      message: "Hello from vein!"
  - id: fetch
    type: http
    config:
      url: https://httpbin.org/json
  - id: done
    type: log
    config:
      message: "Fetched: {{ fetch.body }}"
`;

export function CreateDialog(props: {
  onClose: () => void;
  onCreate: (name: string, yamlStr: string, description: string) => void;
}) {
  const [desc, setDesc] = useState("");
  const [yamlStr, setYamlStr] = useState(EXAMPLE_YAML);
  const [error, setError] = useState("");

  const handleSubmit = () => {
    try {
      const data = yaml.load(yamlStr) as any;
      if (!data?.name) { setError("YAML must have a 'name' field"); return; }
      if (!data?.steps || !Array.isArray(data.steps)) { setError("YAML must have a 'steps' array"); return; }
      if (!/^[a-zA-Z][a-zA-Z0-9_-]*$/.test(data.name)) { setError("Name must be alphanumeric (hyphens/underscores ok)"); return; }
      props.onCreate(data.name, yamlStr, desc);
    } catch (e) {
      setError(`Invalid YAML: ${e instanceof Error ? e.message : String(e)}`);
    }
  };

  return (
    <div class="dialog-backdrop" onClick={(e) => { if (e.target === e.currentTarget) props.onClose(); }}>
      <div class="dialog">
        <div class="dialog-title">Create Workflow</div>
        <div class="dialog-field">
          <label>Description</label>
          <input type="text" value={desc} onInput={(e) => setDesc((e.target as HTMLInputElement).value)} placeholder="What this workflow does" />
        </div>
        <div class="dialog-field">
          <label>Workflow (YAML)</label>
          <textarea value={yamlStr} onInput={(e) => { setYamlStr((e.target as HTMLTextAreaElement).value); setError(""); }} rows={16} />
          <div class="dialog-hint">Define name and steps. Types: http, log, if, loop, subflow, llm. Use depends: to set DAG edges.</div>
        </div>
        {error && <div style="color:var(--danger);font-size:12px;margin-bottom:8px;">{error}</div>}
        <div class="dialog-actions">
          <button class="btn" onClick={props.onClose}>Cancel</button>
          <button class="btn btn-primary" onClick={handleSubmit}>Create</button>
        </div>
      </div>
    </div>
  );
}

