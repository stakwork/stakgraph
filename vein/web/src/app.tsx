import { useState, useEffect, useCallback, useMemo } from "preact/hooks";
import { SystemCanvas } from "system-canvas-react";
import type { CanvasData, CanvasNode } from "system-canvas";
import yaml from "js-yaml";
import * as api from "./api";
import { flowToCanvas, veinTheme } from "./flow-to-canvas";
import type { FlowData, StepData, RunEventData } from "./flow-to-canvas";
import "./styles/base.css";
import "./styles/components.css";

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

const STEP_TYPES = ["http", "log", "if", "loop", "parallel", "subflow", "llm", "wait"];

export function App() {
  const [workflows, setWorkflows] = useState<api.WorkflowEntry[]>([]);
  const [runs, setRuns] = useState<api.RunSummary[]>([]);
  const [selectedWf, setSelectedWf] = useState<string | null>(null);
  const [selectedRun, setSelectedRun] = useState<string | null>(null);
  const [canvas, setCanvas] = useState<CanvasData | null>(null);
  const [events, setEvents] = useState<api.RunEvent[]>([]);
  const [showCreate, setShowCreate] = useState(false);
  const [publishedSteps, setPublishedSteps] = useState<StepData[] | null>(null);
  const [localSteps, setLocalSteps] = useState<StepData[] | null>(null);
  const [flyoutStepId, setFlyoutStepId] = useState<string | null>(null);
  const [flyoutStepIndex, setFlyoutStepIndex] = useState<number | null>(null);

  const isDirty = useMemo(() => {
    if (!publishedSteps || !localSteps) return false;
    return JSON.stringify(publishedSteps) !== JSON.stringify(localSteps);
  }, [publishedSteps, localSteps]);

  const activeVersion = workflows.find((w) => w.name === selectedWf)?.activeVersion;

  // Runs are already filtered by workflow (fetched per-workflow)
  const filteredRuns = runs;

  // Events for the flyout step
  const flyoutEvents = useMemo(() => {
    if (!flyoutStepId || !selectedWf || events.length === 0) return null;
    const path = `${selectedWf}/${flyoutStepId}`;
    const stepEvts = events.filter((e) => e.path === path);
    if (stepEvts.length === 0) return null;
    const start = stepEvts.find((e) => e.type === "step.start");
    const end = stepEvts.find((e) => e.type === "step.end");
    const error = stepEvts.find((e) => e.type === "step.error");
    return { start, end, error, all: stepEvts };
  }, [flyoutStepId, selectedWf, events]);

  const isRunView = selectedRun != null && events.length > 0;

  const refreshWorkflows = useCallback(async () => {
    const wfs = await api.listWorkflows();
    setWorkflows(wfs);
  }, []);

  const refreshRuns = useCallback(async (wf?: string) => {
    const name = wf ?? selectedWf;
    if (!name) { setRuns([]); return; }
    const rs = await api.listRuns(name);
    setRuns(rs);
  }, [selectedWf]);

  useEffect(() => { refreshWorkflows(); }, []);

  // Load workflow + its runs when selected
  useEffect(() => {
    if (!selectedWf) {
      setCanvas(null);
      setPublishedSteps(null);
      setLocalSteps(null);
      setRuns([]);
      setFlyoutStepId(null);
      setFlyoutStepIndex(null);
      return;
    }
    api.getWorkflowFlow(selectedWf).then((flow) => {
      const steps = flow.steps as StepData[];
      setPublishedSteps(steps);
      setLocalSteps(steps);
      rebuildCanvas(selectedWf, steps, null);
    }).catch(() => {
      setPublishedSteps(null);
      setLocalSteps(null);
      setCanvas({
        nodes: [{ id: "err", type: "text", text: `${selectedWf}\n\nCould not load`, x: 100, y: 100, width: 260, height: 70, color: "1" }],
        edges: [], theme: { base: "midnight" },
      });
    });
    refreshRuns(selectedWf);
  }, [selectedWf]);

  // Load run events when a run is selected; clear overlay when deselected
  useEffect(() => {
    if (!selectedRun || !selectedWf) {
      setEvents([]);
      // Rebuild canvas without run overlay
      if (selectedWf && localSteps) {
        rebuildCanvas(selectedWf, localSteps, null);
      }
      return;
    }
    api.getRunEvents(selectedWf, selectedRun).then((evts) => {
      setEvents(evts);
      if (localSteps) {
        rebuildCanvas(selectedWf, localSteps, evts as RunEventData[]);
      }
    }).catch(console.error);
  }, [selectedRun]);

  function rebuildCanvas(wfName: string, steps: StepData[], evts: RunEventData[] | null) {
    setCanvas(flowToCanvas({ name: wfName, steps }, evts ?? undefined));
  }

  function updateLocalSteps(steps: StepData[]) {
    setLocalSteps(steps);
    if (selectedWf) rebuildCanvas(selectedWf, steps, null);
  }

  const handleRun = useCallback(async () => {
    if (!selectedWf || !localSteps) return;
    const wf = selectedWf;
    const steps = localSteps;
    const accumulated: RunEventData[] = [];

    // Show pending status on all nodes immediately
    rebuildCanvas(wf, steps, []);

    const result = await api.runWorkflow(wf, {}, (event) => {
      accumulated.push(event as RunEventData);
      rebuildCanvas(wf, steps, [...accumulated]);
      setEvents([...accumulated] as api.RunEvent[]);
    });

    if (result?.runId) {
      setSelectedRun(result.runId);
    }
    await refreshRuns(wf);
  }, [selectedWf, localSteps, refreshRuns]);

  const handleCreate = useCallback(async (name: string, yamlStr: string, desc: string) => {
    await api.publishWorkflowYaml(name, "v1", yamlStr, desc || undefined);
    await refreshWorkflows();
    setSelectedWf(name);
    setShowCreate(false);
  }, [refreshWorkflows]);

  const handlePublish = useCallback(async () => {
    if (!selectedWf || !localSteps || !activeVersion) return;
    const num = parseInt(activeVersion.replace(/^v/, ""), 10);
    const nextVersion = `v${(isNaN(num) ? 1 : num) + 1}`;
    const yamlStr = yaml.dump(
      { name: selectedWf, steps: localSteps },
      { lineWidth: 120, noRefs: true },
    );
    await api.publishWorkflowYaml(selectedWf, nextVersion, yamlStr);
    await refreshWorkflows();
    const flow = await api.getWorkflowFlow(selectedWf);
    const steps = flow.steps as StepData[];
    setPublishedSteps(steps);
    setLocalSteps(steps);
    rebuildCanvas(selectedWf, steps, null);
  }, [selectedWf, localSteps, activeVersion, refreshWorkflows]);

  const handleNodeClick = useCallback((node: CanvasNode) => {
    const stepId = node.customData?.stepId as string | undefined;
    const stepIndex = node.customData?.stepIndex as number | undefined;
    if (stepId != null) {
      setFlyoutStepId(stepId);
      setFlyoutStepIndex(stepIndex ?? null);
    }
  }, []);

  const handleStepSave = useCallback((index: number, updated: StepData) => {
    if (!localSteps) return;
    const next = [...localSteps];
    next[index] = updated;
    updateLocalSteps(next);
    setFlyoutStepId(null);
    setFlyoutStepIndex(null);
  }, [localSteps, selectedWf]);

  const closeFlyout = () => { setFlyoutStepId(null); setFlyoutStepIndex(null); };

  return (
    <div class="shell">
      {/* Sidebar */}
      <div class="shell-sidebar">
        <div class="sidebar-brand"><span class="brand-dot" /> vein</div>

        <div class="sidebar-section">
          <div class="section-title">
            Workflows
            <button class="btn" style="float:right;padding:1px 8px;font-size:11px;" onClick={() => setShowCreate(true)}>+</button>
          </div>
          <div class="sidebar-scroll">
            {workflows.length === 0 && <div class="empty-sidebar">No workflows yet</div>}
            {workflows.map((wf) => (
              <div key={wf.name} class={`list-item ${selectedWf === wf.name ? "is-active" : ""}`}
                onClick={() => { setSelectedWf(wf.name); setSelectedRun(null); setEvents([]); closeFlyout(); }}>
                <span class="list-item-name">{wf.name}</span>
                <span class="badge badge-accent">{wf.activeVersion}</span>
              </div>
            ))}
          </div>
        </div>

        <div class="sidebar-section">
          <div class="section-title">
            Runs
            {selectedWf && <span style="float:right;font-weight:400;text-transform:none;letter-spacing:0;">{selectedWf}</span>}
          </div>
          <div class="sidebar-scroll">
            {filteredRuns.length === 0 && <div class="empty-sidebar">{selectedWf ? "No runs for this workflow" : "No runs yet"}</div>}
            {filteredRuns.map((run) => (
              <div key={run.runId} class={`list-item ${selectedRun === run.runId ? "is-active" : ""}`}
                onClick={() => { setSelectedRun(run.runId); closeFlyout(); }}>
                <div class="list-item-stack">
                  <span class="list-item-name">{run.runId.slice(0, 8)}</span>
                  <span class="list-item-sub">
                    {run.startedAt ? new Date(run.startedAt).toLocaleTimeString() : "..."}
                    {run.durationMs != null && ` (${run.durationMs}ms)`}
                  </span>
                </div>
                <span class={`badge badge-${statusTone(run.status)}`}>{run.status}</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Topbar */}
      <div class="shell-topbar">
        <span class="topbar-title">
          {selectedWf ?? "Select a workflow"}
          {isDirty && <span class="dirty-dot" style="margin-left:8px;" />}
        </span>
        <div class="topbar-actions">
          {isDirty && <button class="btn btn-publish" onClick={handlePublish}>Publish</button>}
          {selectedWf && <button class="btn btn-primary" onClick={handleRun}>Run</button>}
        </div>
      </div>

      {/* Canvas */}
      <div class="shell-canvas">
        {canvas
          ? <SystemCanvas canvas={canvas} onNodeClick={handleNodeClick} themes={{ vein: veinTheme }} />
          : <div class="empty">{workflows.length === 0 ? "Create a workflow to get started" : "Select a workflow to view its flow graph"}</div>}
      </div>

      {/* Events panel */}
      {events.length > 0 && <EventsPanel events={events} />}

      {/* Create dialog */}
      {showCreate && <CreateDialog onClose={() => setShowCreate(false)} onCreate={handleCreate} />}

      {/* Step flyout */}
      {flyoutStepId != null && flyoutStepIndex != null && localSteps && localSteps[flyoutStepIndex] && (
        isRunView && flyoutEvents
          ? <StepRunFlyout
              step={localSteps[flyoutStepIndex]!}
              events={flyoutEvents}
              onClose={closeFlyout}
            />
          : <StepEditFlyout
              step={localSteps[flyoutStepIndex]!}
              onSave={(updated) => handleStepSave(flyoutStepIndex!, updated)}
              onClose={closeFlyout}
            />
      )}
    </div>
  );
}

// ── Events Panel (expandable rows) ─────────────────────────────────────────

function EventsPanel(props: { events: api.RunEvent[] }) {
  const [expanded, setExpanded] = useState<number | null>(null);

  return (
    <div class="shell-events">
      <div class="events-header">Events ({props.events.length})</div>
      {props.events.map((evt, i) => {
        const hasData = evt.input != null || evt.output != null || evt.error != null;
        const isOpen = expanded === i;
        return (
          <div key={i} class="event-row">
            <div class="event-row-summary" onClick={() => hasData && setExpanded(isOpen ? null : i)}>
              <span class={`event-type event-type-${eventTone(evt.type)}`}>{evt.type}</span>
              <span class="event-path">{evt.path}</span>
              <span class="event-duration">{evt.durationMs != null ? `${evt.durationMs}ms` : ""}</span>
            </div>
            {isOpen && (
              <div class="event-detail">
                {evt.input != null && (
                  <>
                    <div class="event-detail-label">Input</div>
                    <pre class="event-detail-block">{formatJson(evt.input)}</pre>
                  </>
                )}
                {evt.output != null && (
                  <>
                    <div class="event-detail-label">Output</div>
                    <pre class="event-detail-block">{formatJson(evt.output)}</pre>
                  </>
                )}
                {evt.error != null && (
                  <>
                    <div class="event-detail-label">Error</div>
                    <pre class="event-detail-block tone-error">{formatJson(evt.error)}</pre>
                  </>
                )}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}

// ── Step Run Results Flyout (read-only) ────────────────────────────────────

function StepRunFlyout(props: {
  step: StepData;
  events: { start?: api.RunEvent; end?: api.RunEvent; error?: api.RunEvent; all: api.RunEvent[] };
  onClose: () => void;
}) {
  const { step, events } = props;
  const status = events.error ? "error" : events.end ? "success" : "running";

  return (
    <>
      <div class="flyout">
        <div class="flyout-header">
          <div>
            <div class="flyout-eyebrow">Step Results</div>
            <div class="flyout-title">{step.id} <span style="color:var(--text-dim);font-weight:400;">({step.type})</span></div>
          </div>
          <button class="flyout-close" onClick={props.onClose}>x</button>
        </div>
        <div class="flyout-body">
          {/* Status */}
          <div class="flyout-section">
            <div class="flyout-meta-row">
              <span class="flyout-meta-label">Status</span>
              <span class={`badge badge-${statusTone(status)}`}>{status}</span>
            </div>
            {events.end?.durationMs != null && (
              <div class="flyout-meta-row">
                <span class="flyout-meta-label">Duration</span>
                <span class="flyout-meta-value">{events.end.durationMs}ms</span>
              </div>
            )}
            {events.start?.ts && (
              <div class="flyout-meta-row">
                <span class="flyout-meta-label">Started</span>
                <span class="flyout-meta-value">{new Date(events.start.ts).toLocaleTimeString()}</span>
              </div>
            )}
          </div>

          {/* Input (resolved config) */}
          {events.start?.input != null && (
            <div class="flyout-section">
              <div class="flyout-section-title">Input (resolved config)</div>
              <pre class="flyout-json">{formatJson(events.start.input)}</pre>
            </div>
          )}

          {/* Output */}
          {events.end?.output != null && (
            <div class="flyout-section">
              <div class="flyout-section-title">Output</div>
              <pre class="flyout-json">{formatJson(events.end.output)}</pre>
            </div>
          )}

          {/* Error */}
          {events.error?.error != null && (
            <div class="flyout-section">
              <div class="flyout-section-title">Error</div>
              <pre class="flyout-json tone-error">{formatJson(events.error.error)}</pre>
            </div>
          )}

          {/* Original config (from workflow definition) */}
          <div class="flyout-section">
            <div class="flyout-section-title">Config (workflow definition)</div>
            <pre class="flyout-json">{yaml.dump(step.config, { lineWidth: 120, noRefs: true }).trim()}</pre>
          </div>
        </div>
      </div>
    </>
  );
}

// ── Step Edit Flyout ───────────────────────────────────────────────────────

function StepEditFlyout(props: {
  step: StepData;
  onSave: (updated: StepData) => void;
  onClose: () => void;
}) {
  const [stepYaml, setStepYaml] = useState(
    yaml.dump(
      { id: props.step.id, type: props.step.type, config: props.step.config },
      { lineWidth: 120, noRefs: true },
    ),
  );
  const [error, setError] = useState("");

  useEffect(() => {
    setStepYaml(
      yaml.dump(
        { id: props.step.id, type: props.step.type, config: props.step.config },
        { lineWidth: 120, noRefs: true },
      ),
    );
    setError("");
  }, [props.step]);

  const handleSave = () => {
    try {
      const data = yaml.load(stepYaml) as any;
      if (!data?.id) { setError("Step must have an 'id'"); return; }
      if (!data?.type) { setError("Step must have a 'type'"); return; }
      if (!/^[a-zA-Z_][a-zA-Z0-9_]*$/.test(data.id)) {
        setError("ID must match [a-zA-Z_][a-zA-Z0-9_]*");
        return;
      }
      props.onSave({
        id: data.id,
        type: data.type,
        config: data.config ?? {},
        options: props.step.options,
      });
    } catch (e) {
      setError(`Invalid YAML: ${e instanceof Error ? e.message : String(e)}`);
    }
  };

  return (
    <>
      <div class="flyout">
        <div class="flyout-header">
          <div>
            <div class="flyout-eyebrow">Edit Step</div>
            <div class="flyout-title">{props.step.id}</div>
          </div>
          <button class="flyout-close" onClick={props.onClose}>x</button>
        </div>
        <div class="flyout-body">
          <div class="flyout-field">
            <label>Step (YAML)</label>
            <textarea
              value={stepYaml}
              onInput={(e) => { setStepYaml((e.target as HTMLTextAreaElement).value); setError(""); }}
              rows={14}
            />
          </div>
          {error && <div style="color:var(--danger);font-size:12px;">{error}</div>}
        </div>
        <div class="flyout-actions">
          <button class="btn" onClick={props.onClose}>Cancel</button>
          <button class="btn btn-primary" onClick={handleSave}>Save</button>
        </div>
      </div>
    </>
  );
}

// ── Create Workflow Dialog ──────────────────────────────────────────────────

function CreateDialog(props: {
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
          <div class="dialog-hint">Define name and steps. Types: http, log, if, loop, parallel, subflow, llm</div>
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

// ── Helpers ────────────────────────────────────────────────────────────────

function statusTone(s: string) { return s === "success" ? "ok" : s === "error" ? "danger" : "warning"; }
function eventTone(t: string) { return t.includes("error") ? "error" : t.includes("end") ? "end" : t.includes("start") ? "start" : t.includes("retry") ? "retry" : "other"; }
function formatJson(v: unknown): string {
  if (typeof v === "string") return v;
  try { return JSON.stringify(v, null, 2); } catch { return String(v); }
}
