import { useState, useEffect, useCallback, useMemo, useRef, type StateUpdater } from "preact/hooks";
import { SystemCanvas } from "system-canvas-react";
import type { CanvasData, CanvasNode, CanvasEdge } from "system-canvas";
import type { AddNodeButtonRenderProps } from "system-canvas-react";
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

const STEP_TYPES = ["http", "log", "if", "loop", "subflow", "llm", "wait"];

interface StepTypeEntry {
  type: string;
  source: "core" | "lib" | "custom";
  description?: string;
}

export function App() {
  const [workflows, setWorkflows] = useState<api.WorkflowEntry[]>([]);
  const [runs, setRuns] = useState<api.RunSummary[]>([]);
  const [selectedWf, setSelectedWf] = useState<string | null>(null);
  const [selectedRun, setSelectedRun] = useState<string | null>(null);
  const [canvas, setCanvas] = useState<CanvasData | null>(null);
  const [events, setEvents] = useState<api.RunEvent[]>([]);
  const [showCreate, setShowCreate] = useState(false);
  const [showAddStep, setShowAddStep] = useState(false);
  const [stepTypes, setStepTypes] = useState<StepTypeEntry[]>([]);
  const [publishedSteps, setPublishedSteps] = useState<StepData[] | null>(null);
  const [localSteps, setLocalSteps] = useState<StepData[] | null>(null);
  const [flyoutStepId, setFlyoutStepId] = useState<string | null>(null);
  const [flyoutStepIndex, setFlyoutStepIndex] = useState<number | null>(null);
  const [flyoutPath, setFlyoutPath] = useState<string | null>(null);
  const [showChat, setShowChat] = useState(false);

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
    const path = flyoutPath ?? `${selectedWf}/${flyoutStepId}`;
    const stepEvts = events.filter((e) => e.path === path);
    if (stepEvts.length === 0) return null;
    const start = stepEvts.find((e) => e.type === "step.start");
    const end = stepEvts.find((e) => e.type === "step.end");
    const error = stepEvts.find((e) => e.type === "step.error");
    return { start, end, error, all: stepEvts };
  }, [flyoutStepId, flyoutPath, selectedWf, events]);

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

  const refreshStepTypes = useCallback(async () => {
    try {
      const resp = await api.listSteps();
      const entries: StepTypeEntry[] = [
        ...resp.core.map((s) => ({ type: s.type, source: s.source as "core" | "lib" | "custom" })),
        ...resp.workspace.map((s) => ({ type: s.type, source: "custom" as const, description: s.description })),
      ];
      setStepTypes(entries);
    } catch { /* ignore */ }
  }, []);

  useEffect(() => { refreshWorkflows(); refreshStepTypes(); }, []);

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
    const nestedPath = node.customData?.nestedPath as string | undefined;
    if (stepId != null) {
      setFlyoutStepId(stepId);
      setFlyoutStepIndex(stepIndex ?? null);
      setFlyoutPath(nestedPath ?? null);
    }
  }, []);

  const handleNodeAdd = useCallback((_node: CanvasNode) => {
    // Intercepted by renderAddNodeButton — should not fire
  }, []);

  const handleAddStepSelect = useCallback((stepType: string) => {
    if (!localSteps || !selectedWf) return;
    // Generate a unique step id from the type name
    const base = stepType.replace(/[^a-zA-Z0-9_]/g, "_");
    let id = base;
    let n = 1;
    while (localSteps.some((s) => s.id === id)) {
      id = `${base}_${n++}`;
    }
    const newStep: StepData = {
      id,
      type: stepType,
      config: {},
      depends: [],
    };
    const next = [...localSteps, newStep];
    updateLocalSteps(next);
    setShowAddStep(false);
    // Open flyout for the new step
    setFlyoutStepId(id);
    setFlyoutStepIndex(next.length - 1);
  }, [localSteps, selectedWf]);

  const handleEdgeAdd = useCallback((edge: CanvasEdge) => {
    if (!localSteps || !selectedWf) return;
    // Extract step ids from node ids (format: "workflowName/stepId")
    const fromStepId = edge.fromNode.split("/").pop();
    const toStepId = edge.toNode.split("/").pop();
    if (!fromStepId || !toStepId) return;

    const next = localSteps.map((s) => {
      if (s.id !== toStepId) return s;
      // Add the dependency
      const currentDeps = s.depends == null ? [] : Array.isArray(s.depends) ? [...s.depends] : [s.depends];
      if (!currentDeps.includes(fromStepId)) {
        currentDeps.push(fromStepId);
      }
      return { ...s, depends: currentDeps };
    });
    updateLocalSteps(next);
  }, [localSteps, selectedWf]);

  const handleEdgeDelete = useCallback((edgeId: string) => {
    if (!localSteps || !selectedWf) return;
    // Edge id format: "wfName/fromId__to__wfName/toId"
    const parts = edgeId.split("__to__");
    if (parts.length !== 2) return;
    const fromStepId = parts[0]!.split("/").pop();
    const toStepId = parts[1]!.split("/").pop();
    if (!fromStepId || !toStepId) return;

    const next = localSteps.map((s) => {
      if (s.id !== toStepId) return s;
      const currentDeps = s.depends == null ? [] : Array.isArray(s.depends) ? [...s.depends] : [s.depends];
      const filtered = currentDeps.filter((d) => d !== fromStepId);
      return { ...s, depends: filtered };
    });
    updateLocalSteps(next);
  }, [localSteps, selectedWf]);

  const handleNodeDelete = useCallback((nodeId: string) => {
    if (!localSteps || !selectedWf) return;
    const stepId = nodeId.split("/").pop();
    if (!stepId) return;
    // Remove the step and clean up depends references
    const next = localSteps
      .filter((s) => s.id !== stepId)
      .map((s) => {
        if (!s.depends) return s;
        const deps = Array.isArray(s.depends) ? s.depends : [s.depends];
        const filtered = deps.filter((d) => d !== stepId);
        return { ...s, depends: filtered };
      });
    updateLocalSteps(next);
    if (flyoutStepId === stepId) closeFlyout();
  }, [localSteps, selectedWf, flyoutStepId]);

  const handleStepSave = useCallback((index: number, updated: StepData) => {
    if (!localSteps) return;
    const oldId = localSteps[index]!.id;
    const newId = updated.id;
    let next = [...localSteps];
    next[index] = updated;
    // If the ID changed, update all depends references in other steps
    if (oldId !== newId) {
      next = next.map((s, i) => {
        if (i === index || !s.depends) return s;
        const deps = Array.isArray(s.depends) ? s.depends : [s.depends];
        if (!deps.includes(oldId)) return s;
        return { ...s, depends: deps.map((d) => d === oldId ? newId : d) };
      });
    }
    updateLocalSteps(next);
    setFlyoutStepId(null);
    setFlyoutStepIndex(null);
  }, [localSteps, selectedWf]);

  const closeFlyout = () => { setFlyoutStepId(null); setFlyoutStepIndex(null); setFlyoutPath(null); };

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
          <button class="btn" onClick={() => setShowChat(!showChat)}>AI</button>
        </div>
      </div>

      {/* Canvas */}
      <div class="shell-canvas">
        {canvas
          ? <SystemCanvas
              panMode="trackpad"
              canvas={canvas}
              editable={!isRunView}
              onNodeClick={handleNodeClick}
              onNodeAdd={handleNodeAdd}
              onNodeDelete={handleNodeDelete}
              onEdgeAdd={handleEdgeAdd}
              onEdgeDelete={handleEdgeDelete}
              renderAddNodeButton={(_props: AddNodeButtonRenderProps) => (
                selectedWf && !isRunView
                  ? <button class="add-step-fab" onClick={() => setShowAddStep(true)}>+</button>
                  : null
              )}
              themes={{ vein: veinTheme }}
            />
          : <div class="empty">{workflows.length === 0 ? "Create a workflow to get started" : "Select a workflow to view its flow graph"}</div>}
      </div>

      {/* Events panel */}
      {events.length > 0 && <EventsPanel events={events} />}

      {/* Create dialog */}
      {showCreate && <CreateDialog onClose={() => setShowCreate(false)} onCreate={handleCreate} />}

      {/* Add step dialog */}
      {showAddStep && <AddStepDialog stepTypes={stepTypes} onSelect={handleAddStepSelect} onClose={() => setShowAddStep(false)} />}

      {/* Chat flyout */}
      {showChat && (
        <ChatFlyout
          onClose={() => setShowChat(false)}
          onWorkflowCreated={async (name) => {
            await refreshWorkflows();
            setSelectedWf(name);
            setShowChat(false);
          }}
        />
      )}

      {/* Step flyout */}
      {flyoutStepId != null && localSteps && (() => {
        // Find step data: either top-level by index, or nested in if/loop config
        const topStep = flyoutStepIndex != null ? localSteps[flyoutStepIndex] : null;
        const step = topStep ?? findNestedStep(localSteps, flyoutStepId);
        if (!step) return null;

        if (isRunView && flyoutEvents) {
          return <StepRunFlyout step={step} events={flyoutEvents} onClose={closeFlyout} />;
        }
        if (topStep && flyoutStepIndex != null) {
          return (
            <StepEditFlyout
              step={topStep}
              allStepIds={localSteps.map((s) => s.id)}
              onSave={(updated) => handleStepSave(flyoutStepIndex, updated)}
              onClose={closeFlyout}
            />
          );
        }
        return null;
      })()}
    </div>
  );
}

// ── Events Panel (expandable rows) ─────────────────────────────────────────

function EventsPanel(props: { events: api.RunEvent[] }) {
  const [expanded, setExpanded] = useState<number | null>(null);

  // Auto-expand run.end when it arrives
  useEffect(() => {
    const idx = props.events.findIndex((e) => e.type === "run.end");
    if (idx >= 0) setExpanded(idx);
  }, [props.events]);

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
  allStepIds: string[];
  onSave: (updated: StepData) => void;
  onClose: () => void;
}) {
  const [id, setId] = useState(props.step.id);
  const [config, setConfig] = useState<Record<string, any>>({ ...props.step.config });
  const [depends, setDepends] = useState<string[]>(() => {
    if (props.step.depends == null) return [];
    return Array.isArray(props.step.depends) ? [...props.step.depends] : [props.step.depends];
  });
  const [fields, setFields] = useState<api.FieldDesc[]>([]);
  const [error, setError] = useState("");

  // Fetch schema for this step type
  useEffect(() => {
    api.getStepSchema(props.step.type).then((resp) => {
      setFields(resp.fields);
    }).catch(() => setFields([]));
  }, [props.step.type]);

  // Reset state when step changes
  useEffect(() => {
    setId(props.step.id);
    setConfig({ ...props.step.config });
    const deps = props.step.depends == null ? [] : Array.isArray(props.step.depends) ? [...props.step.depends] : [props.step.depends];
    setDepends(deps);
    setError("");
  }, [props.step]);

  const updateConfig = (name: string, value: unknown) => {
    setConfig((prev) => ({ ...prev, [name]: value }));
  };

  const handleSave = () => {
    if (!id) { setError("Step must have an id"); return; }
    if (!/^[a-zA-Z_][a-zA-Z0-9_]*$/.test(id)) {
      setError("ID must match [a-zA-Z_][a-zA-Z0-9_]*");
      return;
    }
    // Strip empty-string values for optional fields
    const cleanConfig: Record<string, any> = {};
    for (const [k, v] of Object.entries(config)) {
      if (v !== "" && v !== undefined) cleanConfig[k] = v;
    }
    const updated: StepData = {
      id,
      type: props.step.type,
      config: cleanConfig,
      options: props.step.options,
    };
    if (depends.length > 0) updated.depends = depends;
    props.onSave(updated);
  };

  // Build YAML preview
  const previewObj: Record<string, any> = { id, type: props.step.type, config };
  if (depends.length > 0) previewObj.depends = depends;
  const yamlPreview = yaml.dump(previewObj, { lineWidth: 120, noRefs: true });

  // Other step ids for depends checkboxes (exclude self)
  const otherStepIds = props.allStepIds.filter((sid) => sid !== props.step.id);

  return (
    <div class="flyout">
      <div class="flyout-header">
        <div>
          <div class="flyout-eyebrow">Edit Step</div>
          <div class="flyout-title">{props.step.type}</div>
        </div>
        <button class="flyout-close" onClick={props.onClose}>x</button>
      </div>
      <div class="flyout-body">
        {/* Step ID */}
        <div class="flyout-field">
          <label>ID</label>
          <input
            type="text"
            value={id}
            onInput={(e) => { setId((e.target as HTMLInputElement).value); setError(""); }}
          />
        </div>

        {/* Config fields from schema */}
        {fields.length > 0 && (
          <div class="flyout-section">
            <div class="flyout-section-title">Config</div>
            {fields.map((f) => (
              <ConfigField
                key={f.name}
                field={f}
                value={config[f.name]}
                onChange={(v) => updateConfig(f.name, v)}
              />
            ))}
          </div>
        )}

        {/* Fallback: if no schema loaded yet, show raw config fields */}
        {fields.length === 0 && Object.keys(config).length > 0 && (
          <div class="flyout-section">
            <div class="flyout-section-title">Config</div>
            {Object.entries(config).map(([key, val]) => (
              <div class="flyout-field" key={key}>
                <label>{key}</label>
                <input
                  type="text"
                  value={typeof val === "string" ? val : JSON.stringify(val)}
                  onInput={(e) => {
                    const raw = (e.target as HTMLInputElement).value;
                    try { updateConfig(key, JSON.parse(raw)); } catch { updateConfig(key, raw); }
                  }}
                />
              </div>
            ))}
          </div>
        )}

        {/* Depends */}
        {otherStepIds.length > 0 && (
          <div class="flyout-section">
            <div class="flyout-section-title">Depends on</div>
            <div class="flyout-checkbox-group">
              {otherStepIds.map((sid) => (
                <label key={sid} class="flyout-checkbox-label">
                  <input
                    type="checkbox"
                    checked={depends.includes(sid)}
                    onChange={(e) => {
                      const checked = (e.target as HTMLInputElement).checked;
                      setDepends((prev) =>
                        checked ? [...prev, sid] : prev.filter((d) => d !== sid)
                      );
                    }}
                  />
                  {sid}
                </label>
              ))}
            </div>
          </div>
        )}

        {/* YAML preview (read-only) */}
        <div class="flyout-section">
          <div class="flyout-section-title">YAML Preview</div>
          <pre class="flyout-yaml-preview">{yamlPreview}</pre>
        </div>

        {error && <div style="color:var(--danger);font-size:12px;">{error}</div>}
      </div>
      <div class="flyout-actions">
        <button class="btn" onClick={props.onClose}>Cancel</button>
        <button class="btn btn-primary" onClick={handleSave}>Save</button>
      </div>
    </div>
  );
}

// ── Config Field Renderer ──────────────────────────────────────────────────

function ConfigField(props: {
  field: api.FieldDesc;
  value: unknown;
  onChange: (v: unknown) => void;
}) {
  const { field, value, onChange } = props;
  const label = `${field.name}${field.required ? "" : " (optional)"}`;

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

// ── Add Step Dialog (searchable) ────────────────────────────────────────────

function AddStepDialog(props: {
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

// ── Helpers ────────────────────────────────────────────────────────────────

// ── Chat Flyout (AI workflow builder) ──────────────────────────────────────

type ChatEntry =
  | { kind: "user"; content: string }
  | { kind: "text"; content: string }
  | { kind: "tool"; calls: api.ToolCallInfo[] };

function ChatFlyout(props: {
  onClose: () => void;
  onWorkflowCreated: (name: string) => void;
}) {
  const [entries, setEntries] = useState<ChatEntry[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => { inputRef.current?.focus(); }, []);

  // Auto-scroll on new content
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [entries]);

  const send = useCallback(async () => {
    const text = input.trim();
    if (!text || loading) return;

    setEntries((prev) => [...prev, { kind: "user", content: text }]);
    setInput("");
    setLoading(true);

    // Build API messages: flatten entries into role/content pairs
    const allEntries = [...entries, { kind: "user" as const, content: text }];
    const apiMessages: api.ChatMessage[] = [];
    for (const e of allEntries) {
      if (e.kind === "user") {
        apiMessages.push({ role: "user", content: e.content });
      } else if (e.kind === "text" && e.content) {
        apiMessages.push({ role: "assistant", content: e.content });
      }
    }

    let textBuf = "";
    let toolBuf: api.ToolCallInfo[] = [];

    try {
      await api.chat(apiMessages, {
        onTextDelta: (delta) => {
          textBuf += delta;
          const content = textBuf;
          setEntries((prev) => {
            const last = prev[prev.length - 1];
            if (last && last.kind === "text") {
              const next = [...prev];
              next[next.length - 1] = { kind: "text", content };
              return next;
            }
            return [...prev, { kind: "text", content }];
          });
        },
        onToolCall: (tc) => {
          toolBuf.push(tc);
          if (tc.name === "create_workflow" && tc.input?.name) {
            props.onWorkflowCreated(tc.input.name);
          }
          const calls = [...toolBuf];
          setEntries((prev) => {
            const last = prev[prev.length - 1];
            if (last && last.kind === "tool") {
              const next = [...prev];
              next[next.length - 1] = { kind: "tool", calls };
              return next;
            }
            return [...prev, { kind: "tool", calls }];
          });
        },
        onStepFinish: () => {
          // Reset buffers so next step starts a fresh bubble
          textBuf = "";
          toolBuf = [];
        },
        onFinish: () => {
          setLoading(false);
        },
      });
    } catch {
      setEntries((prev) => [...prev, { kind: "text", content: "Error connecting to AI." }]);
      setLoading(false);
    }
  }, [input, loading, entries, props.onWorkflowCreated]);

  const handleKeyDown = (e: KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      send();
    }
  };

  return (
    <div class="flyout chat-flyout">
      <div class="flyout-header">
        <div>
          <div class="flyout-eyebrow">AI Builder</div>
          <div class="flyout-title">Create Workflow</div>
        </div>
        <button class="flyout-close" onClick={props.onClose}>x</button>
      </div>
      <div class="chat-messages" ref={scrollRef}>
        {entries.length === 0 && (
          <div class="chat-empty">Describe the workflow you want to build.</div>
        )}
        {entries.map((entry, i) => {
          if (entry.kind === "user") {
            return (
              <div key={i} class="chat-msg chat-msg-user">
                <div class="chat-msg-text">{entry.content}</div>
              </div>
            );
          }
          if (entry.kind === "tool") {
            return (
              <div key={i} class="chat-tool-calls">
                {entry.calls.map((tc, j) => (
                  <div key={j} class="chat-tool-call">
                    <span class="chat-tool-name">{tc.name}</span>
                    <pre class="chat-tool-input">{formatJson(tc.input)}</pre>
                  </div>
                ))}
              </div>
            );
          }
          // kind === "text"
          return (
            <div key={i} class="chat-msg chat-msg-assistant">
              <div class="chat-msg-text">{entry.content}</div>
            </div>
          );
        })}
        {loading && (entries.length === 0 || entries[entries.length - 1]?.kind === "user") && (
          <div class="chat-msg chat-msg-assistant">
            <div class="chat-msg-text chat-thinking">Thinking...</div>
          </div>
        )}
      </div>
      <div class="chat-input-row">
        <input
          ref={inputRef}
          type="text"
          value={input}
          onInput={(e) => setInput((e.target as HTMLInputElement).value)}
          onKeyDown={handleKeyDown}
          placeholder="Describe your workflow..."
          disabled={loading}
        />
        <button class="btn btn-primary" onClick={send} disabled={loading}>Send</button>
      </div>
    </div>
  );
}

// ── Helpers ────────────────────────────────────────────────────────────────

/** Search if/loop configs for a nested step by id. */
function findNestedStep(steps: StepData[], id: string): StepData | null {
  for (const s of steps) {
    if (s.type === "if") {
      const t = s.config.then as StepData | undefined;
      const e = s.config.else as StepData | undefined;
      if (t?.id === id) return t;
      if (e?.id === id) return e;
    }
    if (s.type === "loop") {
      const b = s.config.body as StepData | undefined;
      if (b?.id === id) return b;
    }
  }
  return null;
}

function statusTone(s: string) { return s === "success" ? "ok" : s === "error" ? "danger" : "warning"; }
function eventTone(t: string) { return t.includes("error") ? "error" : t.includes("end") ? "end" : t.includes("start") ? "start" : t.includes("retry") ? "retry" : "other"; }
function formatJson(v: unknown): string {
  if (typeof v === "string") return v;
  try { return JSON.stringify(v, null, 2); } catch { return String(v); }
}
