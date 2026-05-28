import { useState, useEffect, useCallback, useMemo, useRef } from "preact/hooks";
import { SystemCanvas } from "system-canvas-react";
import type { CanvasData, CanvasNode, CanvasEdge } from "system-canvas";
import type { AddNodeButtonRenderProps } from "system-canvas-react";
import yaml from "js-yaml";
import * as api from "./api";
import { flowToCanvas, veinTheme } from "./flow-to-canvas";
import type { StepData, RunEventData } from "./flow-to-canvas";
import "./styles/base.css";
import "./styles/components.css";
import { deepEqual, normalizeSteps, statusTone } from "./helpers";
import { ChatFlyout } from "./components/ChatFlyout";
import { CreateDialog } from "./components/CreateDialog";
import { AddStepDialog, StepTypeEntry } from "./components/AddStepDialog";
import { StepEditFlyout } from "./components/StepEditFlyout";
import { EventsPanel } from "./components/EventsPanel";
import { EventsResizer } from "./components/EventsResizer";
import { StepRunFlyout } from "./components/StepRunFlyout";
import { RunInputPopover, deriveInputBindings } from "./components/RunInputPopover";

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
  const [showChat, setShowChat] = useState(false);
  const [runBindings, setRunBindings] = useState<ReturnType<typeof deriveInputBindings> | null>(null);

  const isDirty = useMemo(() => {
    if (!publishedSteps || !localSteps) return false;
    return !deepEqual(normalizeSteps(publishedSteps), normalizeSteps(localSteps));
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
    const skipped = stepEvts.find((e) => e.type === "step.skipped");
    return { start, end, error, skipped, all: stepEvts };
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

  const submitRun = useCallback(async (input: Record<string, unknown>) => {
    if (!selectedWf || !localSteps) return;
    const wf = selectedWf;
    const steps = localSteps;
    const accumulated: RunEventData[] = [];

    setRunBindings(null);
    // Show pending status on all nodes immediately
    rebuildCanvas(wf, steps, []);

    const result = await api.runWorkflow(wf, input, (event) => {
      accumulated.push(event as RunEventData);
      rebuildCanvas(wf, steps, [...accumulated]);
      setEvents([...accumulated] as api.RunEvent[]);
    });

    if (result?.runId) {
      setSelectedRun(result.runId);
    }
    await refreshRuns(wf);
  }, [selectedWf, localSteps, refreshRuns]);

  const handleRun = useCallback(async () => {
    if (!selectedWf || !localSteps || localSteps.length === 0) return;
    const first = localSteps[0]!;
    try {
      const { fields } = await api.getStepSchema(first.type);
      const bindings = deriveInputBindings(first, fields);
      if (bindings.length === 0) {
        await submitRun({});
      } else {
        setRunBindings(bindings);
      }
    } catch {
      // If we can't load the schema, fall back to running with no input.
      await submitRun({});
    }
  }, [selectedWf, localSteps, submitRun]);

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

  const closeFlyout = () => { setFlyoutStepId(null); setFlyoutStepIndex(null); };

  return (
    <div class="shell">
      {/* Sidebar */}
      <div class="shell-sidebar">
        <div class="sidebar-brand"><span class="brand-dot" /> vein</div>

        <div class="sidebar-section">
          <div class="section-title">
            Workflows
            <button class="btn" style="float:right;padding:1px 8px;font-size:11px;margin-top:-3px;" onClick={() => setShowCreate(true)}>+</button>
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
          {selectedWf && (
            <div class="run-anchor">
              <button class="btn btn-primary" onClick={handleRun}>Run</button>
              {runBindings && selectedWf && (
                <RunInputPopover
                  workflow={selectedWf}
                  bindings={runBindings}
                  onSubmit={submitRun}
                  onClose={() => setRunBindings(null)}
                />
              )}
            </div>
          )}
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
              showNodeToolbar={false}
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

      {/* Events panel + resize handle */}
      {events.length > 0 && <EventsResizer />}
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
      {flyoutStepId != null && localSteps && flyoutStepIndex != null && (() => {
        const step = localSteps[flyoutStepIndex];
        if (!step) return null;

        if (isRunView && flyoutEvents) {
          return <StepRunFlyout step={step} events={flyoutEvents} onClose={closeFlyout} />;
        }
        return (
          <StepEditFlyout
            step={step}
            allSteps={localSteps}
            onSave={(updated) => handleStepSave(flyoutStepIndex, updated)}
            onClose={closeFlyout}
          />
        );
      })()}
    </div>
  );
}
