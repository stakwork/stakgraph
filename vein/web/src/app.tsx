import { useState, useEffect, useCallback, useMemo, useRef } from "preact/hooks";
import { SystemCanvas } from "system-canvas-react";
import type { CanvasData, CanvasNode, CanvasEdge } from "system-canvas";
import type { AddNodeButtonRenderProps } from "system-canvas-react";
import yaml from "js-yaml";
import * as api from "./api";
import { flowToCanvas, stepWorkflow, veinTheme } from "./flow-to-canvas";
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
import { ParamsFlyout } from "./components/ParamsFlyout";
import { PromoteFlyout } from "./components/PromoteFlyout";
import { RunInputPopover, deriveInputBindings } from "./components/RunInputPopover";

// A nested run-execution the user has drilled into. `pathPrefix` is the
// original event-path prefix this child lives under (e.g. `wf/subflowId`),
// `workflow` is the child workflow name, `steps` its loaded definition.
// For a foreach/loop the body runs N times under `<pathPrefix>#<i>`, so the
// frame carries the iteration `count` and the selected `iter`.
interface DrillFrame {
  pathPrefix: string;
  workflow: string;
  steps: StepData[];
  iterations?: number;
  iter?: number;
}

// The actual event-path prefix for a frame, accounting for the selected
// foreach/loop iteration.
function framePrefix(f: DrillFrame): string {
  return f.iterations != null ? `${f.pathPrefix}#${f.iter ?? 0}` : f.pathPrefix;
}

// Count foreach/loop iterations from the run events: bodies execute under
// `<prefix>#<i>`, so the count is (max index + 1) seen across all such paths.
function countIterations(events: api.RunEvent[], prefix: string): number {
  let max = -1;
  for (const e of events) {
    if (!e.path.startsWith(prefix + "#")) continue;
    const rest = e.path.slice(prefix.length + 1); // after `#`
    const num = parseInt(rest, 10);
    if (!isNaN(num) && num > max) max = num;
  }
  return max + 1;
}

export function App() {
  const [workflows, setWorkflows] = useState<api.WorkflowEntry[]>([]);
  const [runs, setRuns] = useState<api.RunSummary[]>([]);
  const [selectedWf, setSelectedWf] = useState<string | null>(null);
  const [selectedRun, setSelectedRun] = useState<string | null>(null);
  const [events, setEvents] = useState<api.RunEvent[]>([]);
  const [running, setRunning] = useState(false);
  // True while an in-tab launched run is streaming live into `events`. Disarmed
  // when the user navigates into another run, so the stream can't clobber it.
  const liveStreamRef = useRef(false);
  const [loadError, setLoadError] = useState(false);
  // Run-view drill-down stack: each frame is a nested child execution. Each
  // carries the original event-path prefix it lives under, the child workflow
  // name, and that workflow's steps. Empty = viewing the root.
  const [runDrill, setRunDrill] = useState<DrillFrame[]>([]);
  const [showCreate, setShowCreate] = useState(false);
  const [showAddStep, setShowAddStep] = useState(false);
  const [stepTypes, setStepTypes] = useState<StepTypeEntry[]>([]);
  const [publishedSteps, setPublishedSteps] = useState<StepData[] | null>(null);
  const [localSteps, setLocalSteps] = useState<StepData[] | null>(null);
  // The workflow `localSteps` were actually loaded for. Until this matches
  // `selectedWf`, the steps belong to the previously-selected workflow (the
  // load is async), so the root canvas must not render with them — otherwise
  // switching workflows briefly mounts/fits the canvas on the OLD content.
  const [loadedWf, setLoadedWf] = useState<string | null>(null);
  const [flyoutStepId, setFlyoutStepId] = useState<string | null>(null);
  const [flyoutStepIndex, setFlyoutStepIndex] = useState<number | null>(null);
  const [showChat, setShowChat] = useState(false);
  const [runBindings, setRunBindings] = useState<ReturnType<typeof deriveInputBindings> | null>(null);
  // The selected workflow's `params` defaults (tunable knobs). `wfParams` is
  // the published baseline; `localParams` is the editable working copy. Editing
  // a param and publishing = a new workflow version (params live in the YAML).
  const [wfParams, setWfParams] = useState<Record<string, unknown> | null>(null);
  const [localParams, setLocalParams] = useState<Record<string, unknown> | null>(null);
  // Whether the Params flyout (editable) is open.
  const [showParams, setShowParams] = useState(false);
  // Declared promotions resolved against the selected run's output (the
  // "promote a winner" review surface) + whether its flyout is open.
  const [promotions, setPromotions] = useState<api.Promotion[]>([]);
  const [showPromote, setShowPromote] = useState(false);
  // Whether every structured param currently parses (the flyout reports this);
  // an invalid YAML param blocks Publish while the flyout is open.
  const [paramsValid, setParamsValid] = useState(true);

  const isDirty = useMemo(() => {
    if (!publishedSteps || !localSteps) return false;
    const stepsDirty = !deepEqual(normalizeSteps(publishedSteps), normalizeSteps(localSteps));
    const paramsDirty = !deepEqual(wfParams ?? {}, localParams ?? {});
    return stepsDirty || paramsDirty;
  }, [publishedSteps, localSteps, wfParams, localParams]);

  const activeVersion = workflows.find((w) => w.name === selectedWf)?.activeVersion;

  // Runs are already filtered by workflow (fetched per-workflow)
  const filteredRuns = runs;

  const isRunView = selectedRun != null && events.length > 0;

  // ── View context: what the canvas + flyout operate on ───────────────────
  // Root view = the selected workflow. When drilled into a run's nested
  // execution, the deepest frame's child workflow + its re-keyed events.
  const drill = runDrill.length > 0 ? runDrill[runDrill.length - 1]! : null;
  const viewWorkflow = drill ? drill.workflow : selectedWf;
  const viewSteps = drill ? drill.steps : localSteps;

  // Re-key the run events into the drilled child's namespace: events under
  // `<prefix>/...` become `<childWorkflow>/...` so the existing path-based
  // status overlay + flyout lookups work unchanged at any depth.
  const viewEvents = useMemo(() => {
    if (!drill) return events;
    const prefix = framePrefix(drill);
    const { workflow } = drill;
    const out: api.RunEvent[] = [];
    for (const e of events) {
      if (e.path === prefix || e.path.startsWith(prefix + "/")) {
        out.push({ ...e, path: workflow + e.path.slice(prefix.length) });
      }
    }
    return out;
  }, [drill, events]);

  // Canvas is derived from the current view (root or drilled child). Run/live
  // events overlay status; in plain edit mode there are no events.
  const canvas = useMemo<CanvasData | null>(() => {
    if (loadError && selectedWf) {
      return {
        nodes: [{ id: "err", type: "text", text: `${selectedWf}\n\nCould not load`, x: 100, y: 100, width: 260, height: 70, color: "1" }],
        edges: [], theme: { base: "midnight" },
      };
    }
    if (!viewWorkflow || !viewSteps) return null;
    // Root view: don't render until the steps belong to the selected workflow.
    // (Drill frames carry their own already-loaded steps, so they're exempt.)
    if (runDrill.length === 0 && loadedWf !== selectedWf) return null;
    const overlay = events.length > 0 || running;
    return flowToCanvas(
      { name: viewWorkflow, steps: viewSteps },
      overlay ? (viewEvents as RunEventData[]) : undefined,
    );
  }, [loadError, selectedWf, loadedWf, runDrill.length, viewWorkflow, viewSteps, viewEvents, events.length, running]);

  // Events for the flyout step (within the current view). Containers show
  // their own aggregate I/O here (subflow → child input/output; foreach →
  // items array / results array); per-child detail is via the drill arrow.
  const flyoutEvents = useMemo(() => {
    if (!flyoutStepId || !viewWorkflow || viewEvents.length === 0) return null;
    const path = `${viewWorkflow}/${flyoutStepId}`;
    const stepEvts = viewEvents.filter((e) => e.path === path);
    if (stepEvts.length === 0) return null;
    const start = stepEvts.find((e) => e.type === "step.start");
    const end = stepEvts.find((e) => e.type === "step.end");
    const error = stepEvts.find((e) => e.type === "step.error");
    const skipped = stepEvts.find((e) => e.type === "step.skipped");
    return { start, end, error, skipped, all: stepEvts };
  }, [flyoutStepId, viewWorkflow, viewEvents]);

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
      // The registry (`resp.core`) already includes disk-discovered custom
      // steps tagged with their real source, so `resp.workspace` overlaps it.
      // Dedupe by type (keying off the registry), folding in workspace
      // descriptions, so each step appears once in the Add Step picker.
      const descByType = new Map(resp.workspace.map((s) => [s.type, s.description]));
      const entries: StepTypeEntry[] = resp.core.map((s) => ({
        type: s.type,
        source: s.source as "core" | "lib" | "custom",
        description: descByType.get(s.type),
      }));
      setStepTypes(entries);
    } catch { /* ignore */ }
  }, []);

  useEffect(() => { refreshWorkflows(); refreshStepTypes(); }, []);

  // Load workflow + its runs when selected
  useEffect(() => {
    setRunDrill([]);
    setLoadError(false);
    setShowParams(false);
    if (!selectedWf) {
      setPublishedSteps(null);
      setLocalSteps(null);
      setLoadedWf(null);
      setWfParams(null);
      setLocalParams(null);
      setRuns([]);
      setFlyoutStepId(null);
      setFlyoutStepIndex(null);
      return;
    }
    api.getWorkflowFlow(selectedWf).then((flow) => {
      const steps = flow.steps as StepData[];
      setPublishedSteps(steps);
      setLocalSteps(steps);
      setLoadedWf(selectedWf);
      setWfParams(flow.params ?? null);
      setLocalParams(flow.params ?? null);
    }).catch(() => {
      setPublishedSteps(null);
      setLocalSteps(null);
      setLoadedWf(selectedWf);
      setLoadError(true);
    });
    refreshRuns(selectedWf);
  }, [selectedWf]);

  // Load run events when a run is selected; clear overlay + drill when deselected
  useEffect(() => {
    setRunDrill([]);
    setFlyoutStepId(null);
    setFlyoutStepIndex(null);
    if (!selectedRun || !selectedWf) {
      setEvents([]);
      return;
    }
    api.getRunEvents(selectedWf, selectedRun).then((evts) => {
      setEvents(evts);
    }).catch(console.error);
  }, [selectedRun]);

  // Resolve the selected run's declared promotions (a winning value → a target
  // param). Drives the topbar Promote button + flyout. Empty unless the
  // workflow declares `promotes` and the run output resolves them.
  useEffect(() => {
    setShowPromote(false);
    setPromotions([]);
    if (!selectedRun || !selectedWf) return;
    api.getPromotions(selectedWf, selectedRun)
      .then(setPromotions)
      .catch(() => setPromotions([]));
  }, [selectedWf, selectedRun]);

  // A container node's ref arrow means "open this workflow" — go-to-definition.
  // We switch the selected workflow rather than drilling into an inline
  // sub-canvas (a subflow target is a standalone, separately-editable workflow).
  // In run view it instead drills into that node's nested execution.
  const handleNavigate = useCallback((ref: string) => {
    if (!ref.startsWith("wf:")) return;
    const name = ref.slice(3);
    if (isRunView) {
      // Run view: the arrow drills into this node's nested execution.
      const step = (viewSteps ?? []).find((s) => stepWorkflow(s) === name);
      if (step) drillIntoStep(step);
      return;
    }
    if (name === selectedWf) return;
    setSelectedWf(name);
    setSelectedRun(null);
    setEvents([]);
    setFlyoutStepId(null);
    setFlyoutStepIndex(null);
  }, [selectedWf, isRunView, viewSteps]);

  // Drill into a container step's nested execution (run view). Loads the child
  // workflow definition and pushes a frame; events are re-keyed by `viewEvents`.
  // foreach/loop bodies run N times under `<prefix>#<i>`, so we count the
  // iterations and start at #0.
  const drillIntoStep = useCallback((step: StepData) => {
    const childName = stepWorkflow(step);
    if (!childName) return;
    const base = runDrill.length > 0 ? framePrefix(runDrill[runDrill.length - 1]!) : (selectedWf ?? "");
    const pathPrefix = `${base}/${step.id}`;
    const isIter = step.type === "foreach" || step.type === "loop";
    api.getWorkflowFlow(childName).then((flow) => {
      const frame: DrillFrame = { pathPrefix, workflow: childName, steps: flow.steps as StepData[] };
      if (isIter) {
        frame.iterations = countIterations(events, pathPrefix);
        frame.iter = 0;
      }
      setRunDrill((prev) => [...prev, frame]);
      setFlyoutStepId(null);
      setFlyoutStepIndex(null);
    }).catch(console.error);
  }, [runDrill, selectedWf, events]);

  // Select a different iteration on the deepest (foreach/loop) drill frame.
  const setDrillIter = useCallback((iter: number) => {
    setRunDrill((prev) => {
      if (prev.length === 0) return prev;
      const next = [...prev];
      next[next.length - 1] = { ...next[next.length - 1]!, iter };
      return next;
    });
    setFlyoutStepId(null);
    setFlyoutStepIndex(null);
  }, []);

  function updateLocalSteps(steps: StepData[]) {
    setLocalSteps(steps);
  }

  const submitRun = useCallback(async (
    input: Record<string, unknown>,
    params?: Record<string, unknown>,
  ) => {
    if (!selectedWf || !localSteps) return;
    const wf = selectedWf;
    const accumulated: RunEventData[] = [];

    setRunBindings(null);
    setRunDrill([]);
    setRunning(true);
    // Empty events + running → pending status overlay on all nodes immediately.
    setEvents([] as api.RunEvent[]);
    liveStreamRef.current = true;

    try {
      // Detached launch (§8): we get the runId up front, so the run can surface
      // in the sidebar as "running" while it's still executing. We don't select
      // it yet — selecting fires an effect that refetches (and would clobber)
      // the live events; we select on completion, as before.
      const { runId } = await api.launchWorkflow(wf, input, params);
      let listedRunning = false;
      await api.streamRun(wf, runId, (event) => {
        // If the user navigated into another run mid-stream (e.g. opened a
        // generation's eval run), stop pushing this run's events over theirs.
        if (!liveStreamRef.current) return;
        accumulated.push(event as RunEventData);
        setEvents([...accumulated] as api.RunEvent[]);
        // First event means the run's log exists on disk → `/runs` now returns
        // it (status "running", no run.json yet). Refresh once to show it live.
        if (!listedRunning) {
          listedRunning = true;
          refreshRuns(wf);
        }
      });

      if (liveStreamRef.current) setSelectedRun(runId);
      await refreshRuns(wf);
    } finally {
      liveStreamRef.current = false;
      setRunning(false);
    }
  }, [selectedWf, localSteps, refreshRuns]);

  // Navigate into another workflow's run (from an Events-panel run-ref link).
  // Disarm any in-tab live stream first so it doesn't clobber the target run's
  // events (or yank us back when it finishes).
  const openRun = useCallback((wf: string, runId: string) => {
    liveStreamRef.current = false;
    setRunning(false);
    closeFlyout();
    if (selectedWf !== wf) setSelectedWf(wf);
    setSelectedRun(runId);
  }, [selectedWf]);

  const handleRun = useCallback(async () => {
    if (!selectedWf || !localSteps || localSteps.length === 0) return;
    const first = localSteps[0]!;
    try {
      const { fields } = await api.getStepSchema(first.type);
      const bindings = deriveInputBindings(first, fields);
      const hasParams = localParams != null && Object.keys(localParams).length > 0;
      if (bindings.length === 0 && !hasParams) {
        await submitRun({});
      } else {
        setRunBindings(bindings);
      }
    } catch {
      // If we can't load the schema, fall back to running with no input.
      await submitRun({});
    }
  }, [selectedWf, localSteps, submitRun, localParams]);

  const handleCreate = useCallback(async (name: string, yamlStr: string, desc: string) => {
    // Server auto-suffixes on collision; navigate to the resolved name.
    const res = await api.createWorkflowYaml(name, yamlStr, desc || undefined);
    await refreshWorkflows();
    setSelectedWf(res.workflow);
    setShowCreate(false);
  }, [refreshWorkflows]);

  const handlePublish = useCallback(async () => {
    if (!selectedWf || !localSteps || !activeVersion) return;
    const num = parseInt(activeVersion.replace(/^v/, ""), 10);
    const nextVersion = `v${(isNaN(num) ? 1 : num) + 1}`;
    const hasParams = localParams != null && Object.keys(localParams).length > 0;
    const yamlStr = yaml.dump(
      { name: selectedWf, steps: localSteps, ...(hasParams ? { params: localParams } : {}) },
      { lineWidth: 120, noRefs: true },
    );
    await api.publishWorkflowYaml(selectedWf, nextVersion, yamlStr);
    await refreshWorkflows();
    const flow = await api.getWorkflowFlow(selectedWf);
    const steps = flow.steps as StepData[];
    setPublishedSteps(steps);
    setLocalSteps(steps);
    setWfParams(flow.params ?? null);
    setLocalParams(flow.params ?? null);
  }, [selectedWf, localSteps, localParams, activeVersion, refreshWorkflows]);

  // Clicking a node (body) always opens its flyout — leaf I/O, or a
  // container's aggregate I/O. Drilling into children is the arrow's job.
  const handleNodeClick = useCallback((node: CanvasNode) => {
    const stepId = node.customData?.stepId as string | undefined;
    const stepIndex = node.customData?.stepIndex as number | undefined;
    if (stepId == null) return;
    setShowParams(false);
    setShowPromote(false);
    setFlyoutStepId(stepId);
    setFlyoutStepIndex(stepIndex ?? null);
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
                  <span class="list-item-name">{run.runId.slice(0, 10)}</span>
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
          {selectedRun && <span class="topbar-run">{selectedRun.slice(0, 10)}</span>}
          {isDirty && <span class="dirty-dot" style="margin-left:8px;" />}
        </span>
        <div class="topbar-actions">
          {isDirty && <button class="btn btn-publish" disabled={showParams && !paramsValid} onClick={handlePublish}>Publish</button>}
          {isRunView && promotions.length > 0 && (
            <button
              class={`btn${showPromote ? " is-active" : ""}`}
              onClick={() => { setShowPromote((s) => !s); setShowParams(false); closeFlyout(); }}
            >Promote</button>
          )}
          {selectedWf && localParams && Object.keys(localParams).length > 0 && (
            <button
              class={`btn${showParams ? " is-active" : ""}`}
              onClick={() => { setShowParams((s) => !s); setShowPromote(false); setFlyoutStepId(null); }}
            >Params</button>
          )}
          {selectedWf && (
            <div class="run-anchor">
              <button class="btn btn-primary" onClick={handleRun}>Run</button>
              {runBindings && selectedWf && (
                <RunInputPopover
                  workflow={selectedWf}
                  bindings={runBindings}
                  params={localParams}
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
        {runDrill.length > 0 && (
          <div class="drill-bar">
            <button class="drill-crumb" onClick={() => setRunDrill([])}>{selectedWf}</button>
            {runDrill.map((f, i) => (
              <span key={f.pathPrefix}>
                <span class="drill-sep">›</span>
                <button
                  class={`drill-crumb${i === runDrill.length - 1 ? " is-active" : ""}`}
                  onClick={() => setRunDrill(runDrill.slice(0, i + 1))}
                >{f.workflow}</button>
              </span>
            ))}
            {drill?.iterations != null && drill.iterations > 0 && (
              <select
                class="drill-iter"
                value={String(drill.iter ?? 0)}
                onChange={(e) => setDrillIter(parseInt((e.target as HTMLSelectElement).value, 10))}
              >
                {Array.from({ length: drill.iterations }, (_, i) => (
                  <option key={i} value={String(i)}>iteration #{i}</option>
                ))}
              </select>
            )}
          </div>
        )}
        {canvas
          ? <SystemCanvas
              // Remount when the viewed workflow changes so the canvas
              // re-fits/re-centers on its content (the library's
              // `autoFit="canvas-change"` only fires on sub-canvas
              // navigation, not when we swap the root `canvas` prop).
              key={viewWorkflow ?? "none"}
              panMode="trackpad"
              canvas={canvas}
              // A container node's ref arrow opens the referenced workflow
              // (go-to-definition) instead of drilling into an inline
              // sub-canvas — onNavigate routes via the sidebar selection.
              externalNavigation
              onNavigate={handleNavigate}
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
      {events.length > 0 && <EventsPanel events={events} onOpenRun={openRun} />}

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
          }}
          onWorkflowRan={async (name, runId) => {
            // Make sure the run's workflow is selected, then surface the new run.
            if (selectedWf !== name) setSelectedWf(name);
            await refreshRuns(name);
            setSelectedRun(runId);
          }}
        />
      )}

      {/* Params flyout — edit the workflow's tunable knobs; Publish persists
          them as a new version (params live in the workflow YAML). */}
      {showParams && selectedWf && localParams && (
        <ParamsFlyout
          key={selectedWf}
          workflow={selectedWf}
          params={localParams}
          onChange={setLocalParams}
          onValidChange={setParamsValid}
          onClose={() => setShowParams(false)}
        />
      )}

      {/* Promote flyout — review + apply a run's declared promotions. */}
      {showPromote && selectedWf && selectedRun && promotions.length > 0 && (
        <PromoteFlyout
          key={`${selectedWf}/${selectedRun}`}
          workflow={selectedWf}
          runId={selectedRun}
          promotions={promotions}
          onClose={() => setShowPromote(false)}
          onPromoted={() => { refreshWorkflows(); }}
        />
      )}

      {/* Step flyout — operates on the current view (root or drilled child) */}
      {flyoutStepId != null && viewSteps && flyoutStepIndex != null && (() => {
        const step = viewSteps[flyoutStepIndex];
        if (!step) return null;

        if (isRunView && flyoutEvents) {
          return <StepRunFlyout step={step} events={flyoutEvents} onClose={closeFlyout} />;
        }
        // Editing only applies at the root (drilled views are read-only run views).
        if (drill || !localSteps) return null;
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
