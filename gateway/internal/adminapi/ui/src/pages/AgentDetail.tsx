// AgentDetail — one agent's cost histogram, budget card and recent
// runs, plus the catalog tabs (prompts / tools / skills / evals) and
// the swarm-wide kill switch.
//
// Data flow
// --------
// The chart is `useHistogramCost` scoped server-side with
// `agent_name=<name>` (observability.go `metadataFilterFromQuery`),
// so the response carries only this agent's series. The "Recent
// runs" table is `useAgentRuns` → `/agents/:name/runs`: one row per
// run in the window, newest activity first, with the user, the
// model(s), spend and call count. Both follow the page's window.

import { useState } from "preact/hooks";
import { Link } from "wouter-preact";

import { CostHistogram } from "../components/charts/CostHistogram";
import { ErrorBoundary } from "../components/ErrorBoundary";
import { WindowPicker } from "../components/controls/WindowPicker";
import { KillConfirmModal } from "../components/KillConfirmModal";
import { StatusBadge, deriveAgentStatus } from "../components/StatusBadge";
import { StopIcon, UserIcon } from "../components/icons";
import type {
  AgentBudgetResponse,
  AgentCatalogResponse,
  AgentStateResponse,
  CatalogPrompt,
  CatalogSkill,
  CatalogTool,
} from "../api/types";
import { ApiCallError, getErrorMessage } from "../api/client";
import {
  useAgentBudget,
  useAgentCatalog,
  useAgentEvals,
  useAgentRuns,
  useAgentState,
  useHistogramCost,
  useKillAgent,
  useToggleSkill,
  useToggleTool,
  useUnkillAgent,
} from "../api/queries";
import type { Window } from "../api/manual";
import { windowToSeconds } from "../api/window";
import { EvalsView } from "./EvalsView";

interface Props {
  name: string;
}

type Tab = "overview" | "prompts" | "tools" | "skills" | "evals";

const fmtUSD = (v: number) => {
  if (v === 0) return "$0.00";
  const digits = Math.abs(v) < 0.01 ? 6 : 2;
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  }).format(v);
};

const fmtInt = (v: number) =>
  new Intl.NumberFormat("en-US").format(Math.round(v));

const fmtTs = (s?: string) => {
  if (!s) return "";
  try {
    return new Date(s).toLocaleString();
  } catch {
    return s;
  }
};

// "3m ago" for the runs table. Same helper as UserDetail's — the
// pages keep their own copies by convention (see AGENTS.md).
function fmtRelative(absISO?: string): string {
  if (!absISO) return "";
  try {
    const then = new Date(absISO).getTime();
    const sec = Math.round((Date.now() - then) / 1000);
    if (sec < 60) return `${sec}s ago`;
    const min = Math.round(sec / 60);
    if (min < 60) return `${min}m ago`;
    const hr = Math.round(min / 60);
    if (hr < 48) return `${hr}h ago`;
    const day = Math.round(hr / 24);
    return `${day}d ago`;
  } catch {
    return "";
  }
}

export function AgentDetail({ name }: Props) {
  const [window, setWindow] = useState<Window>("24h");
  const bucket = window === "1h" ? "5m" : window === "6h" ? "10m" : "1h";
  const windowSeconds = windowToSeconds(window);

  const budget = useAgentBudget(name);
  const histogram = useHistogramCost({
    window,
    bucket,
    dimension: "agent-name",
    agentName: name,
  });
  const runs = useAgentRuns(name, window);

  // Server-filtered, so every series is this agent's; summing all
  // points is the window spend.
  const totalCost =
    histogram.data?.series
      .flatMap((s) => s.points)
      .reduce((acc, p) => acc + p.cost, 0) ?? 0;

  const [tab, setTab] = useState<Tab>("overview");
  const catalog = useAgentCatalog(name);
  const evals = useAgentEvals(name);
  // Redis hot state: kill flag + current-bucket spend vs cap. `null`
  // when the swarm has no Redis (badge hidden, switch disabled).
  const state = useAgentState(name);
  const agentStatus = state.data ? deriveAgentStatus(state.data) : null;
  // 503 ⇒ neo4j not wired on this swarm: the catalog tabs render a
  // "not configured" notice rather than an error banner.
  const catalogUnavailable =
    catalog.error instanceof ApiCallError &&
    catalog.error.code === "catalog_unavailable";
  const cat = catalog.data ?? null;

  return (
    <>
      <div class="page-header">
        <div>
          <div class="crumbs">
            <Link href="/agents">Agents</Link> / {name}
          </div>
          <h1 class="mono">
            {name}
            {cat?.default_model ? (
              <span class="pill pill-accent model-chip" title="Default model">
                {cat.default_model}
              </span>
            ) : null}
            {agentStatus ? <StatusBadge status={agentStatus} /> : null}
          </h1>
        </div>
        <div class="page-actions">
          {tab === "overview" ? (
            <WindowPicker value={window} onChange={setWindow} />
          ) : null}
          <KillAgentSwitch name={name} state={state.data} />
        </div>
      </div>

      <nav class="tabs" role="tablist">
        <TabButton id="overview" active={tab} onSelect={setTab} label="Overview" />
        <TabButton
          id="prompts"
          active={tab}
          onSelect={setTab}
          label="Prompts"
          count={cat?.prompts.length}
        />
        <TabButton
          id="tools"
          active={tab}
          onSelect={setTab}
          label="Tools"
          count={cat?.tools.length}
        />
        <TabButton
          id="skills"
          active={tab}
          onSelect={setTab}
          label="Skills"
          count={cat?.skills.length}
        />
        <TabButton
          id="evals"
          active={tab}
          onSelect={setTab}
          label="Evals"
          count={evals.data?.sets.length}
        />
      </nav>

      {tab === "evals" ? (
        <EvalsView agentName={name} />
      ) : tab !== "overview" ? (
        <CatalogPanel
          tab={tab}
          catalog={cat}
          loading={catalog.isLoading}
          unavailable={catalogUnavailable}
          error={
            catalog.isError && !catalogUnavailable
              ? getErrorMessage(catalog.error)
              : null
          }
        />
      ) : (
        <OverviewTab
          window={window}
          bucket={bucket}
          totalCost={totalCost}
          budget={budget.data}
          histogram={histogram}
          windowSeconds={windowSeconds}
          runs={runs}
        />
      )}
    </>
  );
}

interface OverviewProps {
  window: Window;
  bucket: string;
  totalCost: number;
  budget: AgentBudgetResponse | undefined;
  histogram: ReturnType<typeof useHistogramCost>;
  windowSeconds: number;
  runs: ReturnType<typeof useAgentRuns>;
}

function OverviewTab({
  window,
  bucket,
  totalCost,
  budget,
  histogram,
  windowSeconds,
  runs,
}: OverviewProps) {
  return (
    <>
      <div class="kpi">
        <div class="card kpi-card">
          <div class="kpi-label">Spend ({window})</div>
          <div class="kpi-value">{fmtUSD(totalCost)}</div>
        </div>
      </div>

      {budget && budget.cap_usd != null ? (
        <BudgetCard data={budget} />
      ) : null}

      <section class="chart-frame">
        <div class="card-header">
          <div class="card-title">Cost over time</div>
          <div class="text-dim mono" style="font-size: 11px">
            bucket: {bucket}
          </div>
        </div>
        {histogram.isError ? (
          <div class="error-banner">{getErrorMessage(histogram.error)}</div>
        ) : histogram.data ? (
          <ErrorBoundary>
            <CostHistogram data={histogram.data} windowSeconds={windowSeconds} />
          </ErrorBoundary>
        ) : (
          <div class="loading">Loading…</div>
        )}
      </section>

      <section>
        <h2 style="margin-bottom: 16px">Recent runs</h2>
        {runs.isError ? (
          <div class="error-banner">{getErrorMessage(runs.error)}</div>
        ) : !runs.data ? (
          <div class="loading">Loading…</div>
        ) : runs.data.runs.length === 0 ? (
          <div class="empty">No runs in this window.</div>
        ) : (
          <div class="table-wrap">
            <table class="table">
              <thead>
                <tr>
                  <th>Run</th>
                  <th>User</th>
                  <th>Model</th>
                  <th class="num">Spend</th>
                  <th class="num">Calls</th>
                  <th class="cell-when">Last call</th>
                </tr>
              </thead>
              <tbody>
                {runs.data.runs.map((r) => (
                  <tr key={r.run_id} class="row-link">
                    <td class="cell-trunc" title={r.run_id}>
                      <Link href={`/runs/${encodeURIComponent(r.run_id)}`}>
                        <span class="mono">{r.run_id}</span>
                      </Link>
                    </td>
                    <td>
                      {r.user_id ? (
                        <span class="prov-with-icon" title={r.user_id}>
                          <UserIcon class="prov-icon" />
                          <Link href={`/people/${encodeURIComponent(r.user_id)}`}>
                            <span class="mono">{r.user_id.slice(0, 8)}</span>
                          </Link>
                        </span>
                      ) : (
                        <span class="text-dim">—</span>
                      )}
                    </td>
                    <td>
                      <ModelCell models={r.models} />
                    </td>
                    <td class="num">{fmtUSD(r.total_cost)}</td>
                    <td class="num">{fmtInt(r.request_count)}</td>
                    <td class="text-dim cell-when" title={fmtTs(r.last_seen)}>
                      {fmtRelative(r.last_seen) || fmtTs(r.last_seen)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            {runs.data.total > runs.data.runs.length ? (
              <div class="table-foot text-dim">
                Showing the {runs.data.runs.length} most recent of{" "}
                {runs.data.total} runs in this window.
              </div>
            ) : null}
          </div>
        )}
      </section>
    </>
  );
}

// ModelCell shows the run's most-used model, plus a "+N" pill when
// the run touched more than one (the others ride on the pill's
// title). Empty when no row recorded a model — a run of pure
// failures can look like that.
function ModelCell({ models }: { models: string[] }) {
  if (models.length === 0) return <span class="text-dim">—</span>;
  const [primary, ...rest] = models;
  return (
    <span class="model-cell">
      <span class="mono">{primary}</span>
      {rest.length > 0 ? (
        <span class="pill" title={"also " + rest.join(", ")}>
          +{rest.length}
        </span>
      ) : null}
    </span>
  );
}

// BudgetCard renders the configured cap, the live spend against it,
// remaining headroom, and a coloured progress bar. The same data is
// summarised inline on the Agents list; here it gets the full
// dashboard treatment because operators land on this page to decide
// "is this agent about to be cut off?".
function BudgetCard({ data }: { data: AgentBudgetResponse }) {
  const cap = data.cap_usd ?? 0;
  const spent = data.spent_usd;
  const remaining = data.remaining_usd ?? 0;
  const ratio = data.ratio ?? 0;
  const pct = Math.min(100, Math.max(0, ratio * 100));
  const tone =
    ratio >= 1 ? "danger" : ratio >= 0.8 ? "warning" : "ok";
  return (
    <section class="card budget-card">
      <div class="card-header">
        <div class="card-title">Budget ({data.window})</div>
        <div class="text-dim mono" style="font-size: 11px">
          {data.period_start
            ? "since " + new Date(data.period_start).toLocaleString()
            : ""}
        </div>
      </div>
      <div class="budget-row">
        <div class="budget-figure">
          <div class="budget-figure-label">Spent</div>
          <div class={"budget-figure-val tone-" + tone}>{fmtUSD(spent)}</div>
        </div>
        <div class="budget-figure">
          <div class="budget-figure-label">Cap</div>
          <div class="budget-figure-val">{fmtUSD(cap)}</div>
        </div>
        <div class="budget-figure">
          <div class="budget-figure-label">Remaining</div>
          <div class="budget-figure-val">{fmtUSD(remaining)}</div>
        </div>
        <div class="budget-figure">
          <div class="budget-figure-label">Usage</div>
          <div class={"budget-figure-val tone-" + tone}>{pct.toFixed(1)}%</div>
        </div>
      </div>
      <div class="budget-meter budget-meter-lg">
        <div
          class={`budget-meter-bar tone-${tone}`}
          style={`width:${pct}%`}
        />
      </div>
    </section>
  );
}

// ─── kill switch ───────────────────────────────────────────────────
//
// Kill / Unkill for the whole agent, swarm-wide. `state` is the
// /agents/:name/state snapshot: undefined while loading, null when the
// swarm has no Redis (switch disabled with a tooltip — no hot state
// means no kill keys either). Typed confirmation in the modal because
// the blast radius is every run of this agent, not just the ones on
// this page.
function KillAgentSwitch({
  name,
  state,
}: {
  name: string;
  state: AgentStateResponse | null | undefined;
}) {
  const kill = useKillAgent(name);
  const unkill = useUnkillAgent(name);
  const [modal, setModal] = useState<"kill" | "unkill" | null>(null);
  const active = modal === "kill" ? kill : unkill;
  const unavailable = state === null;

  const open = (which: "kill" | "unkill") => {
    kill.reset();
    unkill.reset();
    setModal(which);
  };

  return (
    <>
      {state?.killed ? (
        <button
          type="button"
          class="btn"
          title="Clear the swarm-wide kill; runs of this agent resume on their next LLM call"
          onClick={() => open("unkill")}
        >
          Unkill agent
        </button>
      ) : (
        <button
          type="button"
          class="btn btn-icon is-danger-solid"
          disabled={!state}
          title={
            unavailable
              ? "Hot state unavailable on this swarm (no Redis) — kill switches are off"
              : "Stop every run of this agent, swarm-wide"
          }
          onClick={() => open("kill")}
        >
          <StopIcon />
          Kill agent
        </button>
      )}
      {modal ? (
        <KillConfirmModal
          target={{ kind: "agent", name }}
          action={modal}
          pending={active.isPending}
          error={active.isError ? getErrorMessage(active.error) : null}
          onConfirm={() => {
            const done = { onSuccess: () => setModal(null) };
            if (modal === "kill") kill.mutate(undefined, done);
            else unkill.mutate(undefined, done);
          }}
          onClose={() => setModal(null)}
        />
      ) : null}
    </>
  );
}

// ─── catalog tabs ──────────────────────────────────────────────────

function TabButton({
  id,
  active,
  onSelect,
  label,
  count,
}: {
  id: Tab;
  active: Tab;
  onSelect: (t: Tab) => void;
  label: string;
  count?: number;
}) {
  return (
    <button
      type="button"
      role="tab"
      aria-selected={active === id}
      class={"tab" + (active === id ? " is-active" : "")}
      onClick={() => onSelect(id)}
    >
      {label}
      {count != null && count > 0 ? <span class="tab-count">{count}</span> : null}
    </button>
  );
}

interface PanelProps {
  tab: Exclude<Tab, "overview">;
  catalog: AgentCatalogResponse | null;
  loading: boolean;
  unavailable: boolean;
  error: string | null;
}

// CatalogPanel handles the shared empty / loading / error chrome for
// the three catalog tabs, then delegates to the per-kind renderer.
function CatalogPanel({ tab, catalog, loading, unavailable, error }: PanelProps) {
  if (unavailable) {
    return (
      <div class="empty">
        Catalog not wired on this swarm. Set <span class="mono">NEO4J_PASSWORD</span>{" "}
        on the gateway to enable prompts, tools and skills.
      </div>
    );
  }
  if (loading) return <div class="loading">Loading…</div>;
  if (error) return <div class="error-banner">{error}</div>;
  if (!catalog) {
    return (
      <div class="empty">
        No catalog for this agent yet. Sources (hive, prompt-manager, goose)
        push manifests as they deploy.
      </div>
    );
  }

  if (tab === "prompts") return <PromptsView prompts={catalog.prompts} />;
  if (tab === "tools") return <ToolsView agentName={catalog.name} tools={catalog.tools} />;
  return <SkillsView agentName={catalog.name} skills={catalog.skills} />;
}

// SourceChip is the source + version provenance marker shared by all
// three kinds — which system contributed this node and at what stamp.
function SourceChip({ source, version }: { source: string; version?: string }) {
  return (
    <span class="pill pill-accent" title={version ? `version ${version}` : undefined}>
      {source}
      {version ? <span class="text-dim">· {version}</span> : null}
    </span>
  );
}

// PromptRoleBadge marks which slot a prompt fills — the SYSTEM prompt
// vs the USER (main/task) prompt. Reuses the role-badge pills that the
// chat-message roles already style.
function PromptRoleBadge({ role }: { role: string }) {
  const lower = role.toLowerCase();
  const cls = lower === "system" ? "badge role-system" : "badge role-user";
  const label = lower === "system" ? "system" : "main";
  return <span class={cls}>{label}</span>;
}

// EnabledSwitch is the accessible toggle shared by the tools and skills
// tabs. `stopPropagation` is set inside <summary> (tools) so flipping
// the switch doesn't also expand/collapse the details card.
function EnabledSwitch({
  enabled,
  pending,
  onChange,
  stopPropagation,
}: {
  enabled: boolean;
  pending: boolean;
  onChange: (enabled: boolean) => void;
  stopPropagation?: boolean;
}) {
  return (
    <label
      class="switch"
      title={enabled ? "Enabled" : "Disabled"}
      onClick={stopPropagation ? (e) => e.stopPropagation() : undefined}
    >
      <input
        type="checkbox"
        checked={enabled}
        disabled={pending}
        onChange={(e) => onChange((e.target as HTMLInputElement).checked)}
      />
      <span class="switch-track" />
      <span class="switch-knob" />
    </label>
  );
}

function PromptsView({ prompts }: { prompts: CatalogPrompt[] }) {
  if (prompts.length === 0) return <div class="empty">No prompts.</div>;
  return (
    <div class="catalog-list">
      {prompts.map((p) => (
        <details key={p.source + "/" + p.name} class="card catalog-card">
          <summary class="catalog-summary">
            <span class="catalog-summary-main">
              <span class="mono">{p.name}</span>
              {p.role ? <PromptRoleBadge role={p.role} /> : null}
            </span>
            <SourceChip source={p.source} />
          </summary>
          <pre class="catalog-body mono">{p.body}</pre>
        </details>
      ))}
    </div>
  );
}

function ToolsView({
  agentName,
  tools,
}: {
  agentName: string;
  tools: CatalogTool[];
}) {
  const toggle = useToggleTool(agentName);
  if (tools.length === 0) return <div class="empty">No tools.</div>;
  return (
    <div class="catalog-list">
      {tools.map((t) => (
        <details
          key={t.source + "/" + t.name}
          class={"card catalog-card" + (t.enabled ? "" : " is-disabled")}
        >
          <summary class="catalog-summary">
            <EnabledSwitch
              enabled={t.enabled}
              pending={toggle.isPending}
              stopPropagation
              onChange={(enabled) =>
                toggle.mutate({ source: t.source, name: t.name, enabled })
              }
            />
            <span class="catalog-summary-main">
              <span class="mono">{t.name}</span>
              <span class="text-dim">{t.description}</span>
            </span>
            <SourceChip source={t.source} version={t.version} />
          </summary>
          {t.schema != null ? (
            <pre class="catalog-body mono">
              {JSON.stringify(t.schema, null, 2)}
            </pre>
          ) : (
            <div class="text-dim" style="padding: 8px 0 0">
              No parameter schema.
            </div>
          )}
        </details>
      ))}
    </div>
  );
}

function SkillsView({
  agentName,
  skills,
}: {
  agentName: string;
  skills: CatalogSkill[];
}) {
  const toggle = useToggleSkill(agentName);
  if (skills.length === 0) return <div class="empty">No skills.</div>;
  return (
    <div class="table-wrap">
      <table class="table">
        <thead>
          <tr>
            <th style="width: 70px">Enabled</th>
            <th>Skill</th>
            <th>Description</th>
            <th>Source</th>
          </tr>
        </thead>
        <tbody>
          {skills.map((s) => (
            <tr
              key={s.source + "/" + s.name}
              class={s.enabled ? undefined : "is-disabled"}
            >
              <td>
                <EnabledSwitch
                  enabled={s.enabled}
                  pending={toggle.isPending}
                  onChange={(enabled) =>
                    toggle.mutate({ source: s.source, name: s.name, enabled })
                  }
                />
              </td>
              <td>
                <span class="mono">{s.name}</span>
              </td>
              <td class="text-dim">{s.description}</td>
              <td>
                <SourceChip source={s.source} version={s.version} />
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
