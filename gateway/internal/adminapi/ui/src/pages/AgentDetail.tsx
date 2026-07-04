// AgentDetail — per-agent cost histogram + recent runs.
//
// Phase-8 limitation
// ------------------
// The backend's /_plugin/histogram/cost endpoint doesn't yet accept
// a per-agent metadata filter; it returns one series per agent
// across the whole window. We work around that here by filtering
// the response client-side to just this agent's series. A future
// backend revision can add a server-side filter without touching
// this page.

import { useMemo, useState } from "preact/hooks";
import { Link } from "wouter-preact";

import { CostHistogram } from "../components/charts/CostHistogram";
import { ErrorBoundary } from "../components/ErrorBoundary";
import { WindowPicker } from "../components/controls/WindowPicker";
import type {
  AgentBudgetResponse,
  AgentCatalogResponse,
  CatalogPrompt,
  CatalogSkill,
  CatalogTool,
} from "../api/types";
import { ApiCallError, getErrorMessage } from "../api/client";
import {
  useAgentBudget,
  useAgentCatalog,
  useHistogramCost,
} from "../api/queries";
import type { HistogramCostResponse, Window } from "../api/types";
import { windowToSeconds } from "../api/window";

interface Props {
  name: string;
}

type Tab = "overview" | "prompts" | "tools" | "skills";

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

export function AgentDetail({ name }: Props) {
  const [window, setWindow] = useState<Window>("24h");
  const bucket = window === "1h" ? "5m" : window === "6h" ? "10m" : "1h";
  const windowSeconds = windowToSeconds(window);

  const budget = useAgentBudget(name);
  const histogram = useHistogramCost({
    window,
    bucket,
    dimension: "agent-name",
  });

  // Filter the histogram down to just this agent's series.
  const filtered = useMemo<HistogramCostResponse | undefined>(() => {
    if (!histogram.data) return undefined;
    return {
      ...histogram.data,
      series: histogram.data.series.filter((s) => s.dimension_value === name),
    };
  }, [histogram.data, name]);

  // Pull run-ids from the same window to derive the "recent runs"
  // table. We don't have a dedicated `/spend/by-run` endpoint yet,
  // so we use the agent dimension histogram to surface activity and
  // a separate dimension histogram by run-id for the same window.
  const runsHistogram = useHistogramCost({
    window,
    bucket,
    dimension: "run-id",
  });
  const recentRuns = useMemo(() => {
    if (!runsHistogram.data) return [];
    // Sort by total cost desc and cap to top 50 — the dashboard plan
    // commits to "first 100 runs in window"; 50 keeps render cheap.
    return [...runsHistogram.data.series]
      .map((s) => ({
        run_id: s.dimension_value,
        cost: s.points.reduce((acc, p) => acc + p.cost, 0),
        calls: s.points.length, // approximate; chart points are by bucket
      }))
      .sort((a, b) => b.cost - a.cost)
      .slice(0, 50);
  }, [runsHistogram.data]);

  const totalCost = filtered?.series[0]?.points.reduce((s, p) => s + p.cost, 0) ?? 0;

  const [tab, setTab] = useState<Tab>("overview");
  const catalog = useAgentCatalog(name);
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
          </h1>
        </div>
        {tab === "overview" ? (
          <WindowPicker value={window} onChange={setWindow} />
        ) : null}
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
      </nav>

      {tab !== "overview" ? (
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
          filtered={filtered}
          windowSeconds={windowSeconds}
          recentRuns={recentRuns}
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
  filtered: HistogramCostResponse | undefined;
  windowSeconds: number;
  recentRuns: { run_id: string; cost: number; calls: number }[];
}

function OverviewTab({
  window,
  bucket,
  totalCost,
  budget,
  histogram,
  filtered,
  windowSeconds,
  recentRuns,
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
        ) : filtered ? (
          <ErrorBoundary>
            <CostHistogram data={filtered} windowSeconds={windowSeconds} />
          </ErrorBoundary>
        ) : (
          <div class="loading">Loading…</div>
        )}
      </section>

      <section>
        <h2 style="margin-bottom: 16px">Recent runs</h2>
        {recentRuns.length === 0 ? (
          <div class="empty">No runs in this window.</div>
        ) : (
          <div class="table-wrap">
            <table class="table">
              <thead>
                <tr>
                  <th>Run</th>
                  <th class="num">Spend</th>
                  <th class="num">Buckets seen</th>
                </tr>
              </thead>
              <tbody>
                {recentRuns.map((r) => (
                  <tr key={r.run_id} class="row-link">
                    <td>
                      <Link href={`/runs/${encodeURIComponent(r.run_id)}`}>
                        <span class="mono">{r.run_id}</span>
                      </Link>
                    </td>
                    <td class="num">{fmtUSD(r.cost)}</td>
                    <td class="num">{fmtInt(r.calls)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </>
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
  if (tab === "tools") return <ToolsView tools={catalog.tools} />;
  return <SkillsView skills={catalog.skills} />;
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

function PromptsView({ prompts }: { prompts: CatalogPrompt[] }) {
  if (prompts.length === 0) return <div class="empty">No prompts.</div>;
  return (
    <div class="catalog-list">
      {prompts.map((p) => (
        <details key={p.source + "/" + p.name} class="card catalog-card">
          <summary class="catalog-summary">
            <span class="catalog-summary-main">
              <span class="mono">{p.name}</span>
            </span>
            <SourceChip source={p.source} />
          </summary>
          <pre class="catalog-body mono">{p.body}</pre>
        </details>
      ))}
    </div>
  );
}

function ToolsView({ tools }: { tools: CatalogTool[] }) {
  if (tools.length === 0) return <div class="empty">No tools.</div>;
  return (
    <div class="catalog-list">
      {tools.map((t) => (
        <details key={t.source + "/" + t.name} class="card catalog-card">
          <summary class="catalog-summary">
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

function SkillsView({ skills }: { skills: CatalogSkill[] }) {
  if (skills.length === 0) return <div class="empty">No skills.</div>;
  return (
    <div class="table-wrap">
      <table class="table">
        <thead>
          <tr>
            <th>Skill</th>
            <th>Description</th>
            <th>Source</th>
          </tr>
        </thead>
        <tbody>
          {skills.map((s) => (
            <tr key={s.source + "/" + s.name}>
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
