// EvalsView — the agent-detail "Evals" tab.
//
// Reads (the set list + each set's requirements/triggers/outputs) come
// from the gateway's neo4j traversal. Writes and runs are delegated by
// the gateway to Hive, so a mutation may fail with "hive_not_connected"
// when this swarm hasn't had its Hive callback config pushed yet — we
// surface that inline rather than as a fatal error.

import { useState } from "preact/hooks";

import { ApiCallError, getErrorMessage } from "../api/client";
import {
  useAgentEvals,
  useCreateOrLinkEvalSet,
  useCreateRequirement,
  useDeleteEvalSet,
  useDeleteRequirement,
  useEvalSet,
  useRunRequirement,
  useUnlinkEvalSet,
} from "../api/queries";
import type {
  EvalRequirementDetail,
  EvalSetSummary,
  EvalTriggerSummary,
} from "../api/types";

export function EvalsView({ agentName }: { agentName: string }) {
  const evals = useAgentEvals(agentName);
  const createOrLink = useCreateOrLinkEvalSet(agentName);
  const [newName, setNewName] = useState("");
  const [openSet, setOpenSet] = useState<string | null>(null);

  const unavailable =
    evals.error instanceof ApiCallError &&
    evals.error.code === "catalog_unavailable";

  if (unavailable) {
    return (
      <div class="empty">
        Eval graph not wired on this swarm. Set{" "}
        <span class="mono">NEO4J_PASSWORD</span> on the gateway to enable evals.
      </div>
    );
  }
  if (evals.isLoading) return <div class="loading">Loading…</div>;
  if (evals.isError) {
    return <div class="error-banner">{getErrorMessage(evals.error)}</div>;
  }

  const sets = evals.data?.sets ?? [];

  const submitCreate = (e: Event) => {
    e.preventDefault();
    const name = newName.trim();
    if (!name) return;
    createOrLink.mutate(
      { name },
      { onSuccess: () => setNewName("") },
    );
  };

  return (
    <div class="evals-view">
      <form class="eval-create" onSubmit={submitCreate}>
        <input
          class="input"
          type="text"
          placeholder="New eval set name…"
          value={newName}
          onInput={(e) => setNewName((e.target as HTMLInputElement).value)}
        />
        <button
          type="submit"
          class="btn is-primary"
          disabled={createOrLink.isPending || !newName.trim()}
        >
          {createOrLink.isPending ? "Creating…" : "Create eval set"}
        </button>
      </form>
      {createOrLink.isError ? (
        <div class="error-banner">{getErrorMessage(createOrLink.error)}</div>
      ) : null}

      {sets.length === 0 ? (
        <div class="empty">No eval sets linked to this agent yet.</div>
      ) : (
        <div class="catalog-list">
          {sets.map((s) => (
            <SetCard
              key={s.ref_id}
              agentName={agentName}
              set={s}
              open={openSet === s.ref_id}
              onToggle={() =>
                setOpenSet(openSet === s.ref_id ? null : s.ref_id)
              }
            />
          ))}
        </div>
      )}
    </div>
  );
}

function SetCard({
  agentName,
  set,
  open,
  onToggle,
}: {
  agentName: string;
  set: EvalSetSummary;
  open: boolean;
  onToggle: () => void;
}) {
  const unlink = useUnlinkEvalSet(agentName);
  const del = useDeleteEvalSet(agentName);

  return (
    <div class="card catalog-card">
      <div class="catalog-summary" style="cursor: default">
        <button type="button" class="eval-disclosure" onClick={onToggle}>
          <span class="mono">{set.name || set.ref_id}</span>
          <span class="pill pill-accent">{set.requirements} req</span>
        </button>
        <div class="btn-group">
          <button
            type="button"
            class="btn"
            title="Remove from this agent (keeps the set)"
            disabled={unlink.isPending}
            onClick={() => unlink.mutate(set.ref_id)}
          >
            Unlink
          </button>
          <button
            type="button"
            class="btn"
            title="Delete the eval set entirely"
            disabled={del.isPending}
            onClick={() => {
              if (confirm(`Delete eval set "${set.name || set.ref_id}"?`)) {
                del.mutate(set.ref_id);
              }
            }}
          >
            Delete
          </button>
        </div>
      </div>
      {del.isError ? (
        <div class="error-banner">{getErrorMessage(del.error)}</div>
      ) : null}
      {open ? <SetDetail setId={set.ref_id} agentName={agentName} /> : null}
    </div>
  );
}

function SetDetail({
  setId,
  agentName,
}: {
  setId: string;
  agentName: string;
}) {
  const detail = useEvalSet(setId);
  const createReq = useCreateRequirement(setId, agentName);
  const [reqName, setReqName] = useState("");

  if (detail.isLoading) return <div class="loading">Loading…</div>;
  if (detail.isError) {
    return <div class="error-banner">{getErrorMessage(detail.error)}</div>;
  }

  const reqs = detail.data?.requirements ?? [];

  const submitReq = (e: Event) => {
    e.preventDefault();
    const name = reqName.trim();
    if (!name) return;
    createReq.mutate({ name }, { onSuccess: () => setReqName("") });
  };

  return (
    <div class="eval-detail">
      {reqs.length === 0 ? (
        <div class="empty">No requirements yet.</div>
      ) : (
        reqs.map((r) => (
          <RequirementRow
            key={r.ref_id}
            setId={setId}
            agentName={agentName}
            req={r}
          />
        ))
      )}

      <form class="eval-create" onSubmit={submitReq}>
        <input
          class="input"
          type="text"
          placeholder="New requirement…"
          value={reqName}
          onInput={(e) => setReqName((e.target as HTMLInputElement).value)}
        />
        <button
          type="submit"
          class="btn"
          disabled={createReq.isPending || !reqName.trim()}
        >
          Add requirement
        </button>
      </form>
      {createReq.isError ? (
        <div class="error-banner">{getErrorMessage(createReq.error)}</div>
      ) : null}
    </div>
  );
}

function RequirementRow({
  setId,
  agentName,
  req,
}: {
  setId: string;
  agentName: string;
  req: EvalRequirementDetail;
}) {
  const run = useRunRequirement(setId);
  const del = useDeleteRequirement(setId, agentName);

  return (
    <div class="eval-req">
      <div class="eval-req-head">
        <span class="eval-req-name">{req.name || req.ref_id}</span>
        <div class="btn-group">
          <button
            type="button"
            class="btn is-primary"
            disabled={run.isPending}
            onClick={() => run.mutate({ reqId: req.ref_id })}
          >
            {run.isPending ? "Running…" : "Run"}
          </button>
          <button
            type="button"
            class="btn"
            disabled={del.isPending}
            onClick={() => del.mutate(req.ref_id)}
          >
            Delete
          </button>
        </div>
      </div>
      {req.description ? (
        <div class="text-dim eval-req-desc">{req.description}</div>
      ) : null}
      {run.isError ? (
        <div class="error-banner">{getErrorMessage(run.error)}</div>
      ) : null}
      {req.triggers.length > 0 ? (
        <div class="eval-triggers">
          {req.triggers.map((t) => (
            <TriggerChip key={t.ref_id} trigger={t} />
          ))}
        </div>
      ) : (
        <div class="text-dim eval-req-desc">No captured triggers.</div>
      )}
    </div>
  );
}

function TriggerChip({ trigger }: { trigger: EvalTriggerSummary }) {
  const result = (trigger.last_result || "").toLowerCase();
  const badge =
    result === "pass"
      ? "badge badge-ok"
      : result === "fail"
        ? "badge badge-warning"
        : "badge";
  const label =
    result === "pass"
      ? "pass"
      : result === "fail"
        ? "fail"
        : "not run";
  return (
    <span class="eval-trigger" title={trigger.last_notes || undefined}>
      <span class={badge}>{label}</span>
      {trigger.source ? (
        <span class="text-dim mono">{trigger.source}</span>
      ) : null}
      {result && typeof trigger.last_score === "number" ? (
        <span class="text-dim">{trigger.last_score.toFixed(2)}</span>
      ) : null}
    </span>
  );
}
