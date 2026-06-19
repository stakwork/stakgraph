import { useState, useEffect, useRef } from "preact/hooks";
import * as api from "../api";
import { eventTone, statusTone } from "../helpers";
import { ValueFields } from "./ValueFields";
import { StepData } from "../flow-to-canvas";
import { CloseIcon } from "../icons";
import yaml from "js-yaml";

// ── Events Panel (expandable rows) ─────────────────────────────────────────

interface RunRef {
  label?: string;
  workflow: string;
  runId: string;
}

/** A step may attach pointers to related runs via `input.runs` / `output.runs`
 *  (e.g. eval/optimize links each generation's eval & reflect runs). Collect
 *  them so the panel can render "open run" links that drill into those logs. */
function runRefs(evt: api.RunEvent): RunRef[] {
  const out: RunRef[] = [];
  for (const src of [evt.input, evt.output]) {
    const runs = (src as { runs?: unknown } | null | undefined)?.runs;
    if (!Array.isArray(runs)) continue;
    for (const r of runs) {
      if (r && typeof r.workflow === "string" && typeof r.runId === "string") {
        out.push({ label: typeof r.label === "string" ? r.label : undefined, workflow: r.workflow, runId: r.runId });
      }
    }
  }
  return out;
}

export function EventsPanel(props: {
  events: api.RunEvent[];
  /** Navigate to another workflow's run (used by run-ref links). */
  onOpenRun?: (workflow: string, runId: string) => void;
}) {
  const [expanded, setExpanded] = useState<number | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  // Whether the user is pinned to the bottom (updated on every scroll). New
  // events only auto-scroll when this is true — scroll up and we leave you be.
  const stickToBottom = useRef(true);

  // Auto-expand run.end when it arrives
  useEffect(() => {
    const idx = props.events.findIndex((e) => e.type === "run.end");
    if (idx >= 0) setExpanded(idx);
  }, [props.events]);

  // Auto-scroll to the bottom on new events, but only if already at the bottom.
  useEffect(() => {
    const el = scrollRef.current;
    if (el && stickToBottom.current) el.scrollTop = el.scrollHeight;
  }, [props.events.length, expanded]);

  const onScroll = () => {
    const el = scrollRef.current;
    if (el) stickToBottom.current = el.scrollHeight - el.scrollTop - el.clientHeight < 40;
  };

  return (
    <div class="shell-events" ref={scrollRef} onScroll={onScroll}>
      <div class="events-header">Events ({props.events.length})</div>
      {props.events.map((evt, i) => {
        const hasData = evt.input != null || evt.output != null || evt.error != null;
        const isOpen = expanded === i;
        const refs = props.onOpenRun ? runRefs(evt) : [];
        return (
          <div key={i} class="event-row">
            <div class="event-row-summary" onClick={() => hasData && setExpanded(isOpen ? null : i)}>
              <span class={`event-type event-type-${eventTone(evt.type)}`}>{evt.type}</span>
              <span class="event-path">{evt.path}</span>
              {refs.map((r, j) => (
                <button
                  key={j}
                  class="event-run-link"
                  title={`Open ${r.workflow} / ${r.runId}`}
                  onClick={(e) => { e.stopPropagation(); props.onOpenRun!(r.workflow, r.runId); }}
                >
                  ↗ {r.label ?? "open run"}
                </button>
              ))}
              <span class="event-duration">{evt.durationMs != null ? `${evt.durationMs}ms` : ""}</span>
            </div>
            {isOpen && (
              <div class="event-detail">
                {evt.input != null && (
                  <>
                    <div class="event-detail-label">Input</div>
                    <ValueFields value={evt.input} blockClass="event-detail-block" />
                  </>
                )}
                {evt.output != null && (
                  <>
                    <div class="event-detail-label">Output</div>
                    <ValueFields value={evt.output} blockClass="event-detail-block" />
                  </>
                )}
                {evt.error != null && (
                  <>
                    <div class="event-detail-label">Error</div>
                    <ValueFields value={evt.error} blockClass="event-detail-block tone-error" />
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
