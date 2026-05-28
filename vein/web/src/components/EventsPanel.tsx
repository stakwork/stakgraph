import { useState, useEffect, useRef } from "preact/hooks";
import * as api from "../api";
import { formatJson, eventTone, statusTone } from "../helpers";
import { StepData } from "../flow-to-canvas";
import { CloseIcon } from "../icons";
import yaml from "js-yaml";

// ── Events Panel (expandable rows) ─────────────────────────────────────────

export function EventsPanel(props: { events: api.RunEvent[] }) {
  const [expanded, setExpanded] = useState<number | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);

  // Auto-expand run.end when it arrives
  useEffect(() => {
    const idx = props.events.findIndex((e) => e.type === "run.end");
    if (idx >= 0) setExpanded(idx);
  }, [props.events]);

  // Auto-scroll to the bottom whenever events change
  useEffect(() => {
    const el = scrollRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [props.events.length, expanded]);

  return (
    <div class="shell-events" ref={scrollRef}>
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
