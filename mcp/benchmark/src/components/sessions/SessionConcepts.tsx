import { useEffect, useState } from "react";
import { CopyableBlock } from "../ui";
import { muted } from "./styles";
import { api } from "../../api";
import type { ReflectedConcept, SessionReflection } from "../../types";

/**
 * The gitree Concepts a session read.
 *
 * Three depths, so the common case costs nothing: a count pill in the header,
 * the names one click in, and a concept's full documentation one click past
 * that. The names list is what you scan — it renders open by default so
 * landing on a session shows what it leaned on without any interaction.
 */

/** Read order is the scan order, and the only field every entry carries. */
function byReadOrder(a: ReflectedConcept, b: ReflectedConcept): number {
  return (a.read_order ?? 0) - (b.read_order ?? 0);
}

function conceptKey(c: ReflectedConcept, i: number): string {
  return c.ref_id ?? c.id ?? c.name ?? String(i);
}

/**
 * Sits first in the session-signal row, alongside `flagged` / `oversized` /
 * `repeats`. Those are inert text, so this one has to carry enough weight —
 * filled violet, a caret, a hover state — to read as the button it is.
 */
export function ConceptsPill({
  count,
  open,
  onToggle,
}: {
  count: number;
  open: boolean;
  onToggle: () => void;
}) {
  return (
    <button
      onClick={onToggle}
      className="concepts-pill"
      title={open ? "Hide concepts read" : "Show the concepts this session read"}
      style={{
        display: "inline-flex",
        alignItems: "center",
        gap: "6px",
        fontSize: "11px",
        fontWeight: 600,
        padding: "3px 10px",
        borderRadius: "6px",
        border: `1px solid ${open ? "#a78bfa" : "#6d28d9"}`,
        backgroundColor: open ? "rgba(109,40,217,0.4)" : "rgba(109,40,217,0.28)",
        color: "#ddd6fe",
        cursor: "pointer",
        fontFamily: "inherit",
      }}
    >
      <span>
        {count} concept{count === 1 ? "" : "s"}
      </span>
      <span style={{ fontSize: "9px" }}>{open ? "▴" : "▾"}</span>
    </button>
  );
}

/** Full documentation for one concept, fetched on first expand. */
function ConceptBody({ concept }: { concept: ReflectedConcept }) {
  const [doc, setDoc] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    // Only gitree ids resolve; a Concept reached solely through the graph has
    // a ref_id and no id, and there is nothing to fetch it with.
    if (!concept.id) return;
    let cancelled = false;
    api.concepts
      .get(concept.id, concept.repo)
      .then((res) => {
        if (cancelled) return;
        setDoc(
          res.concept.documentation ||
            res.concept.description ||
            "No documentation recorded for this concept.",
        );
      })
      .catch((e: Error) => {
        if (!cancelled) setError(e.message);
      });
    return () => {
      cancelled = true;
    };
  }, [concept.id, concept.repo]);

  return (
    <div style={{ padding: "4px 0 10px 0" }}>
      {concept.evidence && (
        <p style={{ ...muted, color: "#d4d4d8", padding: "0 14px 8px 14px" }}>
          {concept.evidence}
        </p>
      )}
      {concept.contradicts && (
        <div style={{ padding: "0 14px 8px 14px" }}>
          <p
            style={{
              margin: "0 0 4px 0",
              fontSize: "10px",
              textTransform: "uppercase",
              letterSpacing: "0.1em",
              color: "#fdba74",
            }}
          >
            Contradicts source
          </p>
          <p style={{ ...muted, color: "#d4d4d8", margin: 0 }}>
            {concept.contradicts}
          </p>
        </div>
      )}
      {!concept.id ? (
        <p style={{ ...muted, padding: "0 14px" }}>
          Read through the graph only — no gitree id to load documentation with.
        </p>
      ) : error ? (
        <p style={{ ...muted, padding: "0 14px", color: "#fca5a5" }}>
          Could not load documentation: {error}
        </p>
      ) : doc === null ? (
        <p style={{ ...muted, padding: "0 14px" }}>Loading documentation…</p>
      ) : (
        <CopyableBlock value={doc} maxHeight="24rem" />
      )}
    </div>
  );
}

function ConceptRow({ concept, index }: { concept: ReflectedConcept; index: number }) {
  return (
    <details style={{ borderTop: "1px solid #1f1f22" }}>
      <summary
        style={{
          listStyle: "none",
          display: "flex",
          alignItems: "center",
          gap: "10px",
          padding: "8px 14px",
          cursor: "pointer",
          userSelect: "none",
        }}
      >
        <span
          style={{
            display: "inline-block",
            width: "24px",
            flexShrink: 0,
            fontSize: "10px",
            fontWeight: 700,
            color: "#52525b",
          }}
        >
          {concept.read_order ?? index + 1}
        </span>
        <span
          style={{
            fontSize: "12px",
            fontWeight: 600,
            color: "#ededed",
            overflowWrap: "anywhere",
          }}
        >
          {concept.name || concept.id || concept.ref_id || "(unnamed)"}
        </span>
        {concept.rank !== null && (
          <span
            title="Agent's ranking of how load-bearing this concept was"
            style={{
              fontSize: "10px",
              fontWeight: 700,
              backgroundColor: "#3f3f46",
              borderRadius: "9999px",
              padding: "1px 6px",
              color: "#ededed",
              flexShrink: 0,
            }}
          >
            #{concept.rank}
          </span>
        )}
        {concept.contradicts && (
          <span
            title="This concept disagreed with what the agent found in the source"
            style={{
              fontSize: "10px",
              lineHeight: 1,
              padding: "3px 6px",
              borderRadius: "9999px",
              border: "1px solid #7c2d12",
              color: "#fdba74",
              backgroundColor: "rgba(124,45,18,0.35)",
              flexShrink: 0,
            }}
          >
            contradicts
          </span>
        )}
      </summary>
      <ConceptBody concept={concept} />
    </details>
  );
}

/**
 * The names list. Rendered next to the header card, below the meta pills, so
 * it reads as part of the session summary rather than as another trace panel.
 */
export function ConceptsPanel({ reflection }: { reflection: SessionReflection }) {
  // Stored rank-first (see mergeReflection); re-sort so the list always reads
  // in the order the session actually visited them.
  const concepts = [...reflection.concepts].sort(byReadOrder);
  return (
    <div
      style={{
        marginTop: "10px",
        border: "1px solid #27272a",
        borderRadius: "8px",
        backgroundColor: "#0d0d0f",
        overflow: "hidden",
      }}
    >
      {concepts.map((c, i) => (
        <ConceptRow key={conceptKey(c, i)} concept={c} index={i} />
      ))}
      {reflection.gap && (
        <div
          style={{
            borderTop: "1px solid #1f1f22",
            padding: "8px 14px 10px 14px",
          }}
        >
          <p
            style={{
              margin: "0 0 4px 0",
              fontSize: "10px",
              textTransform: "uppercase",
              letterSpacing: "0.1em",
              color: "#71717a",
            }}
          >
            Gap — worked out from source, covered by no concept
          </p>
          <p style={{ ...muted, color: "#d4d4d8", margin: 0 }}>
            {reflection.gap}
          </p>
        </div>
      )}
    </div>
  );
}
