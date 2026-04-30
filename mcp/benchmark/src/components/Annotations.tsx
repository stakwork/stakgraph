import { useState, useRef, useEffect } from "react";
import { createPortal } from "react-dom";
import type { Annotation, AnnotationMarker } from "../types";

export type { Annotation, AnnotationMarker };

export const ANNOTATION_COLORS: Record<
  AnnotationMarker,
  { fg: string; bg: string; border: string }
> = {
  inefficient: { fg: "#fdba74", bg: "rgba(124,45,18,0.35)", border: "#7c2d12" },
  bad_search: { fg: "#fca5a5", bg: "rgba(127,29,29,0.35)", border: "#7f1d1d" },
  good_result: { fg: "#86efac", bg: "rgba(21,128,61,0.24)", border: "#166534" },
  loop: { fg: "#fcd34d", bg: "rgba(120,53,15,0.35)", border: "#78350f" },
  wrong_tool: { fg: "#f9a8d4", bg: "rgba(131,24,67,0.35)", border: "#831843" },
  wasted_tokens: {
    fg: "#c4b5fd",
    bg: "rgba(76,29,149,0.35)",
    border: "#4c1d95",
  },
};

export const ANNOTATION_MARKERS: AnnotationMarker[] = [
  "inefficient",
  "bad_search",
  "good_result",
  "loop",
  "wrong_tool",
  "wasted_tokens",
];

export function AnnotationBadge({
  marker,
  note,
}: {
  marker: AnnotationMarker;
  note?: string;
}) {
  const color = ANNOTATION_COLORS[marker];
  const [open, setOpen] = useState(false);
  const [coords, setCoords] = useState<{ top: number; left: number } | null>(
    null,
  );
  const badgeRef = useRef<HTMLSpanElement>(null);

  useEffect(() => {
    if (!open || !badgeRef.current) return;
    const rect = badgeRef.current.getBoundingClientRect();
    setCoords({ top: rect.bottom + 6, left: rect.left });
  }, [open]);

  // Close on scroll/resize
  useEffect(() => {
    if (!open) return;
    const close = () => setOpen(false);
    window.addEventListener("scroll", close, true);
    window.addEventListener("resize", close);
    return () => {
      window.removeEventListener("scroll", close, true);
      window.removeEventListener("resize", close);
    };
  }, [open]);

  return (
    <span style={{ position: "relative", display: "inline-flex" }}>
      <span
        ref={badgeRef}
        onClick={(e) => {
          if (!note) return;
          e.preventDefault();
          e.stopPropagation();
          setOpen((o) => !o);
        }}
        style={{
          fontSize: "10px",
          lineHeight: 1,
          padding: "4px 6px",
          borderRadius: "9999px",
          border: `1px solid ${color.border}`,
          color: color.fg,
          backgroundColor: color.bg,
          textTransform: "lowercase",
          cursor: note ? "pointer" : "default",
          userSelect: "none",
        }}
      >
        {marker.replace(/_/g, " ")}
        {note && (
          <span style={{ marginLeft: "4px", opacity: 0.7, fontSize: "9px" }}>
            {open ? "▲" : "▼"}
          </span>
        )}
      </span>
      {open &&
        note &&
        coords &&
        createPortal(
          <span
            onClick={(e) => e.stopPropagation()}
            style={{
              position: "fixed",
              top: coords.top,
              left: coords.left,
              zIndex: 99999,
              backgroundColor: "#1c1c1f",
              border: `1px solid ${color.border}`,
              borderRadius: "6px",
              padding: "8px 10px",
              fontSize: "11px",
              color: "#ededed",
              whiteSpace: "pre-wrap",
              maxWidth: "340px",
              width: "max-content",
              boxShadow: "0 4px 16px rgba(0,0,0,0.6)",
              lineHeight: 1.5,
            }}
          >
            {note}
          </span>,
          document.body,
        )}
    </span>
  );
}

export function AnnotationForm({
  onSubmit,
  onCancel,
}: {
  onSubmit: (marker: AnnotationMarker, note: string) => void;
  onCancel: () => void;
}) {
  const [marker, setMarker] = useState<AnnotationMarker>("inefficient");
  const [note, setNote] = useState("");

  return (
    <div
      style={{
        padding: "8px 14px",
        borderTop: "1px solid #1f1f22",
        backgroundColor: "rgba(163,230,53,0.04)",
        display: "flex",
        gap: "8px",
        alignItems: "center",
        flexWrap: "wrap",
      }}
    >
      <select
        value={marker}
        onChange={(e) => setMarker(e.target.value as AnnotationMarker)}
        style={{
          fontSize: "11px",
          borderRadius: "5px",
          padding: "4px 6px",
          backgroundColor: "#18181b",
          color: "#ededed",
          border: "1px solid #27272a",
        }}
      >
        {ANNOTATION_MARKERS.map((m) => (
          <option key={m} value={m}>
            {m.replace(/_/g, " ")}
          </option>
        ))}
      </select>
      <input
        placeholder="Note (optional)"
        value={note}
        onChange={(e) => setNote(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter") onSubmit(marker, note);
          if (e.key === "Escape") onCancel();
        }}
        style={{
          fontSize: "11px",
          borderRadius: "5px",
          padding: "4px 8px",
          backgroundColor: "#18181b",
          color: "#ededed",
          border: "1px solid #27272a",
          flex: 1,
          minWidth: "120px",
        }}
      />
      <button
        onClick={() => onSubmit(marker, note)}
        style={{
          fontSize: "11px",
          padding: "4px 10px",
          borderRadius: "5px",
          border: "1px solid #3f3f46",
          backgroundColor: "#27272a",
          color: "#ededed",
          cursor: "pointer",
        }}
      >
        Save
      </button>
      <button
        onClick={onCancel}
        style={{
          fontSize: "11px",
          padding: "4px 8px",
          borderRadius: "5px",
          border: "1px solid #27272a",
          backgroundColor: "transparent",
          color: "#71717a",
          cursor: "pointer",
        }}
      >
        ✕
      </button>
    </div>
  );
}
