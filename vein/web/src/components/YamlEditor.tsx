import { useEffect, useRef } from "preact/hooks";
import { EditorView, basicSetup } from "codemirror";
import { EditorState } from "@codemirror/state";
import { yaml } from "@codemirror/lang-yaml";
import { HighlightStyle, syntaxHighlighting } from "@codemirror/language";
import { tags as t } from "@lezer/highlight";

// ── YamlEditor ──────────────────────────────────────────────────────────────
//
// A thin CodeMirror 6 wrapper (vanilla-DOM, no React/compat) for editing a
// chunk of YAML (or plain text) in a syntax-highlighted, resizable pane. Used
// by the Params flyout to edit structured params (datasets, gold configs) as
// readable YAML instead of escaped JSON, and reusable as a standalone document
// editor (e.g. a future dataset entity). Themed entirely with vein's CSS vars.

// Editor chrome themed to the palette (base.css :root vars).
const cmTheme = EditorView.theme(
  {
    "&": {
      color: "var(--text)",
      backgroundColor: "var(--surface)",
      fontSize: "12.5px",
      border: "1px solid var(--border)",
      borderRadius: "var(--r-md)",
    },
    "&.cm-focused": { outline: "none", borderColor: "var(--accent)" },
    ".cm-scroller": { fontFamily: "var(--font-mono)", maxHeight: "60vh" },
    ".cm-content": { minHeight: "96px", caretColor: "var(--accent)" },
    ".cm-gutters": { backgroundColor: "transparent", color: "var(--text-dim)", border: "none" },
    ".cm-activeLine": { backgroundColor: "var(--surface-2)" },
    ".cm-activeLineGutter": { backgroundColor: "transparent", color: "var(--text-muted)" },
    ".cm-cursor, .cm-dropCursor": { borderLeftColor: "var(--accent)" },
    "&.cm-focused .cm-selectionBackground, .cm-selectionBackground, .cm-content ::selection": {
      backgroundColor: "var(--surface-2)",
    },
    ".cm-foldPlaceholder": { backgroundColor: "var(--surface-2)", color: "var(--text-muted)", border: "none" },
  },
  { dark: true },
);

// Syntax token colors (override basicSetup's default light-ish highlight).
const cmHighlight = HighlightStyle.define([
  { tag: t.keyword, color: "var(--accent-strong)" },
  { tag: [t.propertyName, t.definition(t.propertyName)], color: "var(--accent)" },
  { tag: [t.string, t.special(t.string)], color: "var(--ok)" },
  { tag: [t.number, t.bool, t.null], color: "var(--warning)" },
  { tag: t.comment, color: "var(--text-dim)", fontStyle: "italic" },
  { tag: [t.meta, t.punctuation, t.separator], color: "var(--text-muted)" },
]);

export function YamlEditor(props: {
  value: string;
  onChange: (next: string) => void;
  /** "yaml" → YAML grammar; "text" → no language (plain prompts/strings). */
  language?: "yaml" | "text";
  readOnly?: boolean;
}) {
  const host = useRef<HTMLDivElement>(null);
  const view = useRef<EditorView | null>(null);
  // Keep onChange current without re-creating the editor.
  const onChange = useRef(props.onChange);
  onChange.current = props.onChange;

  // Build the editor once (mount). `value`/`language`/`readOnly` are captured as
  // initial state; external `value` changes are reconciled by the effect below.
  useEffect(() => {
    if (!host.current) return;
    const v = new EditorView({
      parent: host.current,
      state: EditorState.create({
        doc: props.value,
        extensions: [
          basicSetup,
          ...(props.language === "text" ? [] : [yaml()]),
          syntaxHighlighting(cmHighlight),
          cmTheme,
          EditorView.lineWrapping,
          EditorState.readOnly.of(!!props.readOnly),
          EditorView.updateListener.of((u) => {
            if (u.docChanged) onChange.current(u.state.doc.toString());
          }),
        ],
      }),
    });
    view.current = v;
    return () => {
      v.destroy();
      view.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Reconcile external value changes (no-op when it already matches the doc, so
  // a self-originated edit doesn't loop or jump the cursor).
  useEffect(() => {
    const v = view.current;
    if (!v) return;
    const cur = v.state.doc.toString();
    if (props.value !== cur) {
      v.dispatch({ changes: { from: 0, to: cur.length, insert: props.value } });
    }
  }, [props.value]);

  return <div class="yaml-editor" ref={host} />;
}
