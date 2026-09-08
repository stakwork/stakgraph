import { h, Fragment, JSX } from "preact";

// ── Tiny markdown renderer ─────────────────────────────────────────────────
// Supports: headers (#..######), unordered lists (- / *), ordered lists (1.),
// fenced code blocks (```), blockquotes (>), inline code (`x`), bold (**x**),
// italic (*x* / _x_), links [text](url), horizontal rules (---), paragraphs.
// Deliberately small — no external deps.

type Node = JSX.Element | string;

function renderInline(text: string): Node[] {
  const nodes: Node[] = [];
  let i = 0;
  let buf = "";
  const flush = () => { if (buf) { nodes.push(buf); buf = ""; } };

  while (i < text.length) {
    const ch = text[i];
    const rest = text.slice(i);

    // Inline code: `...`
    if (ch === "`") {
      const end = text.indexOf("`", i + 1);
      if (end !== -1) {
        flush();
        nodes.push(<code class="md-code-inline">{text.slice(i + 1, end)}</code>);
        i = end + 1;
        continue;
      }
    }

    // Bold: **...**
    if (ch === "*" && text[i + 1] === "*") {
      const end = text.indexOf("**", i + 2);
      if (end !== -1) {
        flush();
        nodes.push(<strong>{renderInline(text.slice(i + 2, end))}</strong>);
        i = end + 2;
        continue;
      }
    }

    // Italic: *...* or _..._
    if ((ch === "*" || ch === "_") && text[i + 1] !== ch) {
      const end = text.indexOf(ch, i + 1);
      if (end !== -1 && end > i + 1) {
        flush();
        nodes.push(<em>{renderInline(text.slice(i + 1, end))}</em>);
        i = end + 1;
        continue;
      }
    }

    // Link: [text](url)
    if (ch === "[") {
      const closeBracket = text.indexOf("]", i + 1);
      if (closeBracket !== -1 && text[closeBracket + 1] === "(") {
        const closeParen = text.indexOf(")", closeBracket + 2);
        if (closeParen !== -1) {
          const label = text.slice(i + 1, closeBracket);
          const url = text.slice(closeBracket + 2, closeParen);
          flush();
          nodes.push(<a href={url} target="_blank" rel="noopener noreferrer">{renderInline(label)}</a>);
          i = closeParen + 1;
          continue;
        }
      }
    }

    buf += ch;
    i++;
    void rest;
  }
  flush();
  return nodes;
}

export function Markdown(props: { source: string; class?: string }) {
  const src = props.source.replace(/\r\n/g, "\n");
  const lines = src.split("\n");
  const out: JSX.Element[] = [];

  let i = 0;
  let para: string[] = [];
  const flushPara = () => {
    if (para.length) {
      out.push(<p key={out.length}>{renderInline(para.join(" "))}</p>);
      para = [];
    }
  };

  while (i < lines.length) {
    const line = lines[i];

    // Fenced code block
    if (/^```/.test(line)) {
      flushPara();
      const lang = line.slice(3).trim();
      const codeLines: string[] = [];
      i++;
      while (i < lines.length && !/^```/.test(lines[i])) {
        codeLines.push(lines[i]);
        i++;
      }
      i++; // closing fence
      out.push(
        <pre key={out.length} class="md-code-block" data-lang={lang || undefined}>
          <code>{codeLines.join("\n")}</code>
        </pre>
      );
      continue;
    }

    // Headers
    const hMatch = /^(#{1,6})\s+(.*)$/.exec(line);
    if (hMatch) {
      flushPara();
      const level = hMatch[1].length;
      const text = hMatch[2];
      const Tag = `h${level}` as keyof JSX.IntrinsicElements;
      out.push(h(Tag, { key: out.length, class: `md-h md-h${level}` }, renderInline(text)));
      i++;
      continue;
    }

    // Horizontal rule
    if (/^\s*([-*_])\s*\1\s*\1[-*_\s]*$/.test(line)) {
      flushPara();
      out.push(<hr key={out.length} class="md-hr" />);
      i++;
      continue;
    }

    // Unordered list
    if (/^\s*[-*]\s+/.test(line)) {
      flushPara();
      const items: string[] = [];
      while (i < lines.length && /^\s*[-*]\s+/.test(lines[i])) {
        items.push(lines[i].replace(/^\s*[-*]\s+/, ""));
        i++;
      }
      out.push(
        <ul key={out.length} class="md-ul">
          {items.map((it, idx) => <li key={idx}>{renderInline(it)}</li>)}
        </ul>
      );
      continue;
    }

    // Ordered list
    if (/^\s*\d+\.\s+/.test(line)) {
      flushPara();
      const items: string[] = [];
      while (i < lines.length && /^\s*\d+\.\s+/.test(lines[i])) {
        items.push(lines[i].replace(/^\s*\d+\.\s+/, ""));
        i++;
      }
      out.push(
        <ol key={out.length} class="md-ol">
          {items.map((it, idx) => <li key={idx}>{renderInline(it)}</li>)}
        </ol>
      );
      continue;
    }

    // Blockquote
    if (/^>\s?/.test(line)) {
      flushPara();
      const quoted: string[] = [];
      while (i < lines.length && /^>\s?/.test(lines[i])) {
        quoted.push(lines[i].replace(/^>\s?/, ""));
        i++;
      }
      out.push(<blockquote key={out.length} class="md-quote">{renderInline(quoted.join(" "))}</blockquote>);
      continue;
    }

    // Blank line → paragraph break
    if (/^\s*$/.test(line)) {
      flushPara();
      i++;
      continue;
    }

    // Paragraph line
    para.push(line);
    i++;
  }
  flushPara();

  return <div class={`md-view ${props.class || ""}`}>{out}</div>;
}

/** Detect whether a step output is an object with a `markdown` string field. */
export function hasMarkdownField(v: unknown): v is { markdown: string } & Record<string, unknown> {
  return (
    v != null &&
    typeof v === "object" &&
    !Array.isArray(v) &&
    typeof (v as Record<string, unknown>).markdown === "string" &&
    ((v as Record<string, unknown>).markdown as string).length > 0
  );
}
