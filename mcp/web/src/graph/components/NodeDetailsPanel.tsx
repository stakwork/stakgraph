import { memo, useState } from "react";
import { Html } from "@react-three/drei";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { vscDarkPlus } from "react-syntax-highlighter/dist/esm/styles/prism";
import { useGraphData, getColorForType } from "@/stores/useGraphData";
import { nodePositions } from "@/stores/useSimulation";
import type { NodeExtended } from "@/graph/types";

const IMPORTANCE_TAG_COLORS: Record<string, string> = {
  entry_point: "#FFD700",
  hub: "#FF4444",
  utility: "#00BCD4",
  connector: "#888",
  isolated: "#555",
};

const IMPORTANCE_TAG_LABELS: Record<string, string> = {
  entry_point: "Entry Point",
  hub: "Hub",
  utility: "Utility",
  connector: "Connector",
  isolated: "Isolated",
};

function getLanguageFromFile(file: string): string {
  const ext = file.split(".").pop()?.toLowerCase() || "";
  const map: Record<string, string> = {
    ts: "typescript", tsx: "tsx", js: "javascript", jsx: "jsx",
    py: "python", rs: "rust", go: "go", rb: "ruby",
    java: "java", kt: "kotlin", swift: "swift",
    c: "c", cpp: "cpp", h: "c", hpp: "cpp", cs: "csharp",
    css: "css", scss: "scss", html: "html",
    json: "json", yaml: "yaml", yml: "yaml", toml: "toml",
    sql: "sql", sh: "bash", bash: "bash", md: "markdown",
    xml: "xml", dart: "dart", lua: "lua", zig: "zig",
  };
  return map[ext] || "text";
}

function ImportanceBadge({ tag }: { tag: string }) {
  const color = IMPORTANCE_TAG_COLORS[tag] || "#888";
  const label = IMPORTANCE_TAG_LABELS[tag] || tag;
  return (
    <span
      style={{
        display: "inline-flex",
        alignItems: "center",
        gap: 4,
        fontSize: 10,
        padding: "2px 6px",
        borderRadius: 4,
        background: `${color}20`,
        border: `1px solid ${color}40`,
        color,
        fontWeight: 600,
        textTransform: "uppercase",
        letterSpacing: 0.5,
      }}
    >
      <span style={{ width: 6, height: 6, borderRadius: "50%", background: color }} />
      {label}
    </span>
  );
}

function CodeViewer({ body, file, startLine }: { body: string; file: string; startLine: number }) {
  const language = getLanguageFromFile(file);
  return (
    <div
      onWheel={(e) => e.stopPropagation()}
      style={{
        borderRadius: 6,
        border: "1px solid rgba(255,255,255,0.08)",
        maxHeight: 200,
        overflowY: "auto",
        overflowX: "auto",
      }}
    >
      <SyntaxHighlighter
        language={language}
        style={vscDarkPlus}
        showLineNumbers
        startingLineNumber={startLine || 1}
        customStyle={{
          margin: 0,
          padding: "8px 0",
          fontSize: 11,
          lineHeight: 1.5,
          background: "rgba(15, 15, 20, 0.95)",
        }}
        lineNumberStyle={{ minWidth: "2.5em", paddingRight: "0.8em", color: "#555", fontSize: 10 }}
        wrapLines
      >
        {body}
      </SyntaxHighlighter>
    </div>
  );
}

function MetricsRow({ node }: { node: NodeExtended }) {
  const tag = node.properties.importance_tag as string | undefined;
  const pagerank = node.properties.pagerank as number | undefined;
  const inDeg = node.properties.in_degree as number | undefined;
  const outDeg = node.properties.out_degree as number | undefined;

  if (!tag && pagerank == null) return null;

  return (
    <div style={{ display: "flex", flexWrap: "wrap", alignItems: "center", gap: 8, marginBottom: 10 }}>
      {tag && <ImportanceBadge tag={tag} />}
      {pagerank != null && <span style={{ fontSize: 10, color: "#888" }}>pr: {pagerank.toFixed(4)}</span>}
      {inDeg != null && <span style={{ fontSize: 10, color: "#888" }}>in: {inDeg}</span>}
      {outDeg != null && <span style={{ fontSize: 10, color: "#888" }}>out: {outDeg}</span>}
    </div>
  );
}

export const NodeDetailsPanel = memo(() => {
  const selectedNode = useGraphData((s) => s.selectedNode);
  const nodesNormalized = useGraphData((s) => s.nodesNormalized);
  const setSelectedNode = useGraphData((s) => s.setSelectedNode);
  const tracedPath = useGraphData((s) => s.tracedPath);
  const traceCriticalPath = useGraphData((s) => s.traceCriticalPath);
  const clearTrace = useGraphData((s) => s.clearTrace);
  const [showCode, setShowCode] = useState(true);

  if (!selectedNode) return null;

  const pos = nodePositions.get(selectedNode.ref_id);
  if (!pos) return null;

  const relatedIds = [
    ...(selectedNode.sources || []),
    ...(selectedNode.targets || []),
  ];
  const relatedNodes = relatedIds
    .map((id) => nodesNormalized.get(id))
    .filter(Boolean)
    .slice(0, 20);

  const hasBody = !!selectedNode.properties.body;
  const hasTrace = tracedPath && tracedPath.nodeIds.size > 0;
  const isTraceRoot = hasTrace && tracedPath?.rootId === selectedNode.ref_id;

  return (
    <Html
      position={[pos.x + 100, pos.y, pos.z]}
      style={{ pointerEvents: "auto" }}
      className="html-panel"
      distanceFactor={400}
    >
      <div
        onPointerDown={(e) => e.stopPropagation()}
        onPointerUp={(e) => e.stopPropagation()}
        onClick={(e) => e.stopPropagation()}
        onWheel={(e) => e.stopPropagation()}
        style={{
          background: "rgba(16, 16, 24, 0.96)",
          border: "1px solid rgba(255,255,255,0.1)",
          borderRadius: 10,
          padding: 16,
          width: 360,
          maxHeight: 520,
          overflowY: "auto",
          color: "white",
          fontFamily: "system-ui, sans-serif",
          fontSize: 13,
          backdropFilter: "blur(12px)",
          boxShadow: "0 8px 32px rgba(0,0,0,0.5)",
        }}
      >
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "start" }}>
          <div style={{ flex: 1, minWidth: 0 }}>
            <div
              style={{
                fontSize: 10,
                textTransform: "uppercase",
                letterSpacing: 1,
                color: getColorForType(selectedNode.node_type),
                marginBottom: 4,
              }}
            >
              {selectedNode.node_type}
            </div>
            <div style={{ fontSize: 15, fontWeight: 600, marginBottom: 6, wordBreak: "break-word" }}>
              {selectedNode.properties.name}
            </div>
          </div>
          <button
            onClick={() => {
              setSelectedNode(null);
              if (hasTrace) clearTrace();
            }}
            style={{
              background: "none",
              border: "none",
              color: "#888",
              cursor: "pointer",
              fontSize: 18,
              lineHeight: 1,
              padding: "0 4px",
            }}
          >
            ×
          </button>
        </div>

        {selectedNode.properties.file && (
          <div
            style={{
              fontSize: 11,
              color: "#999",
              marginBottom: 8,
              wordBreak: "break-all",
              fontFamily: "monospace",
            }}
          >
            {selectedNode.properties.file}
            {selectedNode.properties.start != null && `:${selectedNode.properties.start}`}
            {selectedNode.properties.end != null && `-${selectedNode.properties.end}`}
          </div>
        )}

        <MetricsRow node={selectedNode} />

        <div style={{ display: "flex", gap: 6, marginBottom: 10 }}>
          {hasBody && (
            <button
              onClick={() => setShowCode(!showCode)}
              style={{
                fontSize: 10,
                padding: "3px 8px",
                borderRadius: 4,
                border: "1px solid rgba(255,255,255,0.15)",
                background: showCode ? "rgba(255,255,255,0.1)" : "transparent",
                color: showCode ? "#fff" : "#888",
                cursor: "pointer",
                textTransform: "uppercase",
                letterSpacing: 0.5,
              }}
            >
              Code
            </button>
          )}
          <button
            onClick={() => {
              if (isTraceRoot) {
                clearTrace();
              } else {
                traceCriticalPath(selectedNode.ref_id);
              }
            }}
            style={{
              fontSize: 10,
              padding: "3px 8px",
              borderRadius: 4,
              border: `1px solid ${isTraceRoot ? "#FFD70060" : "rgba(255,255,255,0.15)"}`,
              background: isTraceRoot ? "#FFD70020" : "transparent",
              color: isTraceRoot ? "#FFD700" : "#888",
              cursor: "pointer",
              textTransform: "uppercase",
              letterSpacing: 0.5,
            }}
          >
            {isTraceRoot ? "Clear Trace" : "Trace Path ↓"}
          </button>
        </div>

        {hasBody && showCode && (
          <div style={{ marginBottom: 10 }}>
            <CodeViewer
              body={selectedNode.properties.body as string}
              file={selectedNode.properties.file || "file.txt"}
              startLine={(selectedNode.properties.start as number) || 1}
            />
          </div>
        )}

        {hasTrace && !isTraceRoot && (
          <div style={{ fontSize: 11, color: "#FFD700", marginBottom: 8, display: "flex", alignItems: "center", gap: 4 }}>
            <span style={{ fontSize: 14 }}>⚡</span>
            Tracing from {nodesNormalized.get(tracedPath!.rootId)?.properties.name || "..."}
          </div>
        )}

        {relatedNodes.length > 0 && (
          <>
            <div
              style={{
                fontSize: 11,
                color: "#888",
                textTransform: "uppercase",
                letterSpacing: 1,
                marginBottom: 6,
              }}
            >
              Connected ({relatedIds.length})
            </div>
            {relatedNodes.map((node) => (
              <div
                key={node!.ref_id}
                onClick={() => setSelectedNode(node!)}
                style={{
                  padding: "5px 8px",
                  marginBottom: 2,
                  borderRadius: 4,
                  cursor: "pointer",
                  display: "flex",
                  alignItems: "center",
                  gap: 8,
                }}
                onMouseEnter={(e) => {
                  (e.currentTarget as HTMLDivElement).style.background = "rgba(255,255,255,0.05)";
                }}
                onMouseLeave={(e) => {
                  (e.currentTarget as HTMLDivElement).style.background = "transparent";
                }}
              >
                <span
                  style={{
                    width: 8,
                    height: 8,
                    borderRadius: "50%",
                    background: getColorForType(node!.node_type),
                    flexShrink: 0,
                  }}
                />
                <span style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", flex: 1 }}>
                  {node!.properties.name}
                </span>
                <span style={{ fontSize: 10, color: "#666", marginLeft: "auto", flexShrink: 0 }}>
                  {node!.node_type}
                </span>
              </div>
            ))}
          </>
        )}
      </div>
    </Html>
  );
});

NodeDetailsPanel.displayName = "NodeDetailsPanel";
