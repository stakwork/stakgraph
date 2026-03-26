import { memo } from "react";
import { Html } from "@react-three/drei";
import { useGraphData, getColorForType } from "@/stores/useGraphData";
import { nodePositions } from "@/stores/useSimulation";

export const NodeDetailsPanel = memo(() => {
  const selectedNode = useGraphData((s) => s.selectedNode);
  const nodesNormalized = useGraphData((s) => s.nodesNormalized);
  const setSelectedNode = useGraphData((s) => s.setSelectedNode);

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

  return (
    <Html
      position={[pos.x + 100, pos.y, pos.z]}
      style={{ pointerEvents: "auto" }}
      className="html-panel"
      distanceFactor={400}
    >
      <div
        style={{
          background: "rgba(20, 20, 30, 0.95)",
          border: "1px solid rgba(255,255,255,0.1)",
          borderRadius: 8,
          padding: 16,
          width: 280,
          maxHeight: 400,
          overflowY: "auto",
          color: "white",
          fontFamily: "system-ui, sans-serif",
          fontSize: 13,
        }}
      >
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "start" }}>
          <div>
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
            <div style={{ fontSize: 15, fontWeight: 600, marginBottom: 8 }}>
              {selectedNode.properties.name}
            </div>
          </div>
          <button
            onClick={() => setSelectedNode(null)}
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
            x
          </button>
        </div>

        {selectedNode.properties.file && (
          <div
            style={{
              fontSize: 11,
              color: "#999",
              marginBottom: 12,
              wordBreak: "break-all",
            }}
          >
            {selectedNode.properties.file}
            {selectedNode.properties.start != null &&
              `:${selectedNode.properties.start}`}
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
              Related ({relatedIds.length})
            </div>
            {relatedNodes.map((node) => (
              <div
                key={node!.ref_id}
                onClick={() => setSelectedNode(node!)}
                style={{
                  padding: "6px 8px",
                  marginBottom: 2,
                  borderRadius: 4,
                  cursor: "pointer",
                  display: "flex",
                  alignItems: "center",
                  gap: 8,
                }}
                onMouseEnter={(e) => {
                  (e.target as HTMLDivElement).style.background =
                    "rgba(255,255,255,0.05)";
                }}
                onMouseLeave={(e) => {
                  (e.target as HTMLDivElement).style.background = "transparent";
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
                <span style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
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
