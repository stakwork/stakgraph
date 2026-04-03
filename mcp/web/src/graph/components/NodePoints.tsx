import { memo, useMemo } from "react";
import { Instance, Instances } from "@react-three/drei";
import { SphereGeometry, BufferGeometry, AdditiveBlending } from "three";
import {
  useGraphData,
  getColorForType,
  type TraceMode,
} from "@/stores/useGraphData";
import { useLayerVisibility } from "@/stores/useLayerVisibility";
import type { NodeExtended } from "@/graph/types";
import { NODE_SIZE } from "@/graph/config";

const DIMMED_COLOR = "#333344";

const IMPORTANCE_TAG_COLORS: Record<string, string> = {
  entry_point: "#FFD700",
  hub: "#FF4444",
  utility: "#00BCD4",
  connector: "#6688AA",
  isolated: "#444455",
};

function getImportanceColor(tag: string): string {
  return IMPORTANCE_TAG_COLORS[tag] || "#6688AA";
}

const TRACE_GLOW_COLORS: Record<TraceMode, string> = {
  down: "#FFD36B",
  up: "#82D7D0",
  both: "#BCAEFF",
};

function getTraceGlowColor(mode: TraceMode): string {
  return TRACE_GLOW_COLORS[mode] || TRACE_GLOW_COLORS.down;
}

const NodePointsComponent = () => {
  const data = useGraphData((s) => s.data);
  const nodeTypes = useGraphData((s) => s.nodeTypes);
  const highlightedFeatureId = useGraphData((s) => s.highlightedFeatureId);
  const highlightedNodeIds = useGraphData((s) => s.highlightedNodeIds);
  const importanceFilter = useGraphData((s) => s.importanceFilter);
  const tracedPath = useGraphData((s) => s.tracedPath);
  const disabledLayers = useLayerVisibility((s) => s.disabledLayers);

  const sharedGeometry = useMemo(
    () => new SphereGeometry(NODE_SIZE / 2, 16, 8),
    []
  );
  const glowGeometry = useMemo(
    () => new SphereGeometry(NODE_SIZE / 2, 16, 8),
    [],
  );

  const importanceActive = importanceFilter.tag !== null;
  const instancesKey = `${importanceFilter.tag || "none"}-${importanceFilter.nodeType || "all"}-${tracedPath?.rootId || ""}-${tracedPath?.mode || "none"}`;

  const nodeInstanceData = useMemo(() => {
    if (!data?.nodes) return [];

    return data.nodes
      .filter((node: NodeExtended) => !disabledLayers.has(node.node_type))
      .map((node: NodeExtended) => {
        const weight = (node.properties?.weight as number) || 1;
        const tag = node.properties?.importance_tag as string | undefined;
        const pagerank = node.properties?.pagerank as number | undefined;

        // Size: boost by pagerank when importance filter active
        let scale: number;
        if (importanceActive && pagerank != null) {
          scale = Math.max(
            0.6,
            Math.min(3.5, 1 + Math.log1p(pagerank * 1000) * 0.4),
          );
        } else {
          scale = Math.max(0.5, Math.min(2, Math.cbrt(weight)));
        }

        const baseColor = getColorForType(node.node_type);

        // Determine match for importance filter
        const matchesImportance =
          importanceActive &&
          tag === importanceFilter.tag &&
          (importanceFilter.nodeType === null ||
            node.node_type === importanceFilter.nodeType);

        // Feature highlight dimming
        const isFeatureDimmed =
          highlightedFeatureId !== null && !highlightedNodeIds.has(node.ref_id);
        // Importance filter dimming
        const isImportanceDimmed = importanceActive && !matchesImportance;
        // Trace path dimming
        const isTraceDimmed =
          tracedPath !== null && !tracedPath.nodeIds.has(node.ref_id);

        const color =
          isFeatureDimmed || isImportanceDimmed || isTraceDimmed
            ? DIMMED_COLOR
            : baseColor;

        return {
          key: node.ref_id,
          color,
          scale,
          node,
          position: [node.x || 0, node.y || 0, node.z || 0] as [
            number,
            number,
            number,
          ],
          glowColor: tracedPath?.nodeIds.has(node.ref_id)
            ? getTraceGlowColor(tracedPath.mode)
            : matchesImportance && tag
              ? getImportanceColor(tag)
              : null,
          glowScale: scale * 1.4,
        };
      });
  }, [
    data?.nodes,
    nodeTypes,
    highlightedFeatureId,
    highlightedNodeIds,
    importanceFilter,
    tracedPath,
    disabledLayers,
    importanceActive,
  ]);

  const glowNodes = useMemo(
    () => nodeInstanceData.filter((n) => n.glowColor !== null),
    [nodeInstanceData],
  );

  return (
    <>
      <Instances
        key={instancesKey}
        geometry={sharedGeometry as BufferGeometry}
        limit={100000}
        range={100000}
        frustumCulled={false}
      >
        <meshBasicMaterial />
        {nodeInstanceData.map(({ key, color, scale, node, position }) => (
          <Instance
            key={key}
            color={color}
            scale={scale}
            position={position}
            userData={node}
          />
        ))}
      </Instances>

      {glowNodes.length > 0 && (
        <Instances
          geometry={glowGeometry as BufferGeometry}
          limit={10000}
          range={10000}
          frustumCulled={false}
        >
          <meshBasicMaterial
            transparent
            opacity={0.18}
            blending={AdditiveBlending}
            depthWrite={false}
          />
          {glowNodes.map(({ key, glowColor, glowScale, position }) => (
            <Instance
              key={`glow-${key}`}
              color={glowColor!}
              scale={glowScale}
              position={position}
            />
          ))}
        </Instances>
      )}
    </>
  );
};

export const NodePoints = memo(NodePointsComponent);
