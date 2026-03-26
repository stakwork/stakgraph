import { memo, useMemo } from "react";
import { Instance, Instances } from "@react-three/drei";
import { SphereGeometry, BufferGeometry } from "three";
import { useGraphData, getColorForType } from "@/stores/useGraphData";
import type { NodeExtended } from "@/graph/types";
import { NODE_SIZE } from "@/graph/config";

const DIMMED_COLOR = "#555566";

const NodePointsComponent = () => {
  const data = useGraphData((s) => s.data);
  const nodeTypes = useGraphData((s) => s.nodeTypes);
  const highlightedFeatureId = useGraphData((s) => s.highlightedFeatureId);
  const highlightedNodeIds = useGraphData((s) => s.highlightedNodeIds);

  const sharedGeometry = useMemo(
    () => new SphereGeometry(NODE_SIZE / 2, 16, 8),
    []
  );

  const nodeInstanceData = useMemo(() => {
    if (!data?.nodes) return [];

    return data.nodes.map((node: NodeExtended) => {
      const weight = (node.properties?.weight as number) || 1;
      const scale = Math.cbrt(weight);
      const baseColor = getColorForType(node.node_type);

      const isDimmed =
        highlightedFeatureId !== null && !highlightedNodeIds.has(node.ref_id);
      const color = isDimmed ? DIMMED_COLOR : baseColor;

      return {
        key: node.ref_id,
        color,
        scale: Math.max(0.5, Math.min(2, scale)),
        node,
        position: [node.x || 0, node.y || 0, node.z || 0] as [
          number,
          number,
          number,
        ],
      };
    });
  }, [data?.nodes, nodeTypes, highlightedFeatureId, highlightedNodeIds]);

  return (
    <Instances
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
  );
};

export const NodePoints = memo(NodePointsComponent);
