import { memo, useMemo } from "react";
import { Instance, Instances } from "@react-three/drei";
import { SphereGeometry, BufferGeometry } from "three";
import { useGraphData, getColorForType } from "@/stores/useGraphData";
import type { NodeExtended } from "@/graph/types";

const NODE_SIZE = 20;

const NodePointsComponent = () => {
  const data = useGraphData((s) => s.data);
  const nodeTypes = useGraphData((s) => s.nodeTypes);

  const sharedGeometry = useMemo(
    () => new SphereGeometry(NODE_SIZE / 2, 16, 8),
    []
  );

  const nodeInstanceData = useMemo(() => {
    if (!data?.nodes) return [];

    return data.nodes.map((node: NodeExtended) => {
      const weight = (node.properties?.weight as number) || 1;
      const scale = Math.cbrt(weight);
      const color = getColorForType(node.node_type);

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
  }, [data?.nodes, nodeTypes]);

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
