import { memo } from "react";
import { Billboard, Text } from "@react-three/drei";
import { useGraphData, getColorForType } from "@/stores/useGraphData";

const LAYER_SPACING = 500;

export const LayerLabels = memo(() => {
  const nodeTypes = useGraphData((s) => s.nodeTypes);
  const totalTypes = nodeTypes.length;
  const startOffset = ((totalTypes - 1) / 2) * LAYER_SPACING;

  return (
    <>
      {nodeTypes.map((nodeType, i) => {
        const yOffset = startOffset - i * LAYER_SPACING;
        return (
          <Billboard
            key={nodeType}
            position={[-1500, yOffset, 0]}
            follow={false}
          >
            <Text
              fontSize={40}
              color={getColorForType(nodeType)}
              anchorX="right"
              anchorY="middle"
              material-toneMapped={false}
            >
              {nodeType}
            </Text>
          </Billboard>
        );
      })}
    </>
  );
});

LayerLabels.displayName = "LayerLabels";
