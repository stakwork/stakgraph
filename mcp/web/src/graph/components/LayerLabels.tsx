import { memo, useMemo } from "react";
import { Billboard, Text } from "@react-three/drei";
import { useGraphData, getColorForType } from "@/stores/useGraphData";
import { useLayerVisibility } from "@/stores/useLayerVisibility";
import { LAYER_SPACING } from "@/graph/config";

export const LayerLabels = memo(() => {
  const nodeTypes = useGraphData((s) => s.nodeTypes);
  const disabledLayers = useLayerVisibility((s) => s.disabledLayers);

  const labels = useMemo(() => {
    const totalTypes = nodeTypes.length;
    const startOffset = ((totalTypes - 1) / 2) * LAYER_SPACING;
    return nodeTypes.map((nodeType, i) => {
      const name = nodeType.replace(/_/g, " ");
      const yPosition = startOffset - i * LAYER_SPACING;
      const halfW = name.length * 12.5 + 16;
      const halfH = 36;
      return { nodeType, name, yPosition, halfW, halfH };
    });
  }, [nodeTypes]);

  return (
    <group>
      {labels.map(({ nodeType, name, yPosition, halfW, halfH }) => {
        const color = getColorForType(nodeType);
        return (
          <Billboard key={nodeType} position={[0, yPosition, 0]}>
            {/* Colored border rect */}
            <lineLoop position={[0, 0, -0.1]}>
              <bufferGeometry>
                <bufferAttribute
                  attach="attributes-position"
                  args={[
                    new Float32Array([
                      -halfW,
                      -halfH,
                      0,
                      halfW,
                      -halfH,
                      0,
                      halfW,
                      halfH,
                      0,
                      -halfW,
                      halfH,
                      0,
                    ]),
                    3,
                  ]}
                />
              </bufferGeometry>
              <lineBasicMaterial
                color={color}
                opacity={disabledLayers.has(nodeType) ? 0.15 : 0.6}
                transparent
              />
            </lineLoop>
            <Text
              fontSize={35}
              color={color}
              anchorX="center"
              anchorY="middle"
              material-toneMapped={false}
              fillOpacity={disabledLayers.has(nodeType) ? 0.2 : 1}
            >
              {name}
            </Text>
          </Billboard>
        );
      })}
    </group>
  );
});

LayerLabels.displayName = "LayerLabels";
