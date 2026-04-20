import { useRef } from "react";
import { Mesh, AdditiveBlending, Color } from "three";
import { useFrame } from "@react-three/fiber";
import { useGraphData } from "@/stores/useGraphData";
import { nodePositions } from "@/stores/useSimulation";
import { NODE_SIZE } from "@/graph/config";

const SELECTED_COLOR = new Color("#9999CC");
const HOVERED_COLOR = new Color("#FFFFFF");
const SELECTED_SCALE = NODE_SIZE * 1.6;
const HOVERED_SCALE = NODE_SIZE * 1.4;

export function NodeHoverHighlight() {
  const hoverRef = useRef<Mesh>(null!);
  const selectedRef = useRef<Mesh>(null!);

  const hoveredNode = useGraphData((s) => s.hoveredNode);
  const selectedNode = useGraphData((s) => s.selectedNode);

  useFrame(() => {
    const hMesh = hoverRef.current;
    const sMesh = selectedRef.current;
    if (!hMesh || !sMesh) return;

    // Selected mesh
    if (selectedNode) {
      const pos = nodePositions.get(selectedNode.ref_id);
      if (pos) {
        sMesh.position.set(pos.x, pos.y, pos.z);
        sMesh.visible = true;
      } else {
        sMesh.visible = false;
      }
    } else {
      sMesh.visible = false;
    }

    // Hovered mesh — hide if same node as selected
    const isSameAsSelected =
      hoveredNode && selectedNode && hoveredNode.ref_id === selectedNode.ref_id;

    if (hoveredNode && !isSameAsSelected) {
      const pos = nodePositions.get(hoveredNode.ref_id);
      if (pos) {
        hMesh.position.set(pos.x, pos.y, pos.z);
        hMesh.visible = true;
      } else {
        hMesh.visible = false;
      }
    } else {
      hMesh.visible = false;
    }
  });

  return (
    <>
      {/* Hovered glow mesh */}
      <mesh ref={hoverRef} visible={false}>
        <sphereGeometry args={[HOVERED_SCALE, 12, 12]} />
        <meshBasicMaterial
          color={HOVERED_COLOR}
          transparent
          blending={AdditiveBlending}
          depthWrite={false}
          opacity={0.35}
        />
      </mesh>
      {/* Selected glow mesh */}
      <mesh ref={selectedRef} visible={false}>
        <sphereGeometry args={[SELECTED_SCALE, 12, 12]} />
        <meshBasicMaterial
          color={SELECTED_COLOR}
          transparent
          blending={AdditiveBlending}
          depthWrite={false}
          opacity={0.35}
        />
      </mesh>
    </>
  );
}
