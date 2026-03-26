import { memo, useRef, useCallback } from "react";
import { Group } from "three";
import { ThreeEvent, useFrame } from "@react-three/fiber";
import { NodePoints } from "./NodePoints";
import { Edges } from "./Edges";
import { LayerLabels } from "./LayerLabels";
import { LayerHoverHighlight } from "./LayerHoverHighlight";
import { NodeDetailsPanel } from "./NodeDetailsPanel";
import { useGraphData } from "@/stores/useGraphData";
import { nodePositions, useSimulation } from "@/stores/useSimulation";

const POINTER_DRAG_THRESHOLD = 5;

export const Graph = memo(() => {
  const groupRef = useRef<Group>(null);
  const downPos = useRef<{ x: number; y: number } | null>(null);
  const positioned = useRef(false);

  const nodesNormalized = useGraphData((s) => s.nodesNormalized);
  const selectedNode = useGraphData((s) => s.selectedNode);
  const setSelectedNode = useGraphData((s) => s.setSelectedNode);
  const setHoveredNode = useGraphData((s) => s.setHoveredNode);
  const ready = useSimulation((s) => s.ready);

  // Set instance positions from the precomputed grid
  useFrame(() => {
    if (!ready || positioned.current || !groupRef.current) return;

    const grPoints = groupRef.current.getObjectByName(
      "node-points-group"
    ) as Group | null;
    if (!grPoints) return;

    const instancedMesh = grPoints.children[0];
    if (!instancedMesh || instancedMesh.children.length === 0) return;

    const data = useGraphData.getState().data;
    if (!data) return;

    const count = Math.min(data.nodes.length, instancedMesh.children.length);

    for (let i = 0; i < count; i++) {
      const node = data.nodes[i];
      const pos = nodePositions.get(node.ref_id);
      if (pos && instancedMesh.children[i]) {
        instancedMesh.children[i].position.set(pos.x, pos.y, pos.z);
      }
    }

    positioned.current = true;
  });

  const handlePointerDown = useCallback((e: ThreeEvent<PointerEvent>) => {
    downPos.current = { x: e.clientX, y: e.clientY };
  }, []);

  const handleClick = useCallback(
    (e: ThreeEvent<PointerEvent>) => {
      e.stopPropagation();

      if (downPos.current) {
        const dx = e.clientX - downPos.current.x;
        const dy = e.clientY - downPos.current.y;
        if (Math.hypot(dx, dy) > POINTER_DRAG_THRESHOLD) return;
      }

      const obj = e.intersections[0]?.object;
      if (!obj?.userData?.ref_id) return;

      const node = nodesNormalized.get(obj.userData.ref_id as string);
      if (node) {
        setSelectedNode(
          selectedNode?.ref_id === node.ref_id ? null : node
        );
      }
    },
    [nodesNormalized, selectedNode, setSelectedNode]
  );

  const handlePointerOver = useCallback(
    (e: ThreeEvent<PointerEvent>) => {
      const obj = e.intersections[0]?.object;
      if (!obj?.userData?.ref_id) return;
      e.stopPropagation();
      const node = nodesNormalized.get(obj.userData.ref_id as string);
      if (node) setHoveredNode(node);
    },
    [nodesNormalized, setHoveredNode]
  );

  const handlePointerOut = useCallback(() => {
    setHoveredNode(null);
  }, [setHoveredNode]);

  return (
    <group ref={groupRef}>
      <group
        name="node-points-group"
        onClick={handleClick}
        onPointerDown={handlePointerDown}
        onPointerOver={handlePointerOver}
        onPointerOut={handlePointerOut}
      >
        <NodePoints />
      </group>
      <Edges />
      <LayerLabels />
      <LayerHoverHighlight />
      <NodeDetailsPanel />
    </group>
  );
});

Graph.displayName = "Graph";
