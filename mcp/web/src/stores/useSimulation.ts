import { create } from "zustand";
import type { NodeExtended } from "@/graph/types";

const LAYER_SPACING = 500;
const GRID_SPACING = 300;

// Mutable position map — read imperatively, never triggers React re-renders
export const nodePositions = new Map<
  string,
  { x: number; y: number; z: number }
>();

// Deterministic square grid layout per layer — ported from hive's calculateGridMap
function calculateGridMap(
  nodes: NodeExtended[],
  nodeTypes: string[]
): Map<string, { x: number; y: number; z: number }> {
  const normalizeType = (type?: string) => (type || "Unknown").trim();
  const nodesByType: Record<string, NodeExtended[]> = {};

  // 1. Group by type
  nodes.forEach((node) => {
    const typeKey = normalizeType(node.node_type);
    if (!nodesByType[typeKey]) nodesByType[typeKey] = [];
    nodesByType[typeKey].push(node);
  });

  const providedOrder = Array.from(new Set(nodeTypes.map(normalizeType)));
  const allTypes = Array.from(new Set(Object.keys(nodesByType)));
  const typeOrder = providedOrder.length ? providedOrder : allTypes;

  if (typeOrder.length === 0) {
    return new Map();
  }

  const typeIndexMap = new Map(
    typeOrder.map((type, index) => [type, index])
  );

  const positionMap = new Map<
    string,
    { x: number; y: number; z: number }
  >();

  // 2. Calculate positions for each type
  nodes.forEach((n) => {
    const typeKey = normalizeType(n.node_type);
    const typeIndex = typeIndexMap.get(typeKey) ?? typeOrder.length - 1;

    // Position layers from top to bottom to match LayerLabels ordering
    const totalTypes = typeOrder.length;
    const startOffset = ((totalTypes - 1) / 2) * LAYER_SPACING;
    const yOffset = startOffset - typeIndex * LAYER_SPACING;

    const sameTypeNodes = nodesByType[typeKey] || [];
    const nodeIndexInType = sameTypeNodes.findIndex(
      (node) => node.ref_id === n.ref_id
    );

    const nodesPerRow = Math.ceil(Math.sqrt(sameTypeNodes.length));

    const row = Math.floor(nodeIndexInType / nodesPerRow);
    const col = nodeIndexInType % nodesPerRow;

    const gridWidth = (nodesPerRow - 1) * GRID_SPACING;
    const gridHeight =
      (Math.ceil(sameTypeNodes.length / nodesPerRow) - 1) * GRID_SPACING;

    const x = col * GRID_SPACING - gridWidth / 2;
    const z = row * GRID_SPACING - gridHeight / 2;

    positionMap.set(n.ref_id, { x, y: yOffset, z });
  });

  // 3. Center the entire grid around (0,0,0)
  const positions = Array.from(positionMap.values());
  if (positions.length > 0) {
    const minX = Math.min(...positions.map((p) => p.x));
    const maxX = Math.max(...positions.map((p) => p.x));
    const minZ = Math.min(...positions.map((p) => p.z));
    const maxZ = Math.max(...positions.map((p) => p.z));

    const centerX = (minX + maxX) / 2;
    const centerZ = (minZ + maxZ) / 2;

    for (const [nodeId, pos] of positionMap.entries()) {
      positionMap.set(nodeId, {
        x: pos.x - centerX,
        y: pos.y,
        z: pos.z - centerZ,
      });
    }
  }

  return positionMap;
}

interface SimulationState {
  ready: boolean;

  layoutNodes: (nodes: NodeExtended[], nodeTypes: string[]) => void;
  destroy: () => void;
}

export const useSimulation = create<SimulationState>((set) => ({
  ready: false,

  layoutNodes: (nodes: NodeExtended[], nodeTypes: string[]) => {
    nodePositions.clear();
    const grid = calculateGridMap(nodes, nodeTypes);
    for (const [id, pos] of grid.entries()) {
      nodePositions.set(id, pos);
    }
    set({ ready: true });
  },

  destroy: () => {
    nodePositions.clear();
    set({ ready: false });
  },
}));
