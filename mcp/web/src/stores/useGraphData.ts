import { create } from "zustand";
import type {
  NodeExtended,
  Link,
  GraphData,
  GraphNode,
  GraphEdge,
} from "@/graph/types";

import { LAYER_ORDER } from "@/graph/config";

// Re-export for consumers that import from here
export { LAYER_ORDER };

// Color palette by node type
const COLORS: Record<string, string> = {
  Function: "#5C6BC0",
  Endpoint: "#0288D1",
  Class: "#9747FF",
  Trait: "#7E57C2",
  Datamodel: "#00887A",
  File: "#689F39",
  Page: "#EC407A",
  Import: "#78909C",
  Library: "#8C6E63",
  Var: "#FFC064",
  Feature: "#FF7F50",
  Directory: "#465A65",
  UnitTest: "#33691E",
  IntegrationTest: "#0098A6",
  Repository: "#BF360C",
};

const DEFAULT_COLOR = "#78909C";

export function getColorForType(nodeType: string): string {
  return COLORS[nodeType] || DEFAULT_COLOR;
}

interface GraphDataState {
  // Data
  data: GraphData | null;
  nodesNormalized: Map<string, NodeExtended>;
  nodeTypes: string[];
  loading: boolean;

  // Selection
  selectedNode: NodeExtended | null;
  hoveredNode: NodeExtended | null;

  // Feature highlight (from sidebar hover)
  highlightedFeatureId: string | null;
  highlightedNodeIds: Set<string>;

  // Actions
  setData: (nodes: GraphNode[], edges: GraphEdge[]) => void;
  setSelectedNode: (node: NodeExtended | null) => void;
  setHoveredNode: (node: NodeExtended | null) => void;
  setHighlightedFeature: (featureRefId: string | null) => void;
  reset: () => void;
}

export const useGraphData = create<GraphDataState>((set) => ({
  data: null,
  nodesNormalized: new Map(),
  nodeTypes: [],
  loading: true,

  selectedNode: null,
  hoveredNode: null,
  highlightedFeatureId: null,
  highlightedNodeIds: new Set(),

  setData: (rawNodes: GraphNode[], rawEdges: GraphEdge[]) => {
    const nodesMap = new Map<string, NodeExtended>();
    // Also index by node_key since edges use node_key as source/target
    const nodeKeyToRefId = new Map<string, string>();

    // Build node map
    const nodes: NodeExtended[] = rawNodes.map((n, i) => {
      const extended: NodeExtended = {
        ...n,
        x: 0,
        y: 0,
        z: 0,
        sources: [],
        targets: [],
        index: i,
      };
      nodesMap.set(n.ref_id, extended);
      // Map node_key -> ref_id for edge resolution
      const nodeKey = n.properties?.node_key as string | undefined;
      if (nodeKey) {
        nodeKeyToRefId.set(nodeKey, n.ref_id);
      }
      return extended;
    });

    // Build links, filtering out edges with missing nodes
    // Edge source/target can be either ref_id or node_key
    const links: Link[] = [];
    for (const e of rawEdges) {
      const rawSource =
        typeof e.source === "string" ? e.source : String(e.source);
      const rawTarget =
        typeof e.target === "string" ? e.target : String(e.target);

      // Resolve to ref_id (try direct match first, then node_key lookup)
      const sourceRefId = nodesMap.has(rawSource)
        ? rawSource
        : nodeKeyToRefId.get(rawSource);
      const targetRefId = nodesMap.has(rawTarget)
        ? rawTarget
        : nodeKeyToRefId.get(rawTarget);

      if (sourceRefId && targetRefId && nodesMap.has(sourceRefId) && nodesMap.has(targetRefId)) {
        links.push({
          source: sourceRefId,
          target: targetRefId,
          ref_id: e.ref_id || `${sourceRefId}-${targetRefId}`,
          edge_type: e.edge_type,
        });

        const sourceNode = nodesMap.get(sourceRefId)!;
        const targetNode = nodesMap.get(targetRefId)!;
        sourceNode.targets = sourceNode.targets || [];
        sourceNode.targets.push(targetRefId);
        targetNode.sources = targetNode.sources || [];
        targetNode.sources.push(sourceRefId);
      }
    }

    

    // Only keep node types in LAYER_ORDER
    const layerSet = new Set(LAYER_ORDER);
    const presentTypes = new Set<string>();
    for (const n of nodes) {
      if (layerSet.has(n.node_type)) presentTypes.add(n.node_type);
    }
    const nodeTypes = LAYER_ORDER.filter((t) => presentTypes.has(t));

    set({
      data: { nodes, links },
      nodesNormalized: nodesMap,
      nodeTypes,
      loading: false,
    });
  },

  setSelectedNode: (node) => set({ selectedNode: node }),
  setHoveredNode: (node) => set({ hoveredNode: node }),

  setHighlightedFeature: (featureRefId: string | null) => {
    if (!featureRefId) {
      set({ highlightedFeatureId: null, highlightedNodeIds: new Set() });
      return;
    }

    const { nodesNormalized, data } = useGraphData.getState();
    const node = nodesNormalized.get(featureRefId);
    if (!node || !data) {
      set({ highlightedFeatureId: null, highlightedNodeIds: new Set() });
      return;
    }

    // Collect the feature + all directly connected nodes + their children
    const connected = new Set<string>();
    connected.add(featureRefId);

    // First level: direct connections
    const directIds: string[] = [];
    for (const id of node.sources || []) {
      connected.add(id);
      directIds.push(id);
    }
    for (const id of node.targets || []) {
      connected.add(id);
      directIds.push(id);
    }

    // Second level: children of all directly connected nodes
    for (const id of directIds) {
      const child = nodesNormalized.get(id);
      if (!child) continue;
      for (const cid of child.sources || []) connected.add(cid);
      for (const cid of child.targets || []) connected.add(cid);
    }

    set({ highlightedFeatureId: featureRefId, highlightedNodeIds: connected });
  },

  reset: () =>
    set({
      data: null,
      nodesNormalized: new Map(),
      nodeTypes: [],
      loading: true,
      selectedNode: null,
      hoveredNode: null,
      highlightedFeatureId: null,
      highlightedNodeIds: new Set(),
    }),
}));
