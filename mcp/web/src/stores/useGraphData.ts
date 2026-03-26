import { create } from "zustand";
import type {
  NodeExtended,
  Link,
  GraphData,
  GraphNode,
  GraphEdge,
} from "@/graph/types";

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

const DEFAULT_COLOR = "#F8F8FF";

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

  // Actions
  setData: (nodes: GraphNode[], edges: GraphEdge[]) => void;
  setSelectedNode: (node: NodeExtended | null) => void;
  setHoveredNode: (node: NodeExtended | null) => void;
  reset: () => void;
}

export const useGraphData = create<GraphDataState>((set) => ({
  data: null,
  nodesNormalized: new Map(),
  nodeTypes: [],
  loading: true,

  selectedNode: null,
  hoveredNode: null,

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

    // Fixed layer order (top to bottom)
    const LAYER_ORDER = [
      "Repository",
      "Directory",
      "File",
      "Feature",
      "PullRequest",
      "Commit",
      "Class",
      "Function",
      "Datamodel",
      "Endpoint",
      "Request",
      "Var",
      "Page",
    ];

    // Collect present node types, ordered by LAYER_ORDER then remainder alphabetically
    const presentTypes = new Set<string>();
    for (const n of nodes) {
      presentTypes.add(n.node_type);
    }
    const ordered = LAYER_ORDER.filter((t) => presentTypes.has(t));
    const rest = Array.from(presentTypes)
      .filter((t) => !LAYER_ORDER.includes(t))
      .sort();
    const nodeTypes = [...ordered, ...rest];

    set({
      data: { nodes, links },
      nodesNormalized: nodesMap,
      nodeTypes,
      loading: false,
    });
  },

  setSelectedNode: (node) => set({ selectedNode: node }),
  setHoveredNode: (node) => set({ hoveredNode: node }),

  reset: () =>
    set({
      data: null,
      nodesNormalized: new Map(),
      nodeTypes: [],
      loading: true,
      selectedNode: null,
      hoveredNode: null,
    }),
}));
