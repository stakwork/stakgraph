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

const API_BASE = import.meta.env.VITE_API_BASE || "";

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

  // Importance filter
  importanceFilter: { tag: string | null; nodeType: string | null };

  // Critical path trace (downstream)
  tracedPath: { rootId: string; nodeIds: Set<string>; edgeKeys: Set<string> } | null;

  // Actions
  setData: (nodes: GraphNode[], edges: GraphEdge[]) => void;
  addNodes: (nodes: GraphNode[], edges: GraphEdge[]) => void;
  setSelectedNode: (node: NodeExtended | null) => void;
  setHoveredNode: (node: NodeExtended | null) => void;
  setHighlightedFeature: (featureRefId: string | null) => void;
  setImportanceFilter: (tag: string | null, nodeType?: string | null) => void;
  fetchNodeBody: (refId: string) => void;
  traceCriticalPath: (refId: string) => void;
  clearTrace: () => void;
  reset: () => void;
}

export const useGraphData = create<GraphDataState>((set) => ({
  data: null,
  nodesNormalized: new Map(),
  nodeTypes: [],
  loading: false,

  selectedNode: null,
  hoveredNode: null,
  highlightedFeatureId: null,
  highlightedNodeIds: new Set(),
  importanceFilter: { tag: null, nodeType: null },
  tracedPath: null,

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

      if (
        sourceRefId &&
        targetRefId &&
        nodesMap.has(sourceRefId) &&
        nodesMap.has(targetRefId)
      ) {
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

  addNodes: (rawNodes: GraphNode[], rawEdges: GraphEdge[]) => {
    const state = useGraphData.getState();
    const nodesMap = new Map(state.nodesNormalized);
    const nodeKeyToRefId = new Map<string, string>();

    // Re-build nodeKey index from existing nodes
    for (const [refId, node] of nodesMap) {
      const nodeKey = node.properties?.node_key as string | undefined;
      if (nodeKey) nodeKeyToRefId.set(nodeKey, refId);
    }

    // Add only new nodes
    const nextIndex = state.data ? state.data.nodes.length : 0;
    const newNodes: NodeExtended[] = [];
    for (let i = 0; i < rawNodes.length; i++) {
      const n = rawNodes[i];
      if (nodesMap.has(n.ref_id)) continue;
      const extended: NodeExtended = {
        ...n,
        x: 0,
        y: 0,
        z: 0,
        sources: [],
        targets: [],
        index: nextIndex + newNodes.length,
      };
      nodesMap.set(n.ref_id, extended);
      const nodeKey = n.properties?.node_key as string | undefined;
      if (nodeKey) nodeKeyToRefId.set(nodeKey, n.ref_id);
      newNodes.push(extended);
    }

    // Add only new edges
    const existingLinks = state.data ? [...state.data.links] : [];
    const existingLinkKeys = new Set(existingLinks.map((l) => l.ref_id));
    for (const e of rawEdges) {
      const rawSource =
        typeof e.source === "string" ? e.source : String(e.source);
      const rawTarget =
        typeof e.target === "string" ? e.target : String(e.target);
      const sourceRefId = nodesMap.has(rawSource)
        ? rawSource
        : nodeKeyToRefId.get(rawSource);
      const targetRefId = nodesMap.has(rawTarget)
        ? rawTarget
        : nodeKeyToRefId.get(rawTarget);
      if (!sourceRefId || !targetRefId) continue;
      const linkId = e.ref_id || `${sourceRefId}-${targetRefId}`;
      if (existingLinkKeys.has(linkId)) continue;
      existingLinkKeys.add(linkId);
      existingLinks.push({
        source: sourceRefId,
        target: targetRefId,
        ref_id: linkId,
        edge_type: e.edge_type,
      });
      const sourceNode = nodesMap.get(sourceRefId)!;
      const targetNode = nodesMap.get(targetRefId)!;
      sourceNode.targets = sourceNode.targets || [];
      sourceNode.targets.push(targetRefId);
      targetNode.sources = targetNode.sources || [];
      targetNode.sources.push(sourceRefId);
    }

    if (
      newNodes.length === 0 &&
      existingLinks.length === (state.data?.links.length ?? 0)
    )
      return;

    const layerSet = new Set(LAYER_ORDER);
    const presentTypes = new Set<string>();
    for (const n of nodesMap.values()) {
      if (layerSet.has(n.node_type)) presentTypes.add(n.node_type);
    }
    const nodeTypes = LAYER_ORDER.filter((t) => presentTypes.has(t));

    set({
      data: { nodes: Array.from(nodesMap.values()), links: existingLinks },
      nodesNormalized: nodesMap,
      nodeTypes,
      loading: false,
    });
  },

  setSelectedNode: (node) => {
    set({ selectedNode: node });
    if (node && !node.properties.body) {
      useGraphData.getState().fetchNodeBody(node.ref_id);
    }
  },
  setHoveredNode: (node) => set({ hoveredNode: node }),

  fetchNodeBody: async (refId: string) => {
    try {
      const res = await fetch(`${API_BASE}/subgraph?ref_id=${encodeURIComponent(refId)}`);
      if (!res.ok) return;
      const result = await res.json();
      const body = result?.node?.properties?.body;
      if (!body) return;
      const { nodesNormalized, selectedNode, data } = useGraphData.getState();
      const existing = nodesNormalized.get(refId);
      if (!existing) return;
      existing.properties.body = body;
      nodesNormalized.set(refId, existing);
      if (data) {
        const idx = data.nodes.findIndex((n) => n.ref_id === refId);
        if (idx >= 0) data.nodes[idx].properties.body = body;
      }
      if (selectedNode?.ref_id === refId) {
        set({ selectedNode: { ...existing } });
      }
    } catch (e) {
      console.error("[fetchNodeBody]", e);
    }
  },

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

  setImportanceFilter: (tag: string | null, nodeType: string | null = null) => {
    set({ importanceFilter: { tag, nodeType } });
  },

  traceCriticalPath: (refId: string) => {
    const { nodesNormalized } = useGraphData.getState();
    const nodeIds = new Set<string>();
    const edgeKeys = new Set<string>();
    const maxDepth = 8;
    const queue: { id: string; depth: number }[] = [{ id: refId, depth: 0 }];

    while (queue.length > 0) {
      const { id, depth } = queue.shift()!;
      if (nodeIds.has(id)) continue;
      nodeIds.add(id);
      if (depth >= maxDepth) continue;
      const node = nodesNormalized.get(id);
      if (!node) continue;
      for (const targetId of node.targets || []) {
        if (!nodeIds.has(targetId)) {
          edgeKeys.add(`${id}->${targetId}`);
          queue.push({ id: targetId, depth: depth + 1 });
        }
      }
    }

    set({ tracedPath: { rootId: refId, nodeIds, edgeKeys } });
  },

  clearTrace: () => set({ tracedPath: null }),

  reset: () =>
    set({
      data: null,
      nodesNormalized: new Map(),
      nodeTypes: [],
      loading: false,
      selectedNode: null,
      hoveredNode: null,
      highlightedFeatureId: null,
      highlightedNodeIds: new Set(),
      importanceFilter: { tag: null, nodeType: null },
      tracedPath: null,
    }),
}));
