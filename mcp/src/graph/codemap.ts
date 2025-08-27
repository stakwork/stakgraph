import { Record } from "neo4j-driver";
import {
  deser_node,
  deser_multi,
  getNodeLabel,
  getNodeSummaryLabel,
} from "./utils.js";
import { TikTokenizer } from "@microsoft/tiktokenizer";
import { BoltInt, EdgeType, Neo4jNode } from "./types.js";

export interface TreeNode {
  label: string;
  nodes: TreeNode[];
}

// list the edge types which are "parents" that we want shown as "children" in the tree
const REVERSE_RELATIONSHIPS = ["OPERAND"];

interface Tree {
  root: TreeNode;
  total_tokens: number;
}

interface Relationship {
  source: BoltInt;
  target: BoltInt;
  type: EdgeType;
  properties: Record<any, any>;
}

export async function buildTree(
  record: Record,
  direction: string = "down",
  tokenizer: TikTokenizer,
  summary: boolean = false
): Promise<Tree> {
  if (!record) {
    throw new Error("failed to get record");
  }

  let total_tokens = 0;

  // Extract data from the record
  const startNode: Neo4jNode = deser_node(record, "startNode");
  const allNodes: Neo4jNode[] = deser_multi(record, "allNodes");
  const relationships: Relationship[] = record.get("relationships");

  // Create maps to store nodes
  const nodeMap = new Map<string, any>(); // Neo4j node by ID
  const treeNodeMap = new Map<string, TreeNode>(); // TreeNode by ID
  const childRelationships = new Map<string, Set<string>>(); // parent -> Set of children

  // Add all nodes to the nodeMap
  for (const node of allNodes) {
    const nodeId =
      node.identity?.toString() || node.properties.node_key.toString();
    nodeMap.set(nodeId, node);
  }

  // Add the root node
  const rootId =
    startNode.identity?.toString() || startNode.properties.node_key.toString();
  nodeMap.set(rootId, startNode);

  // Process relationships
  for (const rel of relationships) {
    const sourceId = rel.source.toString();
    const targetId = rel.target.toString();

    // Determine which is parent and which is child based on direction and relationship type
    let parentId, childId;

    if (direction === "up") {
      // When direction is "up", generally treat target as parent and source as child
      // EXCEPT for relationships in REVERSE_RELATIONSHIPS
      if (REVERSE_RELATIONSHIPS.includes(rel.type)) {
        parentId = sourceId;
        childId = targetId;
      } else {
        parentId = targetId;
        childId = sourceId;
      }
    } else {
      // direction is "down"
      // When direction is "down", generally treat source as parent and target as child
      // EXCEPT for relationships in REVERSE_RELATIONSHIPS
      if (REVERSE_RELATIONSHIPS.includes(rel.type)) {
        parentId = targetId;
        childId = sourceId;
      } else {
        parentId = sourceId;
        childId = targetId;
      }
    }

    if (!childRelationships.has(parentId)) {
      childRelationships.set(parentId, new Set<string>());
    }
    childRelationships.get(parentId)!.add(childId);
  }

  // Create TreeNodes for all Neo4j nodes
  for (const [id, node] of nodeMap.entries()) {
    let label = "";
    if (summary) {
      // more in-depth node-specific summary
      label = getNodeSummaryLabel(node);
    } else {
      label = getNodeLabel(node);
      if (node.properties?.token_count) {
        label = `${label} (${node.properties.token_count})`;
      } else if (node.properties?.body) {
        const tokens = tokenizer.encode(node.properties.body, []);
        total_tokens += tokens.length;
        label = `${label} (${tokens.length})`;
      }
    }
    treeNodeMap.set(id, {
      label,
      nodes: [],
    });
  }

  // Build tree using breadth-first approach to avoid recursion issues
  const processQueue = [rootId];
  const processedNodes = new Set<string>(); // Track processed nodes to avoid cycles
  const nodePlacement = new Map<string, boolean>(); // Track if node has been placed in tree

  while (processQueue.length > 0) {
    const currentId = processQueue.shift()!;

    if (processedNodes.has(currentId)) continue;
    processedNodes.add(currentId);

    const children = childRelationships.get(currentId);
    if (!children) continue;

    const parentNode = treeNodeMap.get(currentId);
    if (!parentNode) continue;

    // Add all child nodes to the parent
    for (const childId of children) {
      // Skip if this would create a cycle back to root
      if (childId === rootId) continue;

      const childNode = treeNodeMap.get(childId);
      if (!childNode) continue;

      // Check if we already added this child to this parent
      if (!parentNode.nodes.some((n) => n.label === childNode.label)) {
        parentNode.nodes.push(childNode);
        nodePlacement.set(childId, true);

        // Add child to processing queue if we haven't processed it yet
        if (!processedNodes.has(childId)) {
          processQueue.push(childId);
        }
      }
    }
  }

  // Second pass - make sure all nodes are included somewhere in the tree
  // For any nodes not yet placed, add them as direct children of the root
  const rootNode = treeNodeMap.get(rootId)!;

  for (const [id, node] of treeNodeMap.entries()) {
    if (id !== rootId && !nodePlacement.get(id)) {
      rootNode.nodes.push(node);
      nodePlacement.set(id, true);
    }
  }

  const root = rootNode || { label: "Root not found", nodes: [] };
  return removeCircularReferences({ root, total_tokens });
  // return { root, total_tokens };
}

export function alphabetizeNodeLabels(
  node: TreeNode,
  visited = new Set<TreeNode>()
) {
  // Prevent infinite recursion by checking if we've already processed this node
  if (!node || visited.has(node)) {
    return node;
  }

  // Mark this node as visited
  visited.add(node);

  if (node.nodes && Array.isArray(node.nodes)) {
    // Sort the nodes array based on the 'label' property
    node.nodes.sort((a, b) => {
      const labelA = a.label?.toUpperCase() || "";
      const labelB = b.label?.toUpperCase() || "";
      return labelA.localeCompare(labelB);
    });

    // Recursively alphabetize the labels of child nodes
    node.nodes.forEach((childNode) => {
      alphabetizeNodeLabels(childNode, visited);
    });
  }

  return node;
}

export function removeCircularReferences(tree: Tree): Tree {
  if (!tree || !tree.root) {
    return {
      root: { label: "empty", nodes: [] },
      total_tokens: tree?.total_tokens || 0,
    };
  }

  function processNode(
    node: TreeNode,
    visited = new Set<TreeNode>(),
    path = new Set<TreeNode>()
  ): TreeNode {
    if (!node) return { label: "unknown", nodes: [] };

    // If we've encountered this node in our current path, it's a circular reference
    if (path.has(node)) {
      return {
        label: node.label,
        nodes: [],
      };
    }

    // If we've already processed this node completely, return a copy
    if (visited.has(node)) {
      return {
        label: node.label,
        nodes: [],
      };
    }

    // Mark as visited and add to current path
    visited.add(node);
    path.add(node);

    const cleanNode: TreeNode = {
      label: node.label,
      nodes: [],
    };

    if (node.nodes && Array.isArray(node.nodes)) {
      cleanNode.nodes = node.nodes
        .filter((child) => child != null)
        .map((child) => processNode(child, visited, path))
        .filter((child) => child != null);
    }

    // Remove from current path (backtrack)
    path.delete(node);

    return cleanNode;
  }

  return {
    root: processNode(tree.root),
    total_tokens: tree.total_tokens,
  };
}
