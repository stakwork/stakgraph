import { formatNode, deser_multi } from "./utils.js";
import { Neo4jNode } from "./types.js";

export function extractNodesFromRecord(
  record: any,
  extra_nodes: Neo4jNode[] = []
): string {
  try {
    if (!record) {
      return "";
    }

    const allNodes = new Map<string, Neo4jNode>();

    // Helper function that properly identifies and adds only valid nodes
    const addNode = (item: any) => {
      try {
        // Skip if not a node object
        if (!item || typeof item !== "object") {
          return;
        }
        // Skip relationships - they have type but not labels
        if (item.type && !item.labels) {
          return;
        }
        // Must have properties and file property to be a valid node for our purposes
        if (!item.properties || !item.properties.file) {
          return;
        }
        const key = `${item.properties.file}:${item.properties.name || ""}:${
          item.properties.start || "0"
        }`;
        allNodes.set(key, item);
      } catch (e) {
        // Silent error handling
      }
    };

    // Process allNodes from the updated query
    if (record.has("allNodes")) {
      const nodes: Neo4jNode[] = deser_multi(record, "allNodes");
      nodes.forEach((node: any, index: number) => {
        addNode(node);
      });
    }

    // Process imports collection
    if (record.has("imports")) {
      const imports: Neo4jNode[] = deser_multi(record, "imports");
      imports.forEach((importNode: any, index: number) => {
        addNode(importNode);
      });
    }

    // Add any extra nodes
    if (extra_nodes && extra_nodes.length > 0) {
      extra_nodes.forEach((node, index) => {
        addNode(node);
      });
    }

    // Convert to array and sort
    const nodeArray = Array.from(allNodes.values());

    // Sort the nodes by file and start position
    const sortedNodes = nodeArray.sort((a, b) => {
      try {
        // Safely get file names with default empty string
        const fileA = (a.properties.file || "").toLowerCase();
        const fileB = (b.properties.file || "").toLowerCase();
        // Sort by filename first
        if (fileA !== fileB) return fileA.localeCompare(fileB);
        // Then by start position
        const startA = Number(a.properties.start?.toString() || 0);
        const startB = Number(b.properties.start?.toString() || 0);
        return startA - startB;
      } catch (e) {
        return 0;
      }
    });

    // Format the nodes
    const formattedNodes = sortedNodes
      .map(formatNode)
      .filter((text) => text.trim() !== "");

    // Return the formatted nodes as a single string
    return formattedNodes.join("\n");
  } catch (error) {
    return "";
  }
}
