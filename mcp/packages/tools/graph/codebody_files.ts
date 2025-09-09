import { formatNode, deser_multi } from "./utils.js";
import { Neo4jNode } from "./types.js";

const FILE_SIZE_THRESHOLD = 5000;

export function extractNodesFromRecord(
  record: any,
  extra_nodes: Neo4jNode[] = []
): string {
  try {
    if (!record) {
      return "";
    }

    const allNodes = new Map<string, Neo4jNode>();
    const fileBodySizes = new Map<string, number>();
    const fileNodes = new Map<string, Neo4jNode>();

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

        // If this is a File node, track its size
        if (item.labels.includes("File")) {
          const filePath = item.properties.file;
          const bodySize = (item.properties.body || "").length;
          fileBodySizes.set(filePath, bodySize);
          fileNodes.set(filePath, item);
          return; // Don't add to allNodes yet, we'll decide later
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
      nodes.forEach((node: any) => {
        addNode(node);
      });
    }

    // Process imports collection
    if (record.has("imports")) {
      const imports: Neo4jNode[] = deser_multi(record, "imports");
      imports.forEach((importNode: any) => {
        addNode(importNode);
      });
    }

    // Process files collection from the updated query
    if (record.has("files")) {
      const files: Neo4jNode[] = deser_multi(record, "files");
      files.forEach((fileNode: any) => {
        addNode(fileNode);
      });
    }

    // Add any extra nodes
    if (extra_nodes && extra_nodes.length > 0) {
      extra_nodes.forEach((node) => {
        addNode(node);
      });
    }

    // Make the final decision about which nodes to include
    const finalNodeMap = new Map<string, Neo4jNode>();

    // First pass: identify which files to include whole vs. by components
    const filesToShowWhole = new Set<string>();
    const filesToShowByComponents = new Set<string>();

    fileBodySizes.forEach((size, filePath) => {
      if (size < FILE_SIZE_THRESHOLD && size > 0) {
        filesToShowWhole.add(filePath);
      } else {
        filesToShowByComponents.add(filePath);
      }
    });

    // Add whole file nodes for small files
    filesToShowWhole.forEach((filePath) => {
      if (fileNodes.has(filePath)) {
        const fileNode = fileNodes.get(filePath)!;
        const key = `${filePath}:File:0`;
        finalNodeMap.set(key, fileNode);
      }
    });

    // Track which files have component nodes
    const filesWithComponentNodes = new Set<string>();

    // Add individual component nodes for large files
    allNodes.forEach((node, key) => {
      const filePath = node.properties.file;

      // If this is a component node for a large file, track it
      if (filesToShowByComponents.has(filePath)) {
        finalNodeMap.set(key, node);
        filesWithComponentNodes.add(filePath);
      }
      // For files not in our file list, always add the component nodes
      else if (!fileNodes.has(filePath)) {
        finalNodeMap.set(key, node);
      }
    });

    // Add whole file nodes for large files that have no component nodes
    filesToShowByComponents.forEach((filePath) => {
      if (!filesWithComponentNodes.has(filePath) && fileNodes.has(filePath)) {
        const fileNode = fileNodes.get(filePath)!;
        const key = `${filePath}:File:0`;
        finalNodeMap.set(key, fileNode);
      }
    });

    // Convert to array and sort
    const nodeArray = Array.from(finalNodeMap.values());

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
