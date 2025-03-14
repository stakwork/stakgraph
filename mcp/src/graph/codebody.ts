import { getNodeLabel } from "./utils.js";

export interface Node {
  properties: NodeProps;
  labels?: string[];
}

export interface NodeProps {
  name: string;
  file: string;
  start: string;
  end: string;
  body: string;
}

export function code_body(
  record: any,
  extra_nodes: Node[],
  include_tests: boolean = false
): string {
  try {
    // Get the starting function and all paths
    const startFunction = record.get("function");
    const paths = record.get("paths") || [];

    console.log("Total paths received:", paths.length);

    // Set to store unique nodes
    const uniqueNodes = new Map<string, Node>();

    // Helper function to generate a unique key for each node
    const getNodeKey = (node: Node): string => {
      return `${node.properties.file || ""}:${node.properties.name || ""}:${
        node.properties.start || ""
      }`;
    };

    // Add the starting function to the unique nodes
    if (startFunction) {
      uniqueNodes.set(getNodeKey(startFunction), startFunction);
    }

    // First, process any direct nodes in the paths array
    for (const path of paths) {
      // Check if this is a direct node object with labels and properties
      if (path && path.labels && path.properties) {
        console.log(`Found direct node with labels: ${path.labels.join(", ")}`);
        uniqueNodes.set(getNodeKey(path), path);
        continue; // Skip to next item
      }

      try {
        // For regular Path objects from Neo4j
        if (path.start && path.end) {
          // This handles paths created with apoc.path.create for Import nodes
          uniqueNodes.set(getNodeKey(path.start), path.start);
          uniqueNodes.set(getNodeKey(path.end), path.end);

          // If the path has segments, process those too
          if (path.segments && Array.isArray(path.segments)) {
            path.segments.forEach((segment: any) => {
              uniqueNodes.set(getNodeKey(segment.start), segment.start);
              uniqueNodes.set(getNodeKey(segment.end), segment.end);
            });
          }
        }
        // If path has nodes method (Neo4j specific)
        else if (typeof path.nodes === "function") {
          const nodes = path.nodes();
          nodes.forEach((node: Node) => {
            if (node) uniqueNodes.set(getNodeKey(node), node);
          });
        }
        // If path is an array
        else if (Array.isArray(path)) {
          path.forEach((node) => {
            if (node) uniqueNodes.set(getNodeKey(node), node);
          });
        }
        // If path has forEach method
        else if (typeof path.forEach === "function") {
          path.forEach((item: any) => {
            if (item.start) uniqueNodes.set(getNodeKey(item.start), item.start);
            if (item.end) uniqueNodes.set(getNodeKey(item.end), item.end);
          });
        }
      } catch (err) {
        console.warn("Error processing path:", err);
      }
    }

    // Helper function to format node
    const formatNode = (node: Node): string => {
      if (node && node.properties) {
        // Regular format for other nodes
        return [
          `<snippet>`,
          `name: ${getNodeLabel(node)}`,
          `file: ${node.properties.file || "Not specified"}`,
          `start: ${node.properties.start || "N/A"}, end: ${
            node.properties.end || "N/A"
          }`,
          node.properties.body ? "```\n" + node.properties.body + "\n```" : "",
          "</snippet>",
          "", // Empty line for spacing
        ].join("\n");
      }
      return "";
    };

    // Convert the Map values to an array
    let nodes = Array.from(uniqueNodes.values());
    nodes.push(...extra_nodes);

    // Filter out test nodes if include_tests is false (as in your original code)
    if (!include_tests) {
      nodes = nodes.filter((node) => {
        const labels = node.labels || [];
        return !labels.includes("Test") && !labels.includes("E2etest");
      });
    }

    // Sort nodes by file path alphabetically, then by start position
    nodes.sort((a, b) => {
      // First compare file paths
      const fileA = (a?.properties?.file || "").toLowerCase();
      const fileB = (b?.properties?.file || "").toLowerCase();

      if (fileA !== fileB) {
        return fileA.localeCompare(fileB);
      }

      // If files are the same, compare start positions
      const startA = parseInt(a?.properties?.start || "0", 10);
      const startB = parseInt(b?.properties?.start || "0", 10);

      return startA - startB;
    });

    const output = nodes.map(formatNode);

    console.log("=> code_body:", output.length, "nodes");

    // Filter out empty strings and join with newlines
    return output.filter((line) => line.trim() !== "").join("\n");
  } catch (error) {
    console.error("Error in code_body:", error);
    throw error;
  }
}
