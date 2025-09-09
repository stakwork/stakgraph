import { db, Data_Bank } from "./neo4j.js";
import type { Direction } from "./neo4j.js";
import archy from "archy";
import { buildTree, alphabetizeNodeLabels } from "./codemap.js";
import { extractNodesFromRecord } from "./codebody_files.js";
import {
  Neo4jNode,
  Neo4jEdge,
  NodeType,
  EdgeType,
  GraphResponse,
} from "./types.js";
import {
  nameFileOnly,
  toReturnNode,
  formatNode,
  clean_node,
  getExtensionsForLanguage,
} from "./utils.js";
import { createByModelName } from "@microsoft/tiktokenizer";
import { generate_services_config } from "./service.js";

export type SearchMethod = "vector" | "fulltext";

export { Data_Bank, Direction };

export async function get_nodes(
  node_type: NodeType,
  concise: boolean,
  ref_ids: string[],
  output: OutputFormat = "snippet",
  language?: string
) {
  let result: Neo4jNode[] = [];
  if (ref_ids.length > 0) {
    result = await db.nodes_by_ref_ids(ref_ids, language);
  } else {
    result = await db.nodes_by_type(node_type, language);
  }

  return toNodes(result, concise, output);
}

export async function get_edges(
  edge_type: EdgeType,
  concise: boolean,
  ref_ids: string[],
  output: OutputFormat = "snippet",
  language?: string
) {
  let result: Neo4jEdge[] = [];
  if (ref_ids.length > 0) {
    result = await db.edges_by_ref_ids(ref_ids, language);
  } else if (edge_type) {
    result = await db.edges_by_type(edge_type, language);
  } else {
    result = await db.all_edges(language);
  }

  return toEdges(result, concise, output);
}

export type OutputFormat = "snippet" | "json";

export async function search(
  query: string,
  limit: number,
  node_types: NodeType[],
  concise: boolean,
  maxTokens: number,
  method: SearchMethod = "fulltext",
  output: OutputFormat = "snippet",
  tests: boolean = false,
  language?: string
) {
  if (method === "vector") {
    const result = await db.vectorSearch(
      query,
      limit,
      node_types,
      0.7,
      language
    );
    return toNodes(result, concise, output);
  } else {
    const skip_node_types = tests
      ? []
      : ["UnitTest", "IntegrationTest", "E2etest"];
    const result = await db.search(
      query,
      limit,
      node_types,
      skip_node_types as NodeType[],
      maxTokens,
      language
    );
    return toNodes(result, concise, output);
  }
}
export async function get_rules_files() {
  const files = await db.get_rules_files();
  return files
    .filter(
      (file) => file.properties.body && file.properties.body.trim() !== ""
    )
    .map(
      (file) =>
        `File: ${file.properties.name}\n Content: \n ${file.properties.body}\n`
    )
    .join("\n");
}

export async function get_services() {
  const pkgFiles = await db.get_pkg_files();
  const allFiles = await db.nodes_by_type("File");
  const envVarNodes = await db.get_env_vars();
  return generate_services_config(pkgFiles, allFiles, envVarNodes);
}

export function toNodes(
  result: Neo4jNode[],
  concise: boolean,
  output: OutputFormat = "snippet"
) {
  if (output === "snippet") {
    let r = "";
    for (const node of result) {
      r += formatNode(node);
    }
    return r;
  }
  return result.map((f) => toNode(f, concise));
}

export function toNode(node: Neo4jNode, concise: boolean): any {
  return concise ? nameFileOnly(node) : toReturnNode(node);
}

export function toEdges(
  result: Neo4jEdge[],
  concise: boolean,
  output: OutputFormat = "snippet"
) {
  if (output === "snippet") {
    let r = "";
    for (const edge of result) {
      r += formatEdge(edge);
    }
    return r;
  }
  return result.map((e) => toEdge(e, concise));
}

export function toEdge(edge: Neo4jEdge, concise: boolean): any {
  return concise ? edgeNameOnly(edge) : edge;
}

function edgeNameOnly(edge: Neo4jEdge): {
  edge_type: string;
  source: string;
  target: string;
} {
  return {
    edge_type: edge.edge_type,
    source: edge.source,
    target: edge.target,
  };
}

function formatEdge(edge: Neo4jEdge): string {
  return [
    `<edge>`,
    `type: ${edge.edge_type}`,
    `ref_id: ${edge.ref_id}`,
    `source: ${edge.source}`,
    `target: ${edge.target}`,
    edge.properties && Object.keys(edge.properties).length > 0
      ? `properties: ${JSON.stringify(edge.properties)}`
      : "",
    "</edge>",
    "",
  ]
    .filter((line) => line !== "")
    .join("\n");
}

export interface MapParams {
  node_type: string;
  name: string;
  ref_id: string;
  tests: boolean;
  depth: number;
  direction: Direction;
  trim: string[];
}

export async function get_subtree(p: MapParams) {
  const r = await db.get_subtree(
    p.node_type,
    p.name,
    p.ref_id,
    p.tests,
    p.depth,
    p.direction,
    p.trim
  );
  return r.records[0];
}

export async function get_repo_map(
  name: string,
  ref_id: string,
  node_type: NodeType = "Repository",
  include_functions_and_classes: boolean = false
): Promise<string> {
  let names = [name];
  if (name === "" && ref_id === "" && node_type === "Repository") {
    const repos = await db.get_repositories();
    names = repos.map((r) => r.properties.name);
  }
  let finalText = "";
  for (const name of names) {
    const r = await db.get_repo_subtree(
      name,
      ref_id,
      node_type,
      include_functions_and_classes
    );
    const record = r.records[0];
    const tokenizer = await createByModelName("gpt-4");
    const tree = await buildTree(record, "down", tokenizer);
    alphabetizeNodeLabels(tree.root);
    const text = archy(tree.root);
    finalText += text;
    finalText += "\n\n";
  }
  return finalText;
}

export async function get_file_map(file_end: string): Promise<string> {
  const f = await db.get_file_ends_with(file_end);
  const tokenizer = await createByModelName("gpt-4");
  const record = await get_subtree({
    node_type: "File",
    name: "",
    ref_id: f.ref_id as string,
    depth: 1,
    tests: false,
    direction: "down",
    trim: [],
  });

  const tree = await buildTree(record, "down", tokenizer, true);
  alphabetizeNodeLabels(tree.root);
  const text = archy(tree.root);
  return text;
}

export async function get_map(params: MapParams): Promise<string> {
  const record = await get_subtree(params);
  const pkg_files = await db.get_pkg_files();
  const tokenizer = await createByModelName("gpt-4");

  const tree = await buildTree(record, params.direction, tokenizer);
  alphabetizeNodeLabels(tree.root);

  const text = archy(tree.root);
  let themap = ``;
  themap += text;
  for (const file of pkg_files) {
    const tokens = tokenizer.encode(file.properties.body || "", []);
    themap += `File: ${toNode(file, true).file} (${tokens.length})\n`;
  }
  const fulltext = extractNodesFromRecord(record, pkg_files);
  const tokens = tokenizer.encode(fulltext, []);
  themap += `Total tokens: ${tokens.length}`;
  return themap;
}

export async function get_code(params: MapParams): Promise<string> {
  const record = await get_subtree(params);
  const pkg_files = await db.get_pkg_files();
  const tokenizer = await createByModelName("gpt-4");
  const text = extractNodesFromRecord(record, pkg_files);
  const tokens = tokenizer.encode(text, []);
  return `Total tokens: ${tokens.length}\n\n${text}`;
}

export async function get_shortest_path(
  start_node_key: string,
  end_node_key: string,
  start_ref_id: string,
  end_ref_id: string
) {
  if (start_ref_id && end_ref_id) {
    const result = await db.get_shortest_path_ref_id(start_ref_id, end_ref_id);
    const record = result.records[0];
    const path: ShortestPath = record.get("path");
    return pathToSnippets(path);
  } else {
    const result = await db.get_shortest_path(start_node_key, end_node_key);
    const record = result.records[0];
    const path: ShortestPath = record.get("path");
    return pathToSnippets(path);
  }
}

interface ShortestPath {
  start: Neo4jNode;
  end: Neo4jNode;
  segments: {
    start: Neo4jNode;
    end: Neo4jNode;
  }[];
}

function pathToSnippets(path: ShortestPath) {
  let r = "";
  for (const segment of path.segments) {
    const snip = formatNode(clean_node(segment.start));
    r += snip;
  }
  const snip = formatNode(clean_node(path.end));
  r += snip;
  return r;
}

export async function get_graph(
  node_type: NodeType,
  edge_type: EdgeType,
  concise: boolean,
  ref_ids: string[],
  include_edges: boolean = false,
  language?: string
): Promise<GraphResponse> {
  const nodes_result = await get_nodes(
    node_type,
    concise,
    ref_ids,
    "json",
    language
  );
  let edges_result: any[] = [];
  if (include_edges) {
    const result = await get_edges(
      edge_type,
      concise,
      ref_ids,
      "json",
      language
    );
    edges_result = Array.isArray(result) ? result : [];
  }
  return {
    nodes: Array.isArray(nodes_result) ? nodes_result : [],
    edges: Array.isArray(edges_result) ? edges_result : [],
    status: "Success",
  };
}
