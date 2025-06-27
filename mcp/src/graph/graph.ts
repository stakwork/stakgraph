import { db, Direction, Data_Bank } from "./neo4j.js";
import archy from "archy";
import { buildTree, alphabetizeNodeLabels } from "./codemap.js";
import { extractNodesFromRecord } from "./codebody_files.js";
import { Neo4jNode, NodeType } from "./types.js";
import { nameFileOnly, toReturnNode, formatNode, clean_node } from "./utils.js";
import { createByModelName } from "@microsoft/tiktokenizer";

export type SearchMethod = "vector" | "fulltext";

export { Data_Bank, Direction };

export async function get_nodes(
  node_type: NodeType,
  concise: boolean,
  ref_ids: string[],
  output: OutputFormat = "snippet"
) {
  let result: Neo4jNode[] = [];
  if (ref_ids.length > 0) {
    result = await db.nodes_by_ref_ids(ref_ids);
  } else {
    result = await db.nodes_by_type(node_type);
  }
  return toNodes(result, concise, output);
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
  tests: boolean = false
) {
  if (method === "vector") {
    const result = await db.vectorSearch(query, limit, node_types);
    return toNodes(result, concise, output);
  } else {
    const skip_node_types = tests ? [] : ["Test", "E2etest"];
    const result = await db.search(
      query,
      limit,
      node_types,
      skip_node_types as NodeType[],
      maxTokens
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
  node_type: NodeType = "Repository"
): Promise<string> {
  let names = [name];
  if (name === "" && ref_id === "" && node_type === "Repository") {
    const repos = await db.get_repositories();
    names = repos.map((r) => r.properties.name);
  }
  let finalText = "";
  for (const name of names) {
    const r = await db.get_repo_subtree(name, ref_id, node_type);
    const record = r.records[0];
    const tokenizer = await createByModelName("gpt-4");
    const tree = await buildTree(record, "down", tokenizer);
    const text = archy(tree.root);
    finalText += text;
    finalText += "\n\n";
  }
  return finalText;
}

export async function get_map(params: MapParams): Promise<string> {
  const record = await get_subtree(params);
  const pkg_files = await db.get_pkg_files();
  const tokenizer = await createByModelName("gpt-4");

  const tree = await buildTree(record, params.direction, tokenizer);
  const sortedTreeNodes = [];
  for (const node of tree.root.nodes) {
    const sortedNode = alphabetizeNodeLabels(node);
    sortedTreeNodes.push(sortedNode);
  }
  tree.root.nodes = sortedTreeNodes;

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
