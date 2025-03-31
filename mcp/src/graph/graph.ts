import { Record } from "neo4j-driver";
import { db, Direction } from "./neo4j.js";
import archy from "archy";
import { buildTree } from "./codemap.js";
import { buildTree as buildTree2 } from "./codemap2.js";
import { code_body, formatNode } from "./codebody.js";
import { extractNodesFromRecord } from "./codebody2.js";
import { Request, Response, NextFunction } from "express";
import { Neo4jNode, NodeType } from "./types.js";
import { nameFileOnly, toReturnNode } from "./utils.js";

export function authMiddleware(
  req: Request,
  res: Response,
  next: NextFunction
) {
  const apiToken = process.env.API_TOKEN;
  if (!apiToken) {
    return next();
  }
  const requestToken = req.header("x-api-token");
  if (!requestToken || requestToken !== apiToken) {
    res.status(401).json({ error: "Unauthorized: Invalid API token" });
    return;
  }
  next();
}

export async function get_nodes(req: Request, res: Response) {
  try {
    console.log("=> get_nodes", req.query);
    const node_type = req.query.node_type as NodeType;
    const concise = req.query.concise === "true";
    const result = await db.nodes_by_type(node_type);
    const nodes = result.map((f) => toNode(f, concise));
    res.json(nodes);
  } catch (error) {
    console.error("Error:", error);
    res.status(500).send("Internal Server Error");
  }
}

export async function search(req: Request, res: Response) {
  try {
    const query = req.query.query as string;
    const limit = parseInt(req.query.limit as string) || 25;
    const concise = req.query.concise === "true";
    let node_types: NodeType[] = [];
    if (req.query.node_types) {
      node_types = (req.query.node_types as string).split(",") as NodeType[];
    }
    const result = await db.search(query, limit, node_types);
    const nodes = result.map((f) => toNode(f, concise));
    res.json(nodes);
  } catch (error) {
    console.error("Error:", error);
    res.status(500).send("Internal Server Error");
  }
}

export function toNode(node: Neo4jNode, concise: boolean): any {
  return concise ? nameFileOnly(node) : toReturnNode(node);
}

const DEFAULT_DEPTH = 7;

interface MapParams {
  node_type: string;
  name: string;
  ref_id: string;
  tests: boolean;
  depth: number;
  direction: Direction;
}

function mapParams(req: Request): MapParams {
  const node_type = req.query.node_type as string;
  const name = req.query.name as string;
  const ref_id = req.query.ref_id as string;
  const name_and_type = node_type && name;
  if (!name_and_type && !ref_id) {
    throw new Error("either node_type+name or ref_id required");
  }
  const direction = req.query.direction as Direction;
  const tests = !(req.query.tests === "false" || req.query.tests === "0");
  const depth = parseInt(req.query.depth as string) || DEFAULT_DEPTH;
  let default_direction = "down";
  if (node_type === "Datamodel") {
    default_direction = "up";
  }
  return {
    node_type: node_type || "",
    name: name || "",
    ref_id: ref_id || "",
    tests,
    depth,
    direction: direction || default_direction,
  };
}

async function get_record_from_query(fn_name: string, req: Request) {
  const { node_type, name, tests, depth, direction, ref_id } = mapParams(req);
  console.log("=>", fn_name, node_type, name, tests, depth, direction);
  const r = await db.get_subtree(
    node_type,
    name,
    ref_id,
    tests,
    depth,
    direction
  );
  return r.records[0];
}

export async function get_map(req: Request, res: Response) {
  try {
    const record = await get_record_from_query("get_map", req);
    const { direction } = mapParams(req);
    const pkg_files = await db.get_pkg_files();

    const tree = await buildTree2(record, direction);
    const text = archy(tree);
    let html = `<pre>`;
    html += text;
    for (const file of pkg_files) {
      html += `File: ${toNode(file, true).file}\n`;
    }
    html += `<pre>`;
    res.send(html);
    // res.send(`<pre>${text}</pre>`);
  } catch (error) {
    console.error("Error:", error);
    res.status(500).send("Internal Server Error");
  }
}

export async function get_code(req: Request, res: Response) {
  try {
    const record = await get_record_from_query("get_code", req);
    const pkg_files = await db.get_pkg_files();
    const text = extractNodesFromRecord(record, pkg_files);
    res.send(text);
  } catch (error) {
    console.error("Error:", error);
    res.status(500).send("Internal Server Error");
  }
}

function toSnippets(path: any) {
  let r = "";
  for (const segment of path.segments) {
    const snip = formatNode(segment.start);
    r += snip;
  }
  const snip = formatNode(path.end);
  r += snip;
  return r;
}

export async function get_shortest_path(req: Request, res: Response) {
  try {
    const start_node_key = req.query.start_node_key as string;
    const end_node_key = req.query.end_node_key as string;
    const result = await db.get_shortest_path(start_node_key, end_node_key);
    const path = result.records[0].get("path");
    console.log(path);
    res.send(toSnippets(path));
  } catch (error) {
    console.error("Error:", error);
    res.status(500).send("Internal Server Error");
  }
}

export async function get_shortest_path_ref_id(req: Request, res: Response) {
  try {
    const start_ref_id = req.query.start_ref_id as string;
    const end_ref_id = req.query.end_ref_id as string;
    const result = await db.get_shortest_path_ref_id(start_ref_id, end_ref_id);
    const path = result.records[0].get("path");
    res.send(toSnippets(path));
  } catch (error) {
    console.error("Error:", error);
    res.status(500).send("Internal Server Error");
  }
}

/*************************
  DEPRECATED ROUTES  ↓↓↓
 *************************/

export async function get_pages(req: Request, res: Response) {
  try {
    const result = await db.get_pages();
    const pages = result.records.map(toPage);
    res.json(pages);
  } catch (error) {
    console.error("Error:", error);
    res.status(500).send("Internal Server Error");
  }
}

export function toPage(rec: Record): any {
  const page = rec.get("page");
  return {
    node_type: page.labels[0],
    ...page.properties,
  };
}

interface Params {
  page_name: string | null;
  function_name: string | null;
  tests: boolean;
  depth: number;
}

function params(req: Request): Params {
  const page_name = req.query.page_name as string;
  const function_name = req.query.function_name as string;
  const tests = !(req.query.tests === "false" || req.query.tests === "0");
  const depth = parseInt(req.query.depth as string) || DEFAULT_DEPTH;
  if (!page_name && !function_name)
    throw new Error("page or function required");
  return {
    page_name: page_name || null,
    function_name: function_name || null,
    tests,
    depth,
  };
}

export async function get_feature_map(req: Request, res: Response) {
  try {
    const { page_name, function_name, tests, depth } = params(req);
    console.log("=> get_feature_map:", page_name, function_name, tests, depth);
    const result = await db.get_function_path(
      page_name,
      function_name,
      tests,
      depth
    );
    const fn = result.records[0];
    const tree = await buildTree(fn);
    const text = archy(tree);
    res.send(`<pre>${text}</pre>`);
  } catch (error) {
    console.error("Error:", error);
    res.status(500).send("Internal Server Error");
  }
}

export async function get_feature_code(req: Request, res: Response) {
  try {
    const { page_name, function_name, tests, depth } = params(req);
    const pkg_files = await db.get_pkg_files();
    const result = await db.get_function_path(
      page_name,
      function_name,
      tests,
      depth
    );
    const text = code_body(result.records[0], pkg_files);
    res.send(text);
  } catch (error) {
    console.error("Error:", error);
    res.status(500).send("Internal Server Error");
  }
}

export async function get_pages_links(req: Request, res: Response) {
  try {
    const is_json = req.query.json === "true";
    const result = await db.get_pages();
    const pages = result.records.map(toPage);
    if (is_json) {
      res.json(createLinksJson(pages, "page"));
    } else {
      const html = createLinksList(pages, "page");
      res.send(html);
    }
  } catch (error) {
    console.error("Error:", error);
    res.status(500).send("Internal Server Error");
  }
}

export async function get_components_links(req: Request, res: Response) {
  try {
    const is_json = req.query.json === "true";
    const result = await db.get_components();
    const components = result.records.map(toComponent);
    if (is_json) {
      res.json(createLinksJson(components, "function"));
    } else {
      const html = createLinksList(components, "function");
      res.send(html);
    }
  } catch (error) {
    console.error("Error:", error);
    res.status(500).send("Internal Server Error");
  }
}

function createLinksList(data: any[], type: "page" | "function") {
  let html = "<ul>";

  data.forEach((item) => {
    function makehref(map_or_code: "map" | "code") {
      return `/feature_${map_or_code}?${type}_name=${encodeURIComponent(
        item.name
      )}`;
    }
    html += `<li>
      <strong style="width: 280px; display: inline-block; text-align: right;">${
        item.name
      }</strong>
      <a href="${makehref("map")}">[map]</a>
      <a href="${makehref("code")}">[code]</a>
      <span>(${item.file})</span>
    </li>`;
  });

  html += "</ul>";
  return html;
}

function createLinksJson(data: any[], type: "page" | "function") {
  return data.map((item) => ({
    name: item.name,
    file: item.file,
  }));
}

function toComponent(rec: Record): any {
  const page = rec.get("component");
  return {
    node_type: page.labels[0],
    ...page.properties,
  };
}
