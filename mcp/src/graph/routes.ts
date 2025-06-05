import { Request, Response, NextFunction } from "express";
import { Neo4jNode, node_type_descriptions, NodeType } from "./types.js";
import { nameFileOnly, toReturnNode, isTrue } from "./utils.js";
import * as G from "./graph.js";

export function schema(_req: Request, res: Response) {
  const schema = node_type_descriptions();
  const schemaArray = Object.entries(schema).map(
    ([node_type, description]) => ({
      node_type,
      description,
    })
  );
  res.json(schemaArray);
}

export function logEndpoint(req: Request, res: Response, next: NextFunction) {
  if (req.headers["x-api-token"]) {
    console.log(`=> ${req.method} ${req.url} ${req.headers["x-api-token"]}`);
  } else {
    console.log(`=> ${req.method} ${req.url}`);
  }
  next();
}

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
    const concise = isTrue(req.query.concise as string);
    let ref_ids: string[] = [];
    if (req.query.ref_ids) {
      ref_ids = (req.query.ref_ids as string).split(",");
    }
    const output = req.query.output as G.OutputFormat;
    const result = await G.get_nodes(node_type, concise, ref_ids, output);
    if (output === "snippet") {
      res.send(result);
    } else {
      res.json(result);
    }
  } catch (error) {
    console.error("Error:", error);
    res.status(500).send("Internal Server Error");
  }
}

export async function get_files(req: Request, res: Response) {
  try {
    const prefix = req.query.prefix as string;
    const limit = parseInt(req.query.limit as string) || 100;
    const result = await G.get_files(prefix, limit);
    res.json(result);
  } catch (error) {
    console.error("Error:", error);
    res.status(500).send("Internal Server Error");
  }
}

export async function search(req: Request, res: Response) {
  try {
    const query = req.query.query as string;
    const limit = parseInt(req.query.limit as string) || 25;
    const concise = isTrue(req.query.concise as string);
    let node_types: NodeType[] = [];
    if (req.query.node_types) {
      node_types = (req.query.node_types as string).split(",") as NodeType[];
    } else if (req.query.node_type) {
      node_types = [req.query.node_type as NodeType];
    }
    const method = req.query.method as G.SearchMethod;
    const output = req.query.output as G.OutputFormat;
    let tests = isTrue(req.query.tests as string);
    const maxTokens = parseInt(req.query.max_tokens as string);
    if (maxTokens) {
      console.log("search with max tokens", maxTokens);
    }
    const result = await G.search(
      query,
      limit,
      node_types,
      concise,
      maxTokens || 100000,
      method,
      output || "json",
      tests
    );
    if (output === "snippet") {
      res.send(result);
    } else {
      res.json(result);
    }
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
  direction: G.Direction;
  trim: string[];
}

function mapParams(req: Request): MapParams {
  const node_type = req.query.node_type as string;
  const name = req.query.name as string;
  const ref_id = req.query.ref_id as string;
  const name_and_type = node_type && name;
  if (!name_and_type && !ref_id) {
    throw new Error("either node_type+name or ref_id required");
  }
  const direction = req.query.direction as G.Direction;
  const tests = !(req.query.tests === "false" || req.query.tests === "0");
  const depth = parseInt(req.query.depth as string) || DEFAULT_DEPTH;
  const default_direction = "both" as G.Direction;
  return {
    node_type: node_type || "",
    name: name || "",
    ref_id: ref_id || "",
    tests,
    depth,
    direction: direction || default_direction,
    trim: ((req.query.trim as string) || "").split(","),
  };
}

export async function get_map(req: Request, res: Response) {
  try {
    const html = await G.get_map(mapParams(req));
    res.send(`<pre>\n${html}\n</pre>`);
  } catch (error) {
    console.error("Error:", error);
    res.status(500).send("Internal Server Error");
  }
}

export async function get_repo_map(req: Request, res: Response) {
  try {
    const name = req.query.name as string;
    const ref_id = req.query.ref_id as string;
    const html = await G.get_repo_map(name, ref_id);
    res.send(`<pre>\n${html}\n</pre>`);
  } catch (error) {
    console.error("Error:", error);
    res.status(500).send("Internal Server Error");
  }
}

export async function get_code(req: Request, res: Response) {
  try {
    const text = await G.get_code(mapParams(req));
    res.send(text);
  } catch (error) {
    console.error("Error:", error);
    res.status(500).send("Internal Server Error");
  }
}

export async function get_shortest_path(req: Request, res: Response) {
  try {
    const result = await G.get_shortest_path(
      req.query.start_node_key as string,
      req.query.end_node_key as string,
      req.query.start_ref_id as string,
      req.query.end_ref_id as string
    );
    res.send(result);
  } catch (error) {
    console.error("Error:", error);
    res.status(500).send("Internal Server Error");
  }
}
