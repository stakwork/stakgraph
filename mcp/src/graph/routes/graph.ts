import { Request, Response } from "express";
import {
  Neo4jNode,
  node_type_descriptions,
  NodeType,
  EdgeType,
  relevant_node_types,
} from "../types.js";
import {
  nameFileOnly,
  toReturnNode,
  buildGraphMeta,
  normalizeRepoParam,
} from "../utils.js";
import * as G from "../graph.js";
import { db } from "../neo4j.js";
import { parseBody, parseQuery } from "./validation.js";
import {
  getNodesQuerySchema,
  postNodesBodySchema,
  getEdgesQuerySchema,
  searchQuerySchema,
  refIdQuerySchema,
  workflowQuerySchema,
  mapQuerySchema,
  repoMapQuerySchema,
  shortestPathQuerySchema,
  graphQuerySchema,
  MapQuery,
} from "./schemas/graph.js";

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

export async function get_nodes(req: Request, res: Response) {
  try {
    console.log("=> get_nodes", req.method, req.path);
    const parsed = parseQuery(req, res, getNodesQuerySchema);
    if (!parsed) return;

    const node_type = parsed.node_type;
    const concise = parsed.concise ?? false;
    const ref_ids = parsed.ref_ids || [];
    const output = parsed.output;
    const language = parsed.language;

    const result = await G.get_nodes(
      node_type,
      concise,
      ref_ids,
      output,
      language
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

export async function post_nodes(req: Request, res: Response) {
  try {
    console.log("=> post_nodes", req.method, req.path);
    const parsed = parseBody(req, res, postNodesBodySchema);
    if (!parsed) return;

    const node_type = parsed.node_type;
    const concise = parsed.concise ?? false;
    const ref_ids = parsed.ref_ids || [];
    const output = parsed.output;
    const language = parsed.language;

    const result = await G.get_nodes(
      node_type,
      concise,
      ref_ids,
      output,
      language
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

export async function get_edges(req: Request, res: Response) {
  try {
    const parsed = parseQuery(req, res, getEdgesQuerySchema);
    if (!parsed) return;

    const edge_type = parsed.edge_type as EdgeType;
    const concise = parsed.concise ?? false;
    const ref_ids = parsed.ref_ids || [];
    const output = parsed.output;
    const language = parsed.language;

    const result = await G.get_edges(
      edge_type,
      concise,
      ref_ids,
      output,
      language
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

export async function search(req: Request, res: Response) {
  try {
    const parsed = parseQuery(req, res, searchQuerySchema);
    if (!parsed) return;

    const query = parsed.query;
    const limit = parsed.limit && parsed.limit > 0 ? parsed.limit : 25;
    const concise = parsed.concise ?? false;
    const node_types = parsed.node_types
      ? parsed.node_types
      : parsed.node_type
        ? [parsed.node_type]
        : [];
    const method = parsed.method;
    const output = parsed.output;
    const tests = parsed.tests ?? false;
    const maxTokens = parsed.max_tokens && parsed.max_tokens > 0 ? parsed.max_tokens : undefined;
    const language = parsed.language;

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
      output || "snippet",
      tests,
      language
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
export async function get_rules_files(req: Request, res: Response) {
  try {
    const snippets = await G.get_rules_files();
    res.json(snippets);
  } catch (error) {
    console.error("Error fetching rules files:", error);
    res.status(500).json({ error: "Failed to fetch rules files" });
  }
}

export async function fetch_node_with_related(req: Request, res: Response) {
  // curl "http://localhost:3355/node?ref_id=bcc79e17-fae9-41d6-8932-40ea60e34b54"
  try {
    const parsed = parseQuery(req, res, refIdQuerySchema);
    if (!parsed) return;

    const ref_id = parsed.ref_id;

    const result = await db.get_node_with_related(ref_id);

    res.json(result);
  } catch (error) {
    console.error("Fetch Node with Related Error:", error);
    res.status(500).json({ error: "Internal Server Error" });
  }
}

export async function fetch_workflow_published_version(
  req: Request,
  res: Response
) {
  // curl "http://localhost:3355/workflow?ref_id=<workflow-ref-id>&concise=true"
  try {
    const parsed = parseQuery(req, res, workflowQuerySchema);
    if (!parsed) return;

    const ref_id = parsed.ref_id;
    const concise = parsed.concise ?? false;
    const result = await db.get_workflow_published_version_subgraph(
      ref_id,
      concise
    );

    res.json(result);
  } catch (error) {
    console.error("Fetch Workflow Published Version Error:", error);
    res.status(500).json({ error: "Internal Server Error" });
  }
}

export function toNode(node: Neo4jNode, concise: boolean): any {
  return concise ? nameFileOnly(node) : toReturnNode(node);
}

export const DEFAULT_DEPTH = 7;

export interface MapParams {
  node_type: string;
  name: string;
  file: string;
  ref_id: string;
  tests: boolean;
  depth: number;
  direction: G.Direction;
  trim: string[];
}

export function mapParams(params: MapQuery): MapParams {
  const default_direction = "both" as G.Direction;
  return {
    node_type: params.node_type || "",
    name: params.name || "",
    file: params.file || "",
    ref_id: params.ref_id || "",
    tests: params.tests ?? true,
    depth: params.depth && params.depth > 0 ? params.depth : DEFAULT_DEPTH,
    direction: (params.direction as G.Direction) || default_direction,
    trim: params.trim || [],
  };
}

export async function get_map(req: Request, res: Response) {
  try {
    const parsed = parseQuery(req, res, mapQuerySchema);
    if (!parsed) return;
    const html = await G.get_map(mapParams(parsed));
    res.send(`<pre>\n${html}\n</pre>`);
  } catch (error) {
    console.error("Error:", error);
    res.status(500).send("Internal Server Error");
  }
}

export async function get_repo_map(req: Request, res: Response) {
  try {
    const parsed = parseQuery(req, res, repoMapQuerySchema);
    if (!parsed) return;

    const name = parsed.name;
    const ref_id = parsed.ref_id;
    const node_type = parsed.node_type;
    const normalizedName =
      (node_type || "Repository") === "Repository"
        ? normalizeRepoParam(name) || name || ""
        : name || "";
    const include_functions_and_classes =
      parsed.include_functions_and_classes ?? false;
    const html = await G.get_repo_map(
      normalizedName,
      ref_id || "",
      node_type || "Repository",
      include_functions_and_classes
    );
    res.send(`<pre>\n${html}\n</pre>`);
  } catch (error) {
    console.error("Error:", error);
    res.status(500).send("Internal Server Error");
  }
}

export async function get_code(req: Request, res: Response) {
  try {
    const parsed = parseQuery(req, res, mapQuerySchema);
    if (!parsed) return;
    const text = await G.get_code(mapParams(parsed));
    res.send(text);
  } catch (error) {
    console.error("Error:", error);
    res.status(500).send("Internal Server Error");
  }
}

export async function get_shortest_path(req: Request, res: Response) {
  try {
    const parsed = parseQuery(req, res, shortestPathQuerySchema);
    if (!parsed) return;
    const result = await G.get_shortest_path(
      parsed.start_node_key || "",
      parsed.end_node_key || "",
      parsed.start_ref_id || "",
      parsed.end_ref_id || ""
    );
    res.send(result);
  } catch (error) {
    console.error("Error:", error);
    res.status(500).send("Internal Server Error");
  }
}

export async function get_graph(req: Request, res: Response) {
  try {
    const parsed = parseQuery(req, res, graphQuerySchema);
    if (!parsed) return;

    const edge_type = parsed.edge_type || ("CALLS" as EdgeType);
    const concise = parsed.concise ?? false;
    const include_edges = parsed.edges ?? false;
    const language = parsed.language;
    const since = parsed.since;
    const limit_param = parsed.limit;
    const limit_mode = parsed.limit_mode || "per_type";
    let labels = parsed.node_types
      ? parsed.node_types
      : parsed.node_type
        ? [parsed.node_type]
        : [];
    if (labels.length === 0) labels = relevant_node_types();

    const perTypeDefault = 100;
    let nodes: any[] = [];
    const ref_ids = parsed.ref_ids || [];
    if (ref_ids.length > 0) {
      nodes = await db.nodes_by_ref_ids(ref_ids, language);
    } else {
      if (limit_mode === "total") {
        nodes = await db.nodes_by_types_total(
          labels,
          limit_param || perTypeDefault,
          since,
          language
        );
      } else {
        nodes = await db.nodes_by_types_per_type(
          labels,
          limit_param || perTypeDefault,
          since,
          language
        );
      }
    }

    let edges: any[] = [];
    if (include_edges) {
      const keys = nodes.map((n) => n.properties.node_key).filter(Boolean);
      edges = await db.edges_between_node_keys(keys);
    }

    res.json({
      nodes: concise
        ? nodes.map((n) => nameFileOnly(n))
        : nodes.map((n) => toReturnNode(n)),
      edges: include_edges
        ? concise
          ? edges.map((e) => ({
              edge_type: e.edge_type,
              source: e.source,
              target: e.target,
            }))
          : edges
        : [],
      status: "Success",
      meta: buildGraphMeta(labels, nodes, limit_param, limit_mode, since),
    });
  } catch (error) {
    console.error("Error:", error);
    res.status(500).send("Internal Server Error");
  }
}
