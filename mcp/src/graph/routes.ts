import { Request, Response, NextFunction } from "express";
import {
  ContainerConfig,
  Neo4jNode,
  node_type_descriptions,
  NodeType,
  EdgeType,
  relevant_node_types,
} from "./types.js";
import {
  nameFileOnly,
  toReturnNode,
  isTrue,
  detectLanguagesAndPkgFiles,
  cloneRepoToTmp,
  extractEnvVarsFromRepo,
  findDockerComposeFiles,
  parseNodeTypes,
  parseRefIds,
  parseSince,
  parseLimit,
  parseLimitMode,
  buildGraphMeta,
} from "./utils.js";
import fs from "fs/promises";
import * as G from "./graph.js";
import { db } from "./neo4j.js";
import { parseServiceFile, extractContainersFromCompose } from "./service.js";
import * as path from "path";
import { get_context } from "../tools/explore/tool.js";
import { callGenerateObject } from "../aieo/src/stream.js";
import { z } from "zod";

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

export async function explore(req: Request, res: Response) {
  const prompt = req.query.prompt as string;
  if (!prompt) {
    res.status(400).json({ error: "Missing prompt" });
    return;
  }
  const result = await get_context(prompt);
  res.json({ result });
}

export async function understand(req: Request, res: Response) {
  const prompt = req.query.prompt as string;
  if (!prompt) {
    res.status(400).json({ error: "Missing prompt" });
    return;
  }
  const ctx = await get_context(prompt);
  const result = await callGenerateObject({
    provider: "anthropic",
    apiKey: process.env.ANTHROPIC_API_KEY as string,
    prompt: ctx,
    schema: z.object({
      function_names: z
        .array(z.string())
        .describe("functions or react components"),
      file_names: z.array(z.string()),
      datamodel_names: z.array(z.string()),
      endpoint_names: z.array(z.string()),
      page_names: z.array(z.string()),
    }),
  });
  res.json({ result });
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
    const language = req.query.language as string;

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
    const edge_type = req.query.edge_type as EdgeType;
    const concise = isTrue(req.query.concise as string);
    let ref_ids: string[] = [];
    if (req.query.ref_ids) {
      ref_ids = (req.query.ref_ids as string).split(",");
    }
    const output = req.query.output as G.OutputFormat;
    const language = req.query.language as string;

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
    const language = req.query.language as string;

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

export async function get_services(req: Request, res: Response) {
  try {
    if (req.query.clone === "true" && req.query.repo_url) {
      const repoUrl = req.query.repo_url as string;
      const username = req.query.username as string | undefined;
      const pat = req.query.pat as string | undefined;
      const commit = req.query.commit as string | undefined;

      const repoDir = await cloneRepoToTmp(repoUrl, username, pat, commit);
      const detected = await detectLanguagesAndPkgFiles(repoDir);

      const envVarsByFile = await extractEnvVarsFromRepo(repoDir);

      const services = [];
      for (const { language, pkgFile } of detected) {
        const body = await fs.readFile(pkgFile, "utf8");
        const service = parseServiceFile(pkgFile, body, language);

        const serviceDir = path.dirname(pkgFile);
        const envVars = new Set<string>();
        for (const [file, vars] of Object.entries(envVarsByFile)) {
          if (file.startsWith(serviceDir)) {
            vars.forEach((v) => envVars.add(v));
          }
        }

        service.env = {};
        envVars.forEach((v) => (service.env[v] = process.env[v] || ""));

        const { pkgFile: _, ...cleanService } = service;
        services.push(cleanService);
      }
      const composeFiles = await findDockerComposeFiles(repoDir);
      let containers: ContainerConfig[] = [];
      for (const composeFile of composeFiles) {
        const found = await extractContainersFromCompose(composeFile);
        containers = containers.concat(found);
      }
      res.json({ services, containers });
      return;
    } else {
      const { services, containers } = await G.get_services();
      res.json({ services, containers });
    }
  } catch (error) {
    console.error("Error getting services config:", error);
    res
      .status(500)
      .json({ error: "Failed to generate services configuration" });
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
    const node_type = req.query.node_type as NodeType;
    const include_functions_and_classes =
      req.query.include_functions_and_classes === "true" ||
      req.query.include_functions_and_classes === "1";
    const html = await G.get_repo_map(
      name || "",
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

export async function get_graph(req: Request, res: Response) {
  try {
    const edge_type =
      (req.query.edge_type as EdgeType) || ("CALLS" as EdgeType);
    const concise = isTrue(req.query.concise as string);
    const include_edges = isTrue(req.query.edges as string);
    const language = req.query.language as string | undefined;
    const since = parseSince(req.query);
    const limit_param = parseLimit(req.query);
    const limit_mode = parseLimitMode(req.query);
    let labels = parseNodeTypes(req.query);
    if (labels.length === 0) labels = relevant_node_types();

    const perTypeDefault = 100;
    let nodes: any[] = [];
    const ref_ids = parseRefIds(req.query);
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
