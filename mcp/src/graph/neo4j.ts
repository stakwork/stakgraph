import neo4j, { Driver, Session } from "neo4j-driver";
import {
  createNeo4jDriver,
  withNeo4jRetry,
  ResilientSession,
} from "../utils/neo4jRetry.js";
import fs from "fs";
import readline from "readline";
import { ImportanceTag, ImportanceTopNode, TaggedNode } from "../importance/types.js";
import {
  NodeType,
  all_node_types,
  Neo4jNode,
  Neo4jEdge,
  EdgeType,
  Node,
  Edge,
  toNum,
} from "./types.js";
import {
  create_node_key,
  deser_node,
  clean_node,
  getExtensionsForLanguage,
  toReturnNode,
  nameFileOnly,
} from "./utils.js";
import * as Q from "./queries.js";
import { vectorizeCodeDocument, vectorizeQuery } from "../vector/index.js";
import { v4 as uuidv4 } from "uuid";
import { createByModelName } from "@microsoft/tiktokenizer";

export type Direction = "up" | "down" | "both";

// Strip glob wildcards (**/, /**, *) to get a substring usable with Cypher CONTAINS.
// e.g. "**/*.ts" → ".ts", "stakwork/hive/**" → "stakwork/hive/", "__tests__" → "__tests__"
function normalizeGlobToContains(pattern: string): string {
  return pattern
    .replace(/^\*\*\//, "")
    .replace(/\/\*\*$/, "/")
    .replace(/\*/g, "");
}

export const Data_Bank = Q.Data_Bank;

const no_db = process.env.NO_DB === "true" || process.env.NO_DB === "1";
if (!no_db) {
  const delay_start = parseInt(process.env.DELAY_START || "0") || 0;
  const retry_interval =
    parseInt(process.env.NEO4J_RETRY_INTERVAL || "5000") || 5000;

  setTimeout(async () => {
    while (true) {
      try {
        await db.createIndexes();
        console.log("===> Neo4j indexes created successfully");
        break;
      } catch (error: any) {
        console.warn(
          `===> Neo4j not ready, retrying in ${retry_interval}ms:`,
          error?.message || error
        );
        await new Promise((resolve) => setTimeout(resolve, retry_interval));
      }
    }
  }, delay_start);
}

class Db {
  private driver: Driver;

  constructor() {
    this.driver = createNeo4jDriver();
    const host = process.env.NEO4J_HOST || "localhost:7687";
    const user = process.env.NEO4J_USER || "neo4j";
    console.log("===> connecting to", `bolt://${host}`, user);
  }

  private resilientSession(): ResilientSession {
    return new ResilientSession(
      () => this.driver,
      (d) => {
        this.driver = d;
      },
    );
  }

  async get_pkg_files(): Promise<Neo4jNode[]> {
    const session = this.resilientSession();
    try {
      const r = await session.run(Q.PKGS_QUERY);
      return r.records.map((record) => deser_node(record, "file"));
    } finally {
      await session.close();
    }
  }

  private sanitizeLabel(label: string): string | null {
    const allowed = new Set<string>(all_node_types() as string[]);
    return allowed.has(label) ? label : null;
  }

  async nodes_by_type(
    label: NodeType,
    language?: string,
    limit: number = 5000,
    since?: number,
  ): Promise<Neo4jNode[]> {
    const safe = this.sanitizeLabel(label);
    if (!safe) return [];
    const session = this.resilientSession();
    try {
      const extensions = language ? getExtensionsForLanguage(language) : [];
      const r = await session.run(Q.listQueryForLabel(safe, since != null), {
        extensions,
        limit,
        since: since ?? null,
      });
      return r.records.map((record) => deser_node(record, "f"));
    } finally {
      await session.close();
    }
  }

  async nodes_by_ref_ids(
    ref_ids: string[],
    language?: string,
    limit?: number,
  ): Promise<Neo4jNode[]> {
    const session = this.resilientSession();
    try {
      const extensions = language ? getExtensionsForLanguage(language) : [];
      const effectiveLimit = limit ?? Math.max(ref_ids.length, 1);
      const r = await session.run(Q.REF_IDS_LIST_QUERY, {
        ref_ids,
        extensions,
        limit: effectiveLimit,
      });
      return r.records.map((record) => deser_node(record, "n"));
    } finally {
      await session.close();
    }
  }

  async nodes_by_types_per_type(
    labels: NodeType[],
    limit_per_type: number,
    since?: number,
    language?: string,
  ): Promise<Neo4jNode[]> {
    const safe = labels.filter((l) => this.sanitizeLabel(l)) as NodeType[];
    if (safe.length === 0) return [];
    const results = await Promise.all(
      safe.map((label) =>
        this.nodes_by_type(label, language, limit_per_type, since)
      )
    );
    return results.flat();
  }

  async nodes_by_types_total(
    labels: NodeType[],
    limit_total: number,
    since?: number,
    language?: string,
  ): Promise<Neo4jNode[]> {
    const safe = labels.filter((l) => this.sanitizeLabel(l)) as NodeType[];
    if (safe.length === 0) return [];
    const results = await Promise.all(
      safe.map((label) =>
        this.nodes_by_type(label, language, limit_total, since)
      )
    );
    return results
      .flat()
      .sort((a, b) => {
        const at = Number(a.properties.date_added_to_graph || 0);
        const bt = Number(b.properties.date_added_to_graph || 0);
        return bt - at;
      })
      .slice(0, limit_total);
  }

  async edges_between_node_keys(keys: string[]): Promise<Neo4jEdge[]> {
    if (keys.length === 0) return [];
    const session = this.resilientSession();
    try {
      const r = await session.run(Q.EDGES_BETWEEN_NODE_KEYS_QUERY, { keys });
      return r.records.map((record) => {
        const edge = record.get("r");
        const source = record.get("a");
        const target = record.get("b");
        return {
          edge_type: edge.type,
          ref_id: edge.properties.ref_id,
          source: source.properties.ref_id,
          target: target.properties.ref_id,
          properties: edge.properties,
        } as Neo4jEdge;
      });
    } finally {
      await session.close();
    }
  }

  skip_string(skips: NodeType[]) {
    return skips.map((skip) => `-${skip}`).join("|");
  }

  async get_subtree(
    node_type: NodeType,
    name: string,
    file: string,
    ref_id: string,
    include_tests: boolean,
    depth: number,
    direction: Direction,
    trim: string[],
  ) {
    let disclude: NodeType[] = ["File", "Directory", "Repository", "Library"];
    if (node_type === "Directory") {
      // remove file and directory from disclude
      disclude = disclude.filter(
        (type) => type !== "File" && type !== "Directory",
      );
    }
    if (include_tests === false) {
      disclude.push("UnitTest", "IntegrationTest", "E2etest");
    }
    const label_filter = this.skip_string(disclude);
    const session = this.resilientSession();
    try {
      return await session.run(Q.SUBGRAPH_QUERY, {
        node_label: node_type,
        node_name: name,
        node_file: file,
        ref_id: ref_id,
        depth,
        direction,
        label_filter,
        trim,
      });
    } finally {
      await session.close();
    }
  }

  async find_node(
    node_type: NodeType,
    name: string,
    file: string,
    ref_id: string,
  ): Promise<Neo4jNode | null> {
    const session = this.resilientSession();
    try {
      const r = await session.run(Q.FIND_NODE_QUERY, {
        node_label: node_type,
        node_name: name,
        node_file: file,
        ref_id: ref_id,
      });
      if (r.records.length === 0) return null;
      return deser_node(r.records[0], "node");
    } finally {
      await session.close();
    }
  }

  async get_repositories(): Promise<Neo4jNode[]> {
    const session = this.resilientSession();
    try {
      const r = await session.run(Q.REPOSITORIES_QUERY);
      return r.records.map((record) => deser_node(record, "r"));
    } finally {
      await session.close();
    }
  }

  async get_file_ends_with(file_end: string): Promise<Neo4jNode> {
    const session = this.resilientSession();
    try {
      const r = await session.run(Q.FILE_ENDS_WITH_QUERY, {
        file_name: file_end,
      });
      return r.records.map((record) => deser_node(record, "f"))[0];
    } finally {
      await session.close();
    }
  }

  async get_repo_subtree(
    name: string,
    ref_id: string,
    node_type: NodeType = "Repository",
    include_functions_and_classes: boolean = false,
  ) {
    // include if functions and classes should be included
    let disclude: NodeType[] = all_node_types().filter(
      (type: NodeType) =>
        type !== "File" && type !== "Directory" && type !== "Repository",
    );
    if (include_functions_and_classes) {
      console.log("including functions and classes");
      disclude = disclude.filter(
        (type) => type !== "Function" && type !== "Class",
      );
    }
    const session = this.resilientSession();
    // console.log("get_repo_subtree", name, ref_id, this.skip_string(disclude));
    try {
      return await session.run(Q.REPO_SUBGRAPH_QUERY, {
        node_label: node_type,
        node_name: name,
        node_file: "",
        ref_id: ref_id || "",
        depth: 10,
        label_filter: this.skip_string(disclude),
        trim: [],
      });
    } finally {
      await session.close();
    }
  }

  async get_shortest_path(start_node_key: string, end_node_key: string) {
    const session = this.resilientSession();
    try {
      return await session.run(Q.SHORTEST_PATH_QUERY, {
        start_node_key,
        end_node_key,
      });
    } finally {
      await session.close();
    }
  }

  async get_shortest_path_ref_id(start_ref_id: string, end_ref_id: string) {
    const session = this.resilientSession();
    try {
      return await session.run(Q.SHORTEST_PATH_QUERY_REF_ID, {
        start_ref_id,
        end_ref_id,
      });
    } finally {
      await session.close();
    }
  }

  async update_all_token_counts() {
    const session = this.resilientSession();
    try {
      const tokenizer = await createByModelName("gpt-4");
      const result = await session.run(
        Q.DATA_BANK_BODIES_QUERY_NO_TOKEN_COUNT,
        {
          do_files: true,
        },
      );
      const data_bank = result.records.map((record) => ({
        node_key: record.get("node_key"),
        body: record.get("body"),
      }));
      console.log(`Found ${data_bank.length} nodes without token counts`);
      const BATCH_SIZE = 256;
      for (let i = 0; i < data_bank.length; i += BATCH_SIZE) {
        const chunk = data_bank.slice(i, i + BATCH_SIZE);
        const batch = chunk.map((node) => {
          const tokens = tokenizer.encode(node.body || "", []);
          return { node_key: node.node_key, token_count: tokens.length };
        });
        await session.run(Q.BULK_UPDATE_TOKEN_COUNT_QUERY, { batch });
        console.log(
          `Updated token counts: ${Math.min(i + BATCH_SIZE, data_bank.length)}/${data_bank.length}`,
        );
      }
    } catch (error) {
      console.error("Error updating token counts:", error);
    } finally {
      await session.close();
    }
  }

  // Main function to process both nodes and edges
  async build_graph_from_files(node_file: string, edge_file: string) {
    try {
      await withNeo4jRetry(
        () => this.driver,
        (d) => {
          this.driver = d;
        },
        async (session) => {
          console.log("Processing nodes...", node_file);
          await process_file(session, node_file, (data) =>
            construct_merge_node_query(data),
          );
          console.log("Processing edges...", edge_file);
          await process_file(session, edge_file, (data) =>
            construct_merge_edge_query(data),
          );
          console.log("Added nodes to graph!");
        },
        "build_graph_from_files",
      );
    } catch (error) {
      console.error("Error:", error);
    }
  }

  async search(
    query: string,
    limit: number,
    node_types: NodeType[],
    skip_node_types: NodeType[],
    maxTokens: number, // Optional parameter for token limit
    language?: string,
    include_patterns?: string[],
    exclude_patterns?: string[],
  ): Promise<Neo4jNode[]> {
    const session = this.resilientSession();

    const q_escaped = prepareFulltextSearchQuery(query);
    // console.log("===> search query escaped:", q_escaped);

    // skip Import nodes
    if (!skip_node_types.includes("Import")) {
      skip_node_types.push("Import");
    }

    const extensions = language ? getExtensionsForLanguage(language) : [];
    const normalized_include = (include_patterns ?? [])
      .map(normalizeGlobToContains)
      .filter(Boolean);
    const normalized_exclude = (exclude_patterns ?? [])
      .map(normalizeGlobToContains)
      .filter(Boolean);

    try {
      const result = await session.run(Q.SEARCH_QUERY_COMPOSITE, {
        query: q_escaped,
        limit,
        node_types,
        skip_node_types,
        extensions,
        include_patterns: normalized_include,
        exclude_patterns: normalized_exclude,
      });
      const nodes = result.records.map((record) => {
        const node: Neo4jNode = deser_node(record, "node");
        return {
          properties: node.properties,
          labels: node.labels,
          score: record.get("score"),
        };
      });
      if (!maxTokens) {
        return nodes;
      }
      // Apply token count filtering if maxTokens is specified
      let totalTokens = 0;
      const filteredNodes: Neo4jNode[] = [];
      for (const node of nodes) {
        const tokenCount = node.properties.token_count
          ? parseInt(node.properties.token_count.toString(), 10)
          : 0;
        if (totalTokens + tokenCount <= maxTokens) {
          totalTokens += tokenCount;
          filteredNodes.push(node);
        } else {
          break;
        }
      }
      return filteredNodes;
    } finally {
      await session.close();
    }
  }

  async vectorSearch(
    query: string,
    limit: number,
    node_types: NodeType[],
    skip_node_types: NodeType[] = [],
    maxTokens: number = 0,
    language?: string,
    include_patterns?: string[],
    exclude_patterns?: string[],
  ): Promise<Neo4jNode[]> {
    let session: ResilientSession | null = null;
    try {
      session = this.resilientSession();
      const embeddings = await vectorizeQuery(query);

      if (!skip_node_types.includes("Import")) {
        skip_node_types.push("Import");
      }

      const extensions = language ? getExtensionsForLanguage(language) : [];
      const normalized_include = (include_patterns ?? [])
        .map(normalizeGlobToContains)
        .filter(Boolean);
      const normalized_exclude = (exclude_patterns ?? [])
        .map(normalizeGlobToContains)
        .filter(Boolean);

      const result = await session.run(Q.VECTOR_SEARCH_QUERY, {
        embeddings,
        limit,
        node_types,
        skip_node_types,
        extensions,
        include_patterns: normalized_include,
        exclude_patterns: normalized_exclude,
      });
      const nodes = result.records.map((record) => {
        const node: Neo4jNode = deser_node(record, "node");
        return {
          properties: node.properties,
          labels: node.labels,
          score: record.get("score"),
        };
      });
      if (!maxTokens) {
        return nodes;
      }
      let totalTokens = 0;
      const filteredNodes: Neo4jNode[] = [];
      for (const node of nodes) {
        const tokenCount = node.properties.token_count
          ? parseInt(node.properties.token_count.toString(), 10)
          : 0;
        if (totalTokens + tokenCount <= maxTokens) {
          totalTokens += tokenCount;
          filteredNodes.push(node);
        } else {
          break;
        }
      }
      return filteredNodes;
    } catch (error) {
      console.error("Error vector searching:", error);
      throw error;
    } finally {
      if (session) {
        await session.close();
      }
    }
  }

  async create_hint(
    question: string,
    answer: string,
    embeddings: number[],
    persona: string = "PM",
  ) {
    const session = this.resilientSession();
    const name = question.slice(0, 80);
    const node_key = create_node_key({
      node_type: "Hint",
      node_data: {
        name,
        file: "hint://generated",
        start: 0,
      },
    } as Node);
    try {
      await session.run(Q.CREATE_HINT_QUERY, {
        node_key,
        name,
        file: "hint://generated",
        body: answer,
        question,
        embeddings,
        ts: Date.now() / 1000,
        persona,
      });
      const r = await session.run(Q.GET_HINT_QUERY, { node_key });
      const record = r.records[0];
      const n = record.get("n");
      return { ref_id: n.properties.ref_id, node_key };
    } finally {
      await session.close();
    }
  }

  async get_connected_hints(prompt_ref_id: string) {
    const session = this.resilientSession();
    try {
      const result = await session.run(Q.GET_CONNECTED_HINTS_QUERY, {
        prompt_ref_id,
      });
      return result.records.map((record) => clean_node(record.get("h")));
    } finally {
      await session.close();
    }
  }

  async create_mock(
    name: string,
    description: string,
    files: string[],
    mocked: boolean,
  ) {
    const session = this.resilientSession();
    const node_key = create_node_key({
      node_type: "Mock",
      node_data: {
        name,
        file: `mock://${name}`,
        start: 0,
      },
    } as Node);
    try {
      await session.run(Q.CREATE_MOCK_QUERY, {
        node_key,
        name,
        file: `mock://${name}`,
        body: JSON.stringify(files),
        description,
        mocked,
        ts: Date.now() / 1000,
      });
      const r = await session.run(Q.GET_MOCK_QUERY, { node_key });
      const record = r.records[0];
      const n = record.get("n");
      return { ref_id: n.properties.ref_id, node_key };
    } finally {
      await session.close();
    }
  }

  async link_mock_to_file(mock_ref_id: string, file_path: string) {
    const session = this.resilientSession();
    try {
      await session.run(Q.LINK_MOCK_TO_FILE_QUERY, {
        mock_ref_id,
        file_path,
      });
    } finally {
      await session.close();
    }
  }

  async update_mock_status(name: string, mocked: boolean, files: string[]) {
    const session = this.resilientSession();
    try {
      await session.run(Q.UPDATE_MOCK_STATUS_QUERY, {
        name,
        mocked,
        body: JSON.stringify(files),
      });
    } finally {
      await session.close();
    }
  }

  async get_all_mocks(): Promise<Neo4jNode[]> {
    const session = this.resilientSession();
    try {
      const result = await session.run(Q.GET_ALL_MOCKS_QUERY);
      return result.records.map((record) => clean_node(record.get("n")));
    } finally {
      await session.close();
    }
  }

  async get_mocks_inventory(repo?: string): Promise<
    {
      name: string;
      ref_id: string;
      description: string;
      linked_files: string[];
      file_count: number;
      mocked: boolean;
    }[]
  > {
    const session = this.resilientSession();
    try {
      const result = await session.run(Q.GET_MOCKS_INVENTORY_QUERY, {
        repo: repo || null,
      });
      return result.records.map((record) => {
        const linked_files: string[] = record.get("linked_files") || [];
        return {
          name: record.get("name") || "",
          ref_id: record.get("ref_id") || "",
          description: record.get("description") || "",
          linked_files,
          file_count:
            record.get("file_count")?.toNumber?.() ||
            record.get("file_count") ||
            0,
          mocked: record.get("mocked") ?? false,
        };
      });
    } finally {
      await session.close();
    }
  }

  async get_node_with_related(ref_id: string) {
    const session = this.resilientSession();
    try {
      const result = await session.run(Q.GET_NODE_WITH_RELATED_QUERY, {
        ref_id,
      });

      if (result.records.length === 0) {
        return { nodes: [], edges: [] };
      }

      const record = result.records[0];
      const allNodesArray = record.get("allNodes");
      const edgesArray = record.get("edges");

      // Process nodes
      const nodes = allNodesArray
        .filter((node: any) => node !== null)
        .map((node: any) => clean_node(node));

      // Process edges - filter out any empty edge objects
      const edges = edgesArray
        .filter((edge: any) => edge.source && edge.target && edge.edge_type)
        .map((edge: any) => ({
          edge_type: edge.edge_type,
          source: edge.source,
          target: edge.target,
          properties: edge.properties || {},
        }));

      return {
        nodes: nodes.map(toReturnNode),
        edges,
      };
    } finally {
      await session.close();
    }
  }

  async get_workflow_published_version_subgraph(
    ref_id: string,
    concise: boolean = false,
  ) {
    const session = this.resilientSession();
    try {
      const result = await session.run(
        Q.GET_WORKFLOW_PUBLISHED_VERSION_SUBGRAPH_QUERY,
        {
          ref_id,
        },
      );

      if (result.records.length === 0) {
        return { nodes: [], edges: [] };
      }

      const record = result.records[0];
      const allNodesArray = record.get("allNodes");
      const edgesArray = record.get("edges");

      // Process nodes
      const nodes = allNodesArray
        .filter((node: any) => node !== null)
        .map((node: any) => clean_node(node));

      // Process edges - filter out any empty edge objects
      const edges = edgesArray
        .filter((edge: any) => edge.source && edge.target && edge.edge_type)
        .map((edge: any) => ({
          edge_type: edge.edge_type,
          source: edge.source,
          target: edge.target,
          properties: edge.properties || {},
        }));

      return {
        nodes: concise
          ? nodes.map((n: any) => nameFileOnly(n))
          : nodes.map(toReturnNode),
        edges,
      };
    } finally {
      await session.close();
    }
  }

  async delete_node_by_ref_id(ref_id: string): Promise<number> {
    const session = this.resilientSession();
    try {
      const result = await session.run(Q.DELETE_NODE_BY_REF_ID_QUERY, {
        ref_id,
      });
      const record = result.records[0];
      return record.get("deleted_count").toNumber();
    } finally {
      await session.close();
    }
  }

  async add_node(node_type: NodeType, node_data: any): Promise<string> {
    const session = this.resilientSession();
    try {
      const node_key = create_node_key({
        node_type,
        node_data: {
          name: node_data.name,
          file: node_data.file,
          start: node_data.start || 0,
        },
      } as Node);

      const now = Date.now();

      const { ref_id, ...properties } = node_data;

      await session.run(Q.ADD_NODE_QUERY(node_type), {
        node_key,
        properties: { ...properties, node_key },
        now,
      });

      const result = await session.run(
        `MATCH (n:${node_type} {node_key: $node_key})
         SET n.ref_id = coalesce(n.ref_id, $ref_id)
         RETURN n.ref_id as ref_id`,
        { node_key, ref_id: ref_id || uuidv4() },
      );

      return result.records[0].get("ref_id");
    } finally {
      await session.close();
    }
  }

  async add_edge(
    edge_type: EdgeType,
    source_ref_id: string,
    target_ref_id: string,
  ): Promise<void> {
    const session = this.resilientSession();
    try {
      await session.run(Q.ADD_EDGE_QUERY(edge_type), {
        source_ref_id,
        target_ref_id,
      });
    } finally {
      await session.close();
    }
  }

  async hints_without_siblings(): Promise<Neo4jNode[]> {
    const session = this.resilientSession();
    try {
      const result = await session.run(Q.HINTS_WITHOUT_SIBLINGS_QUERY);
      return result.records.map((record) => clean_node(record.get("h")));
    } finally {
      await session.close();
    }
  }

  async create_sibling_edge(
    source_ref_id: string,
    target_ref_id: string,
  ): Promise<void> {
    const session = this.resilientSession();
    try {
      await session.run(Q.CREATE_SIBLING_EDGE_QUERY, {
        source_ref_id,
        target_ref_id,
      });
    } finally {
      await session.close();
    }
  }

  async get_hint_siblings(ref_id: string): Promise<Neo4jNode[]> {
    const session = this.resilientSession();
    try {
      const result = await session.run(Q.GET_HINT_SIBLINGS_QUERY, { ref_id });
      return result.records.map((record) => clean_node(record.get("s")));
    } finally {
      await session.close();
    }
  }

  async setHintPersona(ref_id: string, persona: string): Promise<void> {
    const session = this.resilientSession();
    try {
      await session.run(Q.SET_HINT_PERSONA_QUERY, { ref_id, persona });
    } finally {
      await session.close();
    }
  }

  async create_prompt(question: string, answer: string, embeddings: number[]) {
    const session = this.resilientSession();
    const name = question.slice(0, 80);
    const node_key = create_node_key({
      node_type: "Prompt",
      node_data: {
        name,
        file: "prompt://generated",
        start: 0,
      },
    } as Node);
    try {
      await session.run(Q.CREATE_PROMPT_QUERY, {
        node_key,
        name,
        file: "prompt://generated",
        body: answer,
        question,
        embeddings,
        ts: Date.now() / 1000,
      });
      const r = await session.run(Q.GET_PROMPT_QUERY, { node_key });
      const record = r.records[0];
      const n = record.get("n");
      return { ref_id: n.properties.ref_id, node_key };
    } finally {
      await session.close();
    }
  }

  async create_pull_request(
    name: string,
    docs: string,
    embeddings: number[],
    number: string,
  ) {
    const session = this.resilientSession();
    const short_name = name.slice(0, 80);
    const node_key = create_node_key({
      node_type: "PullRequest",
      node_data: {
        name: short_name,
        file: `pr://${number}`,
        start: 0,
      },
    } as Node);
    try {
      await session.run(Q.CREATE_PULL_REQUEST_QUERY, {
        node_key,
        name: short_name,
        file: `pr://${number}`,
        body: docs, // Store docs in body for backward compatibility
        docs,
        number,
        embeddings,
        ts: Date.now() / 1000,
      });
      const r = await session.run(Q.GET_PULL_REQUEST_QUERY, { node_key });
      const record = r.records[0];
      const n = record.get("n");
      return { ref_id: n.properties.ref_id, node_key, number };
    } finally {
      await session.close();
    }
  }

  // === New Learning + Scope system ===

  async upsert_learning(
    id: string,
    rule: string,
    embeddings: number[],
    reason?: string,
  ): Promise<{ ref_id: string; node_key: string }> {
    const session = this.resilientSession();
    const node_key = create_node_key({
      node_type: "Learning",
      node_data: {
        name: id,
        file: "learning://generated",
        start: 0,
      },
    } as Node);
    try {
      const result = await session.run(Q.UPSERT_LEARNING_QUERY, {
        id,
        node_key,
        rule,
        reason: reason || null,
        embeddings,
        ts: Date.now() / 1000,
      });
      const n = result.records[0].get("n");
      return { ref_id: n.properties.ref_id, node_key };
    } finally {
      await session.close();
    }
  }

  async upsert_scope(
    name: string,
    embeddings: number[],
  ): Promise<{ ref_id: string }> {
    const session = this.resilientSession();
    const node_key = create_node_key({
      node_type: "Scope",
      node_data: {
        name,
        file: "scope://generated",
        start: 0,
      },
    } as Node);
    try {
      const result = await session.run(Q.UPSERT_SCOPE_QUERY, {
        name,
        node_key,
        embeddings,
        ts: Date.now() / 1000,
      });
      const s = result.records[0].get("s");
      return { ref_id: s.properties.ref_id };
    } finally {
      await session.close();
    }
  }

  async create_has_scope_edge(
    learning_id: string,
    scope_name: string,
  ): Promise<void> {
    const session = this.resilientSession();
    try {
      await session.run(Q.CREATE_HAS_SCOPE_EDGE_QUERY, {
        learning_id,
        scope_name,
      });
    } finally {
      await session.close();
    }
  }

  async get_all_learnings_with_scopes(): Promise<
    { id: string; rule: string; reason: string | null; scopes: string[] }[]
  > {
    const session = this.resilientSession();
    try {
      const result = await session.run(Q.GET_ALL_LEARNINGS_WITH_SCOPES_QUERY);
      return result.records.map((record) => {
        const n = record.get("l");
        const scopes: string[] = record.get("scopes") || [];
        return {
          id: n.properties.id,
          rule: n.properties.rule,
          reason: n.properties.reason || null,
          scopes,
        };
      });
    } finally {
      await session.close();
    }
  }

  async get_all_scopes(): Promise<string[]> {
    const session = this.resilientSession();
    try {
      const result = await session.run(Q.GET_ALL_SCOPES_QUERY);
      return result.records.map((record) => record.get("name") as string);
    } finally {
      await session.close();
    }
  }

  async get_learnings_by_scopes(
    scope_names: string[],
  ): Promise<
    { id: string; rule: string; reason: string | null; scopes: string[] }[]
  > {
    const session = this.resilientSession();
    try {
      const result = await session.run(Q.GET_LEARNINGS_BY_SCOPES_QUERY, {
        scope_names,
      });
      return result.records.map((record) => {
        const n = record.get("l");
        const scopes: string[] = record.get("scopes") || [];
        return {
          id: n.properties.id,
          rule: n.properties.rule,
          reason: n.properties.reason || null,
          scopes,
        };
      });
    } finally {
      await session.close();
    }
  }

  async createEdgesDirectly(
    hint_ref_id: string,
    weightedRefIds: { ref_id: string; relevancy: number }[],
  ): Promise<{ edges_added: number; linked_ref_ids: string[] }> {
    const session = this.resilientSession();
    try {
      const result = await session.run(Q.CREATE_HINT_EDGES_BY_REF_IDS_QUERY, {
        hint_ref_id,
        weighted_ref_ids: weightedRefIds,
      });

      if (result.records.length > 0) {
        const linkedRefs = result.records[0].get("refs") || [];
        return {
          edges_added: linkedRefs.length,
          linked_ref_ids: linkedRefs,
        };
      }

      return { edges_added: 0, linked_ref_ids: [] };
    } finally {
      await session.close();
    }
  }

  async findNodesByName(name: string, nodeType: string): Promise<Neo4jNode[]> {
    const session = this.resilientSession();
    try {
      if (nodeType !== "File") {
        const query = Q.FIND_NODES_BY_NAME_QUERY.replace("{LABEL}", nodeType);
        const result = await session.run(query, { name });
        return result.records.map((record) => clean_node(record.get("n")));
      } else {
        const query = Q.FIND_FILE_NODES_BY_PATH_QUERY;
        const result = await session.run(query, { file_path: name });
        return result.records.map((record) => clean_node(record.get("n")));
      }
    } finally {
      await session.close();
    }
  }

  async get_orphaned_hints(): Promise<Neo4jNode[]> {
    const session = this.resilientSession();
    try {
      const result = await session.run(Q.ORPHANED_HINTS_QUERY);
      return result.records.map((record) => clean_node(record.get("h")));
    } finally {
      await session.close();
    }
  }

  async createIndexes(): Promise<void> {
    let session: ResilientSession | null = null;
    try {
      session = this.resilientSession();
      // console.log("Creating indexes...");
      // console.log(Q.KEY_INDEX_QUERY);
      // console.log(Q.FULLTEXT_BODY_INDEX_QUERY);
      // console.log(Q.FULLTEXT_NAME_INDEX_QUERY);
      // console.log(Q.FULLTEXT_COMPOSITE_INDEX_QUERY);
      // console.log(Q.VECTOR_INDEX_QUERY);
      await session.run(Q.KEY_INDEX_QUERY);
      await session.run(Q.REF_ID_INDEX_QUERY);
      await session.run(Q.FULLTEXT_BODY_INDEX_QUERY);
      await session.run(Q.FULLTEXT_NAME_INDEX_QUERY);
      await session.run(Q.FULLTEXT_COMPOSITE_INDEX_QUERY);
      await session.run(Q.VECTOR_INDEX_QUERY);
      await session.run(
        "CREATE INDEX agent_session_id_index IF NOT EXISTS FOR (n:AgentSession) ON (n.node_key)",
      );
    } finally {
      if (session) {
        await session.close();
      }
    }
  }
  async get_rules_files(): Promise<Neo4jNode[]> {
    const session = this.resilientSession();
    try {
      const result = await session.run(Q.RULES_FILES_QUERY);
      return result.records.map((record) => deser_node(record, "f"));
    } finally {
      await session.close();
    }
  }

  async get_env_vars(): Promise<Neo4jNode[]> {
    const session = this.resilientSession();
    try {
      const result = await session.run(Q.ENV_VARS_QUERY);
      return result.records.map((record) => deser_node(record, "n"));
    } finally {
      await session.close();
    }
  }

  async edges_by_type(
    edge_type?: EdgeType,
    language?: string,
    limit: number = 1000,
  ): Promise<Neo4jEdge[]> {
    const session = this.resilientSession();
    try {
      const extensions = language ? getExtensionsForLanguage(language) : [];
      const edge_types = edge_type ? [edge_type] : [];
      const result = await session.run(Q.EDGES_BY_TYPE_QUERY, {
        edge_types,
        extensions,
        limit,
      });
      return result.records.map((record) => deser_edge(record));
    } finally {
      await session.close();
    }
  }

  async edges_by_ref_ids(
    ref_ids: string[],
    language?: string,
    limit: number = 1000,
  ): Promise<Neo4jEdge[]> {
    const session = this.resilientSession();
    try {
      const extensions = language ? getExtensionsForLanguage(language) : [];
      const result = await session.run(Q.EDGES_BY_REF_IDS_QUERY, {
        ref_ids,
        extensions,
        limit,
      });
      return result.records.map((record) => deser_edge(record));
    } finally {
      await session.close();
    }
  }

  async all_edges(
    language?: string,
    limit: number = 1000,
  ): Promise<Neo4jEdge[]> {
    const session = this.resilientSession();
    try {
      const extensions = language ? getExtensionsForLanguage(language) : [];
      const result = await session.run(Q.ALL_EDGES_QUERY, {
        extensions,
        limit,
      });
      return result.records.map((record) => deser_edge(record));
    } finally {
      await session.close();
    }
  }
  async update_repository_documentation(ref_id: string, documentation: string) {
    const session = this.resilientSession();
    try {
      await session.run(Q.UPDATE_REPO_DOCS_QUERY, {
        ref_id,
        documentation,
        ts: Date.now() / 1000,
      });
    } finally {
      await session.close();
    }
  }

  async get_nodes_without_description(
    limit: number,
    repo_paths: string[] | null,
    file_paths: string[],
  ): Promise<Neo4jNode[]> {
    const session = this.resilientSession();
    try {
      const result = await session.run(Q.GET_NODES_WITHOUT_DESCRIPTION_QUERY, {
        limit: neo4j.int(limit),
        repo_paths: repo_paths || [],
        file_paths,
      });
      return result.records.map((record) => ({
        properties: record.get("properties"),
        labels: record.get("labels"),
        ref_id: record.get("ref_id"),
      }));
    } finally {
      await session.close();
    }
  }

  async update_node_description_and_embeddings(
    ref_id: string,
    description: string,
    embeddings: number[],
  ) {
    const session = this.resilientSession();
    try {
      await session.run(Q.UPDATE_NODE_DESCRIPTION_AND_EMBEDDINGS_QUERY, {
        ref_id,
        description,
        embeddings,
      });
    } finally {
      await session.close();
    }
  }

  async update_node_description_only(ref_id: string, description: string) {
    const session = this.resilientSession();
    try {
      await session.run(Q.UPDATE_NODE_DESCRIPTION_ONLY_QUERY, {
        ref_id,
        description,
      });
    } finally {
      await session.close();
    }
  }

  async get_nodes_with_description_without_embeddings(
    limit: number,
    repo_paths: string[] | null,
    file_paths: string[],
  ): Promise<{ ref_id: string; description: string }[]> {
    const session = this.resilientSession();
    try {
      const result = await session.run(
        Q.GET_NODES_WITH_DESCRIPTION_WITHOUT_EMBEDDINGS_QUERY,
        {
          limit: neo4j.int(limit),
          repo_paths: repo_paths || [],
          file_paths,
        },
      );
      return result.records.map((record) => ({
        ref_id: record.get("ref_id"),
        description: record.get("description"),
      }));
    } finally {
      await session.close();
    }
  }

  async bulk_update_embeddings(
    batch: { ref_id: string; embeddings: number[] }[],
  ) {
    const session = this.resilientSession();
    try {
      await session.run(Q.BULK_UPDATE_EMBEDDINGS_BY_REF_ID_QUERY, { batch });
    } finally {
      await session.close();
    }
  }

  async bulk_update_descriptions(
    batch: { ref_id: string; description: string }[],
  ) {
    const session = this.resilientSession();
    try {
      await session.run(Q.BULK_UPDATE_DESCRIPTIONS_ONLY_QUERY, { batch });
    } finally {
      await session.close();
    }
  }

  async bulk_update_descriptions_and_embeddings(
    batch: { ref_id: string; description: string; embeddings: number[] }[],
  ) {
    const session = this.resilientSession();
    try {
      await session.run(Q.BULK_UPDATE_DESCRIPTIONS_AND_EMBEDDINGS_QUERY, {
        batch,
      });
    } finally {
      await session.close();
    }
  }

  async count_nodes_with_embeddings(): Promise<number> {
    const session = this.resilientSession();
    try {
      const r = await session.run(Q.COUNT_NODES_WITH_EMBEDDINGS_QUERY);
      return r.records[0].get("c").toNumber();
    } finally {
      await session.close();
    }
  }

  async count_eligible_nodes_for_embeddings(): Promise<number> {
    const session = this.resilientSession();
    try {
      const r = await session.run(Q.COUNT_ELIGIBLE_NODES_FOR_EMBEDDINGS_QUERY);
      return r.records[0].get("c").toNumber();
    } finally {
      await session.close();
    }
  }

  async count_workflow_nodes(): Promise<number> {
    const session = this.resilientSession();
    try {
      const r = await session.run(Q.COUNT_WORKFLOWS_QUERY);
      return r.records[0].get("c").toNumber();
    } finally {
      await session.close();
    }
  }

  async get_all_workflows(): Promise<any[]> {
    const session = this.resilientSession();
    try {
      const r = await session.run(Q.GET_ALL_WORKFLOWS_QUERY);
      return r.records.map((rec) => rec.get("w").properties);
    } finally {
      await session.close();
    }
  }

  async get_workflow_by_key(
    node_key: string,
    ref_id?: string,
  ): Promise<any | null> {
    const session = this.resilientSession();
    try {
      let r = await session.run(Q.GET_WORKFLOW_BY_KEY_QUERY, { node_key });
      if (r.records.length === 0 && ref_id) {
        r = await session.run(Q.GET_WORKFLOW_BY_REF_ID_QUERY, { ref_id });
      }
      return r.records.length > 0 ? r.records[0].get("w").properties : null;
    } finally {
      await session.close();
    }
  }

  async get_workflow_documentation(node_key: string): Promise<any | null> {
    const session = this.resilientSession();
    try {
      const r = await session.run(Q.GET_WORKFLOW_DOCUMENTATION_QUERY, {
        node_key,
      });
      return r.records.length > 0 ? r.records[0].get("d").properties : null;
    } finally {
      await session.close();
    }
  }

  async upsert_workflow_documentation(
    workflow_ref_id: string,
    name: string,
    body: string,
  ): Promise<string> {
    const session = this.resilientSession();
    try {
      const node_key = `workflow_documentation_${workflow_ref_id}`;
      const ts = Date.now();
      const r = await session.run(Q.UPSERT_WORKFLOW_DOCUMENTATION_QUERY, {
        workflow_ref_id,
        node_key,
        name,
        body,
        ts,
      });
      return r.records[0].get("ref_id");
    } finally {
      await session.close();
    }
  }

  async project_importance_graph(graphName: string): Promise<number> {
    const session = this.resilientSession();
    try {
      const r = await session.run(Q.IMPORTANCE_GRAPH_PROJECT_QUERY, {
        graphName,
      });
      return toNum(r.records[0]?.get("nodeCount") ?? 0);
    } finally {
      await session.close();
    }
  }

  async stream_pagerank(
    graphName: string,
  ): Promise<{ ref_id: string; score: number }[]> {
    const session = this.resilientSession();
    try {
      const r = await session.run(Q.IMPORTANCE_PAGERANK_QUERY, { graphName });
      return r.records.map((rec) => ({
        ref_id: rec.get("ref_id"),
        score: toNum(rec.get("score")),
      }));
    } finally {
      await session.close();
    }
  }

  async get_degree_counts(): Promise<
    {
      ref_id: string;
      node_type: string;
      in_degree: number;
      out_degree: number;
    }[]
  > {
    const session = this.resilientSession();
    try {
      const r = await session.run(Q.IMPORTANCE_DEGREE_QUERY);
      return r.records.map((rec) => ({
        ref_id: rec.get("ref_id"),
        node_type: rec.get("node_type") ?? "",
        in_degree: toNum(rec.get("in_degree")),
        out_degree: toNum(rec.get("out_degree")),
      }));
    } finally {
      await session.close();
    }
  }

  async bulk_update_importance(batch: TaggedNode[]): Promise<void> {
    const session = this.resilientSession();
    try {
      await session.run(Q.BULK_UPDATE_IMPORTANCE_QUERY, { batch });
    } finally {
      await session.close();
    }
  }

  async drop_importance_graph(graphName: string): Promise<void> {
    const session = this.resilientSession();
    try {
      await session.run(Q.IMPORTANCE_GRAPH_DROP_QUERY, { graphName });
    } finally {
      await session.close();
    }
  }

  /**
   * Delete all graph nodes belonging to a repo (e.g. "owner/repo").
   * Removes:
   *   - Data_Bank nodes whose file path starts with the repo prefix
   *     (Function, Class, File, Endpoint, Repository, etc.)
   *   - Concept and Clue nodes tagged with f.repo = $repo (and id prefix fallback)
   * Returns counts for each category.
   */
  async delete_repo(
    repo: string,
  ): Promise<{ file_nodes: number; concepts: number; clues: number }> {
    if (!repo || !repo.trim()) {
      throw new Error("repo is required");
    }
    const prefix = repo.trim();
    const session = this.resilientSession();
    try {
      const fileRes = await session.run(
        `
        MATCH (n:Data_Bank)
        WHERE n.file = $prefix OR n.file STARTS WITH $prefixSlash
        WITH collect(n) as nodes, count(n) as deleted_count
        FOREACH (x IN nodes | DETACH DELETE x)
        RETURN deleted_count
        `,
        { prefix, prefixSlash: `${prefix}/` },
      );
      const file_nodes =
        fileRes.records[0]?.get("deleted_count")?.toNumber?.() ?? 0;

      const conceptRes = await session.run(
        `
        MATCH (f)
        WHERE (f:Concept OR f:Feature)
          AND (f.repo = $prefix OR f.id STARTS WITH $idPrefix)
        WITH collect(f) as nodes, count(f) as deleted_count
        FOREACH (x IN nodes | DETACH DELETE x)
        RETURN deleted_count
        `,
        { prefix, idPrefix: `${prefix}/` },
      );
      const concepts =
        conceptRes.records[0]?.get("deleted_count")?.toNumber?.() ?? 0;

      const clueRes = await session.run(
        `
        MATCH (c:Clue)
        WHERE c.repo = $prefix OR c.id STARTS WITH $idPrefix
        WITH collect(c) as nodes, count(c) as deleted_count
        FOREACH (x IN nodes | DETACH DELETE x)
        RETURN deleted_count
        `,
        { prefix, idPrefix: `${prefix}/` },
      );
      const clues = clueRes.records[0]?.get("deleted_count")?.toNumber?.() ?? 0;

      return { file_nodes, concepts, clues };
    } finally {
      await session.close();
    }
  }

  async get_top_nodes_by_importance(
    limit: number,
  ): Promise<ImportanceTopNode[]> {
    const session = this.resilientSession();
    try {
      const r = await session.run(Q.GET_TOP_NODES_BY_IMPORTANCE_QUERY, {
        limit: neo4j.int(limit),
      });
      return r.records.map((rec) => ({
        ref_id: rec.get("ref_id"),
        name: rec.get("name"),
        file: rec.get("file") || "",
        label: rec.get("label"),
        pagerank: toNum(rec.get("pagerank")),
        in_degree: toNum(rec.get("in_degree")),
        out_degree: toNum(rec.get("out_degree")),
        entry_score: toNum(rec.get("entry_score")),
        utility_score: toNum(rec.get("utility_score")),
        hub_score: toNum(rec.get("hub_score")),
        importance_tag: (rec.get("importance_tag") as ImportanceTag) ?? null,
      }));
    } finally {
      await session.close();
    }
  }

  async get_nodes_by_importance_tag(
    tag: ImportanceTag,
    limit: number,
  ): Promise<ImportanceTopNode[]> {
    const session = this.resilientSession();
    try {
      const r = await session.run(Q.GET_NODES_BY_IMPORTANCE_TAG_QUERY, {
        tag,
        limit: neo4j.int(limit),
      });
      return r.records.map((rec) => ({
        ref_id: rec.get("ref_id"),
        name: rec.get("name"),
        file: rec.get("file") || "",
        label: rec.get("label"),
        pagerank: toNum(rec.get("pagerank")),
        in_degree: toNum(rec.get("in_degree")),
        out_degree: toNum(rec.get("out_degree")),
        entry_score: toNum(rec.get("entry_score")),
        utility_score: toNum(rec.get("utility_score")),
        hub_score: toNum(rec.get("hub_score")),
        importance_tag: rec.get("importance_tag") as ImportanceTag,
      }));
    } finally {
      await session.close();
    }
  }

  async upsert_agent_session(params: {
    session_id: string;
    source: string;
    repo: string;
    model: string;
    provider: string;
    start_time: number;
    end_time: number;
    duration_ms: number;
    input_tokens: number;
    cache_read_tokens: number;
    cache_write_tokens: number;
    output_tokens: number;
    total_tokens: number;
    status: string;
    error_message: string;
  }): Promise<void> {
    const session = this.resilientSession();
    try {
      await session.run(Q.UPSERT_AGENT_SESSION_QUERY, {
        ...params,
        ts: Date.now() / 1000,
      });
    } finally {
      await session.close();
    }
  }

  async list_agent_sessions(): Promise<any[]> {
    const session = this.resilientSession();
    try {
      const result = await session.run(Q.LIST_AGENT_SESSIONS_QUERY);
      return result.records.map((r) => ({
        ...r.get("n").properties,
      }));
    } finally {
      await session.close();
    }
  }

  async get_agent_session(session_id: string): Promise<any | null> {
    const neo4jSession = this.resilientSession();
    try {
      const result = await neo4jSession.run(Q.GET_AGENT_SESSION_QUERY, {
        session_id,
      });
      if (!result.records.length) return null;
      return {
        ...result.records[0].get("n").properties,
      };
    } finally {
      await neo4jSession.close();
    }
  }

  async get_session_stats(filters: {
    since?: number | null;
    source?: string | null;
    provider?: string | null;
    model?: string | null;
  }): Promise<any[]> {
    const session = this.resilientSession();
    try {
      const result = await session.run(Q.GET_SESSION_STATS_QUERY, {
        since: filters.since ?? null,
        source: filters.source ?? null,
        provider: filters.provider ?? null,
        model: filters.model ?? null,
      });
      return result.records.map((r) => ({ ...r.get("n").properties }));
    } finally {
      await session.close();
    }
  }

  async close() {
    await this.driver.close();
    console.log("===> driver closed");
  }
}

export let db: Db;

if (!no_db) {
  db = new Db();
}

function deser_edge(record: any): Neo4jEdge {
  const edge_ref_id = record.get("edge_ref_id");
  const edge_id = record.get("edge_id");

  let ref_id: string;
  if (edge_ref_id) {
    ref_id = edge_ref_id;
  } else if (edge_id !== undefined && edge_id !== null) {
    ref_id = edge_id.toString();
  } else {
    ref_id = uuidv4();
  }

  return {
    edge_type: record.get("edge_type"),
    ref_id: ref_id,
    source: record.get("source_ref_id"),
    target: record.get("target_ref_id"),
    properties: record.get("properties") || {},
  };
}

interface MergeQuery {
  query: string;
  parameters: any;
}

// Function to construct node merge query
function construct_merge_node_query(node: Node): MergeQuery {
  const { node_type, node_data } = node;
  const node_key = create_node_key(node);
  const query = `
      MERGE (node:${node_type}:${Data_Bank} {node_key: $node_key})
      ON CREATE SET node += $properties
      ON MATCH SET node += $properties
      SET node.ref_id = coalesce(node.ref_id, $ref_id)
      RETURN node
    `;
  return {
    query,
    parameters: {
      node_key,
      ref_id: uuidv4(),
      properties: { ...node_data, node_key },
    },
  };
}

// Function to construct edge merge query
function construct_merge_edge_query(edge_data: Edge): MergeQuery {
  const { edge, source, target } = edge_data;
  const source_key = create_node_key(source);
  const target_key = create_node_key(target);
  const query = `
      MATCH (source:${source.node_type} {node_key: $source_key})
      MATCH (target:${target.node_type} {node_key: $target_key})
      MERGE (source)-[r:${edge.edge_type}]->(target)
      RETURN r
    `;
  return {
    query,
    parameters: {
      source_key,
      target_key,
    },
  };
}

const BATCH_SIZE = 256;

async function process_file(
  session: Session,
  file_path: string,
  process_fn: (data: any) => any,
) {
  const file_interface = readline.createInterface({
    input: fs.createReadStream(file_path),
    crlfDelay: Infinity,
  });
  let batch = [];
  let count = 0;
  for await (const line of file_interface) {
    try {
      const data = JSON.parse(line);
      // console.log(data);
      const query_data = process_fn(data);
      // console.log(query_data);
      batch.push(query_data);

      if (batch.length >= BATCH_SIZE) {
        await execute_batch(session, batch);
        console.log(`Processed ${(count += batch.length)} items`);
        batch = [];
      }
    } catch (error) {
      console.error(`Error processing line: ${line}`, error);
    }
  }
  // Process remaining items
  if (batch.length > 0) {
    await execute_batch(session, batch);
    console.log(`Processed ${(count += batch.length)} items`);
  }
}

async function execute_batch(session: Session, batch: MergeQuery[]) {
  const tx = session.beginTransaction();
  try {
    for (const { query, parameters } of batch) {
      await tx.run(query, parameters);
    }
    await tx.commit();
  } catch (error) {
    await tx.rollback();
    console.error("Error executing batch:", error);
    throw error;
  }
}

/**
 * Prepares a fulltext search query for Neo4j by properly handling special characters
 */
// Returns true if a raw (unescaped) term is safe to use in Lucene wildcard queries.
// Wildcard queries do not honour escape sequences, so terms containing special
// Lucene characters must never be used in wildcard position.
function isSimpleTerm(raw: string): boolean {
  return /^[\w\-.]+$/.test(raw);
}

export function prepareFulltextSearchQuery(searchTerm: string): string {
  const words = searchTerm.trim().split(/\s+/).filter(Boolean);

  if (words.length === 1) {
    const raw = words[0];
    const word = escapeSearchTerm(raw);
    const simple = isSimpleTerm(raw);
    const fieldMatches = [
      `name:${word}^10`,
      `body:${word}^3`,
      `description:${word}^2`,
      ...(simple
        ? [`file:*${word}*^4`, `name:${word}*^2`, `body:${word}*^1`, `description:${word}*^1`]
        : []),
    ];
    return fieldMatches.join(" OR ");
  }

  // Multi-word: OR individual words for recall, boost exact phrase for precision
  const wordMatches = words.map((w) => {
    const word = escapeSearchTerm(w);
    const simple = isSimpleTerm(w);
    return simple
      ? `(name:${word}^10 OR body:${word}^3 OR description:${word}^2 OR name:${word}*^2 OR body:${word}*^1 OR description:${word}*^1)`
      : `(name:${word}^10 OR body:${word}^3 OR description:${word}^2)`;
  });

  const quotedPhrase = `"${searchTerm.replace(/"/g, '\\"')}"`;
  const exactPhraseMatch = `(name:${quotedPhrase}^5 OR body:${quotedPhrase}^2 OR description:${quotedPhrase}^1)`;

  return `(${wordMatches.join(" OR ")}) OR ${exactPhraseMatch}`;
}

function escapeSearchTerm(term: string): string {
  if (term.includes(" ")) {
    return `"${term.replace(/"/g, '\\"')}"`;
  }
  // Escape backslash first, then remaining Lucene special characters.
  return term
    .replace(/\\/g, "\\\\")
    .replace(/([+\-&|!(){}[\]^"~*?:/])/g, "\\$1");
}
