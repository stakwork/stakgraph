import neo4j, { Driver, Session } from "neo4j-driver";
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

export const Data_Bank = Q.Data_Bank;

const no_db = process.env.NO_DB === "true" || process.env.NO_DB === "1";
if (!no_db) {
  const delay_start = parseInt(process.env.DELAY_START || "0") || 0;
  setTimeout(async () => {
    try {
      await db.createIndexes();
    } catch (error) {
      console.error("Error creating indexes:", error);
    }
  }, delay_start);
}

class Db {
  private driver: Driver;

  constructor() {
    const uri = `neo4j://${process.env.NEO4J_HOST || "localhost:7687"}`;
    const user = process.env.NEO4J_USER || "neo4j";
    const pswd = process.env.NEO4J_PASSWORD || "testtest";
    console.log("===> connecting to", uri, user);
    this.driver = neo4j.driver(uri, neo4j.auth.basic(user, pswd));
  }

  async get_pkg_files(): Promise<Neo4jNode[]> {
    const session = this.driver.session();
    try {
      const r = await session.run(Q.PKGS_QUERY);
      return r.records.map((record) => deser_node(record, "file"));
    } finally {
      await session.close();
    }
  }

  async nodes_by_type(
    label: NodeType,
    language?: string,
  ): Promise<Neo4jNode[]> {
    const session = this.driver.session();
    try {
      const extensions = language ? getExtensionsForLanguage(language) : [];
      const r = await session.run(Q.LIST_QUERY, {
        node_label: label,
        extensions,
      });
      return r.records.map((record) => deser_node(record, "f"));
    } finally {
      await session.close();
    }
  }

  async nodes_by_ref_ids(
    ref_ids: string[],
    language?: string,
  ): Promise<Neo4jNode[]> {
    const session = this.driver.session();
    try {
      const extensions = language ? getExtensionsForLanguage(language) : [];
      const r = await session.run(Q.REF_IDS_LIST_QUERY, {
        ref_ids,
        extensions,
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
    const session = this.driver.session();
    try {
      const extensions = language ? getExtensionsForLanguage(language) : [];
      const r = await session.run(Q.MULTI_TYPE_LATEST_PER_TYPE_QUERY, {
        labels,
        limit_per_type,
        since: since ?? null,
        extensions,
      });
      return r.records.map((record) => deser_node(record, "n"));
    } finally {
      await session.close();
    }
  }

  async nodes_by_types_total(
    labels: NodeType[],
    limit_total: number,
    since?: number,
    language?: string,
  ): Promise<Neo4jNode[]> {
    const session = this.driver.session();
    try {
      const extensions = language ? getExtensionsForLanguage(language) : [];
      const r = await session.run(Q.MULTI_TYPE_LATEST_TOTAL_QUERY, {
        labels,
        labelsSize: labels.length,
        limit_total,
        since: since ?? null,
        extensions,
      });
      return r.records.map((record) => deser_node(record, "n"));
    } finally {
      await session.close();
    }
  }

  async edges_between_node_keys(keys: string[]): Promise<Neo4jEdge[]> {
    if (keys.length === 0) return [];
    const session = this.driver.session();
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
    const session = this.driver.session();
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

  async get_repositories(): Promise<Neo4jNode[]> {
    const session = this.driver.session();
    try {
      const r = await session.run(Q.REPOSITORIES_QUERY);
      return r.records.map((record) => deser_node(record, "r"));
    } finally {
      await session.close();
    }
  }

  async get_file_ends_with(file_end: string): Promise<Neo4jNode> {
    const session = this.driver.session();
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
    const session = this.driver.session();
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
    const session = this.driver.session();
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
    const session = this.driver.session();
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
    const session = this.driver.session();
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
      for (const node of data_bank) {
        const tokens = tokenizer.encode(node.body || "", []);
        const token_count = tokens.length;
        await session.run(Q.UPDATE_TOKEN_COUNT_QUERY, {
          node_key: node.node_key,
          token_count,
        });
      }
    } catch (error) {
      console.error("Error updating token counts:", error);
    } finally {
      await session.close();
    }
  }

  // Main function to process both nodes and edges
  async build_graph_from_files(node_file: string, edge_file: string) {
    const session = this.driver.session();
    try {
      console.log("Processing nodes...", node_file);
      await process_file(session, node_file, (data) =>
        construct_merge_node_query(data),
      );
      console.log("Processing edges...", edge_file);
      await process_file(session, edge_file, (data) =>
        construct_merge_edge_query(data),
      );
      console.log("Added nodes to graph!");
    } catch (error) {
      console.error("Error:", error);
    } finally {
      await session.close();
    }
  }

  async search(
    query: string,
    limit: number,
    node_types: NodeType[],
    skip_node_types: NodeType[],
    maxTokens: number, // Optional parameter for token limit
    language?: string,
  ): Promise<Neo4jNode[]> {
    const session = this.driver.session();

    const q_escaped = prepareFulltextSearchQuery(query);
    // console.log("===> search query escaped:", q_escaped);

    // skip Import nodes
    if (!skip_node_types.includes("Import")) {
      skip_node_types.push("Import");
    }

    const extensions = language ? getExtensionsForLanguage(language) : [];

    try {
      const result = await session.run(Q.SEARCH_QUERY_COMPOSITE, {
        query: q_escaped,
        limit,
        node_types,
        skip_node_types,
        extensions,
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
  ): Promise<Neo4jNode[]> {
    let session: Session | null = null;
    try {
      session = this.driver.session();
      const embeddings = await vectorizeQuery(query);

      if (!skip_node_types.includes("Import")) {
        skip_node_types.push("Import");
      }

      const extensions = language ? getExtensionsForLanguage(language) : [];

      const result = await session.run(Q.VECTOR_SEARCH_QUERY, {
        embeddings,
        limit,
        node_types,
        skip_node_types,
        extensions,
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
    const session = this.driver.session();
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
    const session = this.driver.session();
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
    const session = this.driver.session();
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
    const session = this.driver.session();
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
    const session = this.driver.session();
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
    const session = this.driver.session();
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
    const session = this.driver.session();
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
    const session = this.driver.session();
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
    const session = this.driver.session();
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
    const session = this.driver.session();
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
    const session = this.driver.session();
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

      const result = await session.run(Q.ADD_NODE_QUERY(node_type), {
        node_key,
        properties: { ...node_data, node_key },
        now,
      });

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
    const session = this.driver.session();
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
    const session = this.driver.session();
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
    const session = this.driver.session();
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
    const session = this.driver.session();
    try {
      const result = await session.run(Q.GET_HINT_SIBLINGS_QUERY, { ref_id });
      return result.records.map((record) => clean_node(record.get("s")));
    } finally {
      await session.close();
    }
  }

  async setHintPersona(ref_id: string, persona: string): Promise<void> {
    const session = this.driver.session();
    try {
      await session.run(Q.SET_HINT_PERSONA_QUERY, { ref_id, persona });
    } finally {
      await session.close();
    }
  }

  async create_prompt(question: string, answer: string, embeddings: number[]) {
    const session = this.driver.session();
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
    const session = this.driver.session();
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
    const session = this.driver.session();
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
    const session = this.driver.session();
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
    const session = this.driver.session();
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
    const session = this.driver.session();
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
    const session = this.driver.session();
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
    const session = this.driver.session();
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
    const session = this.driver.session();
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
    const session = this.driver.session();
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
    const session = this.driver.session();
    try {
      const result = await session.run(Q.ORPHANED_HINTS_QUERY);
      return result.records.map((record) => clean_node(record.get("h")));
    } finally {
      await session.close();
    }
  }

  async createIndexes(): Promise<void> {
    let session: Session | null = null;
    try {
      session = this.driver.session();
      // console.log("Creating indexes...");
      // console.log(Q.KEY_INDEX_QUERY);
      // console.log(Q.FULLTEXT_BODY_INDEX_QUERY);
      // console.log(Q.FULLTEXT_NAME_INDEX_QUERY);
      // console.log(Q.FULLTEXT_COMPOSITE_INDEX_QUERY);
      // console.log(Q.VECTOR_INDEX_QUERY);
      await session.run(Q.KEY_INDEX_QUERY);
      await session.run(Q.FULLTEXT_BODY_INDEX_QUERY);
      await session.run(Q.FULLTEXT_NAME_INDEX_QUERY);
      await session.run(Q.FULLTEXT_COMPOSITE_INDEX_QUERY);
      await session.run(Q.VECTOR_INDEX_QUERY);
    } finally {
      if (session) {
        await session.close();
      }
    }
  }
  async get_rules_files(): Promise<Neo4jNode[]> {
    const session = this.driver.session();
    try {
      const result = await session.run(Q.RULES_FILES_QUERY);
      return result.records.map((record) => deser_node(record, "f"));
    } finally {
      await session.close();
    }
  }

  async get_env_vars(): Promise<Neo4jNode[]> {
    const session = this.driver.session();
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
    const session = this.driver.session();
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
    const session = this.driver.session();
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
    const session = this.driver.session();
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
    const session = this.driver.session();
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
    skip_tests: boolean = true,
  ): Promise<Neo4jNode[]> {
    const session = this.driver.session();
    try {
      const result = await session.run(Q.GET_NODES_WITHOUT_DESCRIPTION_QUERY, {
        limit: neo4j.int(limit),
        repo_paths: repo_paths || [],
        file_paths,
        skip_tests,
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
    const session = this.driver.session();
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
    const session = this.driver.session();
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
    const session = this.driver.session();
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
    const session = this.driver.session();
    try {
      await session.run(Q.BULK_UPDATE_EMBEDDINGS_BY_REF_ID_QUERY, { batch });
    } finally {
      await session.close();
    }
  }

  async bulk_update_descriptions(
    batch: { ref_id: string; description: string }[],
  ) {
    const session = this.driver.session();
    try {
      await session.run(Q.BULK_UPDATE_DESCRIPTIONS_ONLY_QUERY, { batch });
    } finally {
      await session.close();
    }
  }

  async bulk_update_descriptions_and_embeddings(
    batch: { ref_id: string; description: string; embeddings: number[] }[],
  ) {
    const session = this.driver.session();
    try {
      await session.run(Q.BULK_UPDATE_DESCRIPTIONS_AND_EMBEDDINGS_QUERY, {
        batch,
      });
    } finally {
      await session.close();
    }
  }

  async count_nodes_with_embeddings(): Promise<number> {
    const session = this.driver.session();
    try {
      const r = await session.run(Q.COUNT_NODES_WITH_EMBEDDINGS_QUERY);
      return r.records[0].get("c").toNumber();
    } finally {
      await session.close();
    }
  }

  async count_eligible_nodes_for_embeddings(): Promise<number> {
    const session = this.driver.session();
    try {
      const r = await session.run(Q.COUNT_ELIGIBLE_NODES_FOR_EMBEDDINGS_QUERY);
      return r.records[0].get("c").toNumber();
    } finally {
      await session.close();
    }
  }

  async count_workflow_nodes(): Promise<number> {
    const session = this.driver.session();
    try {
      const r = await session.run(Q.COUNT_WORKFLOWS_QUERY);
      return r.records[0].get("c").toNumber();
    } finally {
      await session.close();
    }
  }

  async get_all_workflows(): Promise<any[]> {
    const session = this.driver.session();
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
    const session = this.driver.session();
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
    const session = this.driver.session();
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
    const session = this.driver.session();
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
    const session = this.driver.session();
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
    const session = this.driver.session();
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
    const session = this.driver.session();
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
    const session = this.driver.session();
    try {
      await session.run(Q.BULK_UPDATE_IMPORTANCE_QUERY, { batch });
    } finally {
      await session.close();
    }
  }

  async drop_importance_graph(graphName: string): Promise<void> {
    const session = this.driver.session();
    try {
      await session.run(Q.IMPORTANCE_GRAPH_DROP_QUERY, { graphName });
    } finally {
      await session.close();
    }
  }

  async get_top_nodes_by_importance(
    limit: number,
  ): Promise<ImportanceTopNode[]> {
    const session = this.driver.session();
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
    const session = this.driver.session();
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
      RETURN node
    `;
  return {
    query,
    parameters: {
      node_key,
      properties: { ...node_data, node_key, ref_id: uuidv4() },
    },
  };
}

// Function to construct edge merge query
function construct_merge_edge_query(edge_data: Edge): MergeQuery {
  const { edge, source, target } = edge_data;
  const query = `
      MATCH (source:${source.node_type} {name: $source_name, file: $source_file})
      MATCH (target:${target.node_type} {name: $target_name, file: $target_file})
      MERGE (source)-[r:${edge.edge_type}]->(target)
      RETURN r
    `;
  return {
    query,
    parameters: {
      source_name: source.node_data.name,
      source_file: source.node_data.file,
      target_name: target.node_data.name,
      target_file: target.node_data.file,
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
export function prepareFulltextSearchQuery(searchTerm: string): string {
  // console.log("===> prepareFulltextSearchQuery", searchTerm);
  // Escape the raw search term first
  const escapedTerm = escapeSearchTerm(searchTerm);

  // Build the query with proper structure
  const queryParts = [
    `name:${escapedTerm}^10`,
    `file:*${escapedTerm}*^4`,
    `body:${escapedTerm}^3`,
    `name:${escapedTerm}*^2`,
    `body:${escapedTerm}*^1`,
  ];

  return queryParts.join(" OR ");
}

/**
 * Escapes special characters in search terms
 */
function escapeSearchTerm(term: string): string {
  // Handle phrases with spaces by wrapping in quotes
  if (term.includes(" ")) {
    // Escape quotes within the term and wrap the whole thing in quotes
    const escapedTerm = term.replace(/"/g, '\\"');
    return `"${escapedTerm}"`;
  }

  // For single terms, escape special characters
  const charsToEscape = [
    "+",
    "-",
    "&",
    "|",
    "!",
    "(",
    ")",
    "{",
    "}",
    "[",
    "]",
    "^",
    '"',
    "~",
    "?",
    ":",
    "\\",
    "/",
    "*",
  ];

  let result = term;
  for (const char of charsToEscape) {
    const regex = new RegExp(
      char.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"), // Removed the extra \\
      "g",
    );
    result = result.replace(regex, `\\${char}`);
  }
  return result;
}
