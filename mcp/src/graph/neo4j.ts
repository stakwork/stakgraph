import neo4j, { Driver, Session } from "neo4j-driver";
import fs from "fs";
import readline from "readline";
import {
  NodeType,
  all_node_types,
  Neo4jNode,
  Neo4jEdge,
  EdgeType,
  Node,
  Edge,
  HintExtraction,
} from "./types.js";
import {
  create_node_key,
  deser_node,
  clean_node,
  getExtensionsForLanguage,
  toReturnNode,
} from "./utils.js";
import * as Q from "./queries.js";
import { vectorizeCodeDocument, vectorizeQuery } from "../vector/index.js";
import { callGenerateObject } from "../aieo/src/stream.js";
import { getApiKeyForProvider, Provider } from "../aieo/src/provider.js";
import { z } from "zod";
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
    language?: string
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
    language?: string
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
    language?: string
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
    language?: string
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
          source: source.properties.node_key,
          target: target.properties.node_key,
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
    trim: string[]
  ) {
    let disclude: NodeType[] = ["File", "Directory", "Repository", "Library"];
    if (node_type === "Directory") {
      // remove file and directory from disclude
      disclude = disclude.filter(
        (type) => type !== "File" && type !== "Directory"
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
    include_functions_and_classes: boolean = false
  ) {
    // include if functions and classes should be included
    let disclude: NodeType[] = all_node_types().filter(
      (type: NodeType) =>
        type !== "File" && type !== "Directory" && type !== "Repository"
    );
    if (include_functions_and_classes) {
      console.log("including functions and classes");
      disclude = disclude.filter(
        (type) => type !== "Function" && type !== "Class"
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

  async embed_all_data_bank_bodies() {
    const session = this.driver.session();
    try {
      const result = await session.run(Q.DATA_BANK_BODIES_QUERY_NO_EMBEDDINGS);
      const data_bank = result.records.map((record) => ({
        node_key: record.get("node_key"),
        body: record.get("body"),
      }));
      for (const node of data_bank) {
        const embeddings = await vectorizeCodeDocument(node.body);
        await session.run(Q.UPDATE_EMBEDDINGS_QUERY, {
          node_key: node.node_key,
          embeddings,
        });
      }
    } finally {
      await session.close();
    }
  }

  async embed_data_bank_bodies(do_files: boolean) {
    let embed_batch_size = 32; // Adjust based on your memory constraints
    let skip = 0;
    let batchIndex = 0;
    let hasMoreNodes = true;
    console.log("Starting embedding process for data bank bodies");
    while (hasMoreNodes) {
      console.log(`Processing batch #${batchIndex + 1}`);
      const startTime = Date.now();
      const session = this.driver.session();
      try {
        // Fetch a batch of nodes
        const result = await session.run(
          Q.DATA_BANK_BODIES_QUERY_NO_EMBEDDINGS,
          {
            skip: skip,
            limit: embed_batch_size,
            do_files,
          }
        );
        const nodes = result.records.map((record) => ({
          node_key: record.get("node_key"),
          body: record.get("body"),
        }));
        // If we got fewer nodes than batch size, we're on the last batch
        hasMoreNodes = nodes.length === embed_batch_size;
        // Process this batch
        if (nodes.length > 0) {
          console.log(`Batch #${batchIndex + 1}: Starting embeddings`);
          const updateSession = this.driver.session();
          try {
            // Create a batch array to hold all node updates
            const updateBatch = [];
            // Process each node and prepare the batch
            for (const node of nodes) {
              console.log("embedding", node.node_key);
              const embeddings = await vectorizeCodeDocument(node.body);
              updateBatch.push({
                node_key: node.node_key,
                embeddings: embeddings,
              });
            }
            // Perform bulk update with a single query
            await updateSession.run(Q.BULK_UPDATE_EMBEDDINGS_QUERY, {
              batch: updateBatch,
            });
            console.log(`Batch #${batchIndex + 1}: Updated embeddings`);
          } catch (error) {
            console.error(
              `Batch #${batchIndex + 1}: Error updating embeddings:`,
              error
            );
          } finally {
            await updateSession.close();
          }
        }
        // Prepare for next batch
        skip += embed_batch_size;
        batchIndex++;
        const duration = (Date.now() - startTime) / 1000;
        console.log(
          `Batch #${batchIndex} completed in ${duration.toFixed(2)} seconds`
        );
      } catch (error) {
        console.error(`Batch #${batchIndex + 1}: Error fetching nodes:`, error);
        hasMoreNodes = false; // Stop on error
      } finally {
        await session.close();
      }
    }
    console.log(
      `Embedding process completed. Processed ${batchIndex} batches (${skip} nodes).`
    );
  }

  async update_all_token_counts() {
    const session = this.driver.session();
    try {
      const tokenizer = await createByModelName("gpt-4");
      const result = await session.run(
        Q.DATA_BANK_BODIES_QUERY_NO_TOKEN_COUNT,
        {
          do_files: true,
        }
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
        construct_merge_node_query(data)
      );
      console.log("Processing edges...", edge_file);
      await process_file(session, edge_file, (data) =>
        construct_merge_edge_query(data)
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
    language?: string
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
    similarityThreshold: number = 0.7,
    language?: string
  ): Promise<Neo4jNode[]> {
    let session: Session | null = null;
    try {
      session = this.driver.session();
      const embeddings = await vectorizeQuery(query);

      const extensions = language ? getExtensionsForLanguage(language) : [];

      const result = await session.run(Q.VECTOR_SEARCH_QUERY, {
        embeddings,
        limit,
        node_types,
        similarityThreshold,
        extensions,
      });
      return result.records.map((record) => {
        const node: Neo4jNode = deser_node(record, "node");
        return {
          properties: node.properties,
          labels: node.labels,
          score: record.get("score"),
        };
      });
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
    persona: string = "PM"
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
    target_ref_id: string
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
    target_ref_id: string
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

  async create_pull_request(name: string, docs: string, embeddings: number[], number: string) {
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

  async createEdgesDirectly(
    hint_ref_id: string,
    weightedRefIds: { ref_id: string; relevancy: number }[]
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
    limit: number = 1000
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
    limit: number = 1000
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
    limit: number = 1000
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
}

export let db: Db;

if (!no_db) {
  db = new Db();
}

function deser_edge(record: any): Neo4jEdge {
  return {
    edge_type: record.get("edge_type"),
    ref_id: uuidv4(),
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
  process_fn: (data: any) => any
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
      "g"
    );
    result = result.replace(regex, `\\${char}`);
  }

  return result;
}
