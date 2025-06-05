import neo4j, { Driver, Session } from "neo4j-driver";
import fs from "fs";
import readline from "readline";
import { Node, Edge, Neo4jNode, NodeType, all_node_types } from "./types.js";
import { create_node_key } from "./utils.js";
import * as Q from "./queries.js";
import { vectorizeCodeDocument, vectorizeQuery } from "../vector/index.js";
import { v4 as uuidv4 } from "uuid";
import { createByModelName } from "@microsoft/tiktokenizer";

export type Direction = "up" | "down" | "both";

export const Data_Bank = Q.Data_Bank;

const delay_start = parseInt(process.env.DELAY_START || "0") || 0;

setTimeout(async () => {
  try {
    await db.createIndexes();
  } catch (error) {
    console.error("Error creating indexes:", error);
  }
}, delay_start);

class Db {
  private driver: Driver;

  constructor() {
    const uri = `neo4j://${process.env.NEO4J_HOST || "localhost"}`;
    const user = process.env.NEO4J_USER || "neo4j";
    const pswd = process.env.NEO4J_PASSWORD || "testtest";
    console.log("===> connecting to", uri, user, pswd);
    this.driver = neo4j.driver(uri, neo4j.auth.basic(user, pswd));
  }

  async get_pkg_files(): Promise<Neo4jNode[]> {
    const session = this.driver.session();
    try {
      const r = await session.run(Q.PKGS_QUERY);
      return r.records.map((record) => record.get("file"));
    } finally {
      await session.close();
    }
  }

  async nodes_by_type(label: NodeType): Promise<Neo4jNode[]> {
    const session = this.driver.session();
    try {
      const r = await session.run(Q.LIST_QUERY, { node_label: label });
      return r.records.map((record) => record.get("f"));
    } finally {
      await session.close();
    }
  }

  async nodes_by_ref_ids(ref_ids: string[]): Promise<Neo4jNode[]> {
    const session = this.driver.session();
    try {
      const r = await session.run(Q.REF_IDS_LIST_QUERY, { ref_ids });
      return r.records.map((record) => record.get("n"));
    } finally {
      await session.close();
    }
  }

  async files(prefix: string, limit: number): Promise<Neo4jNode[]> {
    const session = this.driver.session();
    try {
      const r = await session.run(Q.FILES_QUERY, { prefix, limit });
      return r.records.map((record) => record.get("path"));
    } finally {
      await session.close();
    }
  }

  async get_function_path(
    page: string | null,
    func: string | null,
    include_tests: boolean,
    depth: number
  ) {
    const session = this.driver.session();
    let page_name = page || null;
    let function_name = func || null;
    try {
      return await session.run(Q.PATH_QUERY, {
        page_name,
        function_name,
        include_tests,
        depth: depth || 7,
      });
    } finally {
      await session.close();
    }
  }

  skip_string(skips: NodeType[]) {
    return skips.map((skip) => `-${skip}`).join("|");
  }

  async get_subtree(
    node_type: string,
    name: string,
    ref_id: string,
    include_tests: boolean,
    depth: number,
    direction: Direction,
    trim: string[]
  ) {
    const disclude: NodeType[] = ["File", "Directory", "Repository"];
    if (include_tests === false) {
      disclude.push("Test", "E2etest");
    }
    const label_filter = this.skip_string(disclude);
    const session = this.driver.session();
    try {
      return await session.run(Q.SUBGRAPH_QUERY, {
        node_label: node_type,
        node_name: name,
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

  async get_repo_subtree(name: string, ref_id: string) {
    const disclude: NodeType[] = all_node_types().filter(
      (type: NodeType) =>
        type !== "File" && type !== "Directory" && type !== "Repository"
    );
    const session = this.driver.session();
    console.log("get_repo_subtree", name, ref_id, this.skip_string(disclude));
    try {
      return await session.run(Q.SUBGRAPH_QUERY, {
        node_label: "Repository",
        node_name: name,
        ref_id: ref_id || "",
        depth: 10,
        direction: "down",
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
    maxTokens: number // Optional parameter for token limit
  ): Promise<Neo4jNode[]> {
    const session = this.driver.session();

    const q = `name:${query}^10 OR file:*${query}^4 OR body:${query}^3 OR name:${query}*^2 OR body:${query}*`;
    console.log("===> search query:", q, `node_types: ${node_types}`);
    const q_escaped = prepareFulltextSearchQuery(q);
    console.log("===> search query escaped:", q_escaped);

    // skip Import nodes
    if (!skip_node_types.includes("Import")) {
      skip_node_types.push("Import");
    }
    try {
      const result = await session.run(Q.SEARCH_QUERY_COMPOSITE, {
        query: q_escaped,
        limit,
        node_types,
        skip_node_types,
      });
      const nodes = result.records.map((record) => {
        const node = record.get("node");
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
    similarityThreshold: number = 0.7
  ): Promise<Neo4jNode[]> {
    let session: Session | null = null;
    try {
      session = this.driver.session();
      const embeddings = await vectorizeQuery(query);
      const result = await session.run(Q.VECTOR_SEARCH_QUERY, {
        embeddings,
        limit,
        node_types,
        similarityThreshold,
      });
      return result.records.map((record) => {
        const node = record.get("node");
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
}

export const db = new Db();

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
 *
 * @param searchQuery - The original search query
 * @returns A properly formatted search query for Neo4j fulltext search
 */
export function prepareFulltextSearchQuery(searchQuery: string): string {
  // If the query is very simple (no special operators), return it as is
  if (!/[:^*]/.test(searchQuery)) {
    return searchQuery;
  }

  // For queries with field specifications and boosting, handle with care
  const parts = searchQuery.split(/ OR /);

  const processedParts = parts.map((part) => {
    // Check if this part has a field specification (contains a colon)
    const colonIndex = part.indexOf(":");
    if (colonIndex === -1) {
      return escapeSearchTerm(part);
    }

    // Extract field name and search value
    const fieldName = part.substring(0, colonIndex);
    let remaining = part.substring(colonIndex + 1);

    // Handle boosting factor if present
    let boostFactor = "";
    const boostMatch = remaining.match(/\^(\d+)$/);
    if (boostMatch) {
      boostFactor = boostMatch[0]; // e.g., "^10"
      remaining = remaining.substring(0, remaining.length - boostFactor.length);
    }

    // Handle wildcards carefully
    if (remaining.includes("*")) {
      // Preserve the wildcards, but escape other special characters
      const segments = remaining.split("*");
      const escapedSegments = segments.map((seg) => escapeSearchTerm(seg));
      remaining = escapedSegments.join("*");
    } else {
      // No wildcards, just escape the term
      remaining = escapeSearchTerm(remaining);
    }

    // Reconstruct the part
    return `${fieldName}:${remaining}${boostFactor}`;
  });

  return processedParts.join(" OR ");
}

/**
 * Escapes only the characters that need escaping in search terms
 * (not field names, operators, or wildcards)
 */
function escapeSearchTerm(term: string): string {
  // Characters that should be escaped in the actual search term
  // Note: * is handled separately to preserve wildcards
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
  ];

  let result = term;
  for (const char of charsToEscape) {
    // Create a regex that globally matches the character
    const regex = new RegExp(`\\${char}`, "g");
    // Replace with escaped version
    result = result.replace(regex, `\\${char}`);
  }

  return result;
}
