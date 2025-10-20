use super::*;
use crate::lang::Node;
use crate::utils::create_node_key;
use crate::utils::create_node_key_from_ref;
use lazy_static::lazy_static;
use lsp::language::Language;
use neo4rs::{query, BoltMap, BoltType, ConfigBuilder, Graph as Neo4jConnection};
use shared::{Error, Result};
use std::str::FromStr;
use tiktoken_rs::{get_bpe_from_model, CoreBPE};
use tracing::{debug, error, info};

lazy_static! {
    static ref TOKENIZER: CoreBPE = get_bpe_from_model("gpt-4").unwrap();
}

const DATA_BANK: &str = "Data_Bank";
const BATCH_SIZE: usize = 4096;

pub struct Neo4jConnectionManager;

impl Neo4jConnectionManager {
    pub async fn initialize(
        uri: &str,
        username: &str,
        password: &str,
        database: &str,
    ) -> Result<Neo4jConnection> {
        // info!("Connecting to Neo4j at {}", uri);
        let config = ConfigBuilder::new()
            .uri(uri)
            .user(username)
            .password(password)
            .db(database)
            .build()?;

        match Neo4jConnection::connect(config).await {
            Ok(connection) => {
                // info!("Successfully connected to Neo4j");
                // *conn_guard = Some(Arc::new(connection));
                Ok(connection)
            }
            Err(_) => Err(Error::Custom("Failed to connect to Neo4j : {e}".into())),
        }
    }
}

pub struct NodeQueryBuilder {
    node_type: NodeType,
    node_data: NodeData,
}

impl NodeQueryBuilder {
    pub fn new(node_type: &NodeType, node_data: &NodeData) -> Self {
        Self {
            node_type: node_type.clone(),
            node_data: node_data.clone(),
        }
    }

    pub fn build(&self) -> (String, BoltMap) {
        let mut properties: BoltMap = (&self.node_data).into();
        let ref_id = if std::env::var("TEST_REF_ID").is_ok() {
            "test_ref_id".to_string()
        } else {
            uuid::Uuid::new_v4().to_string()
        };

        boltmap_insert_str(&mut properties, "ref_id", &ref_id);

        let node_key = create_node_key(&Node::new(self.node_type.clone(), self.node_data.clone()));
        boltmap_insert_str(&mut properties, "node_key", &node_key);

        let token_count = calculate_token_count(&self.node_data.body).unwrap_or(0);
        boltmap_insert_int(&mut properties, "token_count", token_count);

        // println!("[NodeQueryBuilder] node_key: {}", node_key);

        let query = format!(
            "MERGE (node:{}:{} {{node_key: $node_key}})
         ON CREATE SET node += $properties, node.date_added_to_graph = $now
         ON MATCH SET node += $properties, node.date_added_to_graph = $now
         Return node",
            self.node_type.to_string(),
            DATA_BANK,
        );

        (query, properties)
    }
}
pub struct EdgeQueryBuilder {
    edge: Edge,
}

impl EdgeQueryBuilder {
    pub fn new(edge: &Edge) -> Self {
        Self { edge: edge.clone() }
    }

    pub fn build(&self) -> (String, BoltMap) {
        let mut params = BoltMap::new();

        let rel_type = self.edge.edge.to_string();

        let source_type = self.edge.source.node_type.to_string();
        let source_key = create_node_key_from_ref(&self.edge.source);
        boltmap_insert_str(&mut params, "source_key", &source_key);

        let target_type = self.edge.target.node_type.to_string();
        let target_key = create_node_key_from_ref(&self.edge.target);
        boltmap_insert_str(&mut params, "target_key", &target_key);

        // println!(
        //     "[EdgeQueryBuilder] source_key: {}, target_key: {}",
        //     source_key, target_key
        // );

        let query = format!(
            "MATCH (source:{} {{node_key: $source_key}}),
                 (target:{} {{node_key: $target_key}})
            MERGE (source)-[r:{}]->(target)
            RETURN r",
            source_type, target_type, rel_type
        );
        (query, params)
    }
}

pub fn build_batch_edge_queries<I>(edges: I, batch_size: usize) -> Vec<(String, BoltMap)>
where
    I: Iterator<Item = (String, String, EdgeType)>,
{
    use itertools::Itertools;
    use std::collections::HashMap;

    // Group edges by type
    let edges_by_type: HashMap<EdgeType, Vec<(String, String)>> = edges
        .map(|(source, target, edge_type)| (edge_type, (source, target)))
        .into_group_map();

    // Create batched queries for each edge type
    edges_by_type
        .into_iter()
        .flat_map(|(edge_type, type_edges)| {
            // Batch the edges for this type
            let chunks: Vec<Vec<(String, String)>> = type_edges
                .into_iter()
                .chunks(batch_size)
                .into_iter()
                .map(|chunk| chunk.collect())
                .collect();

            chunks
                .into_iter()
                .map(|chunk| {
                    let edges_data: Vec<BoltMap> = chunk
                        .into_iter()
                        .map(|(source, target)| {
                            let mut edge_map = BoltMap::new();
                            boltmap_insert_str(&mut edge_map, "source", &source);
                            boltmap_insert_str(&mut edge_map, "target", &target);
                            edge_map
                        })
                        .collect();

                    let mut params = BoltMap::new();
                    boltmap_insert_list_of_maps(&mut params, "edges", edges_data);

                    // could use MERGE instead of CREATE here...
                    let query = format!(
                        "UNWIND $edges AS edge
                         MATCH (source:Data_Bank {{node_key: edge.source}}), (target:Data_Bank {{node_key: edge.target}})
                         CREATE (source)-[r:{}]->(target)
                         RETURN count(r)",
                        edge_type.to_string()
                    );

                    (query, params)
                })
                .collect::<Vec<_>>()
        })
        .collect()
}

pub struct TransactionManager<'a> {
    conn: &'a Neo4jConnection,
    queries: Vec<(String, BoltMap)>,
}

impl<'a> TransactionManager<'a> {
    pub fn new(conn: &'a Neo4jConnection) -> Self {
        Self {
            conn,
            queries: Vec::new(),
        }
    }

    pub fn add_query(&mut self, query: (String, BoltMap)) -> &mut Self {
        self.queries.push(query);
        self
    }

    pub fn add_node(&mut self, node_type: &NodeType, node_data: &NodeData) -> &mut Self {
        self.queries.push(add_node_query(node_type, node_data));
        self
    }

    pub fn add_edge(&mut self, edge: &Edge) -> &mut Self {
        self.queries.push(add_edge_query(edge));
        self
    }

    pub async fn execute(self) -> Result<()> {
        let mut txn = self.conn.start_txn().await?;
        for (query_str, bolt_map) in self.queries {
            let mut query_obj = query(&query_str);
            if query_str.contains("$properties") {
                if let Some(BoltType::String(node_key)) = bolt_map.value.get("node_key") {
                    query_obj = query_obj.param("node_key", node_key.value.as_str());
                }
                let properties = boltmap_to_bolttype_map(bolt_map);
                query_obj = query_obj.param("properties", properties);
                if query_str.contains("$now") {
                    use std::time::{SystemTime, UNIX_EPOCH};
                    if let Ok(dur) = SystemTime::now().duration_since(UNIX_EPOCH) {
                        let ts = dur.as_secs_f64();
                        query_obj = query_obj
                            .param("now", neo4rs::BoltType::String(format!("{:.7}", ts).into()));
                    }
                }
            } else {
                for (key, value) in bolt_map.value.iter() {
                    query_obj = query_obj.param(key.value.as_str(), value.clone());
                }
            }
            txn.run(query_obj).await?;
        }
        txn.commit().await?;
        Ok(())
    }
}

pub async fn execute_batch(conn: &Neo4jConnection, queries: Vec<(String, BoltMap)>) -> Result<()> {
    use itertools::Itertools;

    let total_chunks = (queries.len() as f64 / BATCH_SIZE as f64).ceil() as usize;

    let chunked_queries: Vec<Vec<_>> = queries
        .into_iter()
        .chunks(BATCH_SIZE)
        .into_iter()
        .map(|chunk| chunk.collect())
        .collect();

    for (i, chunk) in chunked_queries.into_iter().enumerate() {
        info!("Processing chunk {}/{}", i + 1, total_chunks);
        let mut txn = conn.start_txn().await?;

        for (query_str, params) in chunk {
            let mut query_obj = query(&query_str);

            if query_str.contains("$properties") {
                if let Some(BoltType::String(node_key)) = params.value.get("node_key") {
                    query_obj = query_obj.param("node_key", node_key.value.as_str());
                }
                let properties = boltmap_to_bolttype_map(params);
                query_obj = query_obj.param("properties", properties);
                if query_str.contains("$now") {
                    use std::time::{SystemTime, UNIX_EPOCH};
                    if let Ok(dur) = SystemTime::now().duration_since(UNIX_EPOCH) {
                        let ts = dur.as_secs_f64();
                        query_obj = query_obj
                            .param("now", neo4rs::BoltType::String(format!("{:.7}", ts).into()));
                    }
                }
            } else {
                for (key, value) in params.value.iter() {
                    query_obj = query_obj.param(key.value.as_str(), value.clone());
                }
            }

            if let Err(e) = txn.run(query_obj).await {
                error!("Error running query in batch chunk {}: {:?}", i + 1, e);
                txn.rollback().await?; // Attempt to rollback
                return Err(e.into());
            }
        }

        if let Err(e) = txn.commit().await {
            error!("Error committing batch chunk {}: {:?}", i + 1, e);
            return Err(e.into());
        }
        debug!("Successfully committed chunk {}/{}", i + 1, total_chunks);
    }
    Ok(())
}

pub async fn execute_queries_simple(
    conn: &Neo4jConnection,
    queries: Vec<(String, BoltMap)>,
) -> Result<()> {
    let total_queries = queries.len();
    for (i, (query_str, params)) in queries.into_iter().enumerate() {
        info!("Processing query {}/{}", i + 1, total_queries);

        let mut query_obj = query(&query_str);

        // Add parameters
        for (key, value) in params.value.iter() {
            query_obj = query_obj.param(key.value.as_str(), value.clone());
        }

        // Execute in a transaction
        let mut txn = conn.start_txn().await?;
        txn.run(query_obj).await?;
        txn.commit().await?;

        debug!("Successfully executed query {}/{}", i + 1, total_queries);
    }
    Ok(())
}

pub fn add_node_query(node_type: &NodeType, node_data: &NodeData) -> (String, BoltMap) {
    NodeQueryBuilder::new(node_type, node_data).build()
}

pub fn add_edge_query(edge: &Edge) -> (String, BoltMap) {
    EdgeQueryBuilder::new(edge).build()
}

pub async fn execute_node_query(
    conn: &Neo4jConnection,
    query_str: String,
    params: BoltMap,
) -> Vec<NodeData> {
    let mut query_obj = query(&query_str);
    for (key, value) in params.value.iter() {
        query_obj = query_obj.param(key.value.as_str(), value.clone());
    }
    match conn.execute(query_obj).await {
        Ok(mut result) => {
            let mut nodes = Vec::new();
            while let Ok(Some(row)) = result.next().await {
                if let Ok(node) = row.get::<neo4rs::Node>("n") {
                    if let Ok(node_data) = NodeData::try_from(&node) {
                        nodes.push(node_data);
                    }
                }
            }
            nodes
        }
        Err(e) => {
            debug!("Error executing query: {}", e);
            Vec::new()
        }
    }
}

pub async fn execute_nodes_with_coverage_query(
    conn: &Neo4jConnection,
    query_str: String,
    params: BoltMap,
) -> Vec<(NodeData, usize, bool, usize, String)> {
    let mut query_obj = query(&query_str);
    for (key, value) in params.value.iter() {
        query_obj = query_obj.param(key.value.as_str(), value.clone());
    }

    let mut results = Vec::new();
    match conn.execute(query_obj).await {
        Ok(mut result) => {
            while let Ok(Some(row)) = result.next().await {
                if let Ok(node) = row.get::<neo4rs::Node>("n") {
                    if let Ok(node_data) = NodeData::try_from(&node) {
                        let usage_count: i64 = row.get("usage_count").unwrap_or(0);
                        let is_covered: bool = row.get("is_covered").unwrap_or(false);
                        let test_count: i64 = row.get("test_count").unwrap_or(0);
                        let ref_id = extract_ref_id(&node_data);
                        results.push((
                            node_data,
                            usage_count as usize,
                            is_covered,
                            test_count as usize,
                            ref_id,
                        ));
                    }
                }
            }
        }
        Err(e) => {
            debug!("Error executing nodes with coverage query: {}", e);
        }
    }
    results
}

pub fn extract_ref_id(node_data: &NodeData) -> String {
    node_data
        .meta
        .get("ref_id")
        .cloned()
        .unwrap_or_else(|| "placeholder".to_string())
}

pub fn count_nodes_edges_query() -> String {
    "MATCH (n) 
     WITH COUNT(n) as nodes
     MATCH ()-[r]->() 
     RETURN nodes, COUNT(r) as edges"
        .to_string()
}
pub fn graph_node_analysis_query() -> String {
    "MATCH (n) 
     RETURN n.node_key as node_key
     ORDER BY node_key"
        .to_string()
}
pub fn graph_edges_analysis_query() -> String {
    "MATCH (source)-[r]->(target) 
     RETURN source.node_key as source_key, type(r) as edge_type, target.node_key as target_key
     ORDER BY source_key, edge_type, target_key"
        .to_string()
}
pub fn count_edges_by_type_query(edge_type: &EdgeType) -> (String, BoltMap) {
    let mut params = BoltMap::new();
    boltmap_insert_str(&mut params, "edge_type", &edge_type.to_string());

    let query = "MATCH ()-[r]->() 
                WHERE type(r) = $edge_type 
                RETURN COUNT(r) as count";

    (query.to_string(), params)
}

pub fn find_nodes_by_type_query(node_type: &NodeType) -> (String, BoltMap) {
    let mut params = BoltMap::new();
    boltmap_insert_str(&mut params, "node_type", &node_type.to_string());

    let query = format!(
        "MATCH (n:{}) 
         RETURN n",
        node_type.to_string()
    );

    (query, params)
}
pub fn find_nodes_by_name_query(node_type: &NodeType, name: &str, root: &str) -> (String, BoltMap) {
    let mut param = BoltMap::new();
    param
        .value
        .insert("name".into(), BoltType::String(name.into()));
    param
        .value
        .insert("root".into(), BoltType::String(root.into()));

    let query = format!(
        "MATCH (n:{}) 
                       WHERE n.name = $name
                       AND n.file STARTS WITH $root
                       RETURN n",
        node_type.to_string()
    );

    (query, param)
}

pub fn find_node_by_name_file_query(
    node_type: &NodeType,
    name: &str,
    file: &str,
) -> (String, BoltMap) {
    let mut params = BoltMap::new();
    boltmap_insert_str(&mut params, "name", name);
    boltmap_insert_str(&mut params, "file", file);

    let query = format!(
        "MATCH (n:{}) 
                       WHERE n.name = $name AND n.file = $file 
                       RETURN n",
        node_type.to_string()
    );

    (query, params)
}

pub fn find_nodes_by_file_pattern_query(
    node_type: &NodeType,
    file_pattern: &str,
) -> (String, BoltMap) {
    let mut params = BoltMap::new();
    boltmap_insert_str(&mut params, "file_pattern", file_pattern);
    let query = format!(
        "MATCH (n:{}) 
                       WHERE n.file CONTAINS $file_pattern 
                       RETURN n",
        node_type.to_string()
    );

    (query, params)
}

pub fn find_nodes_by_name_contains_query(
    node_type: &NodeType,
    name_part: &str,
) -> (String, BoltMap) {
    let mut params = BoltMap::new();
    boltmap_insert_str(&mut params, "name_part", name_part);

    let query = format!(
        "MATCH (n:{}) 
         WHERE n.name CONTAINS $name_part 
         RETURN n",
        node_type.to_string()
    );

    (query, params)
}
pub fn find_nodes_in_range_query(node_type: &NodeType, file: &str, row: u32) -> (String, BoltMap) {
    let mut params = BoltMap::new();
    boltmap_insert_str(&mut params, "node_type", &node_type.to_string());
    boltmap_insert_str(&mut params, "file", file);
    boltmap_insert_int(&mut params, "row", row as i64);

    let query = format!(
        "MATCH (n:$node_type)
         WHERE n.file = $file AND 
               toInteger(n.start) <= toInteger($row) AND 
               toInteger(n.end) >= toInteger($row)
         RETURN n"
    );

    (query, params)
}
pub fn find_source_edge_by_name_and_file_query(
    edge_type: &EdgeType,
    target_name: &str,
    target_file: &str,
) -> (String, BoltMap) {
    let mut params = BoltMap::new();

    boltmap_insert_str(&mut params, "edge_type", &edge_type.to_string());
    boltmap_insert_str(&mut params, "target_name", target_name);
    boltmap_insert_str(&mut params, "target_file", target_file);
    let query = format!(
        "MATCH (source)-[r:{}]->(target {{name: $target_name, file: $target_file}})
         RETURN source.name as name, source.file as file, source.start as start, source.verb as verb
         LIMIT 1",
        edge_type.to_string()
    );
    (query, params)
}

pub fn find_nodes_with_edge_type_query(
    source_type: &NodeType,
    target_type: &NodeType,
    edge_type: &EdgeType,
) -> (String, BoltMap) {
    let mut params = BoltMap::new();
    boltmap_insert_str(&mut params, "source_type", &source_type.to_string());
    boltmap_insert_str(&mut params, "target_type", &target_type.to_string());
    boltmap_insert_str(&mut params, "edge_type", &edge_type.to_string());
    let query = format!(
        "MATCH (source:{})-[r:{}]->(target:{})
         RETURN source.name as source_name, source.file as source_file, source.start as source_start, \
                target.name as target_name, target.file as target_file, target.start as target_start",
        source_type.to_string(),
        edge_type.to_string(),
        target_type.to_string()
    );

    (query, params)
}

pub fn find_resource_nodes_query(
    node_type: &NodeType,
    verb: &str,
    path: &str,
) -> (String, BoltMap) {
    let mut params = BoltMap::new();
    boltmap_insert_str(&mut params, "verb", verb);
    boltmap_insert_str(&mut params, "path", path);
    let query = format!(
        "MATCH (n:{})
         WHERE n.name CONTAINS $path AND 
               (n.verb IS NULL OR toUpper(n.verb) CONTAINS $verb)
         RETURN n",
        node_type.to_string()
    );

    (query, params)
}

pub fn find_handlers_for_endpoint_query(endpoint: &NodeData) -> (String, BoltMap) {
    let mut params = BoltMap::new();
    boltmap_insert_str(&mut params, "endpoint_name", &endpoint.name);
    boltmap_insert_str(&mut params, "endpoint_file", &endpoint.file);
    boltmap_insert_int(&mut params, "endpoint_start", endpoint.start as i64);

    let query = format!(
        "MATCH (endpoint:Endpoint {{name: $endpoint_name, file: $endpoint_file, start: $endpoint_start}})
        -[:HANDLER]->(handler)
        RETURN handler");

    (query, params)
}

pub fn check_direct_data_model_usage_query(
    function_name: &str,
    data_model: &str,
) -> (String, BoltMap) {
    let mut params = BoltMap::new();

    boltmap_insert_str(&mut params, "function_name", function_name);
    boltmap_insert_str(&mut params, "data_model", data_model);

    let query = format!(
        "MATCH (f:Function {{name: $function_name}})-[:CONTAINS]->(n:Datamodel)
         WHERE n.name CONTAINS $data_model
         RETURN COUNT(n) > 0 as exists"
    );

    (query, params)
}

pub fn find_functions_called_by_query(function: &NodeData) -> (String, BoltMap) {
    let mut params = BoltMap::new();
    boltmap_insert_str(&mut params, "function_name", &function.name);
    boltmap_insert_str(&mut params, "function_file", &function.file);
    boltmap_insert_int(&mut params, "function_start", function.start as i64);

    let query = format!(
        "MATCH (source:Function {{name: $function_name, file: $function_file, start: $function_start}})
        -[:CALLS]->(target:Function)
        RETURN target");

    (query, params)
}

pub fn find_node_at_query(node_type: &NodeType, file: &str, line: u32) -> (String, BoltMap) {
    let mut params = BoltMap::new();
    boltmap_insert_str(&mut params, "node_type", &node_type.to_string());
    boltmap_insert_str(&mut params, "file", file);
    boltmap_insert_int(&mut params, "line", line as i64);

    let query = format!(
        "MATCH (n:{}) 
         WHERE n.file = $file AND 
               toInteger(n.start) <= toInteger($line) AND 
               toInteger(n.end) >= toInteger($line)
         RETURN n",
        node_type.to_string()
    );

    (query, params)
}

pub fn all_nodes_and_edges_query() -> (String, String) {
    let node_query = "MATCH (n) WHERE n.node_key IS NOT NULL RETURN DISTINCT n.node_key as key";
    let edge_query = "MATCH ()-[r]->() RETURN DISTINCT type(r) as edge_type";

    (node_query.to_string(), edge_query.to_string())
}

pub fn all_node_keys_query() -> String {
    "MATCH (n) WHERE n.node_key IS NOT NULL RETURN n.node_key as node_key".to_string()
}

pub fn all_edge_triples_query() -> String {
    "MATCH (s)-[e]->(t) RETURN s.node_key as s_key, type(e) as edge_type, t.node_key as t_key"
        .to_string()
}

pub fn filter_out_nodes_without_children_query(
    parent_type: NodeType,
    child_type: NodeType,
    _child_meta_key: &str,
) -> (String, BoltMap) {
    let mut params = BoltMap::new();

    boltmap_insert_str(&mut params, "parent_type", &parent_type.to_string());
    boltmap_insert_str(&mut params, "child_type", &child_type.to_string());

    let query = format!(
        "MATCH (parent:{})
        WHERE NOT EXISTS {{
            MATCH (parent)<-[:OPERAND]-(child:{})
        }}
        AND NOT EXISTS {{
            MATCH (instance:Instance)-[:OF]->(parent)
        }}
        DETACH DELETE parent",
        parent_type.to_string(),
        child_type.to_string()
    );

    (query, params)
}

pub fn class_inherits_query() -> String {
    "MATCH (c:Class)
    WHERE c.parent IS NOT NULL
    MATCH (parent:Class {name: c.parent})
    MERGE (parent)-[:PARENT_OF]->(c)"
        .to_string()
}
pub fn class_includes_query() -> String {
    "MATCH (c:Class)
    WHERE c.includes IS NOT NULL
    WITH c, split(c.includes, ',') AS modules
    UNWIND modules AS module
    MATCH (m:Class {name: trim(module)})
    MERGE (c)-[:IMPORTS]->(m)"
        .to_string()
}

pub fn add_node_with_parent_query(
    node_type: &NodeType,
    node_data: &NodeData,
    parent_type: &NodeType,
    parent_file: &str,
) -> Vec<(String, BoltMap)> {
    let mut queries = Vec::new();

    queries.push(add_node_query(node_type, node_data));

    let mut params = BoltMap::new();
    boltmap_insert_str(&mut params, "node_name", &node_data.name);
    boltmap_insert_str(&mut params, "node_file", &node_data.file);
    boltmap_insert_int(&mut params, "node_start", node_data.start as i64);
    boltmap_insert_str(&mut params, "parent_file", parent_file);

    let query_str = format!(
        "MATCH (parent:{} {{file: $parent_file}})
         MATCH (node:{} {{name: $node_name, file: $node_file, start: $node_start}})
         MERGE (parent)-[:CONTAINS]->(node)",
        parent_type.to_string(),
        node_type.to_string()
    );

    queries.push((query_str, params));
    queries
}

pub fn add_functions_query(
    function_node: &NodeData,
    method_of: Option<&Operand>,
    reqs: &[NodeData],
    dms: &[Edge],
    trait_operand: Option<&Edge>,
    return_types: &[Edge],
) -> Vec<(String, BoltMap)> {
    let mut queries = Vec::new();

    queries.push(add_node_query(&NodeType::Function, function_node));

    let mut params = BoltMap::new();
    boltmap_insert_str(&mut params, "function_name", &function_node.name);
    boltmap_insert_str(&mut params, "function_file", &function_node.file);
    boltmap_insert_int(&mut params, "function_start", function_node.start as i64);

    let query_str = format!(
        "MATCH (function:Function {{name: $function_name, file: $function_file, start: $function_start}}),
               (file:File {{file: $function_file}})
         MERGE (file)-[:CONTAINS]->(function)"
    );
    queries.push((query_str, params));

    if let Some(operand) = method_of {
        let edge = (*operand).clone().into();
        queries.push(add_edge_query(&edge));
    }

    if let Some(edge) = trait_operand {
        queries.push(add_edge_query(edge));
    }

    for edge in return_types {
        queries.push(add_edge_query(edge));
    }

    for req in reqs {
        queries.push(add_node_query(&NodeType::Request, req));

        let mut params = BoltMap::new();
        boltmap_insert_str(&mut params, "function_name", &function_node.name);
        boltmap_insert_str(&mut params, "function_file", &function_node.file);
        boltmap_insert_int(&mut params, "function_start", function_node.start as i64);
        boltmap_insert_str(&mut params, "req_name", &req.name);
        boltmap_insert_str(&mut params, "req_file", &req.file);
        boltmap_insert_int(&mut params, "req_start", req.start as i64);
        let query_str = format!(
            "MATCH (function:Function {{name: $function_name, file: $function_file, start: $function_start}}),
                   (request:Request {{name: $req_name, file: $req_file, start: $req_start}})
             MERGE (function)-[:CALLS]->(request)"
        );
        queries.push((query_str, params));
    }

    for dm_edge in dms {
        queries.push(add_edge_query(dm_edge));
    }
    queries
}

pub fn add_page_query(page_data: &NodeData, edge_opt: &Option<Edge>) -> Vec<(String, BoltMap)> {
    let mut queries = Vec::new();

    queries.push(add_node_query(&NodeType::Page, page_data));

    if let Some(edge) = edge_opt {
        queries.push(add_edge_query(edge));
    }

    queries
}

pub fn add_pages_query(pages: &[(NodeData, Vec<Edge>)]) -> Vec<(String, BoltMap)> {
    let mut queries = Vec::new();

    for (page_data, edges) in pages {
        queries.push(add_node_query(&NodeType::Page, page_data));

        for edge in edges {
            queries.push(add_edge_query(edge));
        }
    }

    queries
}

pub fn add_endpoints_query(endpoints: &[(NodeData, Option<Edge>)]) -> Vec<(String, BoltMap)> {
    let mut queries = Vec::new();

    for (endpoint_data, handler_edge) in endpoints {
        queries.push(add_node_query(&NodeType::Endpoint, endpoint_data));

        if let Some(edge) = handler_edge {
            queries.push(add_edge_query(edge));
        }
    }

    queries
}

pub fn add_calls_query(
    funcs: &[(Calls, Option<NodeData>, Option<NodeData>)],
    tests: &[(Calls, Option<NodeData>, Option<NodeData>)],
    int_tests: &[Edge],
    extras: &[Edge],
) -> Vec<(String, BoltMap)> {
    let mut queries = Vec::new();

    for (calls, ext_func, class_call) in funcs {
        if let Some(class_call) = class_call {
            let edge = Edge::new(
                EdgeType::Calls,
                NodeRef::from(calls.source.clone(), NodeType::Function),
                NodeRef::from(class_call.into(), NodeType::Class),
            );
            queries.push(add_edge_query(&edge));
        }

        if calls.target.is_empty() {
            continue;
        }
        if let Some(ext_nd) = ext_func {
            queries.push(add_node_query(&NodeType::Function, ext_nd));
            let edge = Edge::uses(calls.source.clone(), ext_nd);
            queries.push(add_edge_query(&edge));
        } else {
            let edge: Edge = calls.clone().into();
            queries.push(add_edge_query(&edge));
        }
    }

    for (test_call, ext_func, class_call) in tests {
        if let Some(ext_nd) = ext_func {
            queries.push(add_node_query(&NodeType::Function, ext_nd));
            let edge = Edge::uses(test_call.source.clone(), ext_nd);
            queries.push(add_edge_query(&edge));
        } else {
            let edge = Edge::from_test_call(test_call);
            queries.push(add_edge_query(&edge));
        }

        if let Some(class_nd) = class_call {
            let edge = Edge::from_test_class_call(test_call, class_nd);
            queries.push(add_edge_query(&edge));

            queries.push(add_node_query(&NodeType::Class, class_nd));
        }
    }

    for edge in int_tests {
        queries.push(add_edge_query(edge));
    }
    for edge in extras {
        queries.push(add_edge_query(edge));
    }

    queries
}
pub fn find_endpoint_query(name: &str, file: &str, verb: &str) -> (String, BoltMap) {
    let mut params = BoltMap::new();
    boltmap_insert_str(&mut params, "name", name);
    boltmap_insert_str(&mut params, "verb", verb.to_uppercase().as_str());
    boltmap_insert_str(&mut params, "file", file);
    boltmap_insert_str(&mut params, "node_type", &NodeType::Endpoint.to_string());

    let query = "MATCH (n:Endpoint {name: $name, file: $file})
         WHERE n.verb IS NULL OR toUpper(n.verb) CONTAINS $verb
         RETURN n";

    (query.to_string(), params)
}
pub fn get_repository_hash_query(repo_url: &str) -> (String, BoltMap) {
    let mut params = BoltMap::new();

    let repo_name = if repo_url.contains('/') {
        let parts: Vec<&str> = repo_url.split('/').collect();
        let name = parts.last().unwrap_or(&repo_url);
        name.trim_end_matches(".git")
    } else {
        repo_url
    };

    boltmap_insert_str(&mut params, "repo_name", repo_name);
    let query = "MATCH (r:Repository) 
                 WHERE r.name CONTAINS $repo_name 
                 RETURN r.hash as hash";

    (query.to_string(), params)
}

pub fn remove_nodes_by_file_query(file_path: &str, root: &str) -> (String, BoltMap) {
    let mut params = BoltMap::new();
    // let file_name = file_path.split('/').last().unwrap_or(file_path);
    boltmap_insert_str(&mut params, "file_name", file_path);
    boltmap_insert_str(&mut params, "root", root);

    let query = "
        MATCH (n)
        WHERE (n.file = $file_name OR n.file ENDS WITH $file_name)
        AND n.file STARTS WITH $root
        WITH DISTINCT n
        OPTIONAL MATCH (n)-[r]-() 
        DELETE r
        WITH n
        DETACH DELETE n
        RETURN count(n) as deleted
    ";

    (query.to_string(), params)
}

pub fn update_repository_hash_query(repo_name: &str, new_hash: &str) -> (String, BoltMap) {
    let mut params = BoltMap::new();
    boltmap_insert_str(&mut params, "repo_name", repo_name);
    boltmap_insert_str(&mut params, "new_hash", new_hash);

    let query = "MATCH (r:Repository) 
                 WHERE r.name CONTAINS $repo_name 
                 SET r.hash = $new_hash";

    (query.to_string(), params)
}

pub fn boltmap_insert_str(map: &mut BoltMap, key: &str, value: &str) {
    map.value.insert(key.into(), BoltType::String(value.into()));
}
pub fn boltmap_insert_map(map: &mut BoltMap, key: &str, value: BoltMap) {
    map.value.insert(key.into(), BoltType::Map(value));
}
pub fn boltmap_insert_list_of_maps(map: &mut BoltMap, key: &str, value: Vec<BoltMap>) {
    let list = neo4rs::BoltList {
        value: value.into_iter().map(|m| BoltType::Map(m)).collect(),
    };
    map.value.insert(key.into(), BoltType::List(list));
}

pub fn boltmap_insert_list(map: &mut BoltMap, key: &str, value: Vec<BoltType>) {
    let list = neo4rs::BoltList { value };
    map.value.insert(key.into(), BoltType::List(list));
}
pub fn boltmap_insert_float(map: &mut BoltMap, key: &str, value: f64) {
    map.value.insert(
        key.into(),
        BoltType::Float(neo4rs::BoltFloat {
            value: value as f64,
        }),
    );
}
pub fn boltmap_insert_int(map: &mut BoltMap, key: &str, value: i64) {
    map.value
        .insert(key.into(), BoltType::Integer(value.into()));
}
fn boltmap_to_bolttype_map(bolt_map: BoltMap) -> BoltType {
    BoltType::Map(bolt_map)
}
pub fn calculate_token_count(body: &str) -> Result<i64> {
    let bpe = &TOKENIZER;
    let token_count = bpe.encode_with_special_tokens(body).len() as i64;
    Ok(token_count)
}
// Add these functions to neo4j_utils.rs

fn unique_functions_filters() -> Vec<String> {
    vec![
        "NOT (n)-[:NESTED_IN]->(:Function)".to_string(),
        "(n.body IS NOT NULL AND n.body <> '')".to_string(),
        "NOT (:Endpoint)-[:HANDLER]->(n)".to_string(),
        "(n.component IS NULL OR n.component <> 'true')".to_string(),
        "n.operand IS NULL".to_string(),
        "EXISTS(()-[:CALLS]->(n))".to_string(),
    ]
}
pub fn find_top_level_functions_query() -> (String, BoltMap) {
    let query = format!(
        "MATCH (n:Function)
        WHERE {}
        RETURN n
    ",
        unique_functions_filters().join(" AND ")
    )
    .to_string();
    (query, BoltMap::new())
}

pub fn find_group_function_query(group_function_name: &str) -> (String, BoltMap) {
    let mut params = BoltMap::new();
    boltmap_insert_str(&mut params, "group_function_name", group_function_name);

    let query = "MATCH (n:Function) 
                 WHERE n.name = $group_function_name 
                 RETURN n";

    (query.to_string(), params)
}

pub fn update_endpoint_name_query(old_name: &str, file: &str, new_name: &str) -> (String, BoltMap) {
    let mut params = BoltMap::new();
    boltmap_insert_str(&mut params, "old_name", old_name);
    boltmap_insert_str(&mut params, "file", file);
    boltmap_insert_str(&mut params, "new_name", new_name);

    let query = "MATCH (n:Endpoint {name: $old_name, file: $file})
                 SET n.name = $new_name
                 RETURN n";

    (query.to_string(), params)
}

pub fn update_endpoint_relationships_query(
    old_name: &str,
    file: &str,
    new_name: &str,
) -> (String, BoltMap) {
    let mut params = BoltMap::new();
    boltmap_insert_str(&mut params, "old_name", old_name);
    boltmap_insert_str(&mut params, "file", file);
    boltmap_insert_str(&mut params, "new_name", new_name);

    let query = "MATCH (source:Endpoint {{name: $old_name, file: $file}})-[r]->(target)
                SET source.name = $new_name
                RETURN r";

    (query.to_string(), params)
}

pub fn process_endpoint_groups_queries(
    groups_with_endpoints: &[(NodeData, Vec<NodeData>)],
) -> Vec<(String, BoltMap)> {
    let mut queries = Vec::new();

    for (group, endpoints) in groups_with_endpoints {
        for endpoint in endpoints {
            let new_name = format!("{}{}", group.name, endpoint.name);

            queries.push(update_endpoint_name_query(
                &endpoint.name,
                &endpoint.file,
                &new_name,
            ));

            queries.push(update_endpoint_relationships_query(
                &endpoint.name,
                &endpoint.file,
                &new_name,
            ));
        }
    }

    queries
}
pub fn add_instance_contains_query(instance: &NodeData) -> (String, BoltMap) {
    let mut params = BoltMap::new();
    boltmap_insert_str(&mut params, "instance_name", &instance.name);
    boltmap_insert_str(&mut params, "instance_file", &instance.file);
    boltmap_insert_int(&mut params, "instance_start", instance.start as i64);

    let query = "MATCH (file:File {file: $instance_file}),
                       (instance:Instance {name: $instance_name, file: $instance_file, start: $instance_start})
                 MERGE (file)-[:CONTAINS]->(instance)";

    (query.to_string(), params)
}

pub fn add_instance_of_query(instance: &NodeData, class_name: &str) -> (String, BoltMap) {
    let mut params = BoltMap::new();
    boltmap_insert_str(&mut params, "instance_name", &instance.name);
    boltmap_insert_str(&mut params, "instance_file", &instance.file);
    boltmap_insert_int(&mut params, "instance_start", instance.start as i64);
    boltmap_insert_str(&mut params, "class_name", class_name);

    let query = "MATCH (instance:Instance {name: $instance_name, file: $instance_file, start: $instance_start}), 
                       (class:Class {name: $class_name}) 
                 MERGE (instance)-[:OF]->(class)";

    (query.to_string(), params)
}

pub fn has_edge_query(source: &Node, target: &Node, edge_type: &EdgeType) -> (String, BoltMap) {
    let mut params = BoltMap::new();
    let source_type = &source.node_type;
    let target_type = &target.node_type;

    boltmap_insert_str(&mut params, "source_type", &source.node_type.to_string());
    boltmap_insert_str(&mut params, "target_type", &target.node_type.to_string());
    boltmap_insert_str(&mut params, "edge_type", &edge_type.to_string());
    boltmap_insert_str(&mut params, "source_name", &source.node_data.name);
    boltmap_insert_str(&mut params, "source_file", &source.node_data.file);
    boltmap_insert_str(&mut params, "target_name", &target.node_data.name);
    boltmap_insert_str(&mut params, "target_file", &target.node_data.file);

    let query = format!(
        "MATCH (source:{})-[r:{}]->(target:{})
         WHERE source.name = $source_name AND source.file = $source_file
           AND target.name = $target_name AND target.file = $target_file
         RETURN COUNT(r) > 0 as exists",
        source_type.to_string(),
        edge_type.to_string(),
        target_type.to_string()
    );

    (query, params)
}

pub fn clear_graph_query() -> String {
    "MATCH (n:Data_Bank) DETACH DELETE n".to_string()
}

pub fn clear_existing_graph_query(root: &str) -> (String, BoltMap) {
    let mut params = BoltMap::new();
    boltmap_insert_str(&mut params, "root", root);
    let query = "MATCH (n:Data_Bank) WHERE n.file STARTS WITH $root DETACH DELETE n";
    (query.to_string(), params)
}

pub fn data_bank_bodies_query_no_embeddings(
    do_files: bool,
    skip: usize,
    limit: usize,
) -> (String, BoltMap) {
    let mut params = BoltMap::new();
    boltmap_insert_int(&mut params, "skip", skip as i64);
    boltmap_insert_int(&mut params, "limit", limit as i64);
    boltmap_insert_str(
        &mut params,
        "do_files",
        if do_files { "true" } else { "false" },
    );
    let query = r#"
        MATCH (n:Data_Bank)
        WHERE n.embeddings IS NULL
          AND (($do_files = 'true') OR NOT n:File)
        RETURN n.node_key as node_key, n.body as body
        SKIP toInteger($skip) LIMIT toInteger($limit)
    "#
    .to_string();
    (query, params)
}
pub fn update_embedding_query(node_key: &str, embedding: &[f32]) -> (String, BoltMap) {
    let mut params = BoltMap::new();
    boltmap_insert_str(&mut params, "node_key", node_key);
    let emb_list = embedding
        .iter()
        .map(|&v| neo4rs::BoltType::Float(neo4rs::BoltFloat { value: v as f64 }))
        .collect::<Vec<_>>();
    boltmap_insert_list(&mut params, "embeddings", emb_list);
    let query = r#"
        MATCH (n:Data_Bank {node_key: $node_key})
        SET n.embeddings = $embeddings
    "#
    .to_string();
    (query, params)
}
pub fn vector_search_query(
    embedding: &[f32],
    limit: usize,
    node_types: Vec<String>,
    similarity_threshold: f32,
    language: Option<String>,
) -> (String, BoltMap) {
    let mut params = BoltMap::new();

    let emb_list = embedding
        .iter()
        .map(|&v| neo4rs::BoltType::Float(neo4rs::BoltFloat { value: v as f64 }))
        .collect::<Vec<_>>();

    boltmap_insert_list(&mut params, "embeddings", emb_list.clone());

    boltmap_insert_int(&mut params, "limit", limit as i64);

    boltmap_insert_float(
        &mut params,
        "similarityThreshold",
        similarity_threshold as f64,
    );

    let node_types_list = node_types
        .into_iter()
        .map(|s| neo4rs::BoltType::String(neo4rs::BoltString::from(s)))
        .collect::<Vec<_>>();

    boltmap_insert_list(&mut params, "node_types", node_types_list);

    let ext_list = if let Some(lang_str) = language.as_ref() {
        if let Ok(lang) = Language::from_str(lang_str) {
            lang.exts()
                .iter()
                .map(|&s| neo4rs::BoltType::String(s.into()))
                .collect()
        } else {
            Vec::new()
        }
    } else {
        Vec::new()
    };

    boltmap_insert_list(&mut params, "extensions", ext_list);

    let query = r#"
        MATCH (node)
        WHERE
          CASE
            WHEN $node_types IS NULL OR size($node_types) = 0 THEN true
            ELSE ANY(label IN labels(node) WHERE label IN $node_types)
          END
          AND node.embeddings IS NOT NULL
          AND
          CASE
            WHEN $extensions IS NULL OR size($extensions) = 0 THEN true
            ELSE node.file IS NOT NULL AND ANY(ext IN $extensions WHERE node.file ENDS WITH ext)
          END
        WITH node, gds.similarity.cosine(node.embeddings, $embeddings) AS score
        WHERE score >= $similarityThreshold
        RETURN node, score
        ORDER BY score DESC
        LIMIT toInteger($limit)
    "#
    .to_string();

    (query, params)
}

pub fn query_nodes_with_count(
    node_type: &NodeType,
    offset: usize,
    limit: usize,
    sort_by_test_count: bool,
    coverage_filter: Option<&str>,
    body_length: bool,
    line_count: bool,
    repo: Option<&str>,
    test_filters: Option<super::TestFilters>,
) -> (String, BoltMap) {
    let mut params = BoltMap::new();
    boltmap_insert_int(&mut params, "offset", offset as i64);
    boltmap_insert_int(&mut params, "limit", limit as i64);

    let ignore_dirs = test_filters
        .as_ref()
        .map(|f| f.ignore_dirs.clone())
        .unwrap_or_default();

    let order_clause = if body_length {
        "ORDER BY size(n.body) DESC, n.name ASC"
    } else if line_count {
        "ORDER BY (n.end - n.start) DESC, n.name ASC"
    } else if sort_by_test_count {
        "ORDER BY test_count DESC, n.name ASC"
    } else {
        "ORDER BY n.name ASC"
    };

    let (test_match_clauses, test_count_expr) = if let Some(filters) = &test_filters {
        let has_filters = !filters.unit_regexes.is_empty()
            || !filters.integration_regexes.is_empty()
            || !filters.e2e_regexes.is_empty();

        if has_filters {
            let mut clauses = Vec::new();
            let mut count_parts = Vec::new();

            if !filters.unit_regexes.is_empty() {
                let regex_conditions: Vec<String> = filters
                    .unit_regexes
                    .iter()
                    .map(|r| format!("ut.file =~ '{}'", r))
                    .collect();
                clauses.push(format!(
                    "OPTIONAL MATCH (ut:UnitTest)-[:CALLS]->(n) WHERE {}",
                    regex_conditions.join(" OR ")
                ));
                count_parts.push("COUNT(DISTINCT ut)");
            }

            if !filters.integration_regexes.is_empty() {
                let regex_conditions: Vec<String> = filters
                    .integration_regexes
                    .iter()
                    .map(|r| format!("it.file =~ '{}'", r))
                    .collect();
                clauses.push(format!(
                    "OPTIONAL MATCH (it:IntegrationTest)-[:CALLS]->(n) WHERE {}",
                    regex_conditions.join(" OR ")
                ));
                count_parts.push("COUNT(DISTINCT it)");
            }

            if !filters.e2e_regexes.is_empty() {
                let regex_conditions: Vec<String> = filters
                    .e2e_regexes
                    .iter()
                    .map(|r| format!("et.file =~ '{}'", r))
                    .collect();
                clauses.push(format!(
                    "OPTIONAL MATCH (et:E2etest)-[:CALLS]->(n) WHERE {}",
                    regex_conditions.join(" OR ")
                ));
                count_parts.push("COUNT(DISTINCT et)");
            }

            let test_count = if count_parts.is_empty() {
                "0".to_string()
            } else {
                count_parts.join(" + ")
            };

            (clauses.join("\n         "), test_count)
        } else {
            (
                "OPTIONAL MATCH (test)-[:CALLS]->(n) WHERE test:UnitTest OR test:IntegrationTest OR test:E2etest".to_string(),
                "count(DISTINCT test)".to_string(),
            )
        }
    } else {
        (
            "OPTIONAL MATCH (test)-[:CALLS]->(n) WHERE test:UnitTest OR test:IntegrationTest OR test:E2etest".to_string(),
            "count(DISTINCT test)".to_string(),
        )
    };

    let coverage_where = match coverage_filter {
        Some("tested") => "WHERE test_count > 0",
        Some("untested") => "WHERE test_count = 0",
        _ => "",
    };

    let repo_filter = if let Some(r) = repo {
        if r.is_empty() || r == "all" {
            String::new()
        } else if r.contains(',') {
            let repos: Vec<&str> = r.split(',').map(|s| s.trim()).collect();
            let conditions: Vec<String> = repos
                .iter()
                .map(|repo_path| format!("n.file STARTS WITH '{}'", repo_path))
                .collect();
            format!("AND ({})", conditions.join(" OR "))
        } else {
            format!("AND n.file STARTS WITH '{}'", r)
        }
    } else {
        String::new()
    };

    let ignore_dirs_filter = if !ignore_dirs.is_empty() {
        let conditions: Vec<String> = ignore_dirs
            .iter()
            .map(|dir| format!("NOT n.file CONTAINS '/{}/'", dir))
            .collect();
        format!("AND {}", conditions.join(" AND "))
    } else {
        String::new()
    };

    let regex_filter = test_filters
        .as_ref()
        .and_then(|f| f.target_regex.as_deref())
        .map(|pattern| format!("AND n.file =~ '{}'", pattern))
        .unwrap_or_default();

    let query = format!(
        "MATCH (n:{})
         WHERE {} {} {} {}
         {}
         WITH n, {} AS test_count
         {} 
         OPTIONAL MATCH (caller)-[:CALLS]->(n)
         WITH n, test_count, count(DISTINCT caller) AS usage_count, (test_count > 0) AS is_covered
         {}
         WITH collect({{
             node: n,
             usage_count: usage_count,
             is_covered: is_covered,
             test_count: test_count,
             body_length: size(n.body),
             line_count: (n.end - n.start + 1)
         }}) AS all_items
         RETURN 
             size(all_items) AS total_count,
             [item IN all_items | item][$offset..($offset + $limit)] AS items",
        node_type.to_string(),
        unique_functions_filters().join(" AND "),
        repo_filter,
        ignore_dirs_filter,
        regex_filter,
        test_match_clauses,
        test_count_expr,
        coverage_where,
        order_clause
    );

    (query, params)
}

pub fn set_missing_data_bank_query() -> String {
    r#"
        MATCH (n)
        WHERE n.Data_Bank IS NULL AND n.name IS NOT NULL AND NOT n:Schema
        SET n.Data_Bank = n.name
        RETURN count(n) as updated_count
    "#
    .to_string()
}

pub fn set_default_namespace_query() -> String {
    r#"
        MATCH (n)
        WHERE n.namespace IS NULL AND NOT n:Schema
        SET n.namespace = "default"
        RETURN count(n) as updated_count
    "#
    .to_string()
}
pub fn find_dynamic_edges_for_file_query(file: &str) -> (String, BoltMap) {
    let mut params = BoltMap::new();
    boltmap_insert_str(&mut params, "file", file);

    let static_types = vec![
        "Repository", "Package", "Language", "Directory", "File", "Import", "Library",
        "Class", "Trait", "Instance", "Function", "Endpoint", "Request", "Datamodel",
        "Feature", "Page", "Var", "UnitTest", "IntegrationTest", "E2etest"
    ];
    
    let static_labels = static_types.iter()
        .map(|t| format!("source:{}", t))
        .collect::<Vec<_>>()
        .join(" OR ");

    let query = format!(
        "MATCH (source)-[r]->(target)
         WHERE target.file ENDS WITH $file 
         AND NOT ({})
         RETURN source.ref_id as source_ref_id, type(r) as edge_type, 
                target.name as target_name, target.file as target_file, labels(target)[0] as target_type",
        static_labels
    );

    (query, params)
}

pub fn find_all_dynamic_edges_query() -> (String, BoltMap) {
    let params = BoltMap::new();

    let static_types = vec![
        "Repository", "Package", "Language", "Directory", "File", "Import", "Library",
        "Class", "Trait", "Instance", "Function", "Endpoint", "Request", "Datamodel",
        "Feature", "Page", "Var", "UnitTest", "IntegrationTest", "E2etest"
    ];
    
    let static_labels = static_types.iter()
        .map(|t| format!("source:{}", t))
        .collect::<Vec<_>>()
        .join(" OR ");

    let query = format!(
        "MATCH (source)-[r]->(target)
         WHERE NOT ({})
         RETURN source.ref_id as source_ref_id, type(r) as edge_type, 
                target.name as target_name, target.file as target_file, labels(target)[0] as target_type",
        static_labels
    );

    (query, params)
}

pub fn restore_dynamic_edge_query(
    source_ref_id: &str,
    edge_type: &str,
    target_name: &str,
    target_file: &str,
    target_type: &str,
) -> (String, BoltMap) {
    let mut params = BoltMap::new();
    boltmap_insert_str(&mut params, "source_ref_id", source_ref_id);
    boltmap_insert_str(&mut params, "edge_type", edge_type);
    boltmap_insert_str(&mut params, "target_name", target_name);
    boltmap_insert_str(&mut params, "target_file", target_file);

    let query = format!(
        "MATCH (source {{ref_id: $source_ref_id}})
         MATCH (target:{} {{name: $target_name, file: $target_file}})
         MERGE (source)-[r:{}]->(target)
         RETURN r",
        target_type, edge_type
    );

    (query, params)
}

