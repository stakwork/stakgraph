use crate::utils::create_node_key;
use anyhow::Result;
use lazy_static::lazy_static;
use neo4rs::{query, BoltMap, BoltType, ConfigBuilder, Graph as Neo4jConnection};
use std::sync::{Arc, Once};
use tiktoken_rs::get_bpe_from_model;
use tracing::{debug, info};
use crate::lang::FunctionCall;

use super::*;

lazy_static! {
    static ref CONNECTION: tokio::sync::Mutex<Option<Arc<Neo4jConnection>>> =
        tokio::sync::Mutex::new(None);
    static ref INIT: Once = Once::new();
}

const DATA_BANK: &str = "Data_Bank";

pub struct Neo4jConnectionManager;

impl Neo4jConnectionManager {
    pub async fn initialize(uri: &str, username: &str, password: &str) -> Result<()> {
        let mut conn_guard = CONNECTION.lock().await;
        if conn_guard.is_some() {
            return Ok(());
        }

        info!("Connecting to Neo4j at {}", uri);
        let config = ConfigBuilder::new()
            .uri(uri)
            .user(username)
            .password(password)
            .build()?;

        match Neo4jConnection::connect(config).await {
            Ok(connection) => {
                info!("Successfully connected to Neo4j");
                *conn_guard = Some(Arc::new(connection));
                Ok(())
            }
            Err(e) => Err(anyhow::anyhow!("Failed to connect to Neo4j: {}", e)),
        }
    }

    pub async fn get_connection() -> Option<Arc<Neo4jConnection>> {
        CONNECTION.lock().await.clone()
    }

    pub async fn clear_connection() {
        let mut conn = CONNECTION.lock().await;
        *conn = None;
    }
    pub async fn initialize_from_env() -> Result<()> {
        let uri =
            std::env::var("NEO4J_URI").unwrap_or_else(|_| "bolt://localhost:7687".to_string());
        let username = std::env::var("NEO4J_USERNAME").unwrap_or_else(|_| "neo4j".to_string());
        let password = std::env::var("NEO4J_PASSWORD").unwrap_or_else(|_| "testtest".to_string());

        Self::initialize(&uri, &username, &password).await
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
        

        let query = format!(
            "MERGE (node:{}:{} {{node_key: $node_key}})
            ON CREATE SET node += $properties
            ON MATCH SET node += $properties
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

        boltmap_insert_str(&mut params, "source_name", &self.edge.source.node_data.name);
        boltmap_insert_str(&mut params, "source_file", &self.edge.source.node_data.file);

        boltmap_insert_str(&mut params, "target_name", &self.edge.target.node_data.name);
        boltmap_insert_str(&mut params, "target_file", &self.edge.target.node_data.file);

        let rel_type = self.edge.edge.to_string();
        let source_type = self.edge.source.node_type.to_string();
        let target_type = self.edge.target.node_type.to_string();

        let query = format!(
            "MATCH (source:{} {{name: $source_name, file: $source_file}}), \
                       (target:{} {{name: $target_name, file: $target_file}}) \
                 MERGE (source)-[r:{}]->(target)",
            source_type, target_type, rel_type
        );
        (query, params)
    }
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
                let properties = boltmap_to_bolttype_map(&bolt_map);
                query_obj = query_obj.param("properties", properties);

                if let Some(BoltType::String(node_key)) = bolt_map.value.get("node_key") {
                    query_obj = query_obj.param("node_key", node_key.value.as_str());
                }
                if let Some(BoltType::String(ref_id)) = bolt_map.value.get("ref_id") {
                    query_obj = query_obj.param("ref_id", ref_id.value.as_str());
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

pub fn count_nodes_edges_query() -> String {
    "MATCH (n) 
     WITH COUNT(n) as nodes
     MATCH ()-[r]->() 
     RETURN nodes, COUNT(r) as edges"
        .to_string()
}
pub fn graph_node_analysis_query() -> String {
    "MATCH (n) 
     RETURN labels(n)[0] as node_type, n.name as name, n.file as file, n.start as start, 
            n.end as end, n.body as body, n.data_type as data_type, n.docs as docs, 
            n.hash as hash, n.meta as meta
     ORDER BY node_type, name"
        .to_string()
}
pub fn graph_edges_analysis_query() -> String {
    "MATCH (source)-[r]->(target) 
     RETURN labels(source)[0] as source_type, source.name as source_name, source.file as source_file, source.start as source_start,
            type(r) as edge_type, labels(target)[0] as target_type, 
            target.name as target_name, target.file as target_file, target.start as target_start
     ORDER BY source_type, source_name, edge_type, target_type, target_name"
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
pub fn find_nodes_by_name_query(node_type: &NodeType, name: &str) -> (String, BoltMap) {
    let mut param = BoltMap::new();
    param
        .value
        .insert("name".into(), BoltType::String(name.into()));

    let query = format!(
        "MATCH (n:{}) 
                       WHERE n.name = $name 
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
pub fn find_nodes_in_range_query(
    node_type: &NodeType,
    file: &str,
    row: u32,
) -> (String, BoltMap) {
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

    let query = 
        format!("MATCH (endpoint:Endpoint {{name: $endpoint_name, file: $endpoint_file, start: $endpoint_start}})-[:HANDLER]->(handler)
         RETURN handler");

    (query, params)
}

pub fn check_direct_data_model_usage_query(
    function_name: &str, 
    data_model: &str
) -> (String, BoltMap) {
    let mut params = BoltMap::new();

    boltmap_insert_str(&mut params, "function_name", function_name);
    boltmap_insert_str(&mut params, "data_model", data_model);

    let query = 
        format!("MATCH (f:Function {{name: $function_name}})-[:CONTAINS]->(n:Datamodel)
         WHERE n.name CONTAINS $data_model
         RETURN COUNT(n) > 0 as exists");

    (query, params)
}

pub fn find_functions_called_by_query(function: &NodeData) -> (String, BoltMap) {
    let mut params = BoltMap::new();
    boltmap_insert_str(&mut params, "function_name", &function.name);
    boltmap_insert_str(&mut params, "function_file", &function.file);
    boltmap_insert_int(&mut params, "function_start", function.start as i64);

    let query = 
        format!("MATCH (source:Function {{name: $function_name, file: $function_file, start: $function_start}})-[:CALLS]->(target:Function)
         RETURN target");

    (query, params)
}

pub fn find_node_at_query(
    node_type: &NodeType,
    file: &str,
    line: u32,
) -> (String, BoltMap) {
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
    format!("MATCH (c:Class)
     WHERE c.meta IS NOT NULL
     WITH c, apoc.convert.fromJsonMap(c.meta) AS meta_map
     WHERE meta_map.parent IS NOT NULL
     MATCH (parent:Class {{name: meta_map.parent}})
     MERGE (parent)-[:PARENT_OF]->(c)")
}
pub fn class_includes_query() -> String {
    format!("MATCH (c:Class)
     WHERE c.meta IS NOT NULL
     WITH c, apoc.convert.fromJsonMap(c.meta) AS meta_map
     WHERE meta_map.includes IS NOT NULL
     WITH c, split(meta_map.includes, ',') AS modules
     UNWIND modules AS module
     MATCH (m:Class {{name: trim(module)}})
     MERGE (c)-[:IMPORTS]->(m)")
}
pub fn prefix_paths_query(root: &str) -> (String, BoltMap) {
    let mut params = BoltMap::new();
    let root = if root.ends_with('/') { root.to_string() } else { format!("{}/", root) };

    boltmap_insert_str(&mut params, "root", &root);

    let query = format!("MATCH (n)
                WHERE n.file IS NOT NULL AND NOT n.file STARTS WITH $root
                SET n.file = $root + n.file");

    (query, params)
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
    boltmap_insert_str(&mut params, "parent_type", &parent_type.to_string());
    boltmap_insert_str(&mut params, "name", &node_data.name);
    boltmap_insert_str(&mut params, "file", &node_data.file);
    boltmap_insert_int(&mut params, "start", node_data.start as i64);
    boltmap_insert_str(&mut params, "parent_file", parent_file);

    let query_str = format!(
        "MATCH (parent:{} {{file: $parent_file}}),
               (node:{} {{name: $name, file: $file, start: $start}})
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
             MERGE (function)-[:HANDLER]->(request)"
        );
        queries.push((query_str, params));
    }

    for dm_edge in dms {
        queries.push(add_edge_query(dm_edge));
    }
    queries
}
pub fn add_test_node_query(
    test_data: &NodeData,
    test_type: &NodeType,
    test_edge: &Option<Edge>
) -> Vec<(String, BoltMap)> {
    let mut queries = Vec::new();

    queries.push(add_node_query(test_type, test_data));

    let mut params = BoltMap::new();
    boltmap_insert_str(&mut params, "test_type", &test_type.to_string());
    boltmap_insert_str(&mut params, "test_name", &test_data.name);
    boltmap_insert_str(&mut params, "test_file", &test_data.file);
    boltmap_insert_int(&mut params, "test_start", test_data.start as i64);

    let query_str = format!(
        "MATCH (test:{} {{name: $test_name, file: $test_file, start: $test_start}}),
               (file:File {{file: $test_file}})
         MERGE (file)-[:CONTAINS]->(test)",
        test_type.to_string()
    );

    queries.push((query_str, params));

    if let Some(edge) = test_edge {
        queries.push(add_edge_query(edge));
    }
    queries
}

pub fn add_page_query(
    page_data: &NodeData,
    edge_opt: &Option<Edge>
) -> Vec<(String, BoltMap)> {
    let mut queries = Vec::new();

    queries.push(add_node_query(&NodeType::Page, page_data));

    if let Some(edge) = edge_opt {
        queries.push(add_edge_query(edge));
    }

    queries
}

pub fn add_pages_query(
    pages: &[(NodeData, Vec<Edge>)]
) -> Vec<(String, BoltMap)> {
    let mut queries = Vec::new();

    for (page_data, edges) in pages {
        queries.push(add_node_query(&NodeType::Page, page_data));

        for edge in edges {
            queries.push(add_edge_query(edge));
        }
    }

    queries
}

pub fn add_endpoints_query(
    endpoints: &[(NodeData, Option<Edge>)]
) -> Vec<(String, BoltMap)> {
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
    funcs: &[FunctionCall],
    tests: &[FunctionCall],
    int_tests: &[Edge]
) -> Vec<(String, BoltMap)> {
    let mut queries = Vec::new();

    for (func_call, ext_func, class_call) in funcs {
        if let Some(class_call) = class_call{
            let edge = Edge::new(
                EdgeType::Calls,
                NodeRef::from(func_call.source.clone(), NodeType::Function),
                NodeRef::from(class_call.into(), NodeType::Class)
            );
            queries.push(add_edge_query(&edge));
        }

         if func_call.target.is_empty() {
            continue;
        }
        if let Some(ext_nd) = ext_func {
            queries.push(add_node_query(&NodeType::Function, ext_nd));
            let edge = Edge::uses(func_call.source.clone(), ext_nd);
            queries.push(add_edge_query(&edge));
        } else {
            let edge = func_call.clone().into();
            queries.push(add_edge_query(&edge));
        }
    }

    for (test_call, ext_func,_class_call) in tests {
        if let Some(ext_nd) = ext_func {

            queries.push(add_node_query(&NodeType::Function, ext_nd));

            let edge = Edge::uses(test_call.source.clone(), ext_nd);
            queries.push(add_edge_query(&edge));
        } else {
            let edge = Edge::new_test_call(test_call.clone());
            queries.push(add_edge_query(&edge));
        }
    }

    for edge in int_tests {
        queries.push(add_edge_query(edge));
    }

    queries
}
pub fn find_endpoint_query(
    name: &str,
    file: &str,
    verb: &str
) -> (String, BoltMap) {
    let mut params = BoltMap::new();
    boltmap_insert_str(&mut params, "name", name);
    boltmap_insert_str(&mut params, "verb", verb.to_uppercase().as_str());
    boltmap_insert_str(&mut params, "file", file);
    boltmap_insert_str(&mut params, "node_type", &NodeType::Endpoint.to_string());

    let query = 
        "MATCH (n:Endpoint {name: $name, file: $file})
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

pub fn remove_nodes_by_file_query(file_path: &str) -> (String, BoltMap) {
    let mut params = BoltMap::new();
    let file_name = file_path.split('/').last().unwrap_or(file_path);
    boltmap_insert_str(&mut params, "file_name", file_name);

    let query = "
        MATCH (n)
        WHERE n.file = $file_name OR n.file ENDS WITH $file_name
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
pub fn boltmap_insert_int(map: &mut BoltMap, key: &str, value: i64) {
    map.value
        .insert(key.into(), BoltType::Integer(value.into()));
}
fn boltmap_to_bolttype_map(bolt_map: &BoltMap) -> BoltType {
    let mut map = std::collections::HashMap::new();
    for (k, v) in &bolt_map.value {
        map.insert(k.clone(), v.clone());
    }
    BoltType::Map(BoltMap { value: map })
}
pub fn calculate_token_count(body: &str) -> Result<i64> {
    let bpe = get_bpe_from_model("gpt-4")?;
    let token_count = bpe.encode_with_special_tokens(body).len() as i64;
    Ok(token_count)
}