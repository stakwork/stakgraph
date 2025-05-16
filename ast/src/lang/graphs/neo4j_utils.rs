use anyhow::Result;
use neo4rs::{query, ConfigBuilder, Graph as Neo4jConnection};
use std::{
    collections::{BTreeMap, HashMap},
    sync::{Arc, Once},
};
use tracing::{debug, info};
use lazy_static::lazy_static;
use crate::{lang::FunctionCall, utils::create_node_key};

use super::*;

lazy_static! {
    static ref CONNECTION: tokio::sync::Mutex<Option<Arc<Neo4jConnection>>> = tokio::sync::Mutex::new(None);
    static ref INIT: Once = Once::new();
}

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
            },
            Err(e) => Err(anyhow::anyhow!("Failed to connect to Neo4j: {}", e))
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
        let uri = std::env::var("NEO4J_URI").unwrap_or_else(|_| "bolt://localhost:7687".to_string());
        let username = std::env::var("NEO4J_USERNAME").unwrap_or_else(|_| "neo4j".to_string());
        let password = std::env::var("NEO4J_PASSWORD").unwrap_or_else(|_| "password".to_string());

        Self::initialize(&uri, &username, &password).await
    }
}

pub struct QueryBuilder {
    query: String,
    params: HashMap<String, String>,
}

impl QueryBuilder {
 pub fn new(query_string: &str) -> Self {
        Self {
            query: query_string.to_string(),
            params: HashMap::new(),
        }
    }
    
    pub fn with_param(mut self, key: &str, value: &str) -> Self {
        self.params.insert(key.to_string(), value.to_string());
        self
    }
    
    pub fn with_params(mut self, params: HashMap<String, String>) -> Self {
        self.params.extend(params);
        self
    }
    
    pub fn build(&self) -> (String, HashMap<String, String>) {
        (self.query.clone(), self.params.clone())
    }
    
    pub fn to_neo4j_query(&self) -> neo4rs::Query {
        let mut query_obj = query(&self.query);
        
        for (key, value) in &self.params {
            query_obj = query_obj.param(key, value.as_str());
        }
        
        query_obj
    }
}

pub async fn execute_query(
    conn: &Neo4jConnection,
    builder: &QueryBuilder,
) -> Result<()> {
    match conn.execute(builder.to_neo4j_query()).await {
        Ok(_) => Ok(()),
        Err(e) => {
            debug!("Neo4j query error: {}", e);
            Err(anyhow::anyhow!("Neo4j query error: {}", e))
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
    
    pub fn build_params(&self) -> HashMap<String, String> {
        let mut params = HashMap::new();

        params.insert("name".to_string(), self.node_data.name.clone());
        params.insert("file".to_string(), self.node_data.file.clone());
        params.insert("start".to_string(), self.node_data.start.to_string());
        params.insert("end".to_string(), self.node_data.end.to_string());
        params.insert("body".to_string(), self.node_data.body.clone());

        if let Some(data_type) = &self.node_data.data_type {
            params.insert("data_type".to_string(), data_type.clone());
        }
        if let Some(docs) = &self.node_data.docs {
            params.insert("docs".to_string(), docs.clone());
        }
        if let Some(hash) = &self.node_data.hash {
            params.insert("hash".to_string(), hash.clone());
        }

        for (key, value) in &self.node_data.meta {
            params.insert(key.clone(), value.clone());
        }
        
        let node_key = create_node_key(&Node::new(self.node_type.clone(), self.node_data.clone()));
        params.insert("key".to_string(), node_key.clone());
        
        params
    }
    
    pub fn build(&self) -> (String, HashMap<String, String>) {
        let params = self.build_params();
        
        let property_list = params
            .keys()
            .filter(|k| k != &"key")
            .map(|k| format!("n.{} = ${}", k, k))
            .collect::<Vec<_>>()
            .join(", ");

        let query = format!(
            "MERGE (n:{} {{key: $key}})
            ON CREATE SET {}
            ON MATCH SET {}",
            self.node_type.to_string(),
            property_list,
            property_list
        );

        (query, params)
    }
}
pub struct EdgeQueryBuilder {
    edge: Edge,
}

impl EdgeQueryBuilder {
    pub fn new(edge: &Edge) -> Self {
        Self {
            edge: edge.clone(),
        }
    }
    
    pub fn build_params(&self) -> HashMap<String, String> {
        let mut params = HashMap::new();

        params.insert("source_name".to_string(), self.edge.source.node_data.name.clone());
        params.insert("source_file".to_string(), self.edge.source.node_data.file.clone());
        params.insert("source_start".to_string(), self.edge.source.node_data.start.to_string());

        if let Some(verb) = &self.edge.source.node_data.verb {
            params.insert("source_verb".to_string(), verb.clone());
        }

        params.insert("target_name".to_string(), self.edge.target.node_data.name.clone());
        params.insert("target_file".to_string(), self.edge.target.node_data.file.clone());
        params.insert("target_start".to_string(), self.edge.target.node_data.start.to_string());

        if let Some(verb) = &self.edge.target.node_data.verb {
            params.insert("target_verb".to_string(), verb.clone());
        }
        
        // Adding edge-specific properties
        match &self.edge.edge {
            EdgeType::Calls(meta) => {
                params.insert("call_start".to_string(), meta.call_start.to_string());
                params.insert("call_end".to_string(), meta.call_end.to_string());

                if let Some(operand) = &meta.operand {
                    params.insert("operand".to_string(), operand.clone());
                }
            }
            _ => {}
        };
        
        params
    }
    
    pub fn build(&self) -> (String, HashMap<String, String>) {
        let mut params = self.build_params();
        let rel_type = self.edge.edge.to_string();
        
        let source_type = self.edge.source.node_type.to_string();
        let target_type = self.edge.target.node_type.to_string();
        
        params.insert("source_name".to_string(), self.edge.source.node_data.name.clone());
        params.insert("source_file".to_string(), self.edge.source.node_data.file.clone());
        params.insert("target_name".to_string(), self.edge.target.node_data.name.clone());
        params.insert("target_file".to_string(), self.edge.target.node_data.file.clone());
    
        let props_clause = match &self.edge.edge {
            EdgeType::Calls(_) if params.contains_key("operand") => {
                "r.call_start = $call_start, r.call_end = $call_end, r.operand = $operand"
            }
            EdgeType::Calls(_) => {
                "r.call_start = $call_start, r.call_end = $call_end"
            }
            _ => "",
        };
    
        
        let query = format!(
            "MATCH (source:{} {{name: $source_name, file: $source_file}}), 
                   (target:{} {{name: $target_name, file: $target_file}})
             MERGE (source)-[r:{}]->(target)
             ON CREATE SET {}
             ON MATCH SET {}",
            source_type,
            target_type,
            rel_type,
            if props_clause.is_empty() { "r.updated = true" } else { props_clause },
            if props_clause.is_empty() { "r.updated = true" } else { props_clause }
        );
    
        (query, params)
    }
}
pub async fn execute_batch(
    conn: &Neo4jConnection,
    queries: Vec<(String, HashMap<String, String>)>,
) -> Result<()> {
    let mut txn = conn.start_txn().await?;
    
     for (i, (query_str, params)) in queries.iter().enumerate() {
        let mut query_obj = query(&query_str);
        for (k, v) in params {
            query_obj = query_obj.param(&k, v.as_str());
        }

            if let Err(e) = txn.run(query_obj).await  {
            println!("Neo4j query #{} {} failed: {}", i, query_str, e);
            txn.rollback().await?;
            return Err(anyhow::anyhow!("Neo4j batch query error: {}", e));
            }
        
    }
    
    txn.commit().await?;
    Ok(())
}
pub struct TransactionManager<'a> {
    conn: &'a Neo4jConnection,
    queries: Vec<(String, HashMap<String, String>)>,
}

impl<'a> TransactionManager<'a> {
    pub fn new(conn: &'a Neo4jConnection) -> Self {
        Self {
            conn,
            queries: Vec::new(),
        }
    }
    
    pub fn add_query(&mut self, query: (String, HashMap<String, String>)) -> &mut Self {
        self.queries.push(query);
        self
    }
    
    pub fn add_queries(&mut self, mut queries: Vec<(String, HashMap<String, String>)>) -> &mut Self {
        self.queries.append(&mut queries);
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
        execute_batch(self.conn, self.queries).await
    }
}


pub fn add_node_query(
    node_type: &NodeType,
    node_data: &NodeData,
) -> (String, HashMap<String, String>) {
   NodeQueryBuilder::new(node_type, node_data).build()
}

pub fn add_edge_query(edge: &Edge) -> (String, HashMap<String, String>) {
    EdgeQueryBuilder::new(edge).build()
}

pub async fn execute_node_query(
    conn: &Neo4jConnection,
    query_str: String,
    params: HashMap<String, String>,
) -> Result<Vec<NodeData>> {
   let mut query_obj = query(&query_str);
    
    for (key, value) in params {
        query_obj = query_obj.param(&key, value);
    }
    
    match conn.execute(query_obj).await {
        Ok(mut result) => {
            let mut nodes = Vec::new();

            while let Some(row) = result.next().await? {
                if let Ok(node) = row.get::<neo4rs::Node>("n") {
                    let name = node.get::<String>("name").unwrap_or_default();
                    let file = node.get::<String>("file").unwrap_or_default();
                    let start = node.get::<i32>("start").unwrap_or_default();
                    let end = node.get::<i32>("end").unwrap_or_default();
                    let body = node.get::<String>("body").unwrap_or_default();
                    let data_type = node.get::<String>("data_type").unwrap_or_default();
                    let docs = node.get::<String>("docs").unwrap_or_default();
                    let hash = node.get::<String>("hash").unwrap_or_default();
                    let meta = node.get::<BTreeMap<String, String>>("meta").unwrap_or_default();
                    let node_data = NodeData {
                        name,
                        file,
                        start: start as usize,
                        end: end as usize,
                        body,
                        data_type: Some(data_type),
                        docs: Some(docs),
                        hash: Some(hash),
                        meta : meta,
                    };
                   
                    nodes.push(node_data);
                }
            }
            Ok(nodes)
        }
        Err(e) => {
            debug!("Error executing query: {}", e);
            Ok(vec![])
        }
    }
}

pub fn find_node_by_key_query(node_key: &str) -> (String, HashMap<String, String>) {
    let mut params = HashMap::new();
    params.insert("key".to_string(), node_key.to_string());

    let query = "MATCH (n {key: $key}) 
                       RETURN n";

    (query.to_string(), params)
}

pub fn node_exists_query(node_key: &str) -> (String, HashMap<String, String>) {
    let mut params = HashMap::new();
    params.insert("key".to_string(), node_key.to_string());

    let query = "MATCH (n {key: $key}) 
                       RETURN COUNT(n) > 0 AS exists";

    (query.to_string(), params)
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
     RETURN labels(n)[0] as node_type, n.name as name, n.file as file 
     ORDER BY node_type, name"
        .to_string()
}
pub fn graph_edges_analysis_query() -> String {
    "MATCH (source)-[r]->(target) 
     RETURN labels(source)[0] as source_type, source.name as source_name, 
            type(r) as edge_type, labels(target)[0] as target_type, 
            target.name as target_name, r.operand as operand 
     ORDER BY source_type, source_name, edge_type, target_type, target_name"
        .to_string()
}
pub fn count_edges_by_type_query(edge_type: &EdgeType) -> (String, HashMap<String, String>) {
    let mut params = HashMap::new();
    params.insert("edge_type".to_string(), edge_type.to_string());

    let query = "MATCH ()-[r]->() 
                WHERE type(r) = $edge_type 
                RETURN COUNT(r) as count";

    (query.to_string(), params)
}

pub fn find_nodes_by_type_query(node_type: &NodeType) -> (String, HashMap<String, String>) {
    let mut params = HashMap::new();
    params.insert("node_type".to_string(), node_type.to_string());

    let query = format!(
        "MATCH (n:{}) 
         RETURN n",
        node_type.to_string()
    );

    (query, params)
}
pub fn find_nodes_by_name_query(
    node_type: &NodeType,
    name: &str,
) -> (String, HashMap<String, String>) {
    let mut params = HashMap::new();
    params.insert("name".to_string(), name.to_string());

    let query = format!(
        "MATCH (n:{}) 
                       WHERE n.name = $name 
                       RETURN n",
        node_type.to_string()
    );

    (query, params)
}

pub fn find_node_by_name_file_query(
    node_type: &NodeType,
    name: &str,
    file: &str,
) -> (String, HashMap<String, String>) {
    let mut params = HashMap::new();
    params.insert("name".to_string(), name.to_string());
    params.insert("file".to_string(), file.to_string());

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
) -> (String, HashMap<String, String>) {
    let mut params = HashMap::new();
    params.insert("file_pattern".to_string(), file_pattern.to_string());
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
) -> (String, HashMap<String, String>) {
    let mut params = HashMap::new();
    params.insert("name_part".to_string(), name_part.to_string());

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
) -> (String, HashMap<String, String>) {
    let mut params = HashMap::new();
    params.insert("file".to_string(), file.to_string());
    params.insert("row".to_string(), row.to_string());

    let query = format!(
        "MATCH (n:{})
         WHERE n.file = $file AND 
               toInteger(n.start) <= toInteger($row) AND 
               toInteger(n.end) >= toInteger($row)
         RETURN n",
        node_type.to_string()
    );

    (query, params)
}

pub fn find_source_edge_by_name_and_file_query(
    edge_type: &EdgeType,
    target_name: &str,
    target_file: &str,
) -> (String, HashMap<String, String>) {
    let mut params = HashMap::new();
    params.insert("edge_type".to_string(), edge_type.to_string());
    params.insert("target_name".to_string(), target_name.to_string());
    params.insert("target_file".to_string(), target_file.to_string());

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
) -> (String, HashMap<String, String>) {
    let mut params = HashMap::new();
    params.insert("edge_type".to_string(), edge_type.to_string());

    let query = format!(
        "MATCH (source:{})-[r:{}]->(target:{})
         RETURN source.key as source_key, target.key as target_key",
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
) -> (String, HashMap<String, String>) {
    let mut params = HashMap::new();
    params.insert("path".to_string(), path.to_string());
    params.insert("verb".to_string(), verb.to_uppercase());

    let query = format!(
        "MATCH (n:{})
         WHERE n.name CONTAINS $path AND 
               (n.verb IS NULL OR toUpper(n.verb) CONTAINS $verb)
         RETURN n",
        node_type.to_string()
    );

    (query, params)
}

pub fn find_handlers_for_endpoint_query(endpoint: &NodeData) -> (String, HashMap<String, String>) {
    let mut params = HashMap::new();
    params.insert("endpoint_name".to_string(), endpoint.name.clone());
    params.insert("endpoint_file".to_string(), endpoint.file.clone());
    
    let query = 
        "MATCH (endpoint:Endpoint {name: $endpoint_name, file: $endpoint_file})-[:HANDLER]->(handler)
         RETURN handler";
    
    (query.to_string(), params)
}

pub fn check_direct_data_model_usage_query(
    function_name: &str, 
    data_model: &str
) -> (String, HashMap<String, String>) {
    let mut params = HashMap::new();
    params.insert("function_name".to_string(), function_name.to_string());
    params.insert("data_model".to_string(), data_model.to_string());
    
    let query = 
        "MATCH (f:Function {name: $function_name})-[:CONTAINS]->(n:Datamodel)
         WHERE n.name CONTAINS $data_model
         RETURN COUNT(n) > 0 as exists";
    
    (query.to_string(), params)
}

pub fn find_functions_called_by_query(function: &NodeData) -> (String, HashMap<String, String>) {
    let mut params = HashMap::new();
    params.insert("function_name".to_string(), function.name.clone());
    params.insert("function_file".to_string(), function.file.clone());
    params.insert("function_start".to_string(), function.start.to_string());
    
    let query = 
        "MATCH (source:Function {name: $function_name, file: $function_file, start: $function_start})-[:CALLS]->(target:Function)
         RETURN target";
    
    (query.to_string(), params)
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
     MERGE (c)-[:CLASS_IMPORTS]->(m)"
        .to_string()
}


pub fn prefix_paths_query(root: &str) -> (String, HashMap<String, String>) {
    let mut params = HashMap::new();
    params.insert("root".to_string(), root.to_string());
    
    let query = "MATCH (n)
                WHERE n.file IS NOT NULL AND NOT n.file STARTS WITH $root
                SET n.file = $root + n.file";
    
    (query.to_string(), params)
}


pub fn add_node_with_parent_query(
    node_type: &NodeType,
    node_data: &NodeData,
    parent_type: &NodeType,
    parent_file: &str,
) -> Vec<(String, HashMap<String, String>)> {
    let mut queries = Vec::new();

    queries.push(add_node_query(node_type, node_data));

    let mut params = HashMap::new();
    params.insert("name".to_string(), node_data.name.clone());
    params.insert("file".to_string(), node_data.file.clone());
    params.insert("start".to_string(), node_data.start.to_string());
    params.insert("parent_file".to_string(), parent_file.to_string());

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
) -> Vec<(String, HashMap<String, String>)> {
    let mut queries = Vec::new();

    queries.push(add_node_query(&NodeType::Function, function_node));

    let mut params = HashMap::new();
    params.insert("function_name".to_string(), function_node.name.clone());
    params.insert("function_file".to_string(), function_node.file.clone());
    params.insert(
        "function_start".to_string(),
        function_node.start.to_string(),
    );

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

        let mut params = HashMap::new();
        params.insert("function_name".to_string(), function_node.name.clone());
        params.insert("function_file".to_string(), function_node.file.clone());
        params.insert(
            "function_start".to_string(),
            function_node.start.to_string(),
        );
        params.insert("req_name".to_string(), req.name.clone());
        params.insert("req_file".to_string(), req.file.clone());
        params.insert("req_start".to_string(), req.start.to_string());
        params.insert("call_start".to_string(), req.start.to_string());
        params.insert("call_end".to_string(), req.end.to_string());

        let query_str = format!(
            "MATCH (function:Function {{name: $function_name, file: $function_file, start: $function_start}}),
                   (request:Request {{name: $req_name, file: $req_file, start: $req_start}})
             MERGE (function)-[:CALLS {{call_start: $call_start, call_end: $call_end}}]->(request)"
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
) -> Vec<(String, HashMap<String, String>)> {
    let mut queries = Vec::new();
   
    queries.push(add_node_query(test_type, test_data));
    
    let mut params = HashMap::new();
    params.insert("test_name".to_string(), test_data.name.clone());
    params.insert("test_file".to_string(), test_data.file.clone());
    params.insert("test_start".to_string(), test_data.start.to_string());
    
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
) -> Vec<(String, HashMap<String, String>)> {
    let mut queries = Vec::new();

    queries.push(add_node_query(&NodeType::Page, page_data));
  
    if let Some(edge) = edge_opt {
        queries.push(add_edge_query(edge));
    }
    
    queries
}

pub fn add_pages_query(
    pages: &[(NodeData, Vec<Edge>)]
) -> Vec<(String, HashMap<String, String>)> {
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
) -> Vec<(String, HashMap<String, String>)> {
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
) -> Vec<(String, HashMap<String, String>)> {
    let mut queries = Vec::new();
    
    for (func_call, ext_func, class_call) in funcs {
        if let Some(class_call) = class_call{
            let edge = Edge::new(
                EdgeType::Calls(CallsMeta::default()),
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
) -> (String, HashMap<String, String>) {
    let mut params = HashMap::new();
    params.insert("name".to_string(), name.to_string());
    params.insert("file".to_string(), file.to_string());
    params.insert("verb".to_string(), verb.to_uppercase());

    let query = 
        "MATCH (n:Endpoint {name: $name, file: $file})
         WHERE n.verb IS NULL OR toUpper(n.verb) CONTAINS $verb
         RETURN n";
    
    (query.to_string(), params)
}

pub fn extract_node_data_from_neo4j_node(node: &neo4rs::Node) -> NodeData {
    let name = node.get::<String>("name").unwrap_or_default();
    let file = node.get::<String>("file").unwrap_or_default();
    let start = node.get::<i32>("start").unwrap_or_default() as usize;
    let end = node.get::<i32>("end").unwrap_or_default() as usize;
    let body = node.get::<String>("body").unwrap_or_default();
    let data_type = node.get::<String>("data_type").ok();
    let docs = node.get::<String>("docs").ok();
    let hash = node.get::<String>("hash").ok();
    let meta = node.get::<BTreeMap<String, String>>("meta").unwrap_or_default();
   
    
    NodeData {
        name,
        file,
        start,
        end,
        body,
        data_type,
        docs,
        hash,
        meta,
    }
}

pub fn get_repository_hash_query(repo_url: &str) -> (String, HashMap<String, String>) {
    let mut params = HashMap::new();
    
    let repo_name = if repo_url.contains('/') {
        let parts: Vec<&str> = repo_url.split('/').collect();
        let name = parts.last().unwrap_or(&repo_url);
        name.trim_end_matches(".git")
    } else {
        repo_url
    };
    
    params.insert("repo_name".to_string(), repo_name.to_string());
    
    let query = "MATCH (r:Repository) 
                 WHERE r.name CONTAINS $repo_name 
                 RETURN r.hash as hash";
    
    (query.to_string(), params)
}

pub fn remove_nodes_by_file_query(file_path: &str) -> (String, HashMap<String, String>) {
    let mut params = HashMap::new();
    let file_name = file_path.split('/').last().unwrap_or(file_path);
    params.insert("file_name".to_string(), file_name.to_string());
    
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

pub fn update_repository_hash_query(repo_name: &str, new_hash: &str) -> (String, HashMap<String, String>) {
    let mut params = HashMap::new();
    params.insert("repo_name".to_string(), repo_name.to_string());
    params.insert("new_hash".to_string(), new_hash.to_string());
    
    let query = "MATCH (r:Repository) 
                 WHERE r.name CONTAINS $repo_name 
                 SET r.hash = $new_hash";
    
    (query.to_string(), params)
}