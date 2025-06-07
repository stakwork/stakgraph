use crate::utils::create_node_key;
use anyhow::Result;
use lazy_static::lazy_static;
use neo4rs::{query, BoltMap, BoltType, ConfigBuilder, Graph as Neo4jConnection};
use serde_json;
use std::{
    collections::{BTreeMap, HashMap},
    sync::{Arc, Once},
};
use tracing::{debug, info};

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

    pub fn build(&self) -> (String, HashMap<String, BoltType>) {
        let bolt_map: BoltMap = (&self.node_data).into();

        let mut params: HashMap<String, BoltType> = HashMap::new();

        for (key, value) in bolt_map.value {
            params.insert(key.to_string(), value);
        }

        let node_key = create_node_key(&Node::new(self.node_type.clone(), self.node_data.clone()));
        params.insert("node_key".to_string(), BoltType::String(node_key.into()));

        let property_list = params
            .keys()
            .filter(|k| k != &"node_key")
            .map(|k| format!("node.{} = ${}", k, k))
            .collect::<Vec<_>>()
            .join(", ");

        let query = format!(
            "MERGE (node:{}:{} {{node_key: $node_key}})
            ON CREATE SET {}
            ON MATCH SET {}
            RETURN node",
            self.node_type.to_string(),
            DATA_BANK,
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
        Self { edge: edge.clone() }
    }

    pub fn build(&self) -> (String, HashMap<String, String>) {
        let mut params = HashMap::new();

        params.insert(
            "source_name".to_string(),
            self.edge.source.node_data.name.clone(),
        );
        params.insert(
            "source_file".to_string(),
            self.edge.source.node_data.file.clone(),
        );
        params.insert(
            "target_name".to_string(),
            self.edge.target.node_data.name.clone(),
        );
        params.insert(
            "target_file".to_string(),
            self.edge.target.node_data.file.clone(),
        );

        let rel_type = self.edge.edge.to_string();
        let source_type = self.edge.source.node_type.to_string();
        let target_type = self.edge.target.node_type.to_string();

        let query = format!(
            "MATCH (source:{} {{name: $source_name, file: $source_file}})
            MATCH (target:{} {{name: $target_name, file: $target_file}})
            MERGE (source)-[r:{}]->(target)
            RETURN r",
            source_type, target_type, rel_type
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

        if let Err(e) = txn.run(query_obj).await {
            println!("Neo4j query #{} {} failed: {}", i, query_str, e);
            txn.rollback().await?;
            return Err(anyhow::anyhow!("Neo4j batch query error: {}", e));
        }
    }

    txn.commit().await?;
    Ok(())
}

pub async fn execute_batch_bolt(
    conn: &Neo4jConnection,
    queries: Vec<(String, HashMap<String, BoltType>)>,
) -> Result<()> {
    let mut txn = conn.start_txn().await?;
    for (i, (query_str, params)) in queries.iter().enumerate() {
        let mut query_obj = query(&query_str);
        for (key, value) in params {
            query_obj = query_obj.param(key, value.clone());
        }

        if let Err(e) = txn.run(query_obj).await {
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
    node_queries: Vec<(String, HashMap<String, BoltType>)>,
    edge_queries: Vec<(String, HashMap<String, String>)>,
}

impl<'a> TransactionManager<'a> {
    pub fn new(conn: &'a Neo4jConnection) -> Self {
        Self {
            conn,
            node_queries: Vec::new(),
            edge_queries: Vec::new(),
        }
    }

    pub fn add_node_query(&mut self, query: (String, HashMap<String, BoltType>)) -> &mut Self {
        self.node_queries.push(query);
        self
    }

    pub fn add_edge_query(&mut self, query: (String, HashMap<String, String>)) -> &mut Self {
        self.edge_queries.push(query);
        self
    }

    pub fn add_node(&mut self, node_type: &NodeType, node_data: &NodeData) -> &mut Self {
        self.node_queries.push(add_node_query(node_type, node_data));
        self
    }

    pub fn add_edge(&mut self, edge: &Edge) -> &mut Self {
        self.edge_queries.push(add_edge_query(edge));
        self
    }

    pub async fn execute(self) -> Result<()> {
        if !self.node_queries.is_empty() {
            execute_batch_bolt(self.conn, self.node_queries).await?;
        }

        if !self.edge_queries.is_empty() {
            execute_batch(self.conn, self.edge_queries).await?;
        }

        Ok(())
    }
}

pub fn add_node_query(
    node_type: &NodeType,
    node_data: &NodeData,
) -> (String, HashMap<String, BoltType>) {
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
                    let meta_json = node.get::<String>("meta").unwrap_or_default();
                    let meta = serde_json::from_str::<BTreeMap<String, String>>(&meta_json)
                        .unwrap_or_default();
                    let node_data = NodeData {
                        name,
                        file,
                        start: start as usize,
                        end: end as usize,
                        body,
                        data_type: Some(data_type),
                        docs: Some(docs),
                        hash: Some(hash),
                        meta: meta,
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
pub fn count_edges_by_type_query(edge_type: &EdgeType) -> (String, HashMap<String, String>) {
    let mut params = HashMap::new();
    params.insert("edge_type".to_string(), edge_type.to_string());

    let query = "MATCH ()-[r]->() 
                WHERE type(r) = $edge_type 
                RETURN COUNT(r) as count";

    (query.to_string(), params)
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

pub fn find_nodes_with_edge_type_query(
    source_type: &NodeType,
    target_type: &NodeType,
    edge_type: &EdgeType,
) -> (String, HashMap<String, String>) {
    let mut params = HashMap::new();
    params.insert("edge_type".to_string(), edge_type.to_string());

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

pub fn extract_node_data_from_neo4j_node(node: &neo4rs::Node) -> NodeData {
    let name = node.get::<String>("name").unwrap_or_default();
    let file = node.get::<String>("file").unwrap_or_default();
    let start = node
        .get::<i64>("start")
        .map(|v| v as usize)
        .unwrap_or_default();
    let end = node
        .get::<i64>("end")
        .map(|v| v as usize)
        .unwrap_or_default();
    let body = node.get::<String>("body").unwrap_or_default();
    let data_type = node.get::<String>("data_type").ok();
    let docs = node.get::<String>("docs").ok();
    let hash = node.get::<String>("hash").ok();
    let meta_json = node.get::<String>("meta").unwrap_or_default();
    let meta: BTreeMap<String, String> = serde_json::from_str(&meta_json).unwrap_or_default();

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

pub fn update_repository_hash_query(
    repo_name: &str,
    new_hash: &str,
) -> (String, HashMap<String, String>) {
    let mut params = HashMap::new();
    params.insert("repo_name".to_string(), repo_name.to_string());
    params.insert("new_hash".to_string(), new_hash.to_string());

    let query = "MATCH (r:Repository) 
                 WHERE r.name CONTAINS $repo_name 
                 SET r.hash = $new_hash";

    (query.to_string(), params)
}
