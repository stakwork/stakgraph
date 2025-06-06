use super::neo4j_utils::*;
use crate::lang::{Edge, EdgeType, NodeData, NodeKeys, NodeRef, NodeType};
use anyhow::Result;
use neo4rs::{query, Graph as Neo4jConnection};
use std::str::FromStr;
use std::{
    sync::{Arc, Mutex},
    time::Duration,
};
use tiktoken_rs::get_bpe_from_model;
use tracing::{debug, info, warn};

#[derive(Clone, Debug)]
pub struct Neo4jConfig {
    pub uri: String,
    pub username: String,
    pub password: String,
    pub connection_timeout: Duration,
    pub max_connections: usize,
}

impl Default for Neo4jConfig {
    fn default() -> Self {
        Neo4jConfig {
            uri: std::env::var("NEO4J_URI").unwrap_or_else(|_| "bolt://localhost:7687".to_string()),
            username: std::env::var("NEO4J_USERNAME").unwrap_or_else(|_| "neo4j".to_string()),
            password: std::env::var("NEO4J_PASSWORD").unwrap_or_else(|_| "testtest".to_string()),
            connection_timeout: Duration::from_secs(30),
            max_connections: 10,
        }
    }
}

#[derive(Clone)]
pub struct Neo4jGraph {
    connection: Option<Arc<Neo4jConnection>>,
    config: Neo4jConfig,
    connected: Arc<Mutex<bool>>,
}

impl Neo4jGraph {
    pub fn with_config(config: Neo4jConfig) -> Self {
        Neo4jGraph {
            connection: None,
            config,
            connected: Arc::new(Mutex::new(false)),
        }
    }

    pub async fn connect(&mut self) -> Result<()> {
        if let Some(conn) = Neo4jConnectionManager::get_connection().await {
            self.connection = Some(conn);
            if let Ok(mut conn_status) = self.connected.lock() {
                *conn_status = true;
            };
            debug!("Using existing connection from connection manager");
            return Ok(());
        }
        if let Ok(conn_status) = self.connected.lock() {
            if *conn_status && self.connection.is_some() {
                debug!("Already connected to Neo4j database");
                return Ok(());
            }
        }

        info!("Connecting to Neo4j database at {}", self.config.uri);

        // Initialize the connection manager with our config
        match Neo4jConnectionManager::initialize(
            &self.config.uri,
            &self.config.username,
            &self.config.password,
        )
        .await
        {
            Ok(_) => {
                if let Some(conn) = Neo4jConnectionManager::get_connection().await {
                    self.connection = Some(conn);

                    if let Ok(mut conn_status) = self.connected.lock() {
                        *conn_status = true;
                    };

                    info!("Successfully connected to Neo4j database");
                    Ok(())
                } else {
                    let error_message =
                        "Failed to get Neo4j connection after initialization".to_string();
                    debug!("{}", error_message);
                    Err(anyhow::anyhow!(error_message))
                }
            }
            Err(e) => {
                let error_message = format!("Failed to connect to Neo4j database: {}", e);
                debug!("{}", error_message);
                Err(anyhow::anyhow!(error_message))
            }
        }
    }

    pub async fn disconnect(&mut self) -> Result<()> {
        if self.connection.is_none() {
            debug!("Not connected to Neo4j database");
            return Ok(());
        }

        self.connection = None;

        if let Ok(mut conn_status) = self.connected.lock() {
            *conn_status = false;
        };

        Neo4jConnectionManager::clear_connection().await;

        info!("Disconnected from Neo4j database");
        Ok(())
    }

    pub fn is_connected(&self) -> bool {
        if let Ok(conn_status) = self.connected.lock() {
            *conn_status
        } else {
            false
        }
    }

    pub async fn ensure_connected(&mut self) -> Result<Arc<Neo4jConnection>> {
        if let Some(conn) = &self.connection {
            return Ok(conn.clone());
        }

        if let Some(conn) = Neo4jConnectionManager::get_connection().await {
            self.connection = Some(conn.clone());
            if let Ok(mut conn_status) = self.connected.lock() {
                *conn_status = true;
            }
            return Ok(conn);
        }

        self.connect().await?;

        match &self.connection {
            Some(conn) => Ok(conn.clone()),
            None => Err(anyhow::anyhow!("Failed to connect to Neo4j")),
        }
    }

    pub fn get_connection(&self) -> Arc<Neo4jConnection> {
        match &self.connection {
            Some(conn) => conn.clone(),
            None => panic!("No Neo4j connection available. Make sure Neo4j is running and connect() was called."),
        }
    }

    pub async fn create_indexes(&mut self) -> anyhow::Result<()> {
        let connection = self.ensure_connected().await?;
        let queries = vec![
            "CREATE INDEX node_key_index IF NOT EXISTS FOR (n) ON (n.node_key)",
            "CREATE FULLTEXT INDEX body_fulltext_index IF NOT EXISTS FOR (n) ON EACH [n.body]",
            "CREATE FULLTEXT INDEX name_fulltext_index IF NOT EXISTS FOR (n) ON EACH [n.name]", 
            "CREATE FULLTEXT INDEX composite_fulltext_index IF NOT EXISTS FOR (n) ON EACH [n.name, n.body, n.file]",
            "CREATE VECTOR INDEX vector_index IF NOT EXISTS FOR (n) ON (n.embeddings) OPTIONS {indexConfig: {`vector.dimensions`: 384, `vector.similarity_function`: 'cosine'}}"
        ];
        for q in queries {
            let _ = connection.run(neo4rs::query(q)).await;
        }
        Ok(())
    }

    pub async fn update_all_token_counts(&mut self) -> anyhow::Result<()> {
        let connection = self.ensure_connected().await?;

        let query_str = "MATCH (n:Data_Bank) 
                         WHERE n.token_count IS NULL AND n.body IS NOT NULL
                         RETURN n.node_key as node_key, n.body as body";

        let mut result = connection.execute(neo4rs::query(&query_str)).await?;
        let mut updates: Vec<(String, String)> = Vec::new();

        while let Some(row) = result.next().await? {
            if let (Ok(node_key), Ok(body)) =
                (row.get::<String>("node_key"), row.get::<String>("body"))
            {
                updates.push((node_key, body));
            }
        }

        info!("Found {} nodes without token counts", updates.len());

        let bpe = get_bpe_from_model("gpt-4")?;
        for (node_key, body) in &updates {
            let token_count = bpe.encode_with_special_tokens(&body).len();

            let update_query = "MATCH (n:Data_Bank {node_key: $node_key})
                               SET n.token_count = $token_count";

            let query_obj = neo4rs::query(&update_query)
                .param("node_key", node_key.as_str())
                .param("token_count", token_count as u32);
            connection.run(query_obj).await?;
        }

        info!("Updated token counts for {} nodes", updates.len());
        Ok(())
    }

    pub async fn clear(&mut self) -> Result<()> {
        let connection = self.ensure_connected().await?;
        let mut txn = connection.start_txn().await?;

        let clear_rels = query("MATCH ()-[r]-() DELETE r");
        if let Err(e) = txn.run(clear_rels).await {
            debug!("Error clearing relationships: {:?}", e);
            txn.rollback().await?;
            return Err(anyhow::anyhow!("Neo4j relationship deletion error: {}", e));
        }

        let clear_nodes = query("MATCH (n) DELETE n");
        if let Err(e) = txn.run(clear_nodes).await {
            debug!("Error clearing nodes: {:?}", e);
            txn.rollback().await?;
            return Err(anyhow::anyhow!("Neo4j node deletion error: {}", e));
        }

        txn.commit().await?;
        Ok(())
    }

    async fn execute_with_transaction<F, T>(&mut self, operation: F) -> Result<T>
    where
        F: FnOnce(&mut TransactionManager) -> Result<T>,
    {
        let connection = self.ensure_connected().await?;
        let mut txn_manager = TransactionManager::new(&connection);

        let result = operation(&mut txn_manager);

        if result.is_ok() {
            match txn_manager.execute().await {
                Ok(_) => result,
                Err(e) => Err(e),
            }
        } else {
            result
        }
    }

    pub async fn get_repository_hash(&mut self, repo_url: &str) -> Result<String> {
        let connection = self.ensure_connected().await?;
        let (query_str, params) = get_repository_hash_query(repo_url);
        let mut query_obj = query(&query_str);
        for (key, value) in &params {
            query_obj = query_obj.param(key, value.as_str());
        }
        let mut result = connection.execute(query_obj).await?;
        if let Some(row) = result.next().await? {
            Ok(row.get::<String>("hash").unwrap_or_default())
        } else {
            Err(anyhow::anyhow!("No hash found for repo"))
        }
    }

    pub async fn remove_nodes_by_file(&mut self, file_path: &str) -> Result<u32> {
        let connection = self.ensure_connected().await?;
        let (query_str, params) = remove_nodes_by_file_query(file_path);
        let mut query_obj = query(&query_str);
        for (k, v) in &params {
            query_obj = query_obj.param(k, v.as_str());
        }
        let mut result = connection.execute(query_obj).await?;
        if let Some(row) = result.next().await? {
            Ok(row.get::<u32>("count").unwrap_or(0))
        } else {
            Ok(0)
        }
    }

    pub async fn update_repository_hash(&mut self, repo_name: &str, new_hash: &str) -> Result<()> {
        self.execute_with_transaction(|txn_manager| {
            let (query, params) = update_repository_hash_query(repo_name, new_hash);
            txn_manager.add_edge_query((query, params));
            Ok(())
        })
        .await
    }

    pub async fn get_incoming_edges_for_file(
        &mut self,
        file: &str,
    ) -> Result<Vec<(Edge, NodeData)>> {
        let connection = self.ensure_connected().await?;
        let query_str = r#"
                MATCH (source)-[r]->(target)
                WHERE target.file = $file AND source.file <> $file
                RETURN source, r, target, labels(source)[0] as source_type, labels(target)[0] as target_type, type(r) as edge_type
            "#;
        let query_obj = query(query_str).param("file", file);
        let mut incoming = Vec::new();
        let mut result = connection.execute(query_obj).await?;
        while let Some(row) = result.next().await? {
            if let (
                Ok(source_node),
                Ok(target_node),
                Ok(source_type),
                Ok(target_type),
                Ok(edge_type),
            ) = (
                row.get::<neo4rs::Node>("source"),
                row.get::<neo4rs::Node>("target"),
                row.get::<String>("source_type"),
                row.get::<String>("target_type"),
                row.get::<String>("edge_type"),
            ) {
                if let (Ok(source_type), Ok(target_type), Ok(edge_type)) = (
                    NodeType::from_str(&source_type),
                    NodeType::from_str(&target_type),
                    EdgeType::from_str(&edge_type),
                ) {
                    let source_data = extract_node_data_from_neo4j_node(&source_node);
                    let target_data = extract_node_data_from_neo4j_node(&target_node);
                    let source_ref = NodeRef::from(NodeKeys::from(&source_data), source_type);
                    let target_ref = NodeRef::from(NodeKeys::from(&target_data), target_type);
                    let edge = Edge::new(edge_type, source_ref, target_ref);
                    incoming.push((edge, target_data));
                }
            }
        }
        Ok(incoming)
    }

    pub async fn all_nodes(&mut self) -> Result<Vec<NodeData>> {
        let connection = self.ensure_connected().await?;
        let query_str = "MATCH (n) RETURN n, labels(n)[0] as node_type";
        let mut nodes = Vec::new();
        let mut result = connection.execute(query(query_str)).await?;
        while let Some(row) = result.next().await? {
            if let Ok(node) = row.get::<neo4rs::Node>("n") {
                nodes.push(extract_node_data_from_neo4j_node(&node));
            }
        }
        Ok(nodes)
    }

    pub async fn all_edges(&mut self) -> Result<Vec<Edge>> {
        let connection = self.ensure_connected().await?;
        let query_str = "MATCH (source)-[r]->(target) \
            RETURN source, r, target, labels(source)[0] as source_type, labels(target)[0] as target_type, type(r) as edge_type";
        let mut edges = Vec::new();
        let mut result = connection.execute(query(query_str)).await?;
        while let Some(row) = result.next().await? {
            if let (
                Ok(source_node),
                Ok(target_node),
                Ok(source_type),
                Ok(target_type),
                Ok(edge_type),
            ) = (
                row.get::<neo4rs::Node>("source"),
                row.get::<neo4rs::Node>("target"),
                row.get::<String>("source_type"),
                row.get::<String>("target_type"),
                row.get::<String>("edge_type"),
            ) {
                if let (Ok(source_type), Ok(target_type), Ok(edge_type)) = (
                    NodeType::from_str(&source_type),
                    NodeType::from_str(&target_type),
                    EdgeType::from_str(&edge_type),
                ) {
                    let source_data = extract_node_data_from_neo4j_node(&source_node);
                    let target_data = extract_node_data_from_neo4j_node(&target_node);
                    let source_ref = NodeRef::from(NodeKeys::from(&source_data), source_type);
                    let target_ref = NodeRef::from(NodeKeys::from(&target_data), target_type);
                    let edge = Edge::new(edge_type, source_ref, target_ref);
                    edges.push(edge);
                }
            }
        }
        Ok(edges)
    }
}

impl Default for Neo4jGraph {
    fn default() -> Self {
        Neo4jGraph {
            connection: None,
            config: Neo4jConfig::default(),
            connected: Arc::new(Mutex::new(false)),
        }
    }
}

impl std::fmt::Debug for Neo4jGraph {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Neo4jGraph")
            .field("config", &self.config)
            .field("connected", &self.connected)
            .field("connection", &"<Neo4jConnection>")
            .finish()
    }
}

impl Neo4jGraph {
    fn _new() -> Self {
        Neo4jGraph {
            connection: None,
            config: Neo4jConfig::default(),
            connected: Arc::new(Mutex::new(false)),
        }
    }

    pub async fn add_node(&mut self, node_type: NodeType, node_data: NodeData) -> Result<()> {
        self.execute_with_transaction(|txn_manager| {
            txn_manager.add_node(&node_type, &node_data);
            Ok(())
        })
        .await
    }

    pub async fn add_edge(&mut self, edge: Edge) -> Result<()> {
        self.execute_with_transaction(|txn_manager| {
            txn_manager.add_edge(&edge);
            Ok(())
        })
        .await
    }

    pub async fn find_nodes_by_name(&self, node_type: NodeType, name: &str) -> Vec<NodeData> {
        let connection = self.get_connection();

        let (query, params) = find_nodes_by_name_query(&node_type, name);

        let query_builder = QueryBuilder::new(&query).with_params(params);
        let (query_str, params_map) = query_builder.build();

        match execute_node_query(&connection, query_str, params_map).await {
            Ok(nodes) => nodes,
            Err(e) => {
                debug!("Error finding nodes by name: {}", e);
                Vec::new()
            }
        }
    }

    pub async fn find_node_by_name_in_file(
        &self,
        node_type: NodeType,
        name: &str,
        file: &str,
    ) -> Option<NodeData> {
        let connection = self.get_connection();

        let (query, params) = find_node_by_name_file_query(&node_type, name, file);

        match execute_node_query(&connection, query, params).await {
            Ok(nodes) => nodes.into_iter().next(),
            Err(e) => {
                debug!("Error finding node by name and file: {}", e);
                None
            }
        }
    }

    pub async fn get_graph_size(&mut self) -> Result<(u32, u32)> {
        let connection = self.ensure_connected().await?;
        let query_str = count_nodes_edges_query();
        let mut result = connection.execute(query(&query_str)).await?;
        if let Some(row) = result.next().await? {
            let nodes = row.get::<u32>("nodes").unwrap_or(0);
            let edges = row.get::<u32>("edges").unwrap_or(0);
            Ok((nodes, edges))
        } else {
            Ok((0, 0))
        }
    }
    pub async fn analysis(&mut self) -> Result<()> {
        let connection = self.get_connection();
        let (nodes, edges) = self.get_graph_size().await?;
        println!("Graph contains {} nodes and {} edges", nodes, edges);

        let query_str = graph_node_analysis_query();
        match connection.execute(query(&query_str)).await {
            Ok(mut result) => {
                while let Some(row) = result.next().await? {
                    if let (Ok(node_type), Ok(name), Ok(file), Ok(start)) = (
                        row.get::<String>("node_type"),
                        row.get::<String>("name"),
                        row.get::<String>("file"),
                        row.get::<String>("start"),
                    ) {
                        println!("Node: \"{}\"-{}-{}-{}", node_type, name, file, start);
                    }
                }
            }
            Err(e) => {
                debug!("Error retrieving node details: {}", e);
            }
        }

        let query_str = graph_edges_analysis_query();
        match connection.execute(query(&query_str)).await {
            Ok(mut result) => {
                while let Some(row) = result.next().await? {
                    if let (
                        Ok(source_type),
                        Ok(source_name),
                        Ok(source_file),
                        Ok(source_start),
                        Ok(edge_type),
                        Ok(target_type),
                        Ok(target_name),
                        Ok(target_file),
                        Ok(target_start),
                    ) = (
                        row.get::<String>("source_type"),
                        row.get::<String>("source_name"),
                        row.get::<String>("source_file"),
                        row.get::<String>("source_start"),
                        row.get::<String>("edge_type"),
                        row.get::<String>("target_type"),
                        row.get::<String>("target_name"),
                        row.get::<String>("target_file"),
                        row.get::<String>("target_start"),
                    ) {
                        println!(
                            "From {}-{}-{}-{} to {}-{}-{}-{} : {}",
                            source_type,
                            source_name,
                            source_file,
                            source_start,
                            target_type,
                            target_name,
                            target_file,
                            target_start,
                            edge_type,
                        );
                    }
                }
            }
            Err(e) => {
                debug!("Error retrieving edge details: {}", e);
            }
        }
        Ok(())
    }

    pub async fn prefix_paths(&mut self, root: &str) -> Result<()> {
        self.execute_with_transaction(|txn_manager| {
            let (query, params) = prefix_paths_query(root);
            txn_manager.add_edge_query((query, params));
            Ok(())
        })
        .await
    }

    pub async fn extend_graph(&mut self, mut other: Self) -> Result<()> {
        let (other_nodes, other_edges) = other.get_graph_size().await?;
        if other_nodes == 0 && other_edges == 0 {
            warn!("Warning: Attempting to extend with an empty graph");
            return Ok(());
        }

        let target_connection = match self.ensure_connected().await {
            Ok(conn) => conn,
            Err(e) => return Err(e),
        };

        let source_connection = other.get_connection();

        let mut txn_manager = TransactionManager::new(&target_connection);

        let nodes_query = "MATCH (n) RETURN n, labels(n)[0] as node_type";

        if let Ok(mut result) = source_connection.execute(query(nodes_query)).await {
            while let Ok(Some(row)) = result.next().await {
                if let (Ok(node), Ok(node_type_str)) =
                    (row.get::<neo4rs::Node>("n"), row.get::<String>("node_type"))
                {
                    if let Ok(node_type) = NodeType::from_str(&node_type_str) {
                        let node_data = extract_node_data_from_neo4j_node(&node);
                        txn_manager.add_node(&node_type, &node_data);
                    }
                }
            }
        }

        let edges_query = "MATCH (source)-[r]->(target) 
                              RETURN source, type(r) as edge_type, r, target, 
                                     labels(source)[0] as source_type, 
                                     labels(target)[0] as target_type";

        if let Ok(mut result) = source_connection.execute(query(edges_query)).await {
            while let Ok(Some(row)) = result.next().await {
                if let (
                    Ok(source_node),
                    Ok(edge_type_str),
                    Ok(target_node),
                    Ok(source_type_str),
                    Ok(target_type_str),
                ) = (
                    row.get::<neo4rs::Node>("source"),
                    row.get::<String>("edge_type"),
                    row.get::<neo4rs::Node>("target"),
                    row.get::<String>("source_type"),
                    row.get::<String>("target_type"),
                ) {
                    if let (Ok(source_type), Ok(target_type)) = (
                        NodeType::from_str(&source_type_str),
                        NodeType::from_str(&target_type_str),
                    ) {
                        if let Ok(edge_type) = EdgeType::from_str(&edge_type_str) {
                            let source_data = extract_node_data_from_neo4j_node(&source_node);
                            let target_data = extract_node_data_from_neo4j_node(&target_node);

                            let source_ref =
                                NodeRef::from(NodeKeys::from(&source_data), source_type);
                            let target_ref =
                                NodeRef::from(NodeKeys::from(&target_data), target_type);

                            let edge = Edge::new(edge_type, source_ref, target_ref);

                            txn_manager.add_edge(&edge);
                        }
                    }
                }
            }
        }

        if let Err(e) = txn_manager.execute().await {
            debug!("Error in extend_graph transaction: {:?}", e);
            return Err(e);
        }

        Ok(())
    }

    pub async fn find_nodes_with_edge_type(
        &mut self,
        source_type: NodeType,
        target_type: NodeType,
        edge_type: EdgeType,
    ) -> Result<Vec<(NodeData, NodeData)>> {
        let connection = self.ensure_connected().await?;
        let (query_str, params) =
            find_nodes_with_edge_type_query(&source_type, &target_type, &edge_type);

        let mut query_obj = query(&query_str);
        for (key, value) in &params {
            query_obj = query_obj.param(&key, value.as_str());
        }
        let mut node_pairs = Vec::new();
        let mut result = connection.execute(query_obj).await?;
        while let Some(row) = result.next().await? {
            let source_name: String = row.get("source_name").unwrap_or_default();
            let source_file: String = row.get("source_file").unwrap_or_default();
            let source_start: i32 = row.get("source_start").unwrap_or_default();
            let target_name: String = row.get("target_name").unwrap_or_default();
            let target_file: String = row.get("target_file").unwrap_or_default();
            let target_start: i32 = row.get("target_start").unwrap_or_default();

            let source_node = NodeData {
                name: source_name,
                file: source_file,
                start: source_start as usize,
                ..Default::default()
            };
            let target_node = NodeData {
                name: target_name,
                file: target_file,
                start: target_start as usize,
                ..Default::default()
            };
            node_pairs.push((source_node, target_node));
        }
        Ok(node_pairs)
    }
}
