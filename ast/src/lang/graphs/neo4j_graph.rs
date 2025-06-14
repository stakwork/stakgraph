use super::{neo4j_utils::*, *};
use crate::utils::sync_fn;
use crate::{
    lang::{Function, FunctionCall},
    Lang,
};
use anyhow::Result;
use neo4rs::{query, Graph as Neo4jConnection};
use std::str::FromStr;
use std::{
    collections::HashSet,
    sync::{Arc, Mutex},
    time::Duration,
};
use tracing::{debug, info};

use super::neo4j_utils::Neo4jConnectionManager;

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
        for (key, value) in params.value.iter() {
            query_obj = query_obj.param(key.value.as_str(), value.clone());
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
        for (k, v) in params.value.iter() {
            query_obj = query_obj.param(k.value.as_str(), v.clone());
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
            txn_manager.add_query((query, params));
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
                    let source_data = NodeData::try_from(&source_node).unwrap();
                    let target_data = NodeData::try_from(&target_node).unwrap();
                    let source_ref = NodeRef::from(NodeKeys::from(&source_data), source_type);
                    let target_ref = NodeRef::from(NodeKeys::from(&target_data), target_type);
                    let edge = Edge::new(edge_type, source_ref, target_ref);
                    incoming.push((edge, target_data));
                }
            }
        }
        Ok(incoming)
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
    pub async fn add_node_async(&mut self, node_type: NodeType, node_data: NodeData) -> Result<()> {
        self.execute_with_transaction(|txn_manager| {
            txn_manager.add_node(&node_type, &node_data);
            Ok(())
        })
        .await
    }

    pub async fn add_edge_async(&mut self, edge: Edge) -> Result<()> {
        self.execute_with_transaction(|txn_manager| {
            txn_manager.add_edge(&edge);
            Ok(())
        })
        .await
    }

    pub async fn find_nodes_by_name_async(&self, node_type: NodeType, name: &str) -> Vec<NodeData> {
        let connection = self.get_connection();

        let (query_str, params_map) = find_nodes_by_name_query(&node_type, name);
        execute_node_query(&connection, query_str, params_map).await
    }

    pub async fn find_node_by_name_in_file_async(
        &self,
        node_type: NodeType,
        name: &str,
        file: &str,
    ) -> Option<NodeData> {
        let connection = self.get_connection();

        let (query, params) = find_node_by_name_file_query(&node_type, name, file);

        execute_node_query(&connection, query, params)
            .await
            .into_iter()
            .next()
    }

    pub async fn get_graph_size_async(&self) -> Result<(u32, u32)> {
        let connection = self.get_connection();
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
    pub async fn analysis_async(&self) -> Result<()> {
        let connection = self.get_connection();
        let (nodes, edges) = self.get_graph_size();
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
    pub async fn count_edges_of_type_async(&self, edge_type: EdgeType) -> usize {
        let connection = self.get_connection();

        let (query_str, params) = count_edges_by_type_query(&edge_type);
        let mut query_obj = query(&query_str);
        for (key, value) in params.value.iter() {
            query_obj = query_obj.param(key.value.as_str(), value.clone());
        }
        match connection.execute(query_obj).await {
            Ok(mut result) => {
                if let Ok(Some(row)) = result.next().await {
                    row.get::<usize>("count").unwrap_or(0)
                } else {
                    0
                }
            }
            Err(e) => {
                debug!("Error counting edges by type: {}", e);
                0
            }
        }
    }

    pub async fn find_nodes_by_type_async(&self, node_type: NodeType) -> Vec<NodeData> {
        let connection = self.get_connection();
        let (query, params) = find_nodes_by_type_query(&node_type);
        execute_node_query(&connection, query, params).await
    }

    pub async fn find_nodes_by_name_contains_async(
        &self,
        node_type: NodeType,
        name_part: &str,
    ) -> Vec<NodeData> {
        let connection = self.get_connection();
        let (query, params) = find_nodes_by_name_contains_query(&node_type, name_part);
        execute_node_query(&connection, query, params).await
    }

    pub async fn find_nodes_by_file_ends_with_async(
        &self,
        node_type: NodeType,
        file: &str,
    ) -> Vec<NodeData> {
        let connection = self.get_connection();
        let (query, params) = find_nodes_by_file_pattern_query(&node_type, file);
        execute_node_query(&connection, query, params).await
    }

    pub async fn find_nodes_with_edge_type_async(
        &self,
        source_type: NodeType,
        target_type: NodeType,
        edge_type: EdgeType,
    ) -> Vec<(NodeData, NodeData)> {
        let connection = self.get_connection();
        let (query_str, params) =
            find_nodes_with_edge_type_query(&source_type, &target_type, &edge_type);

        let mut query_obj = query(&query_str);
        for (key, value) in params.value.iter() {
            query_obj = query_obj.param(key.value.as_str(), value.clone());
        }
        let mut node_pairs = Vec::new();
        match connection.execute(query_obj).await {
            Ok(mut result) => {
                while let Ok(Some(row)) = result.next().await {
                    let source_name: String = row.get("source_name").unwrap_or_default();
                    let source_file: String = row.get("source_file").unwrap_or_default();
                    let source_start: i64 = row.get("source_start").unwrap_or_default();
                    let target_name: String = row.get("target_name").unwrap_or_default();
                    let target_file: String = row.get("target_file").unwrap_or_default();
                    let target_start: i64 = row.get("target_start").unwrap_or_default();

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
            }
            Err(e) => {
                debug!("Error executing find_nodes_with_edge_type query: {}", e);
            }
        }
        node_pairs
    }
    pub async fn find_endpoint_async(
        &self,
        name: &str,
        file: &str,
        verb: &str,
    ) -> Option<NodeData> {
        let connection = self.get_connection();
        let (query, params) = find_endpoint_query(name, file, verb);

        let nodes = execute_node_query(&connection, query, params).await;
        nodes.into_iter().next()
    }
    pub async fn find_resource_nodes_async(
        &self,
        node_type: NodeType,
        verb: &str,
        path: &str,
    ) -> Vec<NodeData> {
        let connection = self.get_connection();
        let (query, params) = find_resource_nodes_query(&node_type, verb, path);

        execute_node_query(&connection, query, params).await
    }
    pub async fn find_handlers_for_endpoint_async(&self, endpoint: &NodeData) -> Vec<NodeData> {
        let connection = self.get_connection();
        let (query, params) = find_handlers_for_endpoint_query(endpoint);

        execute_node_query(&connection, query, params).await
    }

    pub async fn check_direct_data_model_usage_async(
        &self,
        function_name: &str,
        data_model: &str,
    ) -> bool {
        let connection = self.get_connection();
        let (query_str, params) = check_direct_data_model_usage_query(function_name, data_model);

        let mut query_obj = query(&query_str);
        for (key, value) in params.value.iter() {
            query_obj = query_obj.param(key.value.as_str(), value.clone());
        }

        if let Ok(mut result) = connection.execute(query_obj).await {
            if let Ok(Some(row)) = result.next().await {
                return row.get::<bool>("exists").unwrap_or(false);
            }
        }
        false
    }
    pub async fn find_functions_called_by_async(&self, function: &NodeData) -> Vec<NodeData> {
        let connection = self.get_connection();
        let (query, params) = find_functions_called_by_query(function);

        execute_node_query(&connection, query, params).await
    }
    pub async fn find_source_edge_by_name_and_file_async(
        &self,
        edge_type: EdgeType,
        target_name: &str,
        target_file: &str,
    ) -> Option<NodeKeys> {
        let connection = self.get_connection();
        let (query_str, params) =
            find_source_edge_by_name_and_file_query(&edge_type, target_name, target_file);

        let mut query_obj = query(&query_str);
        for (key, value) in params.value.iter() {
            query_obj = query_obj.param(key.value.as_str(), value.clone());
        }

        if let Ok(mut result) = connection.execute(query_obj).await {
            if let Ok(Some(row)) = result.next().await {
                let name = row.get::<String>("name").unwrap_or_default();
                let file = row.get::<String>("file").unwrap_or_default();
                let start = row.get::<i64>("start").unwrap_or_default() as usize;
                let verb = row.get::<String>("verb").ok();

                return Some(NodeKeys {
                    name,
                    file,
                    start,
                    verb,
                });
            }
        }
        None
    }
    pub async fn find_node_in_range_async(
        &self,
        node_type: NodeType,
        row: u32,
        file: &str,
    ) -> Option<NodeData> {
        let connection = self.get_connection();
        let (query, params) = find_nodes_in_range_query(&node_type, file, row);

        let nodes = execute_node_query(&connection, query, params).await;
        nodes.into_iter().next()
    }
    pub async fn find_node_at_async(
        &self,
        node_type: NodeType,
        file: &str,
        line: u32,
    ) -> Option<NodeData> {
        let connection = self.get_connection();
        let (query, params) = find_node_at_query(&node_type, file, line);
        let nodes = execute_node_query(&connection, query, params).await;
        nodes.into_iter().next()
    }
    pub async fn find_node_by_name_and_file_end_with_async(
        &self,
        node_type: NodeType,
        name: &str,
        suffix: &str,
    ) -> Option<NodeData> {
        let nodes = self.find_nodes_by_name_async(node_type, name).await;
        nodes.into_iter().find(|node| node.file.ends_with(suffix))
    }
    pub async fn prefix_paths_async(&mut self, root: &str) -> Result<()> {
        let connection = self.ensure_connected().await?;
        let (query_str, params) = prefix_paths_query(root);

        let mut query_obj = query(&query_str);
        for (key, value) in params.value.iter() {
            query_obj = query_obj.param(key.value.as_str(), value.clone());
        }

        let _ = connection.execute(query_obj).await?;
        Ok(())
    }

    pub async fn add_node_with_parent_async(
        &mut self,
        node_type: NodeType,
        node_data: NodeData,
        parent_type: NodeType,
        parent_file: &str,
    ) -> Result<()> {
        let connection = self.ensure_connected().await?;
        let queries = add_node_with_parent_query(&node_type, &node_data, &parent_type, parent_file);

        let mut txn_manager = TransactionManager::new(&connection);
        for query in queries {
            txn_manager.add_query(query);
        }

        let _ = txn_manager.execute().await;
        Ok(())
    }
    pub async fn class_inherits_async(&mut self) -> Result<()> {
        let connection = self.ensure_connected().await?;
        let query_str = class_inherits_query();

        let _ = connection.execute(query(&query_str)).await?;
        Ok(())
    }

    pub async fn class_includes_async(&mut self) -> Result<()> {
        let connection = self.ensure_connected().await?;
        let query_str = class_includes_query();

        let _ = connection.execute(query(&query_str)).await?;
        Ok(())
    }
    pub async fn add_instances_async(&mut self, nodes: Vec<NodeData>) -> Result<()> {
        let connection = self.ensure_connected().await?;
        let mut txn_manager = TransactionManager::new(&connection);

        for node in nodes {
            txn_manager.add_node(&NodeType::Instance, &node);
        }

        let _ = txn_manager.execute().await;
        Ok(())
    }
    pub async fn add_functions_async(&mut self, functions: Vec<Function>) -> Result<()> {
        let connection = self.ensure_connected().await?;
        let mut txn_manager = TransactionManager::new(&connection);

        for (function_node, method_of, reqs, dms, trait_operand, return_types) in &functions {
            let queries = add_functions_query(
                function_node,
                method_of.as_ref(),
                reqs,
                dms,
                trait_operand.as_ref(),
                return_types,
            );
            for query in queries {
                txn_manager.add_query(query);
            }
        }

        let _ = txn_manager.execute().await;
        Ok(())
    }
    pub async fn add_page_async(&mut self, page: (NodeData, Option<Edge>)) -> Result<()> {
        let connection = self.ensure_connected().await?;
        let queries = add_page_query(&page.0, &page.1);

        let mut txn_manager = TransactionManager::new(&connection);
        for query in queries {
            txn_manager.add_query(query);
        }

        let _ = txn_manager.execute().await;
        Ok(())
    }

    pub async fn add_pages_async(&mut self, pages: Vec<(NodeData, Vec<Edge>)>) -> Result<()> {
        let connection = self.ensure_connected().await?;
        let queries = add_pages_query(&pages);

        let mut txn_manager = TransactionManager::new(&connection);
        for query in queries {
            txn_manager.add_query(query);
        }

        let _ = txn_manager.execute().await;
        Ok(())
    }
    pub async fn add_endpoints_async(
        &mut self,
        endpoints: Vec<(NodeData, Option<Edge>)>,
    ) -> Result<()> {
        let connection = self.ensure_connected().await?;
        let queries = add_endpoints_query(&endpoints);

        let mut txn_manager = TransactionManager::new(&connection);
        for query in queries {
            txn_manager.add_query(query);
        }

        let _ = txn_manager.execute().await;
        Ok(())
    }

    pub async fn add_test_node_async(
        &mut self,
        test_data: NodeData,
        test_type: NodeType,
        test_edge: Option<Edge>,
    ) -> Result<()> {
        let connection = self.ensure_connected().await?;
        let queries = add_test_node_query(&test_data, &test_type, &test_edge);

        let mut txn_manager = TransactionManager::new(&connection);
        for query in queries {
            txn_manager.add_query(query);
        }

        let _ = txn_manager.execute().await;
        Ok(())
    }
    pub async fn add_calls_async(
        &mut self,
        calls: (Vec<FunctionCall>, Vec<FunctionCall>, Vec<Edge>, Vec<Edge>),
    ) -> Result<()> {
        let connection = self.ensure_connected().await?;
        let queries = add_calls_query(&calls.0, &calls.1, &calls.3);

        let mut txn_manager = TransactionManager::new(&connection);
        for query in queries {
            txn_manager.add_query(query);
        }

        let _ = txn_manager.execute().await;
        Ok(())
    }

    pub async fn get_graph_keys_async(&self) -> (HashSet<String>, HashSet<String>) {
        let connection = self.get_connection();
        let mut node_keys = HashSet::new();
        let mut edge_keys = HashSet::new();

        let (node_query, edge_query) = all_nodes_and_edges_query();
        if let Ok(mut result) = connection.execute(query(&node_query)).await {
            while let Ok(Some(row)) = result.next().await {
                if let Ok(key) = row.get::<String>("key") {
                    node_keys.insert(key);
                }
            }
        }

        if let Ok(mut result) = connection.execute(query(&edge_query)).await {
            while let Ok(Some(row)) = result.next().await {
                if let Ok(edge_type) = row.get::<String>("edge_type") {
                    edge_keys.insert(edge_type);
                }
            }
        }

        (node_keys, edge_keys)
    }

    pub async fn filter_out_nodes_without_children_async(
        &mut self,
        parent_type: NodeType,
        child_type: NodeType,
        child_meta_key: &str,
    ) -> Result<()> {
        let connection = self.ensure_connected().await?;
        let queries =
            filter_out_nodes_without_children_query(parent_type, child_type, child_meta_key);

        let mut txn_manager = TransactionManager::new(&connection);
        txn_manager.add_query(queries);
        let _ = txn_manager.execute().await;

        Ok(())
    }
}

impl Graph for Neo4jGraph {
    fn new() -> Self
    where
        Self: Sized,
    {
        Self::default()
    }
    fn with_capacity(_nodes: usize, _edges: usize) -> Self
    where
        Self: Sized,
    {
        Self::default()
    }
    fn analysis(&self) {
        let _ = sync_fn(|| async { self.analysis_async().await });
    }
    fn create_filtered_graph(&self, _final_filter: &[String]) -> Self
    where
        Self: Sized,
    {
        todo!("To be implemented in Neo4jGraph");
    }

    fn extend_graph(&mut self, _other: Self)
    where
        Self: Sized,
    {
        todo!("To be implemented in Neo4jGraph");
    }

    fn get_graph_size(&self) -> (u32, u32) {
        sync_fn(|| async { self.get_graph_size_async().await.unwrap_or_default() })
    }

    fn find_nodes_by_name(&self, node_type: NodeType, name: &str) -> Vec<NodeData> {
        sync_fn(|| async { self.find_nodes_by_name_async(node_type, name).await })
    }
    fn add_node_with_parent(
        &mut self,
        node_type: NodeType,
        node_data: NodeData,
        parent_type: NodeType,
        parent_file: &str,
    ) {
        sync_fn(|| async {
            self.add_node_with_parent_async(node_type, node_data, parent_type, parent_file)
                .await
                .unwrap()
        });
    }
    fn add_edge(&mut self, edge: Edge) {
        sync_fn(|| async { self.add_edge_async(edge).await.unwrap_or_default() });
    }
    fn add_node(&mut self, node_type: NodeType, node_data: NodeData) {
        sync_fn(|| async {
            self.add_node_async(node_type, node_data)
                .await
                .unwrap_or_default()
        })
    }
    fn get_graph_keys(&self) -> (HashSet<String>, HashSet<String>) {
        sync_fn(|| async { self.get_graph_keys_async().await })
    }

    fn find_source_edge_by_name_and_file(
        &self,
        edge_type: EdgeType,
        target_name: &str,
        target_file: &str,
    ) -> Option<NodeKeys> {
        sync_fn(|| async {
            self.find_source_edge_by_name_and_file_async(edge_type, target_name, target_file)
                .await
        })
    }
    fn process_endpoint_groups(&mut self, _eg: Vec<NodeData>, _lang: &Lang) -> Result<()> {
        todo!("To be implemented in Neo4jGraph");
    }
    fn class_inherits(&mut self) {
        sync_fn(|| async { self.class_inherits_async().await.unwrap_or_default() });
    }
    fn class_includes(&mut self) {
        sync_fn(|| async { self.class_includes_async().await.unwrap_or_default() });
    }
    fn add_instances(&mut self, nodes: Vec<NodeData>) {
        sync_fn(|| async { self.add_instances_async(nodes).await.unwrap_or_default() });
    }
    fn add_functions(&mut self, functions: Vec<Function>) {
        sync_fn(|| async {
            self.add_functions_async(functions)
                .await
                .unwrap_or_default()
        });
    }
    fn add_page(&mut self, page: (NodeData, Option<Edge>)) {
        sync_fn(|| async { self.add_page_async(page).await.unwrap_or_default() });
    }
    fn add_pages(&mut self, pages: Vec<(NodeData, Vec<Edge>)>) {
        sync_fn(|| async { self.add_pages_async(pages).await.unwrap_or_default() });
    }
    fn add_endpoints(&mut self, endpoints: Vec<(NodeData, Option<Edge>)>) {
        sync_fn(|| async {
            self.add_endpoints_async(endpoints)
                .await
                .unwrap_or_default()
        });
    }
    fn add_test_node(&mut self, test_data: NodeData, test_type: NodeType, test_edge: Option<Edge>) {
        sync_fn(|| async {
            self.add_test_node_async(test_data, test_type, test_edge)
                .await
                .unwrap_or_default()
        });
    }
    fn add_calls(&mut self, calls: (Vec<FunctionCall>, Vec<FunctionCall>, Vec<Edge>, Vec<Edge>)) {
        sync_fn(|| async { self.add_calls_async(calls).await.unwrap_or_default() });
    }
    fn filter_out_nodes_without_children(
        &mut self,
        parent_type: NodeType,
        child_type: NodeType,
        child_meta_key: &str,
    ) {
        sync_fn(|| async {
            self.filter_out_nodes_without_children_async(parent_type, child_type, child_meta_key)
                .await
                .unwrap_or_default()
        });
    }
    fn get_data_models_within(&mut self, _lang: &Lang) {
        todo!("To be implemented in Neo4jGraph");
    }
    fn prefix_paths(&mut self, root: &str) {
        sync_fn(|| async { self.prefix_paths_async(root).await.unwrap_or_default() });
    }

    fn find_endpoint(&self, name: &str, file: &str, verb: &str) -> Option<NodeData> {
        sync_fn(|| async { self.find_endpoint_async(name, file, verb).await })
    }

    fn find_resource_nodes(&self, node_type: NodeType, verb: &str, path: &str) -> Vec<NodeData> {
        sync_fn(|| async { self.find_resource_nodes_async(node_type, verb, path).await })
    }
    fn find_handlers_for_endpoint(&self, endpoint: &NodeData) -> Vec<NodeData> {
        sync_fn(|| async { self.find_handlers_for_endpoint_async(endpoint).await })
    }
    fn check_direct_data_model_usage(&self, function_name: &str, data_model: &str) -> bool {
        sync_fn(|| async {
            self.check_direct_data_model_usage_async(function_name, data_model)
                .await
        })
    }
    fn find_functions_called_by(&self, function: &NodeData) -> Vec<NodeData> {
        sync_fn(|| async { self.find_functions_called_by_async(function).await })
    }
    fn find_nodes_by_type(&self, node_type: NodeType) -> Vec<NodeData> {
        sync_fn(|| async { self.find_nodes_by_type_async(node_type).await })
    }
    fn find_nodes_with_edge_type(
        &self,
        source_type: NodeType,
        target_type: NodeType,
        edge_type: EdgeType,
    ) -> Vec<(NodeData, NodeData)> {
        sync_fn(|| async {
            self.find_nodes_with_edge_type_async(source_type, target_type, edge_type)
                .await
        })
    }
    fn count_edges_of_type(&self, edge_type: EdgeType) -> usize {
        sync_fn(|| async { self.count_edges_of_type_async(edge_type).await })
    }
    fn find_nodes_by_name_contains(&self, node_type: NodeType, name: &str) -> Vec<NodeData> {
        sync_fn(|| async {
            self.find_nodes_by_name_contains_async(node_type, name)
                .await
        })
    }

    fn find_node_by_name_in_file(
        &self,
        node_type: NodeType,
        name: &str,
        file: &str,
    ) -> Option<NodeData> {
        sync_fn(|| async {
            self.find_node_by_name_in_file_async(node_type, name, file)
                .await
        })
    }

    fn find_nodes_by_file_ends_with(&self, node_type: NodeType, file: &str) -> Vec<NodeData> {
        sync_fn(|| async {
            self.find_nodes_by_file_ends_with_async(node_type, file)
                .await
        })
    }

    fn find_node_by_name_and_file_end_with(
        &self,
        node_type: NodeType,
        name: &str,
        suffix: &str,
    ) -> Option<NodeData> {
        sync_fn(|| async {
            self.find_node_by_name_and_file_end_with_async(node_type, name, suffix)
                .await
        })
    }

    fn find_node_in_range(&self, node_type: NodeType, row: u32, file: &str) -> Option<NodeData> {
        sync_fn(|| async { self.find_node_in_range_async(node_type, row, file).await })
    }

    fn find_node_at(&self, node_type: NodeType, file: &str, line: u32) -> Option<NodeData> {
        sync_fn(|| async { self.find_node_at_async(node_type, file, line).await })
    }
}
