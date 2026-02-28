use crate::lang::graphs::{
    connection::Neo4jConnectionManager, executor::*, helpers::*, operations::*, queries::*,
};
use crate::lang::{
    asg::TestRecord, Calls, Edge, EdgeType, Graph, NodeData, NodeKeys, NodeType, TestFilters,
};
use crate::utils::{create_node_key, sync_fn};
use crate::{lang::Function, lang::Node, Lang};
use lsp::Language;
use neo4rs::{query, BoltMap, Graph as Neo4jConnection};
use shared::{Context, Error, Result};
use std::str::FromStr;
use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
    time::Duration,
};
use tokio::sync::Mutex;
use tracing::{debug, info, warn};

#[derive(Clone)]
pub struct Neo4jGraph {
    pub connection: Arc<Mutex<Option<Neo4jConnection>>>,
    pub config: Neo4jConfig,
    pub root: String,
    pub lang_kind: Language,
}

#[derive(Clone, Debug)]
pub struct Neo4jConfig {
    pub uri: String,
    pub username: String,
    pub password: String,
    pub database: String,
    pub connection_timeout: Duration,
    pub max_connections: usize,
}

impl Default for Neo4jConfig {
    fn default() -> Self {
        Neo4jConfig {
            uri: std::env::var("NEO4J_URI").unwrap_or_else(|_| "bolt://localhost:7687".to_string()),
            username: std::env::var("NEO4J_USERNAME").unwrap_or_else(|_| "neo4j".to_string()),
            password: std::env::var("NEO4J_PASSWORD").unwrap_or_else(|_| "testtest".to_string()),
            database: std::env::var("NEO4J_DATABASE").unwrap_or_else(|_| "neo4j".to_string()),
            connection_timeout: Duration::from_secs(30),
            max_connections: 10,
        }
    }
}

impl Default for Neo4jGraph {
    fn default() -> Self {
        Neo4jGraph {
            connection: Arc::new(Mutex::new(None)),
            config: Neo4jConfig::default(),
            root: String::new(),
            lang_kind: Language::Typescript,
        }
    }
}

impl std::fmt::Debug for Neo4jGraph {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Neo4jGraph")
            .field("config", &self.config)
            .field("connection", &"<Neo4jConnection>")
            .finish()
    }
}

impl Neo4jGraph {
    pub fn with_config(config: Neo4jConfig, root: String, lang_kind: Language) -> Self {
        Neo4jGraph {
            connection: Arc::new(Mutex::new(None)),
            config,
            root,
            lang_kind,
        }
    }

    pub async fn connect(&self) -> Result<()> {
        if self.connection.lock().await.is_some() {
            debug!("Already connected to Neo4j database");
            return Ok(());
        }

        // info!("Connecting to Neo4j database at {}", self.config.uri);

        //global connection manager
        let conn = Neo4jConnectionManager::initialize(
            &self.config.uri,
            &self.config.username,
            &self.config.password,
            &self.config.database,
        )
        .await?;

        *self.connection.lock().await = Some(conn);
        // info!("Successfully connected to Neo4j database");
        Ok(())
    }

    pub async fn disconnect(&self) -> Result<()> {
        if self.connection.lock().await.is_none() {
            debug!("Not connected to Neo4j database");
            return Ok(());
        }

        *self.connection.lock().await = None;

        info!("Disconnected from Neo4j database");
        Ok(())
    }

    pub async fn is_connected(&self) -> bool {
        self.connection.lock().await.is_some()
    }

    pub async fn ensure_connected(&self) -> Result<Neo4jConnection> {
        if !self.is_connected().await {
            self.connect().await?;
        }
        self.connection
            .lock()
            .await
            .clone()
            .context("Neo4j Connection is not established")
    }

    pub async fn add_node_async(&self, node_type: NodeType, node_data: NodeData) -> Result<()> {
        let connection = self.ensure_connected().await?;
        let mut txn_manager = TransactionManager::new(&connection);

        txn_manager.add_node(&node_type, &node_data);

        txn_manager.execute().await
    }

    pub async fn add_edge_async(&self, edge: Edge) -> Result<()> {
        let connection = self.ensure_connected().await?;
        let mut txn_manager = TransactionManager::new(&connection);

        txn_manager.add_edge(&edge);

        txn_manager.execute().await
    }

    pub async fn add_node_with_parent_async(
        &self,
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

        txn_manager.execute().await
    }
}

// Graph Operations
impl Neo4jGraph {
    pub async fn find_nodes_by_name_contains_async(
        &self,
        node_type: NodeType,
        name_part: &str,
    ) -> Vec<NodeData> {
        let Ok(connection) = self.ensure_connected().await else {
            warn!("Failed to connect to Neo4j in find_nodes_by_name_contains_async");
            return vec![];
        };
        let (query, params) = find_nodes_by_name_contains_query(&node_type, name_part);
        let nodes = execute_node_query(&connection, query, params).await;

        let lang_nodes: Vec<NodeData> = nodes
            .into_iter()
            .filter(|n| self.lang_kind.is_from_language(&n.file))
            .collect();

        lang_nodes
    }
    pub async fn find_nodes_by_type_async(&self, node_type: NodeType) -> Vec<NodeData> {
        let Ok(connection) = self.ensure_connected().await else {
            warn!("Failed to connect to Neo4j in find_nodes_by_type_async");
            return vec![];
        };
        let (query, params) = find_nodes_by_type_query(&node_type);
        execute_node_query(&connection, query, params).await
    }
    pub async fn find_nodes_by_file_ends_with_async(
        &self,
        node_type: NodeType,
        file: &str,
    ) -> Vec<NodeData> {
        let Ok(connection) = self.ensure_connected().await else {
            warn!("Failed to connect to Neo4j in find_nodes_by_file_ends_with_async");
            return vec![];
        };
        let (query, params) = find_nodes_by_file_pattern_query(&node_type, file);
        let nodes = execute_node_query(&connection, query, params).await;
        let lang_nodes: Vec<NodeData> = nodes
            .into_iter()
            .filter(|n| self.lang_kind.is_from_language(&n.file))
            .collect();

        lang_nodes
    }

    pub async fn find_nodes_with_edge_type_async(
        &self,
        source_type: NodeType,
        target_type: NodeType,
        edge_type: EdgeType,
    ) -> Vec<(NodeData, NodeData)> {
        let Ok(connection) = self.ensure_connected().await else {
            warn!("Failed to connect to Neo4j in find_nodes_with_edge_type_async");
            return vec![];
        };
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

    pub async fn find_nodes_by_name_async(&self, node_type: NodeType, name: &str) -> Vec<NodeData> {
        let Ok(connection) = self.ensure_connected().await else {
            warn!("Failed to connect to Neo4j in find_nodes_by_name_async");
            return vec![];
        };

        let (query_str, params_map) = find_nodes_by_name_query(&node_type, name, &self.root);

        let nodes = execute_node_query(&connection, query_str, params_map).await;

        let lang_nodes: Vec<NodeData> = nodes
            .into_iter()
            .filter(|n| self.lang_kind.is_from_language(&n.file))
            .collect();

        lang_nodes
    }
    pub async fn find_nodes_by_name_any_language(
        &self,
        node_type: NodeType,
        name: &str,
    ) -> Vec<NodeData> {
        let Ok(connection) = self.ensure_connected().await else {
            warn!("Failed to connect to Neo4j in find_nodes_by_name_async");
            return vec![];
        };

        let (query_str, params_map) = find_nodes_by_name_query(&node_type, name, &self.root);

        let nodes = execute_node_query(&connection, query_str, params_map).await;

        nodes
    }

    pub async fn find_node_by_name_in_file_async(
        &self,
        node_type: NodeType,
        name: &str,
        file: &str,
    ) -> Option<NodeData> {
        let Ok(connection) = self.ensure_connected().await else {
            warn!("Failed to connect to Neo4j in find_node_by_name_in_file_async");
            return None;
        };

        let (query, params) = find_node_by_name_file_query(&node_type, name, file);

        let nodes = execute_node_query(&connection, query, params)
            .await
            .into_iter();

        let lang_nodes: Vec<NodeData> = nodes
            .filter(|n| self.lang_kind.is_from_language(&n.file))
            .collect();

        lang_nodes.into_iter().next()
    }

    pub async fn find_endpoint_async(
        &self,
        name: &str,
        file: &str,
        verb: &str,
    ) -> Option<NodeData> {
        let Ok(connection) = self.ensure_connected().await else {
            warn!("Failed to connect to Neo4j in find_endpoint_async");
            return None;
        };
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
        let Ok(connection) = self.ensure_connected().await else {
            warn!("Failed to connect to Neo4j in find_resource_nodes_async");
            return vec![];
        };
        let (query, params) = find_resource_nodes_query(&node_type, verb, path);

        execute_node_query(&connection, query, params).await
    }
    pub async fn find_handlers_for_endpoint_async(&self, endpoint: &NodeData) -> Vec<NodeData> {
        let Ok(connection) = self.ensure_connected().await else {
            warn!("Failed to connect to Neo4j in find_handlers_for_endpoint_async");
            return vec![];
        };
        let (query, params) = find_handlers_for_endpoint_query(endpoint);

        execute_node_query(&connection, query, params).await
    }

    pub async fn check_direct_data_model_usage_async(
        &self,
        function_name: &str,
        data_model: &str,
    ) -> bool {
        let Ok(connection) = self.ensure_connected().await else {
            warn!("Failed to connect to Neo4j in check_direct_data_model_usage_async");
            return false;
        };
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
        let Ok(connection) = self.ensure_connected().await else {
            warn!("Failed to connect to Neo4j in find_functions_called_by_async");
            return vec![];
        };
        let (query, params) = find_functions_called_by_query(function);

        execute_node_query(&connection, query, params).await
    }
    pub async fn find_source_edge_by_name_and_file_async(
        &self,
        edge_type: EdgeType,
        target_name: &str,
        target_file: &str,
    ) -> Option<NodeKeys> {
        let Ok(connection) = self.ensure_connected().await else {
            warn!("Failed to connect to Neo4j in find_source_edge_by_name_and_file_async");
            return None;
        };
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
                let verb = match row.get::<String>("verb") {
                    Ok(value) => Some(value),
                    Err(_) => None,
                };

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
        let Ok(connection) = self.ensure_connected().await else {
            warn!("Failed to connect to Neo4j in find_node_in_range_async");
            return None;
        };
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
        let Ok(connection) = self.ensure_connected().await else {
            warn!("Failed to connect to Neo4j in find_node_at_async");
            return None;
        };
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

    pub async fn find_node_by_name_and_file_contains_async(
        &self,
        node_type: NodeType,
        name: &str,
        file_part: &str,
    ) -> Option<NodeData> {
        let nodes = self.find_nodes_by_name_async(node_type, name).await;
        nodes.into_iter().find(|node| node.file.contains(file_part))
    }

    pub async fn class_inherits_async(&self) -> Result<()> {
        let connection = self.ensure_connected().await?;
        let mut txn_manager = TransactionManager::new(&connection);

        let query_str = class_inherits_query();
        txn_manager.add_query((query_str, BoltMap::new()));

        txn_manager.execute().await
    }

    pub async fn class_includes_async(&self) -> Result<()> {
        let connection = self.ensure_connected().await?;
        let mut txn_manager = TransactionManager::new(&connection);

        let query_str = class_includes_query();
        txn_manager.add_query((query_str, BoltMap::new()));

        txn_manager.execute().await
    }
}

// Operations

impl Neo4jGraph {
    pub async fn count_edges_of_type_async(&self, edge_type: EdgeType) -> usize {
        let Ok(connection) = self.ensure_connected().await else {
            warn!("Failed to connect to Neo4j in count_edges_of_type_async");
            return 0;
        };

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

    pub async fn get_graph_keys_async(&self) -> (HashSet<String>, HashSet<String>) {
        let Ok(connection) = self.ensure_connected().await else {
            warn!("Failed to connect to Neo4j in get_graph_keys_async");
            return (HashSet::new(), HashSet::new());
        };
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
                if let (Ok(source_key), Ok(target_key), Ok(edge_type)) = (
                    row.get::<String>("source_key"),
                    row.get::<String>("target_key"),
                    row.get::<String>("edge_type"),
                ) {
                    let edge_key = format!(
                        "{}-{}-{}",
                        source_key.to_lowercase(),
                        target_key.to_lowercase(),
                        edge_type
                    );
                    edge_keys.insert(edge_key);
                }
            }
        }

        (node_keys, edge_keys)
    }

    pub async fn get_graph_size_async(&self) -> Result<(u32, u32)> {
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
    pub async fn analysis_async(&self) -> Result<()> {
        let connection = self.ensure_connected().await?;

        let query_str = graph_node_analysis_query();
        match connection.execute(query(&query_str)).await {
            Ok(mut result) => {
                while let Some(row) = result.next().await? {
                    if let Ok(node_key) = row.get::<String>("node_key") {
                        println!("Node: {}", node_key);
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
                    if let (Ok(source_key), Ok(edge_type), Ok(target_key)) = (
                        row.get::<String>("source_key"),
                        row.get::<String>("edge_type"),
                        row.get::<String>("target_key"),
                    ) {
                        println!("Edge: {} - {} -> {}", source_key, edge_type, target_key);
                    }
                }
            }
            Err(e) => {
                debug!("Error retrieving edge details: {}", e);
            }
        }
        Ok(())
    }

    pub async fn filter_out_nodes_without_children_async(
        &self,
        parent_type: NodeType,
        child_type: NodeType,
        child_meta_key: &str,
    ) -> Result<()> {
        let connection = self.ensure_connected().await?;
        let mut txn_manager = TransactionManager::new(&connection);

        let (query, params) =
            filter_out_nodes_without_children_query(parent_type, child_type, child_meta_key);
        txn_manager.add_query((query, params));

        txn_manager.execute().await
    }

    pub async fn remove_node_async(&self, node_type: NodeType, node_data: NodeData) -> Result<()> {
        let connection = self.ensure_connected().await?;
        let mut txn_manager = TransactionManager::new(&connection);

        let (query, params) = remove_node_query(node_type, &node_data);
        txn_manager.add_query((query, params));

        txn_manager.execute().await
    }

    pub async fn deduplicate_nodes_async(
        &self,
        remove_type: NodeType,
        keep_type: NodeType,
        _operation: &str,
    ) -> Result<()> {
        let nodes_to_check = self.find_nodes_by_type_async(remove_type.clone()).await;
        let keep_nodes = self.find_nodes_by_type_async(keep_type.clone()).await;

        let mut keep_nodes_map: HashMap<(String, String), NodeData> = HashMap::new();
        for node in &keep_nodes {
            let key = (node.name.clone(), node.file.clone());
            keep_nodes_map.insert(key, node.clone());
        }

        let operand_edges = self
            .find_nodes_with_edge_type_async(
                keep_type.clone(),
                NodeType::Function,
                EdgeType::Operand,
            )
            .await;

        let mut nodes_with_methods: HashSet<(String, String)> = HashSet::new();
        for (src, _) in operand_edges {
            nodes_with_methods.insert((src.name.clone(), src.file.clone()));
        }

        for remove_node in nodes_to_check {
            let lookup_key = (remove_node.name.clone(), remove_node.file.clone());

            if keep_nodes_map.contains_key(&lookup_key) && nodes_with_methods.contains(&lookup_key)
            {
                self.remove_node_async(remove_type.clone(), remove_node)
                    .await?;
            }
        }

        Ok(())
    }

    pub async fn process_endpoint_groups_async(&self, eg: &[NodeData], lang: &Lang) -> Result<()> {
        if eg.is_empty() {
            return Ok(());
        }

        let connection = self.ensure_connected().await?;

        let endpoints = self.find_nodes_by_type_async(NodeType::Endpoint).await;

        let imports = self.find_nodes_by_type_async(NodeType::Import).await;

        let find_import_node = |file: &str| -> Option<NodeData> {
            imports.iter().find(|node| node.file == file).cloned()
        };

        let matches = lang
            .lang()
            .match_endpoint_groups(&eg, &endpoints, &find_import_node);
        let mut best_matches: HashMap<(String, String, usize, String), (NodeData, String)> =
            HashMap::new();

        for (endpoint, prefix) in matches {
            let endpoint_verb = endpoint.meta.get("verb").cloned().unwrap_or_default();
            let key = (
                endpoint.name.clone(),
                endpoint.file.clone(),
                endpoint.start,
                endpoint_verb,
            );

            match best_matches.get(&key) {
                Some((_existing_ep, existing_prefix)) if prefix.len() > existing_prefix.len() => {
                    best_matches.insert(key, (endpoint, prefix));
                }
                None => {
                    best_matches.insert(key, (endpoint, prefix));
                }
                _ => {}
            }
        }

        let mut txn_manager = TransactionManager::new(&connection);

        for ((_, _, _, _), (endpoint, prefix)) in best_matches {
            let full_path = format!("{}{}", prefix, endpoint.name);
            let mut new_node_data = endpoint.clone();
            new_node_data.name = full_path.clone();
            let new_key = create_node_key(&Node::new(NodeType::Endpoint, new_node_data));

            let endpoint_verb = endpoint.meta.get("verb").map(|v| v.as_str());
            let mut update_query = String::from(
                "MATCH (e:Endpoint) WHERE e.name = $old_name AND e.file = $file AND e.start = $start",
            );
            if endpoint_verb.is_some() {
                update_query.push_str(" AND e.verb = $verb");
            }
            update_query.push_str(" SET e.name = $new_name, e.node_key = $new_key RETURN e.name");

            let mut params = BoltMap::new();
            boltmap_insert_str(&mut params, "old_name", &endpoint.name);
            boltmap_insert_str(&mut params, "file", &endpoint.file);
            boltmap_insert_int(&mut params, "start", endpoint.start as i64);
            boltmap_insert_str(&mut params, "new_name", &full_path);
            boltmap_insert_str(&mut params, "new_key", &new_key);
            if let Some(verb) = endpoint_verb {
                boltmap_insert_str(&mut params, "verb", verb);
            }

            txn_manager.add_query((update_query, params));
        }

        txn_manager.execute().await?;
        Ok(())
    }

    pub async fn find_top_level_functions_async(&self) -> Vec<NodeData> {
        let Ok(connection) = self.ensure_connected().await else {
            warn!("Failed to connect to Neo4j in find_top_level_functions_async");
            return vec![];
        };
        let (query, params) = find_top_level_functions_query();
        execute_node_query(&connection, query, params).await
    }

    pub async fn set_missing_data_bank(&self) -> Result<u32> {
        let connection = self.ensure_connected().await?;
        let query_str = set_missing_data_bank_query();
        let mut result = connection.execute(query(&query_str)).await?;
        if let Some(row) = result.next().await? {
            let count = row.get::<i64>("updated_count").unwrap_or(0);
            info!("Set Data_Bank property for {} nodes", count);
            Ok(count as u32)
        } else {
            Ok(0)
        }
    }

    pub async fn set_default_namespace(&self) -> Result<u32> {
        let connection = self.ensure_connected().await?;
        let query_str = set_default_namespace_query();
        let mut result = connection.execute(query(&query_str)).await?;
        if let Some(row) = result.next().await? {
            let count = row.get::<i64>("updated_count").unwrap_or(0);
            info!("Set namespace property to 'default' for {} nodes", count);
            Ok(count as u32)
        } else {
            Ok(0)
        }
    }

    pub async fn get_data_models_within_async(&self, lang: &Lang) -> Result<()> {
        let connection = self.ensure_connected().await?;
        let mut txn_manager = TransactionManager::new(&connection);

        let data_models = self.find_nodes_by_type_async(NodeType::DataModel).await;

        for data_model in data_models {
            let edges = lang.lang().data_model_within_finder(&data_model, &|file| {
                sync_fn(|| async {
                    self.find_nodes_by_file_ends_with_async(NodeType::Function, file)
                        .await
                })
            });

            for edge in edges {
                txn_manager.add_edge(&edge);
            }
        }

        txn_manager.execute().await
    }

    pub async fn has_edge_async(&self, source: &Node, target: &Node, edge_type: EdgeType) -> bool {
        let Ok(connection) = self.ensure_connected().await else {
            warn!("Failed to connect to Neo4j in has_edge_async");
            return false;
        };

        let (query_str, params) = has_edge_query(source, target, &edge_type);

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
    pub async fn fetch_nodes_without_embeddings(
        &self,
        do_files: bool,
        skip: usize,
        limit: usize,
    ) -> Result<Vec<(String, String)>> {
        let connection = self.ensure_connected().await?;
        let (query_str, params) = data_bank_bodies_query_no_embeddings(do_files, skip, limit);
        let mut query_obj = query(&query_str);
        for (k, v) in params.value.iter() {
            query_obj = query_obj.param(k.value.as_str(), v.clone());
        }
        let mut result = connection.execute(query_obj).await?;
        let mut nodes = Vec::new();
        while let Some(row) = result.next().await? {
            let node_key = row.get::<String>("node_key").unwrap_or_default();
            let body = row.get::<String>("body").unwrap_or_default();
            nodes.push((node_key, body));
        }
        Ok(nodes)
    }
    pub async fn update_embedding(&self, node_key: &str, embedding: &[f32]) -> Result<()> {
        let connection = self.ensure_connected().await?;
        let (query_str, params) = update_embedding_query(node_key, embedding);
        let mut query_obj = query(&query_str);
        for (k, v) in params.value.iter() {
            query_obj = query_obj.param(k.value.as_str(), v.clone());
        }
        let mut txn = connection.start_txn().await?;
        txn.run(query_obj).await?;
        txn.commit().await?;
        Ok(())
    }

    pub async fn vector_search(
        &self,
        embedding: &[f32],
        limit: usize,
        node_types: Vec<String>,
        similarity_threshold: f32,
        language: Option<&str>,
    ) -> Result<Vec<(NodeData, f64)>> {
        let connection = self.ensure_connected().await?;

        let (query_str, params) = vector_search_query(
            embedding,
            limit,
            node_types,
            similarity_threshold,
            language.map(|s| s.to_string()),
        );

        let mut query_obj = query(&query_str);
        for (key, value) in params.value.iter() {
            query_obj = query_obj.param(key.value.as_str(), value.clone());
        }

        let mut result = connection.execute(query_obj).await?;
        let mut nodes = Vec::new();
        while let Some(row) = result.next().await? {
            let node: neo4rs::Node = row
                .get("node")
                .map_err(|e| Error::Custom(format!("Failed to get node {e}")))?;
            let score: f64 = row
                .get("score")
                .map_err(|e| Error::Custom(format!("Failed to get score {e}")))?;

            if let Ok(node_data) = NodeData::try_from(&node) {
                nodes.push((node_data, score));
            }
        }
        Ok(nodes)
    }
    pub async fn fetch_all_node_keys(&self) -> Result<Vec<String>> {
        use all_node_keys_query;
        let connection = self.ensure_connected().await?;
        let mut result = connection.execute(query(&all_node_keys_query())).await?;
        let mut keys = Vec::new();
        while let Some(row) = result.next().await? {
            if let Ok(k) = row.get::<String>("node_key") {
                keys.push(k);
            }
        }
        Ok(keys)
    }
    pub async fn fetch_all_edge_triples(&self) -> Result<Vec<(String, String, EdgeType)>> {
        let connection = self.ensure_connected().await?;
        let mut result = connection.execute(query(&all_edge_triples_query())).await?;
        let mut triples = Vec::new();
        while let Some(row) = result.next().await? {
            if let (Ok(s), Ok(et), Ok(t)) = (
                row.get::<String>("s_key"),
                row.get::<String>("edge_type"),
                row.get::<String>("t_key"),
            ) {
                if let Ok(edge_type) = EdgeType::from_str(&et) {
                    triples.push((s, t, edge_type));
                }
            }
        }
        Ok(triples)
    }

    pub async fn query_nodes_with_count_async(
        &self,
        node_types: &[NodeType],
        offset: usize,
        limit: usize,
        sort_by_test_count: bool,
        coverage_filter: Option<&str>,
        body_length: bool,
        line_count: bool,
        repo: Option<&str>,
        test_filters: Option<TestFilters>,
        search: Option<&str>,
        is_muted: Option<bool>,
    ) -> (
        usize,
        Vec<(
            NodeType,
            NodeData,
            usize,
            bool,
            usize,
            String,
            Option<i64>,
            Option<i64>,
            Option<bool>,
        )>,
    ) {
        let Ok(connection) = self.ensure_connected().await else {
            warn!("Failed to connect to Neo4j in query_nodes_with_count_async");
            return (0, vec![]);
        };

        let (query_str, params) = query_nodes_with_count(
            &node_types,
            offset,
            limit,
            sort_by_test_count,
            coverage_filter,
            body_length,
            line_count,
            repo,
            test_filters,
            search,
            is_muted,
        );

        let mut query_obj = query(&query_str);
        for (key, value) in params.value.iter() {
            query_obj = query_obj.param(key.value.as_str(), value.clone());
        }

        match connection.execute(query_obj).await {
            Ok(mut result) => {
                if let Ok(Some(row)) = result.next().await {
                    let total_count = row.get::<i64>("total_count").unwrap_or(0) as usize;

                    let items: Vec<BoltMap> = row.get("items").unwrap_or_default();

                    let nodes: Vec<(
                        NodeType,
                        NodeData,
                        usize,
                        bool,
                        usize,
                        String,
                        Option<i64>,
                        Option<i64>,
                        Option<bool>,
                    )> = items
                        .into_iter()
                        .filter_map(|item| {
                            let node: neo4rs::Node = match item.get("node") {
                                Ok(node) => node,
                                Err(err) => {
                                    debug!("Missing node in combined query row: {}", err);
                                    return None;
                                }
                            };
                            let node_type =
                                node.labels()
                                    .iter()
                                    .find_map(|label| match NodeType::from_str(label) {
                                        Ok(node_type) => Some(node_type),
                                        Err(_) => None,
                                    })?;

                            let node_data = match NodeData::try_from(&node) {
                                Ok(node_data) => node_data,
                                Err(err) => {
                                    debug!("Failed to convert Neo4j node to NodeData: {}", err);
                                    return None;
                                }
                            };

                            let usage_count: i64 = item.get("usage_count").unwrap_or(0);
                            let is_covered: bool = item.get("is_covered").unwrap_or(false);
                            let test_count: i64 = item.get("test_count").unwrap_or(0);
                            let body_length: Option<i64> = match item.get("body_length") {
                                Ok(value) => Some(value),
                                Err(_) => None,
                            };
                            let line_count: Option<i64> = match item.get("line_count") {
                                Ok(value) => Some(value),
                                Err(_) => None,
                            };
                            let is_muted: Option<bool> = match item.get("is_muted") {
                                Ok(value) => Some(value),
                                Err(_) => None,
                            };

                            let ref_id = extract_ref_id(&node_data);

                            Some((
                                node_type,
                                node_data,
                                usage_count as usize,
                                is_covered,
                                test_count as usize,
                                ref_id,
                                body_length,
                                line_count,
                                is_muted,
                            ))
                        })
                        .collect();

                    (total_count, nodes)
                } else {
                    (0, vec![])
                }
            }
            Err(e) => {
                debug!("Error executing combined query: {}", e);
                (0, vec![])
            }
        }
    }
}

impl Graph for Neo4jGraph {
    fn new(root: String, lang_kind: Language) -> Self
    where
        Self: Sized,
    {
        Neo4jGraph {
            connection: Arc::new(Mutex::new(None)),
            config: Neo4jConfig::default(),
            root,
            lang_kind,
        }
    }
    fn with_capacity(_nodes: usize, _edges: usize, root: String, lang_kind: Language) -> Self
    where
        Self: Sized,
    {
        Neo4jGraph {
            connection: Arc::new(Mutex::new(None)),
            config: Neo4jConfig::default(),
            root,
            lang_kind,
        }
    }
    fn analysis(&self) {
        let _ = sync_fn(|| async { self.analysis_async().await });
    }
    fn create_filtered_graph(self, _final_filter: &[String], _lang_kind: Language) -> Self
    where
        Self: Sized,
    {
        self
    }

    fn extend_graph(&mut self, _other: Self)
    where
        Self: Sized,
    {
    }

    fn get_graph_size(&self) -> (u32, u32) {
        sync_fn(|| async { self.get_graph_size_async().await.unwrap_or_default() })
    }

    fn find_nodes_by_name(&self, node_type: NodeType, name: &str) -> Vec<NodeData> {
        sync_fn(|| async { self.find_nodes_by_name_async(node_type, name).await })
    }
    fn add_node_with_parent(
        &mut self,
        node_type: &NodeType,
        node_data: &NodeData,
        parent_type: &NodeType,
        parent_file: &str,
    ) {
        let node_type = node_type.clone();
        let node_data = node_data.clone();
        let parent_type = parent_type.clone();
        sync_fn(|| async move {
            self.add_node_with_parent_async(node_type, node_data, parent_type, parent_file)
                .await
                .unwrap_or_default()
        });
    }
    fn add_edge(&mut self, edge: &Edge) {
        let edge = edge.clone();
        sync_fn(|| async move { self.add_edge_async(edge).await.unwrap_or_default() });
    }
    fn add_node(&mut self, node_type: &NodeType, node_data: &NodeData) {
        let node_type = node_type.clone();
        let node_data = node_data.clone();
        sync_fn(|| async move {
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

    fn process_endpoint_groups(&mut self, eg: &[NodeData], lang: &Lang) -> Result<()> {
        sync_fn(|| async {
            self.process_endpoint_groups_async(eg, lang)
                .await
                .unwrap_or_default()
        });
        Ok(())
    }
    fn class_inherits(&mut self) {
        sync_fn(|| async { self.class_inherits_async().await.unwrap_or_default() });
    }
    fn class_includes(&mut self) {
        sync_fn(|| async { self.class_includes_async().await.unwrap_or_default() });
    }
    fn add_instances(&mut self, nodes: &[NodeData]) {
        sync_fn(|| async { self.add_instances_async(nodes).await.unwrap_or_default() });
    }
    fn add_functions(&mut self, functions: &[Function]) {
        sync_fn(|| async {
            self.add_functions_async(functions)
                .await
                .unwrap_or_default()
        });
    }
    fn add_page(&mut self, page: (NodeData, Option<Edge>)) {
        sync_fn(|| async { self.add_page_async(page).await.unwrap_or_default() });
    }
    fn add_pages(&mut self, pages: &[(NodeData, Vec<Edge>)]) {
        sync_fn(|| async { self.add_pages_async(pages).await.unwrap_or_default() });
    }
    fn add_endpoints(&mut self, endpoints: &[(NodeData, Option<Edge>)]) {
        sync_fn(|| async {
            self.add_endpoints_async(endpoints)
                .await
                .unwrap_or_default()
        });
    }
    fn add_tests(&mut self, tests: &[TestRecord]) {
        for tr in tests {
            self.add_node_with_parent(&tr.kind, &tr.node, &NodeType::File, &tr.node.file);
            for e in &tr.edges {
                self.add_edge(e);
            }
        }
    }
    fn add_calls(
        &mut self,
        calls: (
            &[(Calls, Option<NodeData>, Option<NodeData>)],
            &[(Calls, Option<NodeData>, Option<NodeData>)],
            &[Edge],
            &[Edge],
        ),
        lang: &Lang,
    ) {
        sync_fn(|| async { self.add_calls_async(calls, lang).await.unwrap_or_default() });
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

    fn remove_node(&mut self, node_type: NodeType, node_data: &NodeData) {
        sync_fn(|| async {
            self.remove_node_async(node_type, node_data.clone())
                .await
                .unwrap_or_default()
        });
    }

    fn deduplicate_nodes(&mut self, remove_type: NodeType, keep_type: NodeType, operation: &str) {
        sync_fn(|| async {
            self.deduplicate_nodes_async(remove_type, keep_type, operation)
                .await
                .unwrap_or_default()
        });
    }

    fn get_data_models_within(&mut self, lang: &Lang) {
        sync_fn(|| async {
            self.get_data_models_within_async(lang)
                .await
                .unwrap_or_default()
        });
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

    fn find_node_by_name_and_file_contains(
        &self,
        node_type: NodeType,
        name: &str,
        file_part: &str,
    ) -> Option<NodeData> {
        sync_fn(|| async {
            self.find_node_by_name_and_file_contains_async(node_type, name, file_part)
                .await
        })
    }

    fn find_node_in_range(&self, node_type: NodeType, row: u32, file: &str) -> Option<NodeData> {
        sync_fn(|| async { self.find_node_in_range_async(node_type, row, file).await })
    }

    fn find_node_at(&self, node_type: NodeType, file: &str, line: u32) -> Option<NodeData> {
        sync_fn(|| async { self.find_node_at_async(node_type, file, line).await })
    }
    fn has_edge(&self, source: &Node, target: &Node, edge_type: EdgeType) -> bool {
        sync_fn(|| async { self.has_edge_async(source, target, edge_type).await })
    }

    fn get_edges_vec(&self) -> Vec<Edge> {
        Vec::new()
    }
    fn set_allow_unverified_calls(&mut self, _allow: bool) {
        false;
    }
    fn get_allow_unverified_calls(&self) -> bool {
        false
    }
}
