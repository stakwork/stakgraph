use crate::{
    lang::{Function, FunctionCall},
    Lang,
};

use super::{graph::Graph, neo4j_utils::*, *};
use anyhow::Result;
use neo4rs::{query, ConfigBuilder, Graph as Neo4jConnection};
use std::{
    collections::{HashMap, HashSet},
    sync::{Arc, Mutex},
    time::Duration,
};
use tokio::runtime::Handle;
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
        if let Some(conn) = Neo4jConnectionManager::get_connection() {
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
                if let Some(conn) = Neo4jConnectionManager::get_connection() {
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

    pub fn disconnect(&mut self) -> Result<()> {
        if self.connection.is_none() {
            debug!("Not connected to Neo4j database");
            return Ok(());
        }

        self.connection = None;

        if let Ok(mut conn_status) = self.connected.lock() {
            *conn_status = false;
        };

        Neo4jConnectionManager::clear_connection();

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

        if let Some(conn) = Neo4jConnectionManager::get_connection() {
            self.connection = Some(conn.clone());
            if let Ok(mut conn_status) = self.connected.lock() {
                *conn_status = true;
            };
            return Ok(conn);
        }

        self.connect().await?;

        match &self.connection {
            Some(conn) => Ok(conn.clone()),
            None => {
                debug!("Failed to connect to Neo4j database");
                Err(anyhow::anyhow!("Failed to connect to Neo4j database"))
            }
        }
    }

    fn find_node_by_key(&self, key: &str) -> Option<NodeData> {
        let connection = self.get_connection();

        let (query, params) = find_node_by_key_query(key);

        match block_in_place(execute_node_query(&connection, query, params)) {
            Ok(nodes) => nodes.into_iter().next(),
            Err(e) => {
                debug!("Error finding node by key: {}", e);
                None
            }
        }
    }

    fn get_connection(&self) -> Arc<Neo4jConnection> {
        if let Some(conn) = &self.connection {
            return conn.clone();
        }

        if let Some(conn) = Neo4jConnectionManager::get_connection() {
            return conn;
        }

        debug!("No connection available, creating a fallback connection");
        self.graph_fallback()
    }

    fn graph_fallback(&self) -> Arc<Neo4jConnection> {
        let config = ConfigBuilder::new()
            .uri(&self.config.uri)
            .user(&self.config.username)
            .password(&self.config.password)
            .build()
            .expect("Failed to build Neo4j config");

        match block_in_place(Neo4jConnection::connect(config)) {
            Ok(graph) => {
                debug!("Successfully created fallback connection");
                Arc::new(graph)
            }
            Err(e) => {
                panic!(
                    "Failed to connect to Neo4j: {}. Please ensure Neo4j is running at {}.",
                    e, self.config.uri
                );
            }
        }
    }

    pub fn clear(&mut self) {
        if let Err(e) = block_in_place(async {
            let connection = match self.ensure_connected().await {
                Ok(conn) => conn,
                Err(e) => return Err(anyhow::anyhow!("Connection error: {}", e)),
            };
            
            let mut txn = connection.start_txn().await?;

            // Delete relationships first, then nodes
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

            // Convert the Result<(), neo4rs::Error> to Result<(), anyhow::Error>
            match txn.commit().await {
                Ok(()) => Ok(()),
                Err(e) => Err(anyhow::anyhow!("Neo4j commit error: {}", e)),
            }
        }) {
            debug!("Error clearing graph: {:?}", e);
        }
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

impl Graph for Neo4jGraph {
    fn new() -> Self {
        Neo4jGraph {
            connection: None,
            config: Neo4jConfig::default(),
            connected: Arc::new(Mutex::new(false)),
        }
    }

    fn with_capacity(_nodes: usize, _edges: usize) -> Self {
        Neo4jGraph::default()
    }

    fn add_node(&mut self, node_type: NodeType, node_data: NodeData) {
        let node_builder = NodeQueryBuilder::new(&node_type, &node_data);
        let (query, params) = node_builder.build();
        if let Err(e) = block_in_place(async {
            let connection = match self.ensure_connected().await {
                Ok(conn) => conn,
                Err(e) => return Err(anyhow::anyhow!("Connection error: {}", e)),
            };
            let query_builder = QueryBuilder::new(&query).with_params(params);
            execute_query(&connection, &query_builder).await
        }) {
            debug!("Error adding node: {:?}", e);
        }
    }
    
    fn add_edge(&mut self, edge: Edge) {
        let edge_builder = EdgeQueryBuilder::new(&edge);
        let (query, params) = edge_builder.build();
        if let Err(e) = block_in_place(async {
            let connection = match self.ensure_connected().await {
                Ok(conn) => conn,
                Err(e) => return Err(anyhow::anyhow!("Connection error: {}", e)),
            };
            let query_builder = QueryBuilder::new(&query).with_params(params);
            execute_query(&connection, &query_builder).await
        }) {
            debug!("Error adding edge: {:?}", e);
        }
    }

    fn add_node_with_parent(
        &mut self,
        node_type: NodeType,
        node_data: NodeData,
        parent_type: NodeType,
        parent_file: &str,
    ) {
        if let Err(e) = block_in_place(async {
            let connection = match self.ensure_connected().await {
                Ok(conn) => conn,
                Err(e) => return Err(anyhow::anyhow!("Connection error: {}", e)),
            };
            let mut txn_manager = TransactionManager::new(&connection);

            txn_manager.add_node(&node_type, &node_data);

            let query_str = format!(
                "MATCH (parent:{} {{file: $parent_file}}),
                   (node:{} {{name: $name, file: $file, start: $start}})
             MERGE (parent)-[:CONTAINS]->(node)",
                parent_type.to_string(),
                node_type.to_string()
            );

            let query_builder = QueryBuilder::new(&query_str)
                .with_param("name", &node_data.name)
                .with_param("file", &node_data.file)
                .with_param("start", &node_data.start.to_string())
                .with_param("parent_file", parent_file);

            txn_manager.add_query(query_builder.build());

            txn_manager.execute().await
        }) {
            debug!("Error in add_node_with_parent: {:?}", e);
        }
    }
    fn find_nodes_by_name(&self, node_type: NodeType, name: &str) -> Vec<NodeData> {
        let connection = self.get_connection();

        let (query, params) = find_nodes_by_name_query(&node_type, name);
        let query_builder = QueryBuilder::new(&query).with_params(params);
        let (query_str, params_map) = query_builder.build();

        match block_in_place(execute_node_query(&connection, query_str, params_map)) {
            Ok(nodes) => nodes,
            Err(e) => {
                debug!("Error finding nodes by name: {}", e);
                Vec::new()
            }
        }
    }
    fn find_node_by_name_in_file(
        &self,
        node_type: NodeType,
        name: &str,
        file: &str,
    ) -> Option<NodeData> {
        let connection = self.get_connection();

        let (query, params) = find_node_by_name_file_query(&node_type, name, file);

        match block_in_place(execute_node_query(&connection, query, params)) {
            Ok(nodes) => nodes.into_iter().next(),
            Err(e) => {
                debug!("Error finding node by name and file: {}", e);
                None
            }
        }
    }

    fn get_graph_size(&self) -> (u32, u32) {
        let connection = self.get_connection();

        let query_str = count_nodes_edges_query();

        match block_in_place(connection.execute(query(&query_str))) {
            Ok(mut result) => {
                if let Ok(Some(row)) = block_in_place(result.next()) {
                    let nodes = row.get::<u32>("nodes").unwrap_or(0);
                    let edges = row.get::<u32>("edges").unwrap_or(0);
                    (nodes, edges)
                } else {
                    (0, 0)
                }
            }
            Err(e) => {
                debug!("Error getting graph size: {}", e);
                (0, 0)
            }
        }
    }
    fn get_graph_keys(&self) -> (HashSet<&str>, HashSet<&str>) {
        (HashSet::new(), HashSet::new())
    }
    fn analysis(&self) {
        let connection = self.get_connection();

        let query_str = graph_analysis_query();

        match block_in_place(connection.execute(query(&query_str))) {
            Ok(mut result) => {
                info!("Neo4j graph analysis:");
                while let Ok(Some(row)) = block_in_place(result.next()) {
                    if let (Ok(node_type), Ok(count)) =
                        (row.get::<Vec<String>>("node_type"), row.get::<i64>("count"))
                    {
                        info!("  {}: {}", node_type.join(","), count);
                    }
                }
            }
            Err(e) => {
                debug!("Error analyzing graph: {}", e);
            }
        }
    }
    fn count_edges_of_type(&self, edge_type: EdgeType) -> usize {
        let connection = self.get_connection();

        let (query_str, params) = count_edges_by_type_query(&edge_type);
        let mut query_obj = query(&query_str);
        for (key, value) in params {
            query_obj = query_obj.param(&key, value);
        }
        match block_in_place(connection.execute(query_obj)) {
            Ok(mut result) => {
                if let Ok(Some(row)) = block_in_place(result.next()) {
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
    fn add_functions(&mut self, functions: Vec<Function>) {
        for (function_node, method_of, reqs, dms, trait_operand, return_types) in functions {
            let queries = add_functions_query(
                &function_node,
                method_of.as_ref(),
                &reqs,
                &dms,
                trait_operand.as_ref(),
                &return_types,
            );

            if let Err(e) = block_in_place(async {
                let connection = match self.ensure_connected().await {
                    Ok(conn) => conn,
                    Err(e) => return Err(anyhow::anyhow!("Connection error: {}", e)),
                };
                execute_batch(&connection, queries).await
            }) {
                debug!("Error adding functions batch: {:?}", e);
            }
        }
    }
    fn add_test_node(&mut self, test_data: NodeData, test_type: NodeType, test_edge: Option<Edge>) {
        if let Err(e) = block_in_place(async {
            let connection = match self.ensure_connected().await {
                Ok(conn) => conn,
                Err(e) => return Err(anyhow::anyhow!("Connection error: {}", e)),
            };
            let mut txn_manager = TransactionManager::new(&connection);

            let queries = add_test_node_query(&test_data, &test_type, &test_edge);
            txn_manager.add_queries(queries);

            txn_manager.execute().await
        }) {
            debug!("Error adding test node: {:?}", e);
        }
    }
    fn add_page(&mut self, page: (NodeData, Option<Edge>)) {
        let (page_data, edge_opt) = page;

        if let Err(e) = block_in_place(async {
            let connection = match self.ensure_connected().await {
                Ok(conn) => conn,
                Err(e) => return Err(anyhow::anyhow!("Connection error: {}", e)),
            };
            let mut txn_manager = TransactionManager::new(&connection);

            // Get the queries from the utility
            let queries = add_page_query(&page_data, &edge_opt);
            txn_manager.add_queries(queries);

            txn_manager.execute().await
        }) {
            debug!("Error adding page: {:?}", e);
        }
    }
    fn add_pages(&mut self, pages: Vec<(NodeData, Vec<Edge>)>) {
        if let Err(e) = block_in_place(async {
            let connection = match self.ensure_connected().await {
                Ok(conn) => conn,
                Err(e) => return Err(anyhow::anyhow!("Connection error: {}", e)),
            };
            let mut txn_manager = TransactionManager::new(&connection);

            // Get the queries from the utility
            let queries = add_pages_query(&pages);
            txn_manager.add_queries(queries);

            txn_manager.execute().await
        }) {
            debug!("Error adding pages: {:?}", e);
        }
    }

    fn add_endpoints(&mut self, endpoints: Vec<(NodeData, Option<Edge>)>) {
        if let Err(e) = block_in_place(async {
            let connection = match self.ensure_connected().await {
                Ok(conn) => conn,
                Err(e) => return Err(anyhow::anyhow!("Connection error: {}", e)),
            };
            let mut txn_manager = TransactionManager::new(&connection);

            let queries = add_endpoints_query(&endpoints);
            txn_manager.add_queries(queries);

            txn_manager.execute().await
        }) {
            debug!("Error adding endpoints: {:?}", e);
        }
    }

    fn add_calls(&mut self, calls: (Vec<FunctionCall>, Vec<FunctionCall>, Vec<Edge>)) {
        let (funcs, tests, int_tests) = calls;

        if let Err(e) = block_in_place(async {
            let connection = match self.ensure_connected().await {
                Ok(conn) => conn,
                Err(e) => return Err(anyhow::anyhow!("Connection error: {}", e)),
            };
            let mut txn_manager = TransactionManager::new(&connection);

            let queries = add_calls_query(&funcs, &tests, &int_tests);
            txn_manager.add_queries(queries);

            txn_manager.execute().await
        }) {
            debug!("Error in add_calls: {:?}", e);
        }
    }

    fn find_nodes_by_type(&self, node_type: NodeType) -> Vec<NodeData> {
        let connection = self.get_connection();

        let (query, params) = find_nodes_by_type_query(&node_type);

        match block_in_place(execute_node_query(&connection, query, params)) {
            Ok(nodes) => nodes,
            Err(e) => {
                debug!("Error finding nodes by type: {}", e);
                Vec::new()
            }
        }
    }
    fn find_nodes_by_name_contains(&self, node_type: NodeType, name_part: &str) -> Vec<NodeData> {
        let connection = self.get_connection();

        let (query, params) = find_nodes_by_name_contains_query(&node_type, name_part);

        match block_in_place(execute_node_query(&connection, query, params)) {
            Ok(nodes) => nodes,
            Err(e) => {
                debug!("Error finding nodes by name contains: {}", e);
                Vec::new()
            }
        }
    }
    fn find_nodes_by_file_ends_with(&self, node_type: NodeType, file: &str) -> Vec<NodeData> {
        let connection = self.get_connection();

        let (query, params) = find_nodes_by_file_pattern_query(&node_type, file);

        match block_in_place(execute_node_query(&connection, query, params)) {
            Ok(nodes) => nodes,
            Err(e) => {
                debug!("Error finding nodes by file ends with: {}", e);
                Vec::new()
            }
        }
    }
    fn find_nodes_in_range(&self, node_type: NodeType, row: u32, file: &str) -> Option<NodeData> {
        let connection = self.get_connection();

        let (query, params) = find_nodes_in_range_query(&node_type, file, row);

        match block_in_place(execute_node_query(&connection, query, params)) {
            Ok(nodes) => nodes.into_iter().next(),
            Err(e) => {
                debug!("Error finding nodes in range: {}", e);
                None
            }
        }
    }

    fn find_resource_nodes(&self, node_type: NodeType, verb: &str, path: &str) -> Vec<NodeData> {
        let connection = self.get_connection();

        let (query, params) = find_resource_nodes_query(&node_type, verb, path);

        match block_in_place(execute_node_query(&connection, query, params)) {
            Ok(nodes) => nodes,
            Err(e) => {
                debug!("Error finding resource nodes: {}", e);
                Vec::new()
            }
        }
    }

    fn find_handlers_for_endpoint(&self, endpoint: &NodeData) -> Vec<NodeData> {
        let connection = self.get_connection();

        let (query, params) = find_handlers_for_endpoint_query(&endpoint);

        match block_in_place(execute_node_query(&connection, query, params)) {
            Ok(nodes) => nodes,
            Err(e) => {
                debug!("Error finding handlers for endpoint: {}", e);
                Vec::new()
            }
        }
    }
    fn find_functions_called_by(&self, function: &NodeData) -> Vec<NodeData> {
        let connection = self.get_connection();

        let (query, params) = find_functions_called_by_query(&function);

        match block_in_place(execute_node_query(&connection, query, params)) {
            Ok(nodes) => nodes,
            Err(e) => {
                debug!("Error finding functions called by: {}", e);
                Vec::new()
            }
        }
    }
    fn filter_out_nodes_without_children(
        &mut self,
        parent_type: NodeType,
        child_type: NodeType,
        child_meta_key: &str,
    ) {
        if let Err(e) = block_in_place(async {
            let connection = self.ensure_connected().await?;

            let (query, params) =
                filter_nodes_without_children_query(&parent_type, &child_type, child_meta_key);
            let query_builder = QueryBuilder::new(&query).with_params(params);

            execute_query(&connection, &query_builder).await
        }) {
            debug!("Error filtering nodes without children: {:?}", e);
        }
    }

    fn class_includes(&mut self) {
        if let Err(e) = block_in_place(async {
            let connection = self.ensure_connected().await?;

            let query = class_includes_query();
            let query_builder = QueryBuilder::new(&query).with_params(HashMap::new());

            execute_query(&connection, &query_builder).await
        }) {
            debug!("Error in class includes: {:?}", e);
        }
    }

    fn class_inherits(&mut self) {
        if let Err(e) = block_in_place(async {
            let connection = self.ensure_connected().await?;

            let query = class_inherits_query();
            let query_builder = QueryBuilder::new(&query).with_params(HashMap::new());

            execute_query(&connection, &query_builder).await
        }) {
            debug!("Error in class inherits: {:?}", e);
        }
    }

    fn check_direct_data_model_usage(&self, function_name: &str, data_model: &str) -> bool {
        let connection = self.get_connection();

        let (query_str, params) = check_direct_data_model_usage_query(function_name, data_model);

        let mut query_obj = query(&query_str);
        for (key, value) in params {
            query_obj = query_obj.param(&key, value);
        }
        match block_in_place(connection.execute(query_obj)) {
            Ok(mut result) => {
                if let Ok(Some(row)) = block_in_place(result.next()) {
                    row.get::<bool>("exists").unwrap_or(false)
                } else {
                    false
                }
            }
            Err(e) => {
                debug!("Error checking direct data model usage: {}", e);
                false
            }
        }
    }
    fn prefix_paths(&mut self, root: &str) {
        if let Err(e) = block_in_place(async {
            let connection = self.ensure_connected().await?;

            let (query, params) = prefix_paths_query(root);
            let query_builder = QueryBuilder::new(&query).with_params(params);

            execute_query(&connection, &query_builder).await
        }) {
            debug!("Error prefixing paths: {:?}", e);
        }
    }
    fn create_filtered_graph(&self, _final_filter: &[String]) -> Self {
        // For Neo4j, create a new graph with the same configuration
        let filtered_graph = Neo4jGraph::default();

        //TODO: think through implementation
        // Connect to the same database
        // if let Err(e) = block_in_place(filtered_graph.connect()) {
        //     debug!("Failed to connect filtered graph: {}", e);
        //     return filtered_graph;
        // }

        //debug!("Creating filtered Neo4j graph with {} files", final_filter.len());

        filtered_graph
    }
    fn extend_graph(&mut self, _other: Self) {
        //TODO:
        // Implement logic to extend the current graph with another graph
    }
    fn find_source_edge_by_name_and_file(
        &self,
        edge_type: EdgeType,
        target_name: &str,
        target_file: &str,
    ) -> Option<NodeKeys> {
        let connection = self.get_connection();
        let (query_str, params) =
            find_source_edge_by_name_and_file_query(&edge_type, target_name, target_file);
        let mut query_obj = query(&query_str);
        for (key, value) in params {
            query_obj = query_obj.param(&key, value);
        }
        match block_in_place(connection.execute(query_obj)) {
            Ok(mut result) => {
                if let Ok(Some(row)) = block_in_place(result.next()) {
                    let name = row.get::<String>("name").unwrap_or_default();
                    let file = row.get::<String>("file").unwrap_or_default();
                    let start = row.get::<i32>("start").unwrap_or(0) as usize;
                    let verb = row.get::<String>("verb").ok();

                    Some(NodeKeys {
                        name,
                        file,
                        start,
                        verb,
                    })
                } else {
                    None
                }
            }
            Err(e) => {
                debug!("Error finding source edge: {}", e);
                None
            }
        }
    }
    fn add_instances(&mut self, instances: Vec<NodeData>) {
        let connection = self.get_connection();

        for inst in instances {
            if let Some(of) = &inst.data_type {
                let mut class_params = HashMap::new();
                class_params.insert("class_name".to_string(), of.clone());

                let class_query = format!(
                    "MATCH (c:Class {{name: $class_name}})
                     RETURN c
                     LIMIT 1"
                );

                let class_nodes =
                    block_in_place(execute_node_query(&connection, class_query, class_params))
                        .unwrap_or_default();

                if let Some(class_node_data) = class_nodes.first() {
                    self.add_node_with_parent(
                        NodeType::Instance,
                        inst.clone(),
                        NodeType::File,
                        &inst.file,
                    );

                    let edge = Edge::of(&inst, class_node_data);
                    self.add_edge(edge);
                }
            }
        }
    }
    fn get_data_models_within(&mut self, lang: &Lang) {
        let connection = self.get_connection();

        let query = "MATCH (dm:Datamodel) RETURN dm.name, dm.file, dm.start, dm.end, properties(dm) as props";

        let data_model_nodes = block_in_place(execute_node_query(
            &connection,
            query.to_string(),
            HashMap::new(),
        ))
        .unwrap_or_default();

        for data_model in data_model_nodes {
            let edges = lang.lang().data_model_within_finder(&data_model, &|file| {
                let func_query = format!(
                    "MATCH (f:Function)
                     WHERE f.file ENDS WITH $file_pattern
                     RETURN f.name, f.file, f.start, f.end, properties(f) as props"
                );

                let mut params = HashMap::new();
                params.insert("file_pattern".to_string(), file.to_string());

                block_in_place(execute_node_query(&connection, func_query, params))
                    .unwrap_or_default()
            });

            // Add all the edges found
            for edge in edges {
                self.add_edge(edge);
            }
        }
    }

    fn find_nodes_with_edge_type(
        &self,
        source_type: NodeType,
        target_type: NodeType,
        edge_type: EdgeType,
    ) -> Vec<(NodeData, NodeData)> {
        let connection = self.get_connection();

        let pairs = Vec::new();

        let (query_str, params) =
            find_nodes_with_edge_type_query(&source_type, &target_type, &edge_type);

        let mut query_obj = query(&query_str);
        for (key, value) in params {
            query_obj = query_obj.param(&key, value);
        }
        match block_in_place(connection.execute(query_obj)) {
            Ok(mut result) => {
                let mut node_pairs = Vec::new();

                while let Ok(Some(row)) = block_in_place(result.next()) {
                    let source_key: String = row.get("source_key").unwrap_or_default();
                    let target_key: String = row.get("target_key").unwrap_or_default();
                    if let (Some(source_node), Some(target_node)) = (
                        self.find_node_by_key(&source_key),
                        self.find_node_by_key(&target_key),
                    ) {
                        node_pairs.push((source_node, target_node));
                    }
                }
                node_pairs
            }
            Err(e) => {
                debug!("Error finding nodes with edge type: {}", e);
                Vec::new()
            }
        };

        pairs
    }
    fn find_endpoint(&self, name: &str, file: &str, verb: &str) -> Option<NodeData> {
        let connection = self.get_connection();

        let (query, params) = find_endpoint_query(name, file, verb);

        match block_in_place(execute_node_query(&connection, query, params)) {
            Ok(nodes) => nodes.into_iter().next(),
            Err(e) => {
                debug!("Error finding endpoint: {}", e);
                None
            }
        }
    }

    fn process_endpoint_groups(&mut self, _eg: Vec<NodeData>, _lang: &Lang) -> Result<()> {
        //TODO: Implement this function
        Ok(())
    }
}

//to handle async code in blocking context - sync
fn block_in_place<F, T>(future: F) -> T
where
    F: std::future::Future<Output = T>,
{
    match Handle::try_current() {
        Ok(handle) => tokio::task::block_in_place(|| handle.block_on(future)),
        Err(_) => {
            let r = tokio::runtime::Runtime::new().expect("Failed to create runtime");
            r.block_on(future)
        }
    }
}
