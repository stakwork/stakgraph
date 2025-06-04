use crate::lang::{Function, FunctionCall};

use super::{neo4j_utils::*, *};
use anyhow::Result;
use neo4rs::{query, Graph as Neo4jConnection};
use std::str::FromStr;
use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
    time::Duration,
};
use tracing::{debug, info, warn};

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

    pub async fn add_node_with_parent(
        &mut self,
        node_type: NodeType,
        node_data: NodeData,
        parent_type: NodeType,
        parent_file: &str,
    ) -> Result<()> {
        self.execute_with_transaction(|txn_manager| {
            txn_manager.add_node(&node_type, &node_data);

            let mut params = HashMap::new();
            params.insert("name".to_string(), node_data.name.clone());
            params.insert("file".to_string(), node_data.file.clone());
            params.insert("start".to_string(), node_data.start.to_string());
            params.insert("parent_file".to_string(), parent_file.to_string());

            let query = format!(
                "MATCH (parent:{} {{file: $parent_file}}),
                   (node:{} {{name: $name, file: $file, start: $start}})
                 MERGE (parent)-[:CONTAINS]->(node)",
                parent_type.to_string(),
                node_type.to_string()
            );

            txn_manager.add_query((query, params));
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
    pub async fn count_edges_of_type(&self, edge_type: EdgeType) -> usize {
        let connection = self.get_connection();

        let (query_str, params) = count_edges_by_type_query(&edge_type);
        let mut query_obj = query(&query_str);
        for (key, value) in params {
            query_obj = query_obj.param(&key, value.as_str());
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
    pub async fn add_functions(&mut self, functions: Vec<Function>) -> Result<()> {
        self.execute_with_transaction(|txn_manager| {
            for (function_node, method_of, reqs, dms, trait_operand, return_types) in &functions {
                let queries = add_functions_query(
                    function_node,
                    method_of.as_ref(),
                    reqs,
                    dms,
                    trait_operand.as_ref(),
                    return_types,
                );
                for (query, params) in queries {
                    txn_manager.add_query((query, params));
                }
            }
            Ok(())
        })
        .await
    }

    pub async fn add_test_node(
        &mut self,
        test_data: NodeData,
        test_type: NodeType,
        test_edge: Option<Edge>,
    ) -> Result<()> {
        let connection = self.ensure_connected().await?;
        let mut txn_manager = TransactionManager::new(&connection);
        let queries = add_test_node_query(&test_data, &test_type, &test_edge);
        txn_manager.add_queries(queries);
        txn_manager.execute().await
    }

    pub async fn add_page(&mut self, page: (NodeData, Option<Edge>)) -> Result<()> {
        let (page_data, edge_opt) = page;
        let connection = self.ensure_connected().await?;
        let mut txn_manager = TransactionManager::new(&connection);
        let queries = add_page_query(&page_data, &edge_opt);
        txn_manager.add_queries(queries);
        txn_manager.execute().await
    }

    pub async fn add_pages(&mut self, pages: Vec<(NodeData, Vec<Edge>)>) -> Result<()> {
        let connection = self.ensure_connected().await?;
        let mut txn_manager = TransactionManager::new(&connection);
        let queries = add_pages_query(&pages);
        txn_manager.add_queries(queries);
        txn_manager.execute().await
    }

    pub async fn add_endpoints(&mut self, endpoints: Vec<(NodeData, Option<Edge>)>) -> Result<()> {
        use std::collections::HashSet;
        let mut to_add = Vec::new();
        let mut seen = HashSet::new();
        for (endpoint_data, handler_edge) in &endpoints {
            if endpoint_data.meta.get("handler").is_some() {
                let default_verb = "".to_string();
                let verb = endpoint_data.meta.get("verb").unwrap_or(&default_verb);
                let key = (
                    endpoint_data.name.clone(),
                    endpoint_data.file.clone(),
                    verb.clone(),
                );
                if seen.contains(&key) {
                    continue;
                }
                let exists = self
                    .find_endpoint(&endpoint_data.name, &endpoint_data.file, verb)
                    .await
                    .is_some();
                if !exists {
                    to_add.push((endpoint_data.clone(), handler_edge.clone()));
                    seen.insert(key);
                }
            }
        }
        self.execute_with_transaction(|txn_manager| {
            for (endpoint_data, handler_edge) in &to_add {
                txn_manager.add_node(&NodeType::Endpoint, endpoint_data);
                if let Some(edge) = handler_edge {
                    txn_manager.add_edge(edge);
                }
            }
            Ok(())
        })
        .await
    }

    pub async fn add_calls(
        &mut self,
        calls: (Vec<FunctionCall>, Vec<FunctionCall>, Vec<Edge>),
    ) -> Result<()> {
        let (funcs, tests, int_tests) = calls;
        let connection = self.ensure_connected().await?;
        let mut txn_manager = TransactionManager::new(&connection);
        for (func_call, ext_func, class_call) in &funcs {
            if let Some(cls_call) = &class_call {
                let edge = Edge::new(
                    EdgeType::Calls,
                    NodeRef::from(func_call.source.clone(), NodeType::Function),
                    NodeRef::from(cls_call.into(), NodeType::Class),
                );
                txn_manager.add_edge(&edge);
            }
            if func_call.target.is_empty() {
                continue;
            }
            if let Some(ext_nd) = ext_func {
                txn_manager.add_node(&NodeType::Function, ext_nd);
                let edge = Edge::uses(func_call.source.clone(), ext_nd);
                txn_manager.add_edge(&edge);
            } else {
                let edge = func_call.clone().into();
                txn_manager.add_edge(&edge);
            }
        }
        for (test_call, ext_func, _class_call) in &tests {
            if let Some(ext_nd) = ext_func {
                txn_manager.add_node(&NodeType::Function, ext_nd);
                let edge = Edge::uses(test_call.source.clone(), ext_nd);
                txn_manager.add_edge(&edge);
            } else {
                let edge = test_call.clone().into();
                txn_manager.add_edge(&edge);
            }
        }
        for edge in int_tests {
            txn_manager.add_edge(&edge);
        }
        txn_manager.execute().await
    }

    pub async fn find_nodes_by_type(&mut self, node_type: NodeType) -> Result<Vec<NodeData>> {
        let connection = self.ensure_connected().await?;
        let (query, params) = find_nodes_by_type_query(&node_type);
        execute_node_query(&connection, query, params).await
    }

    pub async fn find_nodes_by_name_contains(
        &mut self,
        node_type: NodeType,
        name_part: &str,
    ) -> Result<Vec<NodeData>> {
        let connection = self.ensure_connected().await?;
        let (query, params) = find_nodes_by_name_contains_query(&node_type, name_part);
        execute_node_query(&connection, query, params).await
    }

    pub async fn find_nodes_by_file_ends_with(
        &mut self,
        node_type: NodeType,
        file: &str,
    ) -> Result<Vec<NodeData>> {
        let connection = self.ensure_connected().await?;
        let (query, params) = find_nodes_by_file_pattern_query(&node_type, file);
        execute_node_query(&connection, query, params).await
    }

    pub async fn find_nodes_in_range(
        &mut self,
        node_type: NodeType,
        row: u32,
        file: &str,
    ) -> Result<Option<NodeData>> {
        let connection = self.ensure_connected().await?;
        let (query, params) = find_nodes_in_range_query(&node_type, file, row);
        let nodes = execute_node_query(&connection, query, params).await?;
        Ok(nodes.into_iter().next())
    }

    pub async fn find_resource_nodes(
        &mut self,
        node_type: NodeType,
        verb: &str,
        path: &str,
    ) -> Result<Vec<NodeData>> {
        let connection = self.ensure_connected().await?;
        let (query, params) = find_resource_nodes_query(&node_type, verb, path);
        execute_node_query(&connection, query, params).await
    }

    pub async fn find_handlers_for_endpoint(
        &mut self,
        endpoint: &NodeData,
    ) -> Result<Vec<NodeData>> {
        let connection = self.ensure_connected().await?;
        let (query, params) = find_handlers_for_endpoint_query(&endpoint);
        execute_node_query(&connection, query, params).await
    }

    pub async fn find_functions_called_by(&mut self, function: &NodeData) -> Result<Vec<NodeData>> {
        let connection = self.ensure_connected().await?;
        let (query, params) = find_functions_called_by_query(&function);
        execute_node_query(&connection, query, params).await
    }

    pub async fn filter_out_nodes_without_children(
        &mut self,
        parent_type: NodeType,
        child_type: NodeType,
        _child_meta_key: &str,
    ) -> Result<()> {
        self.execute_with_transaction(|txn_manager| {
            let query = format!(
                "MATCH (parent:{}) \\n                 WHERE NOT EXISTS {{
                     MATCH (parent)<-[:OPERAND]-(child:{})
                 }}
                 AND NOT EXISTS {{
                     MATCH (instance:Instance)-[:OF]->(parent)
                 }}
                 DETACH DELETE parent",
                parent_type.to_string(),
                child_type.to_string()
            );
            txn_manager.add_query((query, HashMap::new()));
            Ok(())
        })
        .await
    }

    pub async fn class_includes(&mut self) -> Result<()> {
        self.execute_with_transaction(|txn_manager| {
            let query = class_includes_query();
            txn_manager.add_query((query, HashMap::new()));
            Ok(())
        })
        .await
    }

    pub async fn class_inherits(&mut self) -> Result<()> {
        self.execute_with_transaction(|txn_manager| {
            let query = class_inherits_query();
            txn_manager.add_query((query, HashMap::new()));
            Ok(())
        })
        .await
    }

    pub async fn check_direct_data_model_usage(
        &mut self,
        function_name: &str,
        data_model: &str,
    ) -> Result<bool> {
        let connection = self.ensure_connected().await?;
        let (query_str, params) = check_direct_data_model_usage_query(function_name, data_model);
        let mut query_obj = query(&query_str);
        for (key, value) in params {
            query_obj = query_obj.param(&key, value.as_str());
        }
        let mut result = connection.execute(query_obj).await?;
        if let Some(row) = result.next().await? {
            Ok(row.get::<bool>("exists").unwrap_or(false))
        } else {
            Ok(false)
        }
    }

    pub async fn prefix_paths(&mut self, root: &str) -> Result<()> {
        self.execute_with_transaction(|txn_manager| {
            let (query, params) = prefix_paths_query(root);
            txn_manager.add_query((query, params));
            Ok(())
        })
        .await
    }

    pub async fn create_filtered_graph(&self, final_filter: &[String]) -> Result<Self> {
        if final_filter.is_empty() {
            return Ok(self.clone());
        }
        let mut filtered_graph = Neo4jGraph::with_config(self.config.clone());
        filtered_graph.connect().await?;
        filtered_graph.clear().await?;
        let source_connection = self.get_connection();
        let target_connection = filtered_graph.ensure_connected().await?;
        let mut txn_manager = TransactionManager::new(&target_connection);
        let files_list = final_filter
            .iter()
            .map(|f| format!("'{}'", f.replace("'", "\\'")))
            .collect::<Vec<_>>()
            .join(",");
        let repo_query = "MATCH (n:Repository) RETURN n";
        let mut result = source_connection.execute(query(repo_query)).await?;
        while let Some(row) = result.next().await? {
            if let Ok(node) = row.get::<neo4rs::Node>("n") {
                let node_data = extract_node_data_from_neo4j_node(&node);
                txn_manager.add_node(&NodeType::Repository, &node_data);
            }
        }
        let filtered_nodes_query = format!(
            "MATCH (n) \n WHERE n.file IN [{}] \n RETURN n, labels(n)[0] as node_type",
            files_list
        );
        let mut result = source_connection
            .execute(query(&filtered_nodes_query))
            .await?;
        while let Some(row) = result.next().await? {
            if let (Ok(node), Ok(node_type_str)) =
                (row.get::<neo4rs::Node>("n"), row.get::<String>("node_type"))
            {
                if let Ok(node_type) = NodeType::from_str(&node_type_str) {
                    let node_data = extract_node_data_from_neo4j_node(&node);
                    txn_manager.add_node(&node_type, &node_data);
                }
            }
        }
        let edges_query = format!(
            "MATCH (source)-[r]->(target)\n WHERE (source.file IN [{}] OR labels(source)[0] = 'Repository') AND (target.file IN [{}] OR labels(target)[0] = 'Repository')\n RETURN source, type(r) as edge_type, r, target, labels(source)[0] as source_type, labels(target)[0] as target_type",
            files_list, files_list
        );
        let mut result = source_connection.execute(query(&edges_query)).await?;
        while let Some(row) = result.next().await? {
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
                        let source_ref = NodeRef::from(NodeKeys::from(&source_data), source_type);
                        let target_ref = NodeRef::from(NodeKeys::from(&target_data), target_type);
                        let edge = Edge::new(edge_type, source_ref, target_ref);
                        txn_manager.add_edge(&edge);
                    }
                }
            }
        }
        txn_manager.execute().await?;
        Ok(filtered_graph)
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

    pub async fn find_source_edge_by_name_and_file(
        &mut self,
        edge_type: EdgeType,
        target_name: &str,
        target_file: &str,
    ) -> Option<NodeKeys> {
        let connection = match self.ensure_connected().await {
            Ok(conn) => conn,
            Err(e) => {
                debug!("Connection error: {}", e);
                return None;
            }
        };
        let (query_str, params) =
            find_source_edge_by_name_and_file_query(&edge_type, target_name, target_file);
        let mut query_obj = query(&query_str);
        for (key, value) in params {
            query_obj = query_obj.param(&key, value.as_str());
        }
        match connection.execute(query_obj).await {
            Ok(mut result) => {
                if let Some(row) = result.next().await.ok().flatten() {
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

    pub async fn add_instances(&mut self, instances: Vec<NodeData>) -> Result<()> {
        let connection = self.ensure_connected().await?;
        let mut txn_manager = TransactionManager::new(&connection);

        for inst in &instances {
            if let Some(of) = &inst.data_type {
                txn_manager.add_node(&NodeType::Instance, inst);

                let contains_query = format!(
                    "MATCH (file:File {{file: $file}}),
                    (instance:Instance {{name: $name, file: $file}})
                     MERGE (file)-[:CONTAINS]->(instance)"
                );

                let mut contains_params = HashMap::new();
                contains_params.insert("name".to_string(), inst.name.clone());
                contains_params.insert("file".to_string(), inst.file.clone());

                txn_manager.add_query((contains_query, contains_params));

                let of_query = format!(
                    "MATCH (instance:Instance {{name: $name, file: $file}}), 
                               (class:Class {{name: $class_name}}) 
                         MERGE (instance)-[:OF]->(class)"
                );

                let mut of_params = HashMap::new();
                of_params.insert("name".to_string(), inst.name.clone());
                of_params.insert("file".to_string(), inst.file.clone());
                of_params.insert("class_name".to_string(), of.clone());

                txn_manager.add_query((of_query, of_params));
            }
        }

        if let Err(e) = txn_manager.execute().await {
            println!("Transaction failed in add_instances: {:?}", e);
            return Err(e);
        }

        Ok(())
    }

    // pub async fn get_data_models_within(&mut self, lang: &Lang) -> Result<()> {
    //     let connection = self.ensure_connected().await?;
    //     let mut txn_manager = TransactionManager::new(&connection);

    //     let data_models = self.find_nodes_by_type(NodeType::DataModel).await?;

    //     for data_model in data_models {
    //         let edges = lang.lang().data_model_within_finder(&data_model, &|file| {
    //             futures::executor::block_on(
    //                 self.find_nodes_by_file_ends_with(NodeType::Function, file)
    //                     .unwrap(),
    //             )
    //         });

    //         for edge in edges {
    //             txn_manager.add_edge(&edge);
    //         }
    //     }

    //     if let Err(e) = txn_manager.execute().await {
    //         println!("Transaction failed in get_data_models_within: {:?}", e);
    //         return Err(e);
    //     }

    //     Ok(())
    // }

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
        for (key, value) in params {
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

    pub async fn find_endpoint(&self, name: &str, file: &str, verb: &str) -> Option<NodeData> {
        let connection = self.get_connection();
        let (query, params) = find_endpoint_query(name, file, verb);
        match execute_node_query(&connection, query, params).await {
            Ok(nodes) => nodes.into_iter().next(),
            Err(e) => {
                debug!("Error finding endpoint: {}", e);
                None
            }
        }
    }

    // pub async fn process_endpoint_groups(&mut self, eg: Vec<NodeData>, lang: &Lang) -> Result<()> {
    //     // Pre-process the data we need before entering the transaction
    //     let mut group_data = Vec::new();

    //     for group in &eg {
    //         if let Some(group_function_name) = group.meta.get("group") {
    //             if let Some(group_function) = self
    //                 .find_nodes_by_name(NodeType::Function, group_function_name)
    //                 .await
    //                 .first()
    //                 .cloned()
    //             {
    //                 for finder_query in lang.lang().endpoint_finders() {
    //                     if let Ok(endpoints) = lang.get_query_opt::<Self>(
    //                         Some(finder_query),
    //                         &group_function.body,
    //                         &group_function.file,
    //                         NodeType::Endpoint,
    //                     ) {
    //                         group_data.push((group.clone(), endpoints));
    //                     }
    //                 }
    //             }
    //         }
    //     }

    //     self.execute_with_transaction(|txn_manager|  {
    //         for (group, endpoints) in &group_data {
    //             for endpoint in endpoints {
    //                 let update_node_query = format!(
    //                     "MATCH (n:Endpoint {{name: $old_name, file: $file}})
    //                     SET n.name = $new_name
    //                     RETURN n"
    //                 );
    //                 let mut node_params = HashMap::new();
    //                 node_params.insert("old_name".to_string(), endpoint.name.clone());
    //                 node_params.insert("file".to_string(), endpoint.file.clone());
    //                 node_params.insert(
    //                     "new_name".to_string(),
    //                     format!("{}{}", group.name, endpoint.name),
    //                 );
    //                 txn_manager.add_query((update_node_query, node_params.clone()));
    //                 let update_rels_query = format!(
    //                     "MATCH (source:Endpoint {{name: $old_name, file: $file}})-[r]->(target)
    //                     SET source.name = $new_name
    //                     RETURN r"
    //                 );
    //                 txn_manager.add_query((update_rels_query, node_params));
    //             }
    //         }
    //         Ok(())
    //     })
    //     .await
    // }
}
