use neo4rs::{query, BoltMap, BoltType, Graph as Neo4jConnection};
use shared::Result;
use crate::lang::{Edge, NodeData, NodeType, graphs::{queries::*, helpers::*}};
use std::str::FromStr;

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
        println!("Processing chunk {}/{}", i + 1, total_chunks);
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
                println!("Error running query in batch chunk {}: {:?}", i + 1, e);
                txn.rollback().await?; // Attempt to rollback
                return Err(e.into());
            }
        }

        if let Err(e) = txn.commit().await {
            eprintln!("Error committing batch chunk {}: {:?}", i + 1, e);
            return Err(e.into());
        }
        println!("Successfully committed chunk {}/{}", i + 1, total_chunks);
    }
    Ok(())
}

pub async fn execute_queries_simple(
    conn: &Neo4jConnection,
    queries: Vec<(String, BoltMap)>,
) -> Result<()> {
    let total_queries = queries.len();
    for (i, (query_str, params)) in queries.into_iter().enumerate() {
        println!("Processing query {}/{}", i + 1, total_queries);

        let mut query_obj = query(&query_str);

        // Add parameters
        for (key, value) in params.value.iter() {
            query_obj = query_obj.param(key.value.as_str(), value.clone());
        }

        // Execute in a transaction
        let mut txn = conn.start_txn().await?;
        txn.run(query_obj).await?;
        txn.commit().await?;

        println!("Successfully executed query {}/{}", i + 1, total_queries);
    }
    Ok(())
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
            println!("Error executing query: {}", e);
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
            eprintln!("Error executing nodes with coverage query: {}", e);
        }
    }
    results
}

pub async fn execute_muted_nodes_query(
    conn: &Neo4jConnection,
    query_str: String,
    params: BoltMap,
) -> Vec<MutedNodeIdentifier> {
    let mut query_obj = query(&query_str);
    for (key, value) in params.value.iter() {
        query_obj = query_obj.param(key.value.as_str(), value.clone());
    }
    match conn.execute(query_obj).await {
        Ok(mut stream) => {
            let mut results = Vec::new();
            while let Ok(Some(row)) = stream.next().await {
                if let (Ok(name), Ok(node_type_str), Ok(file)) = (
                    row.get::<String>("name"),
                    row.get::<String>("node_type"),
                    row.get::<String>("file"),
                ) {
                    if let Ok(node_type) = NodeType::from_str(&node_type_str) {
                        results.push(MutedNodeIdentifier {
                            node_type,
                            name,
                            file,
                        });
                    }
                }
            }
            results
        }
        Err(_) => {
            Vec::new()
        }
    }
}

pub async fn execute_count_query(
    conn: &Neo4jConnection,
    query_str: String,
    params: BoltMap,
) -> usize {
    let mut query_obj = query(&query_str);
    for (key, value) in params.value.iter() {
        query_obj = query_obj.param(key.value.as_str(), value.clone());
    }
    
    match conn.execute(query_obj).await {
        Ok(mut stream) => {
            if let Ok(Some(row)) = stream.next().await {
                let count = row.get::<i64>("updated_count")
                    .or_else(|_| row.get::<i64>("restored_count"))
                    .unwrap_or(0) as usize;
                count
            } else {
                0
            }
        }
        Err(_) => {     
            0
        }
    }
}

pub async fn execute_boolean_query(
    conn: &Neo4jConnection,
    query_str: String,
    params: BoltMap,
) -> bool {
    let mut query_obj = query(&query_str);
    for (key, value) in params.value.iter() {
        query_obj = query_obj.param(key.value.as_str(), value.clone());
    }
    if let Ok(mut stream) = conn.execute(query_obj).await {
            if let Ok(Some(row)) = stream.next().await {
                let is_muted = row.get::<bool>("is_muted").unwrap_or(false);
                is_muted
            } else {
                false
            }
        }
    else {
        false  
    }
}
