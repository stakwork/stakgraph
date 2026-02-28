use crate::lang::graphs::helpers::MutedNodeIdentifier;
use crate::lang::{
    graphs::{helpers::*, queries::*},
    Edge, NodeData, NodeType,
};
use neo4rs::{query, BoltMap, BoltType, Graph as Neo4jConnection, Query};
use shared::Result;
use std::str::FromStr;

fn bind_parameters(query_str: &str, params: BoltMap) -> Query {
    let mut query_obj = query(query_str);

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
                query_obj =
                    query_obj.param("now", neo4rs::BoltType::String(format!("{:.7}", ts).into()));
            }
        }
    } else {
        for (key, value) in params.value.iter() {
            query_obj = query_obj.param(key.value.as_str(), value.clone());
        }
    }
    query_obj
}

fn extract_node_data(row: &neo4rs::Row) -> Result<NodeData> {
    let node = row.get::<neo4rs::Node>("n")?;
    NodeData::try_from(&node).map_err(|e| e.into())
}

fn extract_count(row: &neo4rs::Row) -> Result<usize> {
    let count = row
        .get::<i64>("updated_count")
        .or_else(|_| row.get::<i64>("restored_count"))
        .unwrap_or(0) as usize;
    Ok(count)
}

fn extract_muted_identifier(row: &neo4rs::Row) -> Result<MutedNodeIdentifier> {
    let name = row.get::<String>("name")?;
    let node_type_str = row.get::<String>("node_type")?;
    let file = row.get::<String>("file")?;
    let node_type = NodeType::from_str(&node_type_str)?;
    Ok(MutedNodeIdentifier {
        node_type,
        name,
        file,
    })
}

fn extract_boolean(row: &neo4rs::Row) -> Result<bool> {
    let is_muted = row.get::<bool>("is_muted").unwrap_or(false);
    Ok(is_muted)
}

fn extract_coverage_data(row: &neo4rs::Row) -> Result<(NodeData, usize, bool, usize, String)> {
    let node = row.get::<neo4rs::Node>("n")?;
    let node_data = NodeData::try_from(&node)?;
    let usage_count: i64 = row.get("usage_count").unwrap_or(0);
    let is_covered: bool = row.get("is_covered").unwrap_or(false);
    let test_count: i64 = row.get("test_count").unwrap_or(0);
    let ref_id = extract_ref_id(&node_data);
    Ok((
        node_data,
        usage_count as usize,
        is_covered,
        test_count as usize,
        ref_id,
    ))
}

async fn execute_query<T>(
    conn: &Neo4jConnection,
    query_str: String,
    params: BoltMap,
    extractor: impl Fn(&neo4rs::Row) -> Result<T>,
) -> Result<Vec<T>> {
    let query_obj = bind_parameters(&query_str, params);
    let mut results = Vec::new();

    match conn.execute(query_obj).await {
        Ok(mut stream) => {
            while let Ok(Some(row)) = stream.next().await {
                match extractor(&row) {
                    Ok(item) => results.push(item),
                    Err(_) => continue,
                }
            }
        }
        Err(e) => {
            return Err(e.into());
        }
    }

    Ok(results)
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
        println!("Processing chunk {}/{}", i + 1, total_chunks);

        let mut txn_manager = TransactionManager::new(conn);
        for query in chunk {
            txn_manager.add_query(query);
        }

        if let Err(e) = txn_manager.execute().await {
            println!("Error executing batch chunk {}: {:?}", i + 1, e);
            return Err(e);
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
    for (i, query) in queries.into_iter().enumerate() {
        println!("Processing query {}/{}", i + 1, total_queries);

        let mut txn_manager = TransactionManager::new(conn);
        txn_manager.add_query(query);
        txn_manager.execute().await?;

        println!("Successfully executed query {}/{}", i + 1, total_queries);
    }
    Ok(())
}

pub async fn execute_node_query(
    conn: &Neo4jConnection,
    query_str: String,
    params: BoltMap,
) -> Vec<NodeData> {
    execute_query(conn, query_str, params, extract_node_data)
        .await
        .unwrap_or_else(|e| {
            println!("Error executing query: {}", e);
            Vec::new()
        })
}

pub async fn execute_nodes_with_coverage_query(
    conn: &Neo4jConnection,
    query_str: String,
    params: BoltMap,
) -> Vec<(NodeData, usize, bool, usize, String)> {
    execute_query(conn, query_str, params, extract_coverage_data)
        .await
        .unwrap_or_else(|e| {
            eprintln!("Error executing nodes with coverage query: {}", e);
            Vec::new()
        })
}

pub async fn execute_muted_nodes_query(
    conn: &Neo4jConnection,
    query_str: String,
    params: BoltMap,
) -> Vec<MutedNodeIdentifier> {
    execute_query(conn, query_str, params, extract_muted_identifier)
        .await
        .unwrap_or_else(|_| Vec::new())
}

pub async fn execute_count_query(
    conn: &Neo4jConnection,
    query_str: String,
    params: BoltMap,
) -> usize {
    execute_query(conn, query_str, params, extract_count)
        .await
        .unwrap_or_else(|_| Vec::new())
        .first()
        .copied()
        .unwrap_or(0)
}

pub async fn execute_boolean_query(
    conn: &Neo4jConnection,
    query_str: String,
    params: BoltMap,
) -> bool {
    execute_query(conn, query_str, params, extract_boolean)
        .await
        .unwrap_or_else(|_| Vec::new())
        .first()
        .copied()
        .unwrap_or(false)
}
