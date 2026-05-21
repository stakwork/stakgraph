use crate::lang::graphs::helpers::MutedNodeIdentifier;
use crate::lang::{
    graphs::{helpers::*, queries::*, Neo4jGraph},
    Edge, NodeData, NodeType,
};
use neo4rs::{query, BoltMap, BoltType, Graph as Neo4jConnection, Query};
use shared::Result;
use std::future::Future;
use std::str::FromStr;
use std::time::Duration;
use tracing::{info, warn};

/// Returns true if the error string indicates a Neo4j transient error
/// (deadlock, leader switch, lock client stopped, etc.) — Neo4j explicitly
/// marks these as safe-to-retry.
///
/// We match on the formatted error string because `neo4rs::Error` does not
/// expose a structured error code in a stable way across versions.
fn is_transient_neo4j_error(err: &shared::Error) -> bool {
    let s = err.to_string();
    s.contains("TransientError")
        || s.contains("DeadlockDetected")
        || s.contains("LeaderSwitch")
        || s.contains("LockClientStopped")
}

/// Retry an async operation while it returns a transient Neo4j error.
/// Uses exponential backoff with jitter (50ms, 100ms, 200ms, ...).
/// Each attempt is bounded by a per-attempt timeout so stuck lock operations
/// surface as retryable failures instead of hanging the bolt connection.
/// Defaults: 5 attempts, 120s per-attempt timeout. Tunable via
/// `NEO4J_RETRY_ATTEMPTS` and `NEO4J_ATTEMPT_TIMEOUT_SECS` env vars.
pub async fn with_transient_retry<F, Fut, T>(label: &str, op: F) -> Result<T>
where
    F: FnMut() -> Fut,
    Fut: Future<Output = Result<T>>,
{
    with_transient_retry_inner(None, label, op).await
}

/// Same as `with_transient_retry`, but when an attempt times out we drop the
/// cached `neo4rs::Graph` on the provided `Neo4jGraph` so the next attempt
/// builds a fresh bolt connection pool. Without this, a single wedged bolt
/// socket inside neo4rs's pool can cause every retry to hang the same way
/// (we have seen this happen in production: one long-lived bolt connection
/// stops responding and the per-attempt timeout fires for all N retries).
pub async fn with_transient_retry_reconnect<F, Fut, T>(
    graph: &Neo4jGraph,
    label: &str,
    op: F,
) -> Result<T>
where
    F: FnMut() -> Fut,
    Fut: Future<Output = Result<T>>,
{
    with_transient_retry_inner(Some(graph), label, op).await
}

async fn with_transient_retry_inner<F, Fut, T>(
    graph: Option<&Neo4jGraph>,
    label: &str,
    mut op: F,
) -> Result<T>
where
    F: FnMut() -> Fut,
    Fut: Future<Output = Result<T>>,
{
    let max_attempts: u32 = std::env::var("NEO4J_RETRY_ATTEMPTS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(5);
    let attempt_timeout_secs: u64 = std::env::var("NEO4J_ATTEMPT_TIMEOUT_SECS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(120);
    let mut attempt: u32 = 0;
    loop {
        let result = if attempt_timeout_secs == 0 {
            // Per-attempt timeout disabled — preserve original behavior.
            Ok(op().await)
        } else {
            tokio::time::timeout(Duration::from_secs(attempt_timeout_secs), op()).await
        };
        match result {
            Ok(Ok(v)) => return Ok(v),
            Ok(Err(e)) if is_transient_neo4j_error(&e) && attempt + 1 < max_attempts => {
                attempt += 1;
                let backoff_ms = 50u64.saturating_mul(1u64 << (attempt - 1).min(6));
                warn!(
                    "[neo4j-retry] transient error on '{}' (attempt {}/{}), retrying in {}ms: {}",
                    label, attempt, max_attempts, backoff_ms, e
                );
                tokio::time::sleep(Duration::from_millis(backoff_ms)).await;
                continue;
            }
            Ok(Err(e)) => return Err(e),
            Err(_elapsed) if attempt + 1 < max_attempts => {
                attempt += 1;
                let backoff_ms = 50u64.saturating_mul(1u64 << (attempt - 1).min(6));
                warn!(
                    "[neo4j-retry] attempt timed out after {}s on '{}' (attempt {}/{}), retrying in {}ms",
                    attempt_timeout_secs, label, attempt, max_attempts, backoff_ms
                );
                // The bolt socket(s) used by the in-flight future may be stuck.
                // Reset the cached pool so the next attempt gets fresh sockets.
                if let Some(g) = graph {
                    if let Err(reconnect_err) = g.force_reconnect().await {
                        warn!(
                            "[neo4j-retry] force_reconnect after timeout failed on '{}': {}",
                            label, reconnect_err
                        );
                    }
                }
                tokio::time::sleep(Duration::from_millis(backoff_ms)).await;
                continue;
            }
            Err(_elapsed) => {
                return Err(shared::Error::dependency(format!(
                    "Neo4j operation '{}' timed out after {}s (all {} attempts)",
                    label, attempt_timeout_secs, max_attempts
                )));
            }
        }
    }
}

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

async fn execute_query<T: Send + 'static>(
    graph: &Neo4jGraph,
    query_str: String,
    params: BoltMap,
    extractor: impl Fn(&neo4rs::Row) -> Result<T> + Send + Sync + 'static,
) -> Result<Vec<T>> {
    let label = query_str.chars().take(80).collect::<String>();
    let extractor = std::sync::Arc::new(extractor);
    with_transient_retry_reconnect(graph, &label, || {
        let query_str = query_str.clone();
        let params = params.clone();
        let extractor = extractor.clone();
        async move {
            let conn = graph.ensure_connected().await?;
            let query_obj = bind_parameters(&query_str, params);
            let mut results = Vec::new();
            let mut stream = conn.execute(query_obj).await.map_err(|e| {
                shared::Error::dependency(format!("[neo4j-read] {}: {}", query_str.chars().take(80).collect::<String>(), e))
            })?;
            while let Ok(Some(row)) = stream.next().await {
                match extractor(&row) {
                    Ok(item) => results.push(item),
                    Err(_) => continue,
                }
            }
            Ok(results)
        }
    })
    .await
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
        let conn = self.conn;
        let queries = self.queries;
        // Retry the entire transaction on Neo4j transient errors (e.g. deadlocks).
        // Neo4j explicitly marks these as safe-to-retry — see
        // https://neo4j.com/docs/operations-manual/current/clustering/disaster-recovery/
        with_transient_retry("TransactionManager::execute", || {
            let queries = queries.clone();
            async move {
                let mut txn = conn.start_txn().await?;
                for (query_str, bolt_map) in queries {
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
                                query_obj = query_obj.param(
                                    "now",
                                    neo4rs::BoltType::String(format!("{:.7}", ts).into()),
                                );
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
        })
        .await
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
        info!("Processing chunk {}/{}", i + 1, total_chunks);

        let mut txn_manager = TransactionManager::new(conn);
        for query in chunk {
            txn_manager.add_query(query);
        }

        if let Err(e) = txn_manager.execute().await {
            warn!("Error executing batch chunk {}: {:?}", i + 1, e);
            return Err(e);
        }

        info!("Successfully committed chunk {}/{}", i + 1, total_chunks);
    }
    Ok(())
}

pub async fn execute_queries_simple(
    conn: &Neo4jConnection,
    queries: Vec<(String, BoltMap)>,
) -> Result<()> {
    let total_queries = queries.len();
    for (i, query) in queries.into_iter().enumerate() {
        info!("Processing query {}/{}", i + 1, total_queries);

        let mut txn_manager = TransactionManager::new(conn);
        txn_manager.add_query(query);
        txn_manager.execute().await?;

        info!("Successfully executed query {}/{}", i + 1, total_queries);
    }
    Ok(())
}

pub async fn execute_node_query(
    graph: &Neo4jGraph,
    query_str: String,
    params: BoltMap,
) -> Vec<NodeData> {
    let q = query_str.clone();
    execute_query(graph, query_str, params, extract_node_data)
        .await
        .unwrap_or_else(|e| {
            warn!(query = %q.chars().take(120).collect::<String>(), error = %e, "neo4j read failed after retries");
            Vec::new()
        })
}

pub async fn execute_nodes_with_coverage_query(
    graph: &Neo4jGraph,
    query_str: String,
    params: BoltMap,
) -> Vec<(NodeData, usize, bool, usize, String)> {
    let q = query_str.clone();
    execute_query(graph, query_str, params, extract_coverage_data)
        .await
        .unwrap_or_else(|e| {
            warn!(query = %q.chars().take(120).collect::<String>(), error = %e, "neo4j coverage read failed after retries");
            Vec::new()
        })
}

pub async fn execute_muted_nodes_query(
    graph: &Neo4jGraph,
    query_str: String,
    params: BoltMap,
) -> Vec<MutedNodeIdentifier> {
    let q = query_str.clone();
    execute_query(graph, query_str, params, extract_muted_identifier)
        .await
        .unwrap_or_else(|e| {
            warn!(query = %q.chars().take(120).collect::<String>(), error = %e, "neo4j muted-nodes read failed after retries");
            Vec::new()
        })
}

pub async fn execute_count_query(
    graph: &Neo4jGraph,
    query_str: String,
    params: BoltMap,
) -> usize {
    let q = query_str.clone();
    execute_query(graph, query_str, params, extract_count)
        .await
        .unwrap_or_else(|e| {
            warn!(query = %q.chars().take(120).collect::<String>(), error = %e, "neo4j count read failed after retries");
            Vec::new()
        })
        .first()
        .copied()
        .unwrap_or(0)
}

pub async fn execute_boolean_query(
    graph: &Neo4jGraph,
    query_str: String,
    params: BoltMap,
) -> bool {
    let q = query_str.clone();
    execute_query(graph, query_str, params, extract_boolean)
        .await
        .unwrap_or_else(|e| {
            warn!(query = %q.chars().take(120).collect::<String>(), error = %e, "neo4j boolean read failed after retries");
            Vec::new()
        })
        .first()
        .copied()
        .unwrap_or(false)
}
