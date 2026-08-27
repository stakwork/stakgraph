use neo4rs::{query, BoltMap, ConfigBuilder, Graph as Neo4jConnection};
use shared::{Error, Result};
use tracing::warn;

use crate::lang::graphs::{
    executor::{execute_batch, execute_queries_simple},
    migration::clear_graph_query,
    Neo4jConfig,
};

use crate::lang::Neo4jGraph;
pub struct Neo4jConnectionManager;

/// Cache key for the shared pool. If a caller ever passes a `Neo4jConfig`
/// pointing at a different server/db, it gets a fresh pool (replacing the
/// cached one) instead of silently reusing the wrong connection.
type PoolKey = (String, String, String);

struct CachedPool {
    key: PoolKey,
    pool: Neo4jConnection,
}

/// Process-wide shared pool. `neo4rs::Graph` is an Arc'd pool handle, so
/// cloning it is cheap and every clone talks to the same set of bolt sockets.
/// The tokio Mutex is held across pool construction on purpose: concurrent
/// first callers serialize, a FAILED connect is never cached, and the next
/// caller retries.
static SHARED_POOL: tokio::sync::Mutex<Option<CachedPool>> = tokio::sync::Mutex::const_new(None);

#[cfg(test)]
pub static POOL_BUILDS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

impl Neo4jConnectionManager {
    /// Return the process-wide `neo4rs::Graph` (a pooled bolt connection
    /// handle), building it on first use.
    ///
    /// Every `Neo4jGraph`/`GraphOps` in the process shares this one pool:
    /// previously each call built a brand-new pool, so per-request
    /// `GraphOps::new().connect()` callers opened `max_connections` fresh bolt
    /// sockets per operation and concurrent load could hold ~70 connections
    /// against a single Neo4j instance (enough to OOM a small swarm node).
    ///
    /// IMPORTANT: this must keep configuring the pool explicitly. At one point
    /// it ignored `Neo4jConfig::max_connections` and `fetch_size`, so the
    /// service ran on neo4rs's default pool. In production this manifested as
    /// a single long-lived bolt socket carrying every query — when that socket
    /// got into a bad state (idle middlebox, server-side bolt thread stuck,
    /// etc.) every subsequent query hung until our tokio per-attempt timeout
    /// fired, and retries reused the same wedged socket. Configuring an
    /// explicit pool size means a wedged socket can't take down the whole
    /// service. (`Neo4jGraph::force_reconnect` handles the wedged case for the
    /// shared pool by calling `invalidate` below.)
    pub async fn initialize(cfg: &Neo4jConfig) -> Result<Neo4jConnection> {
        let key: PoolKey = (cfg.uri.clone(), cfg.username.clone(), cfg.database.clone());

        let mut cached = SHARED_POOL.lock().await;
        if let Some(c) = cached.as_ref() {
            if c.key == key {
                return Ok(c.pool.clone());
            }
        }

        let pool = Self::build_pool(cfg).await?;
        *cached = Some(CachedPool {
            key,
            pool: pool.clone(),
        });
        Ok(pool)
    }

    /// Drop the cached shared pool so the next `initialize` builds a fresh
    /// one. Used by `Neo4jGraph::force_reconnect` when the pool's bolt
    /// socket(s) may be wedged — without this, re-connecting would just hand
    /// back the same stuck pool.
    pub async fn invalidate() {
        *SHARED_POOL.lock().await = None;
    }

    async fn build_pool(cfg: &Neo4jConfig) -> Result<Neo4jConnection> {
        #[cfg(test)]
        POOL_BUILDS.fetch_add(1, std::sync::atomic::Ordering::SeqCst);

        let max_connections = std::env::var("NEO4J_MAX_CONNECTIONS")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(cfg.max_connections)
            .max(1);
        let fetch_size = std::env::var("NEO4J_FETCH_SIZE")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(500);

        let config = ConfigBuilder::new()
            .uri(&cfg.uri)
            .user(&cfg.username)
            .password(&cfg.password)
            .db(cfg.database.as_str())
            .max_connections(max_connections)
            .fetch_size(fetch_size)
            .build()?;

        Neo4jConnection::connect(config)
            .map_err(|e| Error::dependency(format!("Failed to connect to Neo4j: {e}")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::Ordering;

    #[tokio::test]
    async fn initialize_reuses_shared_pool() {
        let cfg = Neo4jConfig::default();
        if Neo4jConnectionManager::initialize(&cfg).await.is_err() {
            eprintln!("skipping initialize_reuses_shared_pool: no Neo4j reachable");
            return;
        }
        let builds = POOL_BUILDS.load(Ordering::SeqCst);
        for _ in 0..3 {
            Neo4jConnectionManager::initialize(&cfg).await.unwrap();
        }
        assert_eq!(
            POOL_BUILDS.load(Ordering::SeqCst),
            builds,
            "repeated initialize() calls must reuse the cached pool"
        );

        Neo4jConnectionManager::invalidate().await;
        Neo4jConnectionManager::initialize(&cfg).await.unwrap();
        assert_eq!(
            POOL_BUILDS.load(Ordering::SeqCst),
            builds + 1,
            "invalidate() must force the next initialize() to build a fresh pool"
        );
    }
}

impl Neo4jGraph {
    pub async fn create_indexes(&self) -> Result<()> {
        let connection: Neo4jConnection = self.ensure_connected().await?;
        let queries = vec![
            "CREATE INDEX data_bank_node_key_index IF NOT EXISTS FOR (n:Data_Bank) ON (n.node_key)",
            "CREATE INDEX data_bank_ref_id_index IF NOT EXISTS FOR (n:Data_Bank) ON (n.ref_id)",
            // Range index on `file` so incremental sync deletions
            // (`remove_nodes_by_files_query` -> `n.file IN $files`) and other
            // file-scoped lookups can do an index seek instead of a full
            // `:Data_Bank` label scan. Without this an incremental sync had
            // to walk every project node for every modified file.
            "CREATE INDEX data_bank_file_index IF NOT EXISTS FOR (n:Data_Bank) ON (n.file)",
            "CREATE FULLTEXT INDEX bodyIndex IF NOT EXISTS FOR (n:Data_Bank) ON EACH [n.body]",
            "CREATE FULLTEXT INDEX nameIndex IF NOT EXISTS FOR (n:Data_Bank) ON EACH [n.name]",
            "CREATE FULLTEXT INDEX nameBodyFileIndex IF NOT EXISTS FOR (n:Data_Bank) ON EACH [n.name, n.body, n.file]",
            "CREATE VECTOR INDEX vectorIndex IF NOT EXISTS FOR (n:Data_Bank) ON (n.embeddings) OPTIONS {indexConfig: {`vector.dimensions`: 384, `vector.similarity_function`: 'cosine'}}"
        ];

        for q in queries {
            if let Err(e) = connection.run(neo4rs::query(q)).await {
                tracing::warn!("Error creating index: {:?}", e);
            }
        }
        Ok(())
    }

    pub async fn clear(&self) -> Result<()> {
        let connection = self.ensure_connected().await?;
        let mut txn = connection.start_txn().await?;

        let clear_query = clear_graph_query();
        let query_obj = query(&clear_query);

        if let Err(e) = txn.run(query_obj).await {
            warn!("Error clearing stakgraph nodes: {:?}", e);
            txn.rollback().await?;
            return Err(Error::internal(format!("Neo4j clear graph error: {}", e)));
        }

        txn.commit().await?;
        Ok(())
    }

    pub async fn execute_batch(&self, queries: Vec<(String, BoltMap)>) -> Result<()> {
        let connection = self.ensure_connected().await?;
        execute_batch(&connection, queries).await
    }

    pub async fn execute_simple(&self, queries: Vec<(String, BoltMap)>) -> Result<()> {
        let connection = self.ensure_connected().await?;
        execute_queries_simple(&connection, queries).await
    }
}
