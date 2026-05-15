use neo4rs::{query, BoltMap, ConfigBuilder, Graph as Neo4jConnection};
use shared::{Error, Result};

use crate::lang::graphs::{
    executor::{execute_batch, execute_queries_simple},
    migration::clear_graph_query,
    Neo4jConfig,
};

use crate::lang::Neo4jGraph;
pub struct Neo4jConnectionManager;

impl Neo4jConnectionManager {
    /// Build a `neo4rs::Graph` (a pooled bolt connection handle).
    ///
    /// IMPORTANT: previously this ignored `Neo4jConfig::max_connections` and
    /// `fetch_size`, so the entire service ran on neo4rs's default pool. In
    /// production this manifested as a single long-lived bolt socket carrying
    /// every query — when that socket got into a bad state (idle middlebox,
    /// server-side bolt thread stuck, etc.) every subsequent query hung until
    /// our tokio per-attempt timeout fired, and retries reused the same wedged
    /// socket. Configuring an explicit pool size means a wedged socket can't
    /// take down the whole service.
    pub async fn initialize(cfg: &Neo4jConfig) -> Result<Neo4jConnection> {
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
            .await
            .map_err(|e| Error::dependency(format!("Failed to connect to Neo4j: {e}")))
    }
}

impl Neo4jGraph {
    pub async fn create_indexes(&self) -> Result<()> {
        let connection: Neo4jConnection = self.ensure_connected().await?;
        let queries = vec![
            "CREATE INDEX data_bank_node_key_index IF NOT EXISTS FOR (n:Data_Bank) ON (n.node_key)",
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
            eprintln!("Error clearing stakgraph nodes: {:?}", e);
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
