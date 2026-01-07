use neo4rs::{query, BoltMap, ConfigBuilder, Graph as Neo4jConnection};
use shared::{Error, Result};

use crate::lang::graphs::{
    executor::{execute_batch, execute_queries_simple},
    migration::clear_graph_query,
};

use crate::lang::Neo4jGraph;
pub struct Neo4jConnectionManager;

impl Neo4jConnectionManager {
    pub async fn initialize(
        uri: &str,
        username: &str,
        password: &str,
        database: &str,
    ) -> Result<Neo4jConnection> {
        // info!("Connecting to Neo4j at {}", uri);
        let config = ConfigBuilder::new()
            .uri(uri)
            .user(username)
            .password(password)
            .db(database)
            .build()?;

        match Neo4jConnection::connect(config).await {
            Ok(connection) => {
                // info!("Successfully connected to Neo4j");
                // *conn_guard = Some(Arc::new(connection));
                Ok(connection)
            }
            Err(_) => Err(Error::Custom("Failed to connect to Neo4j : {e}".into())),
        }
    }
}

impl Neo4jGraph {
    pub async fn create_indexes(&self) -> Result<()> {
        let connection: Neo4jConnection = self.ensure_connected().await?;
        let queries = vec![
            "CREATE INDEX data_bank_node_key_index IF NOT EXISTS FOR (n:Data_Bank) ON (n.node_key)",
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

        // If namespace is "default", clear entire graph (backward compatible)
        // Otherwise, clear only nodes with this namespace (test isolation)
        let (clear_query_str, params) = if self.namespace == "default" {
            (clear_graph_query(), BoltMap::new())
        } else {
            crate::lang::graphs::migration::clear_graph_by_namespace_query(&self.namespace)
        };

        let mut query_obj = query(&clear_query_str);
        for (key, value) in params.value.iter() {
            query_obj = query_obj.param(key.value.as_str(), value.clone());
        }

        if let Err(e) = txn.run(query_obj).await {
            eprintln!("Error clearing stakgraph nodes: {:?}", e);
            txn.rollback().await?;
            return Err(Error::Custom(format!("Neo4j clear graph error: {}", e)));
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
