use crate::lang::graphs::{
    executor::{with_transient_retry_reconnect, TransactionManager},
    helpers::{boltmap_insert_list, boltmap_insert_str},
    queries::*,
    Edge, EdgeType, Neo4jGraph, NodeData, NodeKeys, NodeRef, NodeType,
};
use neo4rs::{query, BoltMap};
use shared::{Error, Result};
use std::str::FromStr;

impl Neo4jGraph {
    pub async fn get_repository_hash(&self, repo_url: &str) -> Result<String> {
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
            Err(Error::not_found(format!(
                "No hash found for REPO {}",
                repo_url
            )))
        }
    }

    pub async fn remove_nodes_by_files(&self, file_paths: &[String]) -> Result<u32> {
        if file_paths.is_empty() {
            return Ok(0);
        }

        // Two-phase delete:
        //   1. count the matching nodes (cheap, uses `data_bank_file_index`)
        //   2. detach-delete them via `CALL { ... } IN TRANSACTIONS`, which
        //      commits in chunks and releases locks between chunks instead of
        //      acquiring write locks for every node + every relationship in
        //      one giant transaction.
        //
        // Why this matters: a previous version ran a single
        //   MATCH (n:Data_Bank) WHERE n.file IN $files DETACH DELETE n RETURN count(n)
        // which, for a modified-file set with high-degree nodes (e.g. a `File`
        // with thousands of CONTAINS rels), produces a lock set large enough
        // to stall on internal locking or hit the bolt-stream timeout. With
        // CALL { ... } IN TRANSACTIONS each chunk commits independently, so
        // worst case we lose progress on the current chunk on retry rather
        // than blocking for minutes on the whole delete.
        //
        // Note: CALL { ... } IN TRANSACTIONS cannot run inside an explicit
        // transaction — we deliberately use `connection.execute` (auto-commit)
        // here, NOT `start_txn`.

        let chunk_size: u32 = std::env::var("NEO4J_DELETE_CHUNK_ROWS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1000);

        // Phase 1: count.
        let count_files = file_paths.to_vec();
        let count = with_transient_retry_reconnect(self, "remove_nodes_by_files.count", || {
            let count_files = count_files.clone();
            async move {
                let connection = self.ensure_connected().await?;
                let (q, params) = count_nodes_by_files_query(&count_files);
                let mut query_obj = query(&q);
                for (k, v) in params.value.iter() {
                    query_obj = query_obj.param(k.value.as_str(), v.clone());
                }
                let mut result = connection.execute(query_obj).await?;
                if let Some(row) = result.next().await? {
                    Ok(row.get::<u32>("total").unwrap_or(0))
                } else {
                    Ok(0u32)
                }
            }
        })
        .await?;

        if count == 0 {
            return Ok(0);
        }

        // Phase 2: chunked detach-delete.
        // `connection.run` is the fire-and-forget auto-commit variant; the
        // chunked delete returns no rows, so there is nothing to stream.
        // (Crucially, this is NOT `start_txn` — `CALL { ... } IN TRANSACTIONS`
        // is rejected inside an explicit transaction.)
        let delete_files = file_paths.to_vec();
        with_transient_retry_reconnect(self, "remove_nodes_by_files.delete", || {
            let delete_files = delete_files.clone();
            async move {
                let connection = self.ensure_connected().await?;
                let (q, params) =
                    remove_nodes_by_files_chunked_query(&delete_files, chunk_size);
                let mut query_obj = query(&q);
                for (k, v) in params.value.iter() {
                    query_obj = query_obj.param(k.value.as_str(), v.clone());
                }
                connection.run(query_obj).await?;
                Ok::<(), shared::Error>(())
            }
        })
        .await?;

        Ok(count)
    }

    pub async fn update_repository_hash(&self, repo_url: &str, new_hash: &str) -> Result<()> {
        let connection = self.ensure_connected().await?;
        let mut txn_manager = TransactionManager::new(&connection);

        let (query, params) = update_repository_hash_query(repo_url, new_hash);
        txn_manager.add_query((query, params));

        txn_manager.execute().await
    }

    pub async fn get_incoming_edges_for_file(&self, file: &str) -> Result<Vec<(Edge, NodeData)>> {
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
                    let source_data = NodeData::try_from(&source_node).unwrap_or_default();
                    let target_data = NodeData::try_from(&target_node).unwrap_or_default();
                    let source_ref = NodeRef::from(NodeKeys::from(&source_data), source_type);
                    let target_ref = NodeRef::from(NodeKeys::from(&target_data), target_type);
                    let edge = Edge::new(edge_type, source_ref, target_ref);
                    incoming.push((edge, target_data));
                }
            }
        }
        Ok(incoming)
    }

    pub async fn clear_existing_graph(&self, root: &str) -> Result<()> {
        let connection = self.ensure_connected().await?;
        println!("Clearing existing graph for root: {}", root);
        let (query_str, params) = clear_existing_graph_query(root);
        let mut txn = connection.start_txn().await?;
        let mut query_obj = query(&query_str);
        for (key, value) in params.value.iter() {
            query_obj = query_obj.param(key.value.as_str(), value.clone());
        }
        if let Err(e) = txn.run(query_obj).await {
            txn.rollback().await?;
            return Err(Error::internal(format!(
                "Neo4j clear existing graph error: {}",
                e
            )));
        }
        txn.commit().await?;
        Ok(())
    }

    pub async fn get_dynamic_edges_for_file(
        &self,
        file: &str,
    ) -> Result<Vec<(String, String, String, String, String)>> {
        let connection = self.ensure_connected().await?;
        let (query_str, params) = find_dynamic_edges_for_file_query(file);
        let mut query_obj = query(&query_str);
        for (k, v) in params.value.iter() {
            query_obj = query_obj.param(k.value.as_str(), v.clone());
        }
        let mut edges = Vec::new();
        let mut result = connection.execute(query_obj).await?;
        while let Some(row) = result.next().await? {
            if let (
                Ok(source_ref_id),
                Ok(edge_type),
                Ok(target_name),
                Ok(target_file),
                Ok(target_type),
            ) = (
                row.get::<String>("source_ref_id"),
                row.get::<String>("edge_type"),
                row.get::<String>("target_name"),
                row.get::<String>("target_file"),
                row.get::<String>("target_type"),
            ) {
                edges.push((
                    source_ref_id,
                    edge_type,
                    target_name,
                    target_file,
                    target_type,
                ));
            }
        }
        Ok(edges)
    }

    pub async fn get_all_dynamic_edges(
        &self,
    ) -> Result<Vec<(String, String, String, String, String)>> {
        let connection = self.ensure_connected().await?;
        let (query_str, params) = find_all_dynamic_edges_query();
        let mut query_obj = query(&query_str);
        for (k, v) in params.value.iter() {
            query_obj = query_obj.param(k.value.as_str(), v.clone());
        }
        let mut edges = Vec::new();
        let mut result = connection.execute(query_obj).await?;
        while let Some(row) = result.next().await? {
            if let (
                Ok(source_ref_id),
                Ok(edge_type),
                Ok(target_name),
                Ok(target_file),
                Ok(target_type),
            ) = (
                row.get::<String>("source_ref_id"),
                row.get::<String>("edge_type"),
                row.get::<String>("target_name"),
                row.get::<String>("target_file"),
                row.get::<String>("target_type"),
            ) {
                edges.push((
                    source_ref_id,
                    edge_type,
                    target_name,
                    target_file,
                    target_type,
                ));
            }
        }
        Ok(edges)
    }

    pub async fn restore_dynamic_edges(
        &self,
        edges: Vec<(String, String, String, String, String)>,
    ) -> Result<usize> {
        if edges.is_empty() {
            return Ok(0);
        }

        let connection = self.ensure_connected().await?;
        let mut restored_count = 0;

        for (source_ref_id, edge_type, target_name, target_file, target_type) in edges {
            let (query_str, params) = restore_dynamic_edge_query(
                &source_ref_id,
                &edge_type,
                &target_name,
                &target_file,
                &target_type,
            );
            let mut query_obj = query(&query_str);
            for (k, v) in params.value.iter() {
                query_obj = query_obj.param(k.value.as_str(), v.clone());
            }

            match connection.execute(query_obj).await {
                Ok(mut result) => {
                    if result.next().await?.is_some() {
                        restored_count += 1;
                    }
                }
                Err(e) => {
                    println!(
                        "Failed to restore edge {} -> {} -> {}:{}: {}",
                        source_ref_id, edge_type, target_type, target_name, e
                    );
                }
            }
        }

        Ok(restored_count)
    }
}

pub fn get_repository_hash_query(repo_url: &str) -> (String, BoltMap) {
    let mut params = BoltMap::new();

    let repo_name = if repo_url.contains('/') {
        let parts: Vec<&str> = repo_url.split('/').collect();
        let name = parts.last().unwrap_or(&repo_url);
        name.trim_end_matches(".git")
    } else {
        repo_url
    };

    boltmap_insert_str(&mut params, "repo_name", repo_name);
    let query = "MATCH (r:Repository) 
                 WHERE r.name CONTAINS $repo_name 
                 RETURN r.hash as hash";

    (query.to_string(), params)
}

/// Single-statement detach-delete. Kept for callers that explicitly want the
/// all-in-one-transaction semantics; the incremental sync path prefers the
/// chunked variant below to avoid massive lock sets.
pub fn remove_nodes_by_files_query(file_paths: &[String]) -> (String, BoltMap) {
    let mut params = BoltMap::new();
    let files: Vec<neo4rs::BoltType> = file_paths
        .iter()
        .map(|p| neo4rs::BoltType::String(p.clone().into()))
        .collect();
    boltmap_insert_list(&mut params, "files", files);

    let query = "
        MATCH (n:Code)
        WHERE n.file IN $files
        DETACH DELETE n
        RETURN count(n) AS deleted
    ";

    (query.to_string(), params)
}

/// Count `:Data_Bank` nodes belonging to the given file set. Uses the range
/// index `data_bank_file_index` on `:Data_Bank(file)`, so it's cheap even for
/// large file lists.
pub fn count_nodes_by_files_query(file_paths: &[String]) -> (String, BoltMap) {
    let mut params = BoltMap::new();
    let files: Vec<neo4rs::BoltType> = file_paths
        .iter()
        .map(|p| neo4rs::BoltType::String(p.clone().into()))
        .collect();
    boltmap_insert_list(&mut params, "files", files);

    let query = "
        MATCH (n:Data_Bank)
        WHERE n.file IN $files
        RETURN count(n) AS total
    ";

    (query.to_string(), params)
}

/// Chunked detach-delete using `CALL { ... } IN TRANSACTIONS`. Each chunk
/// commits independently, so a single failing chunk does not roll back the
/// whole delete and locks are released between chunks. MUST be run as an
/// auto-commit query (`connection.execute` / `connection.run`), NOT inside
/// `start_txn` — Neo4j rejects `CALL IN TRANSACTIONS` inside an explicit txn.
pub fn remove_nodes_by_files_chunked_query(
    file_paths: &[String],
    chunk_rows: u32,
) -> (String, BoltMap) {
    let mut params = BoltMap::new();
    let files: Vec<neo4rs::BoltType> = file_paths
        .iter()
        .map(|p| neo4rs::BoltType::String(p.clone().into()))
        .collect();
    boltmap_insert_list(&mut params, "files", files);

    // `chunk_rows` is interpolated rather than passed as a bolt parameter
    // because Neo4j does not accept parameters in the `OF $x ROWS` clause.
    // It is clamped + sourced from env / a typed `u32`, so there is no
    // injection vector.
    let chunk_rows = chunk_rows.max(1);
    let query = format!(
        "
        MATCH (n:Data_Bank)
        WHERE n.file IN $files
        CALL {{ WITH n DETACH DELETE n }} IN TRANSACTIONS OF {chunk_rows} ROWS
        "
    );

    (query, params)
}
pub fn migrate_code_labels_query() -> String {
        "
                MATCH (n:Data_Bank)
                WHERE NOT 'Code' IN labels(n)
                    AND any(label IN labels(n) WHERE label IN [
                        'Repository', 'Package', 'Language', 'Directory', 'File', 'Import',
                        'Library', 'Class', 'Trait', 'Instance', 'Function', 'Endpoint',
                        'Request', 'DataModel', 'Page', 'Var', 'UnitTest',
                        'IntegrationTest', 'E2eTest'
                    ])
                CALL { WITH n SET n:Code } IN TRANSACTIONS OF 1000 ROWS
        "
        .to_string()
}

pub fn update_repository_hash_query(repo_url: &str, new_hash: &str) -> (String, BoltMap) {
    let mut params = BoltMap::new();

    let name = if repo_url.contains('/') {
        let parts: Vec<&str> = repo_url.split('/').collect();
        let n = parts.last().unwrap_or(&repo_url);
        n.trim_end_matches(".git")
    } else {
        repo_url
    };

    boltmap_insert_str(&mut params, "repo_name", name);
    boltmap_insert_str(&mut params, "new_hash", new_hash);

    let query = "MATCH (r:Repository) 
                 WHERE r.name CONTAINS $repo_name 
                 SET r.hash = $new_hash";

    (query.to_string(), params)
}

pub fn update_endpoint_name_query(
    old_name: &str,
    file: &str,
    verb: Option<&str>,
    new_name: &str,
    new_key: &str,
) -> (String, BoltMap) {
    let mut params = BoltMap::new();
    boltmap_insert_str(&mut params, "old_name", old_name);
    boltmap_insert_str(&mut params, "file", file);
    boltmap_insert_str(&mut params, "new_name", new_name);
    boltmap_insert_str(&mut params, "new_key", new_key);
    let verb_clause = if let Some(v) = verb {
        boltmap_insert_str(&mut params, "verb", v);
        " AND n.verb = $verb"
    } else {
        ""
    };

    let query = format!(
        "MATCH (n:Endpoint {{name: $old_name, file: $file}}){}\n                 SET n.name = $new_name, n.node_key = $new_key\n                 RETURN n",
        verb_clause
    );

    (query, params)
}

pub fn update_endpoint_relationships_query(
    old_name: &str,
    file: &str,
    verb: Option<&str>,
    new_name: &str,
) -> (String, BoltMap) {
    let mut params = BoltMap::new();
    boltmap_insert_str(&mut params, "old_name", old_name);
    boltmap_insert_str(&mut params, "file", file);
    boltmap_insert_str(&mut params, "new_name", new_name);
    let verb_clause = if let Some(v) = verb {
        boltmap_insert_str(&mut params, "verb", v);
        " AND source.verb = $verb"
    } else {
        ""
    };

    let query = format!(
        "MATCH (source:Endpoint {{name: $old_name, file: $file}}){}-[r]->(target)\n                SET source.name = $new_name\n                RETURN r",
        verb_clause
    );

    (query, params)
}

pub fn clear_graph_query() -> String {
    "MATCH (n:Data_Bank) DETACH DELETE n".to_string()
}

pub fn clear_existing_graph_query(root: &str) -> (String, BoltMap) {
    let mut params = BoltMap::new();
    boltmap_insert_str(&mut params, "root", root);
    let query = "MATCH (n:Data_Bank) WHERE n.file STARTS WITH $root DETACH DELETE n";
    (query.to_string(), params)
}

pub fn set_missing_data_bank_query() -> String {
    r#"
        MATCH (n)
        WHERE n.Data_Bank IS NULL AND n.name IS NOT NULL AND NOT n:Schema
        SET n.Data_Bank = n.name
        RETURN count(n) as updated_count
    "#
    .to_string()
}

pub fn set_default_namespace_query() -> String {
    r#"
        MATCH (n)
        WHERE n.namespace IS NULL AND NOT n:Schema
        SET n.namespace = "default"
        RETURN count(n) as updated_count
    "#
    .to_string()
}
