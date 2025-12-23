use crate::lang::graphs::{queries::*, NodeData, NodeKeys, NodeType, NodeRef, Edge, EdgeType, executor::TransactionManager, helpers::boltmap_insert_str, Neo4jGraph};
use neo4rs::{query, BoltMap};
use shared::{Error, Result};
use std::str::FromStr;


impl Neo4jGraph{
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
            Err(Error::Custom(format!(
                "No hash found for REPO {}",
                repo_url
            )))
        }
    }

    pub async fn remove_nodes_by_file(&self, file_path: &str) -> Result<u32> {
        let connection = self.ensure_connected().await?;
        let (query_str, params) = remove_nodes_by_file_query(file_path, &self.root);
        let mut query_obj = query(&query_str);
        for (k, v) in params.value.iter() {
            query_obj = query_obj.param(k.value.as_str(), v.clone());
        }
        let mut result = connection.execute(query_obj).await?;
        if let Some(row) = result.next().await? {
            Ok(row.get::<u32>("count").unwrap_or(0))
        } else {
            Ok(0)
        }
    }


    pub async fn update_repository_hash(&self, repo_name: &str, new_hash: &str) -> Result<()> {
        let connection = self.ensure_connected().await?;
        let mut txn_manager = TransactionManager::new(&connection);

        let (query, params) = update_repository_hash_query(repo_name, new_hash);
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
            return Err(Error::Custom(format!(
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

pub fn remove_nodes_by_file_query(file_path: &str, root: &str) -> (String, BoltMap) {
    let mut params = BoltMap::new();
    // let file_name = file_path.split('/').last().unwrap_or(file_path);
    boltmap_insert_str(&mut params, "file_name", file_path);
    boltmap_insert_str(&mut params, "root", root);

    let query = "
        MATCH (n)
        WHERE (n.file = $file_name OR n.file ENDS WITH $file_name)
        AND n.file STARTS WITH $root
        WITH DISTINCT n
        OPTIONAL MATCH (n)-[r]-() 
        DELETE r
        WITH n
        DETACH DELETE n
        RETURN count(n) as deleted
    ";

    (query.to_string(), params)
}
pub fn update_repository_hash_query(repo_name: &str, new_hash: &str) -> (String, BoltMap) {
    let mut params = BoltMap::new();
    boltmap_insert_str(&mut params, "repo_name", repo_name);
    boltmap_insert_str(&mut params, "new_hash", new_hash);

    let query = "MATCH (r:Repository) 
                 WHERE r.name CONTAINS $repo_name 
                 SET r.hash = $new_hash";

    (query.to_string(), params)
}

pub fn update_endpoint_name_query(old_name: &str, file: &str, new_name: &str) -> (String, BoltMap) {
    let mut params = BoltMap::new();
    boltmap_insert_str(&mut params, "old_name", old_name);
    boltmap_insert_str(&mut params, "file", file);
    boltmap_insert_str(&mut params, "new_name", new_name);

    let query = "MATCH (n:Endpoint {name: $old_name, file: $file})
                 SET n.name = $new_name
                 RETURN n";

    (query.to_string(), params)
}

pub fn update_endpoint_relationships_query(
    old_name: &str,
    file: &str,
    new_name: &str,
) -> (String, BoltMap) {
    let mut params = BoltMap::new();
    boltmap_insert_str(&mut params, "old_name", old_name);
    boltmap_insert_str(&mut params, "file", file);
    boltmap_insert_str(&mut params, "new_name", new_name);

    let query = "MATCH (source:Endpoint {{name: $old_name, file: $file}})-[r]->(target)
                SET source.name = $new_name
                RETURN r";

    (query.to_string(), params)
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

