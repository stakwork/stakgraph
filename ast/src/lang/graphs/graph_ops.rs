use crate::lang::graphs::graph::Graph;
use crate::lang::graphs::neo4j_graph::Neo4jGraph;
use crate::lang::graphs::{BTreeMapGraph, Edge, NodeKeys, NodeRef};
use crate::lang::neo4j_utils::TransactionManager;
use crate::repo::{check_revs_files, Repo};
use anyhow::Result;
use tracing::info;

#[derive(Debug, Clone)]
pub struct GraphOps {
    pub graph: Neo4jGraph,
}

impl GraphOps {
    pub fn new() -> Self {
        Self {
            graph: Neo4jGraph::default(),
        }
    }

    pub async fn connect(&mut self) -> Result<()> {
        self.graph.connect().await
    }

    pub async fn clear(&mut self) -> Result<(u32, u32)> {
        self.graph.clear().await?;
        let (nodes, edges) = self.graph.get_graph_size().await?;
        info!("Graph cleared - Nodes: {}, Edges: {}", nodes, edges);
        Ok((nodes, edges))
    }

    pub async fn update_incremental(
        &mut self,
        repo_url: &str,
        repo_path: &str,
        current_hash: &str,
        stored_hash: &str,
    ) -> Result<(u32, u32)> {
        let revs = vec![stored_hash.to_string(), current_hash.to_string()];
        if let Some(modified_files) = check_revs_files(repo_path, revs.clone()) {
            info!(
                "Processing {} changed files between commits",
                modified_files.len()
            );

            if !modified_files.is_empty() {
                let mut all_incoming_edges = Vec::new();
                for file in &modified_files {
                    // Collect incoming edges before removing nodes
                    let incoming = self.graph.get_incoming_edges_for_file(file).await?;
                    all_incoming_edges.extend(incoming);
                    self.graph.remove_nodes_by_file(file).await?;
                }

                let file_repos = Repo::new_multi_detect(
                    repo_path,
                    Some(repo_url.to_string()),
                    modified_files.clone(),
                    Vec::new(),
                )
                .await?;

                for repo in &file_repos.0 {
                    // Build in-memory graph for this file
                    let file_graph = repo.build_graph_inner::<BTreeMapGraph>().await?;
                    // Upload to Neo4j
                    self.upload_btreemap_to_neo4j(&file_graph).await?;
                    self.graph.create_indexes().await?;

                    // Re-add incoming edges if both nodes exist
                    for (edge, _target_data) in &all_incoming_edges {
                        let source_exists = self
                            .graph
                            .find_nodes_by_name(
                                edge.source.node_type.clone(),
                                &edge.source.node_data.name,
                            )
                            .await
                            .iter()
                            .any(|n| n.file == edge.source.node_data.file);
                        let target_exists = self
                            .graph
                            .find_nodes_by_name(
                                edge.target.node_type.clone(),
                                &edge.target.node_data.name,
                            )
                            .await
                            .iter()
                            .any(|n| n.file == edge.target.node_data.file);
                        if source_exists && target_exists {
                            self.graph.add_edge(edge.clone()).await?;
                        }
                    }

                    let (nodes_after, edges_after) = self.graph.get_graph_size().await?;
                    info!(
                        "Updated files: added {} nodes and {} edges",
                        nodes_after, edges_after
                    );
                }
            }
        }
        self.graph
            .update_repository_hash(repo_url, current_hash)
            .await?;
        self.graph.get_graph_size().await
    }

    pub async fn update_full(
        &mut self,
        repo_url: &str,
        repo_path: &str,
        current_hash: &str,
    ) -> Result<(u32, u32)> {
        let repos = Repo::new_multi_detect(
            repo_path,
            Some(repo_url.to_string()),
            Vec::new(),
            Vec::new(),
        )
        .await?;

        let temp_graph = repos.build_graphs_inner::<BTreeMapGraph>().await?;

        temp_graph.analysis();

        self.upload_btreemap_to_neo4j(&temp_graph).await?;
        self.graph.create_indexes().await?;

        self.graph
            .update_repository_hash(repo_url, current_hash)
            .await?;
        self.graph.get_graph_size().await
    }
    pub async fn upload_btreemap_to_neo4j(
        &mut self,
        btree_graph: &BTreeMapGraph,
    ) -> anyhow::Result<(u32, u32)> {
        let connection = self.graph.ensure_connected().await?;

        let mut nodes_txn_manager = TransactionManager::new(&connection);
        for node in btree_graph.nodes.values() {
            nodes_txn_manager.add_node(&node.node_type, &node.node_data);
        }
        nodes_txn_manager.execute().await?;

        let mut edges_txn_manager = TransactionManager::new(&connection);
        for (src_key, dst_key, edge_type) in &btree_graph.edges {
            let src_base_key = src_key.rsplitn(2, '-').nth(1).unwrap_or(src_key);
            let dst_base_key = dst_key.rsplitn(2, '-').nth(1).unwrap_or(dst_key);

            let src_node = btree_graph
                .nodes
                .iter()
                .find(|(key, _)| key.starts_with(src_base_key))
                .map(|(_, node)| node);

            let dst_node = btree_graph
                .nodes
                .iter()
                .find(|(key, _)| key.starts_with(dst_base_key))
                .map(|(_, node)| node);

            if let (Some(src_node), Some(dst_node)) = (src_node, dst_node) {
                let edge = Edge::new(
                    edge_type.clone(),
                    NodeRef::from((&src_node.node_data).into(), src_node.node_type.clone()),
                    NodeRef::from((&dst_node.node_data).into(), dst_node.node_type.clone()),
                );
                edges_txn_manager.add_edge(&edge);
            }
        }
        edges_txn_manager.execute().await?;

        let (nodes, edges) = self.graph.get_graph_size().await?;
        Ok((nodes, edges))
    }
}
