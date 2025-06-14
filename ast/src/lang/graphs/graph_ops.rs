use crate::lang::graphs::graph::Graph;
use crate::lang::graphs::neo4j_graph::Neo4jGraph;
use crate::lang::graphs::BTreeMapGraph;
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
        let (nodes, edges) = self.graph.get_graph_size();
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
                    let file_graph = repo.build_graph_inner::<BTreeMapGraph>().await?;

                    self.upload_btreemap_to_neo4j(&file_graph).await?;
                    self.graph.create_indexes().await?;

                    for (edge, _target_data) in &all_incoming_edges {
                        let source_exists = self
                            .graph
                            .find_nodes_by_name(
                                edge.source.node_type.clone(),
                                &edge.source.node_data.name,
                            )
                            .iter()
                            .any(|n| n.file == edge.source.node_data.file);
                        let target_exists = self
                            .graph
                            .find_nodes_by_name(
                                edge.target.node_type.clone(),
                                &edge.target.node_data.name,
                            )
                            .iter()
                            .any(|n| n.file == edge.target.node_data.file);
                        if source_exists && target_exists {
                            self.graph.add_edge(edge.clone());
                        }
                    }

                    let (nodes_after, edges_after) = self.graph.get_graph_size();
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
        Ok(self.graph.get_graph_size())
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

        self.graph.clear().await?;
        self.upload_btreemap_to_neo4j(&temp_graph).await?;
        self.graph.create_indexes().await?;

        self.graph
            .update_repository_hash(repo_url, current_hash)
            .await?;
        Ok(self.graph.get_graph_size())
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
        let edges = btree_graph.to_array_graph_edges();
        for edge in edges {
            edges_txn_manager.add_edge(&edge);
        }
        edges_txn_manager.execute().await?;

        let (nodes, edges) = self.graph.get_graph_size();
        Ok((nodes, edges))
    }
}
