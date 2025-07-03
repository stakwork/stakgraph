use crate::lang::graphs::graph::Graph;
use crate::lang::graphs::neo4j_graph::Neo4jGraph;
use crate::lang::graphs::BTreeMapGraph;
use crate::lang::neo4j_utils::{add_node_query, EdgeQueryBuilder};
use crate::lang::{NodeData, NodeType};
use crate::repo::{check_revs_files, Repo};
use anyhow::Result;
use neo4rs::BoltMap;
use tracing::{debug, info};

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

    pub async fn fetch_repo(&mut self, repo_name: &str) -> Result<NodeData> {
        let repo = self
            .graph
            .find_nodes_by_name_async(NodeType::Repository, repo_name)
            .await;
        if repo.is_empty() {
            return Err(anyhow::anyhow!("Repo not found"));
        }
        Ok(repo[0].clone())
    }

    pub async fn update_incremental(
        &mut self,
        repo_url: &str,
        username: Option<String>,
        pat: Option<String>,
        current_hash: &str,
        stored_hash: &str,
        commit: Option<&str>,
    ) -> Result<(u32, u32)> {
        let revs = vec![stored_hash.to_string(), current_hash.to_string()];
        let repo_path = Repo::get_path_from_url(repo_url)?;
        if let Some(modified_files) = check_revs_files(&repo_path, revs.clone()) {
            info!(
                "Processing {} changed files between commits",
                modified_files.len()
            );

            if !modified_files.is_empty() {
                for file in &modified_files {
                    self.graph.remove_nodes_by_file(file).await?;
                }

                let subgraph_repos = Repo::new_clone_multi_detect(
                    repo_url,
                    username.clone(),
                    pat.clone(),
                    modified_files.clone(),
                    Vec::new(),
                    commit,
                )
                .await?;

                for repo in &subgraph_repos.0 {
                    self.graph = repo.build_graph_inner::<Neo4jGraph>().await?;
                }

                let (nodes_after, edges_after) = self.graph.get_graph_size_async().await?;
                info!(
                    "Updated files: total {} nodes and {} edges",
                    nodes_after, edges_after
                );
            }
        }
        self.graph
            .update_repository_hash(repo_url, current_hash)
            .await?;
        self.graph.get_graph_size_async().await
    }

    pub async fn update_full(
        &mut self,
        repo_url: &str,
        username: Option<String>,
        pat: Option<String>,
        current_hash: &str,
        commit: Option<&str>,
    ) -> Result<(u32, u32)> {
        let repos = Repo::new_clone_multi_detect(
            repo_url,
            username.clone(),
            pat.clone(),
            Vec::new(),
            Vec::new(),
            commit,
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
        Ok(self.graph.get_graph_size_async().await?)
    }

    pub async fn upload_btreemap_to_neo4j(
        &mut self,
        btree_graph: &BTreeMapGraph,
    ) -> anyhow::Result<(u32, u32)> {
        self.graph.ensure_connected().await?;

        debug!("preparing node upload {}", btree_graph.nodes.len());
        let node_queries: Vec<(String, BoltMap)> = btree_graph
            .nodes
            .values()
            .map(|node| add_node_query(&node.node_type, &node.node_data))
            .collect();

        debug!("executing node upload in batches");
        self.graph.execute_batch(node_queries).await?;
        debug!("node upload complete");

        debug!("preparing edge upload {}", btree_graph.edges.len());
        let edge_queries: Vec<(String, BoltMap)> = btree_graph
            .edges
            .iter()
            .map(|(source_key, target_key, edge_type)| {
                EdgeQueryBuilder::build_from_keys(source_key, target_key, edge_type)
            })
            .collect();

        debug!("executing edge upload in batches");
        self.graph.execute_batch(edge_queries).await?;
        debug!("edge upload complete!");

        let (nodes, edges) = self.graph.get_graph_size_async().await?;
        debug!("upload complete! nodes: {}, edges: {}", nodes, edges);
        Ok((nodes, edges))
    }
}
