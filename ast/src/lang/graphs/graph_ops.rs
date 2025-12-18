use std::collections::HashSet;
use std::time::Duration;

use crate::lang::embedding::{vectorize_code_document, vectorize_query};
use crate::lang::graphs::graph::Graph;
use crate::lang::graphs::neo4j_graph::Neo4jGraph;
use crate::lang::graphs::utils::tests_sources;
use crate::lang::graphs::BTreeMapGraph;
use crate::lang::neo4j_utils::{add_node_query, build_batch_edge_queries};
use crate::lang::{EdgeType, Node, NodeData, NodeType};
use crate::repo::{check_revs_files, Repo, StatusUpdate};
use crate::utils::create_node_key;
use neo4rs::BoltMap;
use shared::error::{Error, Result};
use tokio::sync::broadcast::Sender;
use tracing::{debug, error, info};

#[derive(Debug, Clone)]
pub struct GraphOps {
    pub graph: Neo4jGraph,
}

#[derive(Debug, Clone)]
pub struct CoverageStat {
    pub total: usize,
    pub total_tests: usize,
    pub covered: usize,
    pub percent: f64,
    pub total_lines: usize,
    pub covered_lines: usize,
    pub line_percent: f64,
}

#[derive(Debug, Clone)]
pub struct MockStat {
    pub total: usize,
    pub mocked: usize,
    pub percent: f64,
}

#[derive(Debug, Clone)]
pub struct MutedNodeIdentifier {
    pub node_type: NodeType,
    pub name: String,
    pub file: String,
}

#[derive(Debug, Clone)]
pub struct GraphCoverage {
    pub language: Option<String>,
    pub unit_tests: Option<CoverageStat>,
    pub integration_tests: Option<CoverageStat>,
    pub e2e_tests: Option<CoverageStat>,
    pub mocks: Option<MockStat>,
}

impl GraphOps {
    pub fn new() -> Self {
        Self {
            graph: Neo4jGraph::default(),
        }
    }

    pub async fn get_graph_size(&self) -> Result<(u32, u32)> {
        self.graph.get_graph_size_async().await
    }
    pub async fn fetch_all_node_keys(&self) -> Result<Vec<String>> {
        self.graph.fetch_all_node_keys().await
    }
    pub async fn fetch_all_edge_triples(&self) -> Result<Vec<(String, String, EdgeType)>> {
        self.graph.fetch_all_edge_triples().await
    }

    pub async fn connect(&mut self) -> Result<()> {
        self.graph.connect().await
    }

    pub async fn check_connection(&mut self) -> Result<()> {
        self.connect().await?;
        let check_timeout = Duration::from_secs(5);
        info!(
            "Verifying database connection with a {} second timeout...",
            check_timeout.as_secs()
        );

        match tokio::time::timeout(check_timeout, self.graph.get_graph_size_async()).await {
            Ok(Ok(_)) => {
                info!("Database connection verified successfully.");
                Ok(())
            }
            Ok(Err(e)) => {
                error!("Database query failed during connection check: {}", e);
                Err(e)
            }
            Err(_) => {
                let err_msg = format!(
                    "Database connection check timed out after {} seconds.",
                    check_timeout.as_secs()
                );
                error!("{}", err_msg);
                Err(Error::Custom(err_msg))
            }
        }
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
            return Err(Error::Custom("Repo not found".into()));
        }
        Ok(repo[0].clone())
    }

    pub async fn fetch_repos(&mut self) -> Vec<NodeData> {
        self.graph
            .find_nodes_by_type_async(NodeType::Repository)
            .await
    }

    pub async fn update_incremental(
        &mut self,
        repo_url: &str,
        username: Option<String>,
        pat: Option<String>,
        current_hash: &str,
        stored_hash: &str,
        commit: Option<&str>,
        branch: Option<&str>,
        use_lsp: Option<bool>,
        status_tx: Option<Sender<StatusUpdate>>,
    ) -> Result<(u32, u32)> {
        let revs = vec![stored_hash.to_string(), current_hash.to_string()];
        let repo_path = Repo::get_path_from_url(repo_url)?;
        if let Some(modified_files) = check_revs_files(&repo_path, revs.clone()) {
            info!(
                "Processing {} changed files between commits",
                modified_files.len()
            );

            if !modified_files.is_empty() {
                let mut all_dynamic_edges = Vec::new();

                for file in &modified_files {
                    let dynamic_edges = self.graph.get_dynamic_edges_for_file(file).await?;
                    if !dynamic_edges.is_empty() {
                        info!(
                            "Found {} dynamic edges for file: {}",
                            dynamic_edges.len(),
                            file
                        );
                        all_dynamic_edges.extend(dynamic_edges);
                    }
                }

                if all_dynamic_edges.is_empty() {
                    info!(
                        "No dynamic edges found (no Hint/Prompt nodes connected to modified files)"
                    );
                } else {
                    info!("Total dynamic edges collected: {}", all_dynamic_edges.len());
                }

                let muted_nodes = self.collect_muted_nodes_for_files(&modified_files).await?;

                for file in &modified_files {
                    self.graph.remove_nodes_by_file(file).await?;
                }

                let mut subgraph_repos = Repo::new_multi_detect(
                    &repo_path,
                    Some(repo_url.to_string()),
                    modified_files,
                    vec![stored_hash.to_string(), current_hash.to_string()],
                    use_lsp,
                )
                .await?;

                if let Some(tx) = status_tx {
                    subgraph_repos.set_status_tx(tx).await;
                }

                subgraph_repos.build_graphs_inner::<Neo4jGraph>().await?;

                if !all_dynamic_edges.is_empty() {
                    let restored_count =
                        self.graph.restore_dynamic_edges(all_dynamic_edges).await?;
                    info!("Restored {} dynamic edges after rebuild", restored_count);
                }

                if !muted_nodes.is_empty() {
                    self.restore_muted_nodes(muted_nodes).await?;
                }
            }
            info!("Setting Data_Bank property for nodes missing it...");
            if let Err(e) = self.graph.set_missing_data_bank().await {
                tracing::warn!("Error setting Data_Bank property: {:?}", e);
            }

            info!("Setting default namespace for nodes missing it...");
            if let Err(e) = self.graph.set_default_namespace().await {
                tracing::warn!("Error setting default namespace: {:?}", e);
            }

            self.graph
                .update_repository_hash(repo_url, current_hash)
                .await?;
        } else if stored_hash.is_empty() && !current_hash.is_empty() {
            info!("Processing new repository with hash: {}", current_hash);
            let mut repos = Repo::new_clone_multi_detect(
                repo_url,
                username.clone(),
                pat.clone(),
                Vec::new(),
                Vec::new(),
                commit,
                branch,
                use_lsp,
            )
            .await?;

            if let Some(tx) = status_tx {
                repos.set_status_tx(tx).await;
            }

            let _graph = repos.build_graphs_inner::<Neo4jGraph>().await?;

            info!("Setting Data_Bank property for nodes missing it...");
            if let Err(e) = self.graph.set_missing_data_bank().await {
                tracing::warn!("Error setting Data_Bank property: {:?}", e);
            }

            info!("Setting default namespace for nodes missing it...");
            if let Err(e) = self.graph.set_default_namespace().await {
                tracing::warn!("Error setting default namespace: {:?}", e);
            }

            self.graph
                .update_repository_hash(repo_url, current_hash)
                .await?;
        }

        self.graph.get_graph_size_async().await
    }

    pub async fn update_full(
        &mut self,
        repo_url: &str,
        username: Option<String>,
        pat: Option<String>,
        current_hash: &str,
        commit: Option<&str>,
        branch: Option<&str>,
        use_lsp: Option<bool>,
        streaming: Option<bool>,
    ) -> Result<(u32, u32)> {
        let all_dynamic_edges = self.graph.get_all_dynamic_edges().await?;
        if all_dynamic_edges.is_empty() {
            info!("No dynamic edges found (no Hint/Prompt nodes in graph)");
        } else {
            info!(
                "Found {} dynamic edges to preserve",
                all_dynamic_edges.len()
            );
        }

        let repos = Repo::new_clone_multi_detect(
            repo_url,
            username.clone(),
            pat.clone(),
            Vec::new(),
            Vec::new(),
            commit,
            branch,
            use_lsp,
        )
        .await?;

        let streaming = streaming.unwrap_or(false);
        let temp_graph = if streaming {
            repos.build_graphs_btree_with_streaming(streaming).await?
        } else {
            repos.build_graphs_inner::<BTreeMapGraph>().await?
        };

        temp_graph.analysis();

        self.graph.clear().await?;
        if !streaming {
            self.upload_btreemap_to_neo4j(&temp_graph, None).await?;
        }
        self.graph.create_indexes().await?;

        if !all_dynamic_edges.is_empty() {
            info!("Restoring {} dynamic edges...", all_dynamic_edges.len());
            let restored_count = self.graph.restore_dynamic_edges(all_dynamic_edges).await?;
            info!(
                "Successfully restored {} dynamic edges after full rebuild",
                restored_count
            );
        }

        info!("Setting Data_Bank property for nodes missing it...");
        if let Err(e) = self.graph.set_missing_data_bank().await {
            tracing::warn!("Error setting Data_Bank property: {:?}", e);
        }

        info!("Setting default namespace for nodes missing it...");
        if let Err(e) = self.graph.set_default_namespace().await {
            tracing::warn!("Error setting default namespace: {:?}", e);
        }

        self.graph
            .update_repository_hash(repo_url, current_hash)
            .await?;

        Ok(self.graph.get_graph_size_async().await?)
    }

    pub async fn get_coverage(
        &mut self,
        repo: Option<&str>,
        test_filters: Option<super::TestFilters>,
        is_muted: Option<bool>,
    ) -> Result<GraphCoverage> {
        self.graph.ensure_connected().await?;

        use super::coverage::CoverageLanguage;
        let coverage_lang = CoverageLanguage::from_graph(&self.graph).await;

        let ignore_dirs = test_filters
            .as_ref()
            .map(|f| f.ignore_dirs.clone())
            .unwrap_or_default();
        let regex = test_filters
            .as_ref()
            .and_then(|f| f.target_regex.as_deref());

        let in_scope = |n: &NodeData| {
            let repo_match = if let Some(r) = repo {
                if r.is_empty() || r == "all" {
                    true
                } else if r.contains(',') {
                    let repos: Vec<&str> = r.split(',').map(|s| s.trim()).collect();
                    repos.iter().any(|repo_path| n.file.starts_with(repo_path))
                } else {
                    n.file.starts_with(r)
                }
            } else {
                true
            };
            let not_ignored = !ignore_dirs.iter().any(|dir| n.file.contains(dir.as_str()));
            let regex_match = if let Some(pattern) = regex {
                if let Ok(re) = regex::Regex::new(pattern) {
                    re.is_match(&n.file)
                } else {
                    true
                }
            } else {
                true
            };
            let is_muted_match = match is_muted {
                Some(true) => n.meta.get("is_muted").map_or(false, |v| v == "true" || v == "True" || v == "TRUE"),
                Some(false) => !n.meta.get("is_muted").map_or(false, |v| v == "true" || v == "True" || v == "TRUE"),
                None => true,
            };
            repo_match && not_ignored && regex_match && is_muted_match
        };

        coverage_lang.get_coverage(&self.graph, in_scope).await
    }

    pub async fn upload_btreemap_to_neo4j(
        &mut self,
        btree_graph: &BTreeMapGraph,
        status_tx: Option<tokio::sync::broadcast::Sender<crate::repo::StatusUpdate>>,
    ) -> Result<(u32, u32)> {
        self.graph.ensure_connected().await?;
        self.graph.create_indexes().await?;

        if let Some(tx) = &status_tx {
            let _ = tx.send(crate::repo::StatusUpdate {
                status: "".to_string(),
                message: "Storing code context in graph database".to_string(),
                step: 15,
                total_steps: 16,
                progress: 0,
                stats: None,
                step_description: Some("Storing code context in graph database".to_string()),
            });
        }

        info!("preparing node upload {}", btree_graph.nodes.len());
        let node_queries: Vec<(String, BoltMap)> = btree_graph
            .nodes
            .values()
            .map(|node| add_node_query(&node.node_type, &node.node_data))
            .collect();

        debug!("executing node upload in batches");
        self.graph.execute_batch(node_queries).await?;
        info!("node upload complete");

        if let Some(tx) = &status_tx {
            let _ = tx.send(crate::repo::StatusUpdate {
                status: "".to_string(),
                message: "Storing relationships in graph database".to_string(),
                step: 16,
                total_steps: 16,
                progress: 0,
                stats: None,
                step_description: Some("Storing relationships in graph database".to_string()),
            });
        }

        info!("preparing edge upload {}", btree_graph.edges.len());
        let edges_with_ref_ids = btree_graph.edges.iter().map(|(src, tgt, et)| {
            use uuid::Uuid;
            let ref_id = Uuid::new_v4().to_string();
            (src.clone(), tgt.clone(), et.clone(), ref_id)
        });
        let edge_queries = build_batch_edge_queries(edges_with_ref_ids, 256);

        debug!("executing edge upload in batches");
        self.graph.execute_simple(edge_queries).await?;
        info!("edge upload complete!");

        let (nodes, edges) = self.graph.get_graph_size_async().await?;
        debug!("upload complete! nodes: {}, edges: {}", nodes, edges);
        Ok((nodes, edges))
    }

    pub async fn clear_existing_graph(&mut self, root: &str) -> Result<()> {
        self.graph.clear_existing_graph(root).await?;
        Ok(())
    }

    pub async fn set_missing_data_bank(&mut self) -> Result<u32> {
        self.graph.set_missing_data_bank().await
    }

    pub async fn set_default_namespace(&mut self) -> Result<u32> {
        self.graph.set_default_namespace().await
    }

    pub async fn embed_data_bank_bodies(&mut self, do_files: bool) -> Result<()> {
        let batch_size = 32;
        // let mut skip = 0;
        loop {
            let nodes = self
                .graph
                .fetch_nodes_without_embeddings(do_files, 0, batch_size)
                .await?;
            if nodes.is_empty() {
                break;
            }
            for (node_key, body) in &nodes {
                let embedding = vectorize_code_document(body).await?;
                self.graph.update_embedding(node_key, &embedding).await?;
            }
            // let mut batch = Vec::new();
            // for (node_key, body) in &nodes {
            //     let embedding = vectorize_code_document(body).await?;
            //     batch.push((node_key.clone(), embedding));
            // }
            // self.graph.bulk_update_embeddings(batch).await?;
            // skip += batch_size;
        }
        Ok(())
    }
    pub async fn vector_search(
        &mut self,
        query: &str,
        limit: usize,
        node_types: Vec<String>,
        similarity_threshold: f32,
        language: Option<&str>,
    ) -> Result<Vec<(NodeData, f64)>> {
        let embedding = vectorize_query(query).await?;
        let results = self
            .graph
            .vector_search(
                &embedding,
                limit,
                node_types,
                similarity_threshold,
                language,
            )
            .await?;
        Ok(results)
    }

    pub async fn query_nodes_with_count(
        &mut self,
        node_types: &[NodeType],
        offset: usize,
        limit: usize,
        sort_by_test_count: bool,
        coverage_filter: Option<&str>,
        body_length: bool,
        line_count: bool,
        repo: Option<&str>,
        test_filters: Option<super::TestFilters>,
        search: Option<&str>,
        is_muted: Option<bool>,
    ) -> Result<(
        usize,
        Vec<(
            NodeType,
            NodeData,
            usize,
            bool,
            usize,
            String,
            Option<i64>,
            Option<i64>,
            Option<bool>,
        )>,
    )> {
        self.graph.ensure_connected().await?;
        Ok(self
            .graph
            .query_nodes_with_count_async(
                node_types,
                offset,
                limit,
                sort_by_test_count,
                coverage_filter,
                body_length,
                line_count,
                repo,
                test_filters,
                search,
                is_muted,
            )
            .await)
    }

    pub async fn has_coverage(
        &mut self,
        node_type: NodeType,
        name: &str,
        file: &str,
        start: Option<usize>,
        root: Option<&str>,
        tests_filter: Option<&str>,
    ) -> Result<bool> {
        self.graph.ensure_connected().await?;
        let in_scope = |n: &NodeData| {
            if let Some(r) = root {
                if r.is_empty() || r == "all" {
                    true
                } else if r.contains(',') {
                    let roots: Vec<&str> = r.split(',').map(|s| s.trim()).collect();
                    roots.iter().any(|root_path| n.file.starts_with(root_path))
                } else {
                    n.file.starts_with(r)
                }
            } else {
                true
            }
        };

        if node_type == NodeType::Function {
            let target = if let Some(s) = start {
                self.graph
                    .find_node_by_name_in_file_async(NodeType::Function, name, file)
                    .await
                    .filter(|n| n.start == s)
            } else {
                self.graph
                    .find_node_by_name_in_file_async(NodeType::Function, name, file)
                    .await
            };
            if target.is_none() {
                return Ok(false);
            }
            let t = target.unwrap();
            if !in_scope(&t) {
                return Ok(false);
            }
            let sources = tests_sources(tests_filter);
            let mut all_pairs: Vec<(NodeData, NodeData)> = Vec::new();
            for source_nt in sources {
                let pairs = self
                    .graph
                    .find_nodes_with_edge_type_async(
                        source_nt.clone(),
                        NodeType::Function,
                        EdgeType::Calls,
                    )
                    .await;
                all_pairs.extend(pairs);
            }
            for (_, dst) in all_pairs.into_iter() {
                if in_scope(&dst)
                    && dst.name == t.name
                    && dst.file == t.file
                    && dst.start == t.start
                {
                    return Ok(true);
                }
            }
            return Ok(false);
        }

        if node_type == NodeType::Endpoint {
            let ep = self.graph.find_endpoint_async(name, file, "").await;
            if let Some(endpoint) = ep {
                if !in_scope(&endpoint) {
                    return Ok(false);
                }
                let handlers = self.graph.find_handlers_for_endpoint_async(&endpoint).await;
                if handlers.is_empty() {
                    return Ok(false);
                }
                let sources = tests_sources(tests_filter);
                let mut covered_funcs: HashSet<String> = HashSet::new();
                for source_nt in sources {
                    let pairs = self
                        .graph
                        .find_nodes_with_edge_type_async(
                            source_nt.clone(),
                            NodeType::Function,
                            EdgeType::Calls,
                        )
                        .await;
                    for (_s, f) in pairs.into_iter() {
                        if in_scope(&f) {
                            covered_funcs
                                .insert(create_node_key(&Node::new(NodeType::Function, f)));
                        }
                    }
                }
                for h in handlers {
                    if in_scope(&h)
                        && covered_funcs
                            .contains(&create_node_key(&Node::new(NodeType::Function, h)))
                    {
                        return Ok(true);
                    }
                }
            }
            return Ok(false);
        }

        Ok(false)
    }

    pub async fn collect_muted_nodes_for_files(&self, files: &[String]) -> Result<Vec<MutedNodeIdentifier>> {
        if files.is_empty() {
            return Ok(vec![]);
        }

        let muted_nodes = self.graph.get_muted_nodes_for_files_async(files).await?;
        
        if muted_nodes.is_empty() {
            println!("No muted nodes found in {} files", files.len());
        } else {
            println!("Found {} muted nodes in {} files to preserve", muted_nodes.len(), files.len());
        }
        
        Ok(muted_nodes)
    }

    pub async fn restore_muted_nodes(&self, identifiers: Vec<MutedNodeIdentifier>) -> Result<usize> {
        if identifiers.is_empty() {
            return Ok(0);
        }

        let restored_count = self.graph.restore_muted_nodes_async(&identifiers).await?;
        
        if restored_count > 0 {
            println!("Successfully restored muted status for {} nodes after rebuild", restored_count);
        } else {
            println!("No nodes matched for muted status restoration ({} identifiers provided)", identifiers.len());
        }
        
        Ok(restored_count)
    }

    pub async fn set_node_muted(&self, node_type: &NodeType, name: &str, file: &str, is_muted: bool) -> Result<usize> {
        self.graph.set_node_muted_async(node_type, name, file, is_muted).await
    }

    pub async fn is_node_muted(&self, node_type: &NodeType, name: &str, file: &str) -> Result<bool> {
        self.graph.is_node_muted_async(node_type, name, file).await
    }
}
