use std::collections::HashSet;
use std::time::Duration;

use crate::lang::embedding::{vectorize_code_document, vectorize_query};
use crate::lang::graphs::graph::Graph;
use crate::lang::graphs::neo4j_graph::Neo4jGraph;
use crate::lang::graphs::utils::tests_sources;
use crate::lang::graphs::BTreeMapGraph;
use crate::lang::linker::{
    extract_test_ids, infer_lang, normalize_backend_path, normalize_frontend_path, paths_match,
    verbs_match,
};
use crate::lang::neo4j_utils::{add_edge_query, add_node_query, build_batch_edge_queries};
use crate::lang::{Edge, EdgeType, Node, NodeData, NodeType};
use crate::repo::{check_revs_files, Repo};
use crate::utils::create_node_key;
use neo4rs::BoltMap;
use shared::error::{Error, Result};
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
}

#[derive(Debug, Clone)]
pub struct GraphCoverage {
    pub unit_tests: Option<CoverageStat>,
    pub integration_tests: Option<CoverageStat>,
    pub e2e_tests: Option<CoverageStat>,
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

                let subgraph_repos = Repo::new_multi_detect(
                    &repo_path,
                    Some(repo_url.to_string()),
                    modified_files,
                    vec![stored_hash.to_string(), current_hash.to_string()],
                    use_lsp,
                )
                .await?;

                let (nodes_before_reassign, edges_before_reassign) = self.graph.get_graph_size();
                info!(
                    "[DEBUG]  Graph  BEFORE build {} nodes, {} edges",
                    nodes_before_reassign, edges_before_reassign
                );

                subgraph_repos.build_graphs_inner::<Neo4jGraph>().await?;

                let (nodes_after_reassign, edges_after_reassign) = self.graph.get_graph_size();
                info!(
                    "[DEBUG]  Graph  AFTER build {} nodes, {} edges",
                    nodes_after_reassign, edges_after_reassign
                );

                let (api_links, e2e_links) = self.link_cross_repo_relations().await?;
                info!(
                    "Linked cross-repo relations: api_links={}, e2e_links={}",
                    api_links, e2e_links
                );

                let (nodes_after, edges_after) = self.graph.get_graph_size_async().await?;
                info!(
                    "Updated files: total {} nodes and {} edges",
                    nodes_after, edges_after
                );
            }
            self.graph
                .update_repository_hash(repo_url, current_hash)
                .await?;
        } else if stored_hash.is_empty() && !current_hash.is_empty() {
            info!("Processing new repository with hash: {}", current_hash);
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

            let graph = repos.build_graphs_inner::<Neo4jGraph>().await?;
            let (api_links, e2e_links) = self.link_cross_repo_relations().await?;
            info!(
                "Linked cross-repo relations: api_links={}, e2e_links={}",
                api_links, e2e_links
            );
            let (nodes_after, edges_after) = graph.get_graph_size_async().await?;
            info!(
                "Procesed new repository with {} nodes and {} edges",
                nodes_after, edges_after
            );
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
    ) -> Result<(u32, u32)> {
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

        let temp_graph = repos.build_graphs_inner::<BTreeMapGraph>().await?;

        temp_graph.analysis();

        self.graph.clear().await?;
        if std::env::var("STREAM_UPLOAD").is_err() {
            self.upload_btreemap_to_neo4j(&temp_graph, None).await?;
        }
        self.graph.create_indexes().await?;

        self.graph
            .update_repository_hash(repo_url, current_hash)
            .await?;

        Ok(self.graph.get_graph_size_async().await?)
    }

    pub async fn get_coverage(&mut self, repo: Option<&str>) -> Result<GraphCoverage> {
        self.graph.ensure_connected().await?;

        let in_scope = |n: &NodeData| repo.map_or(true, |r| n.file.starts_with(r));

        let unit_tests = self
            .graph
            .find_nodes_by_type_async(NodeType::UnitTest)
            .await;
        let integration_tests = self
            .graph
            .find_nodes_by_type_async(NodeType::IntegrationTest)
            .await;
        let e2e_tests = self.graph.find_nodes_by_type_async(NodeType::E2eTest).await;

        let functions = self.graph.find_top_level_functions_async().await;
        let endpoints = self
            .graph
            .find_nodes_by_type_async(NodeType::Endpoint)
            .await;
        let pages = self.graph.find_nodes_by_type_async(NodeType::Page).await;

        let unit_calls_funcs = self
            .graph
            .find_nodes_with_edge_type_async(
                NodeType::UnitTest,
                NodeType::Function,
                EdgeType::Calls,
            )
            .await;
        let integration_calls_endpoints = self
            .graph
            .find_nodes_with_edge_type_async(
                NodeType::IntegrationTest,
                NodeType::Endpoint,
                EdgeType::Calls,
            )
            .await;
        let e2e_calls_pages = self
            .graph
            .find_nodes_with_edge_type_async(NodeType::E2eTest, NodeType::Page, EdgeType::Calls)
            .await;

        let collect_targets = |calls: &Vec<(NodeData, NodeData)>| -> HashSet<String> {
            calls
                .iter()
                .map(|(_, tgt)| format!("{}:{}:{}", tgt.name, tgt.file, tgt.start))
                .collect()
        };

        let unit_target_functions = collect_targets(&unit_calls_funcs);
        let integration_target_endpoints = collect_targets(&integration_calls_endpoints);
        let e2e_target_pages = collect_targets(&e2e_calls_pages);

        let unit_functions_in_scope: Vec<NodeData> = functions
            .into_iter()
            .filter(|n| in_scope(n))
            .filter(|n| {
                if n.body.trim().is_empty() {
                    return false;
                }
                let is_component = n
                    .meta
                    .get("component")
                    .map(|v| v == "true")
                    .unwrap_or(false);
                !is_component
            })
            .collect();
        let integration_endpoints_in_scope: Vec<NodeData> =
            endpoints.into_iter().filter(|n| in_scope(n)).collect();
        let pages_in_scope: Vec<NodeData> = pages.into_iter().filter(|n| in_scope(n)).collect();
        let unit_tests_in_scope: Vec<NodeData> =
            unit_tests.into_iter().filter(|n| in_scope(n)).collect();
        let integration_tests_in_scope: Vec<NodeData> = integration_tests
            .into_iter()
            .filter(|n| in_scope(n))
            .collect();
        let e2e_tests_in_scope: Vec<NodeData> =
            e2e_tests.into_iter().filter(|n| in_scope(n)).collect();

        let e2e_pages_in_scope = pages_in_scope.clone();

        let build_stat = |total_nodes: &Vec<NodeData>,
                          total_tests: &Vec<NodeData>,
                          covered_set: &HashSet<String>|
         -> Option<CoverageStat> {
            if total_nodes.is_empty() {
                return None;
            }
            let covered_count = total_nodes
                .iter()
                .filter(|n| covered_set.contains(&format!("{}:{}:{}", n.name, n.file, n.start)))
                .count();
            let percent = if total_nodes.is_empty() {
                0.0
            } else {
                (covered_count as f64 / total_nodes.len() as f64) * 100.0
            };
            Some(CoverageStat {
                total: total_nodes.len(),
                total_tests: total_tests.len(),
                covered: covered_count,
                percent: (percent * 100.0).round() / 100.0,
            })
        };

        Ok(GraphCoverage {
            unit_tests: build_stat(
                &unit_functions_in_scope,
                &unit_tests_in_scope,
                &unit_target_functions,
            ),
            integration_tests: build_stat(
                &integration_endpoints_in_scope,
                &integration_tests_in_scope,
                &integration_target_endpoints,
            ),
            e2e_tests: build_stat(&e2e_pages_in_scope, &e2e_tests_in_scope, &e2e_target_pages),
        })
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
                message: "Step 15: Uploading nodes to Neo4j".to_string(),
                step: 15,
                total_steps: 16,
                progress: 0,
                stats: None,
                step_description: Some("Uploading nodes to Neo4j".to_string()),
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
                message: "Step 16: Uploading edges to Neo4j".to_string(),
                step: 16,
                total_steps: 16,
                progress: 0,
                stats: None,
                step_description: Some("Uploading edges to Neo4j".to_string()),
            });
        }

        info!("preparing edge upload {}", btree_graph.edges.len());
        let edge_queries = build_batch_edge_queries(btree_graph.edges.iter().cloned(), 256);

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

    pub async fn link_cross_repo_relations(&mut self) -> Result<(usize, usize)> {
        let api = self.link_cross_repo_api_nodes().await?;
        let e2e = self.link_cross_repo_e2e_tests().await?;
        Ok((api, e2e))
    }

    pub async fn link_cross_repo_api_nodes(&mut self) -> Result<usize> {
        self.graph.ensure_connected().await?;

        let requests = self.graph.find_nodes_by_type_async(NodeType::Request).await;
        let endpoints = self
            .graph
            .find_nodes_by_type_async(NodeType::Endpoint)
            .await;

        if requests.is_empty() || endpoints.is_empty() {
            return Ok(0);
        }

        let mut queries: Vec<(String, BoltMap)> = Vec::new();
        let mut count = 0;

        for req in &requests {
            let Some(req_path) = normalize_frontend_path(&req.name) else {
                continue;
            };
            for endpoint in &endpoints {
                let backend_norm =
                    normalize_backend_path(&endpoint.name).unwrap_or_else(|| endpoint.name.clone());
                if paths_match(&req_path, &backend_norm) && verbs_match(req, endpoint) {
                    count += 1;
                    let edge = Edge::calls(NodeType::Request, req, NodeType::Endpoint, endpoint);
                    queries.push(add_edge_query(&edge));
                }
            }
        }

        if queries.is_empty() {
            return Ok(0);
        }
        self.graph.execute_simple(queries).await?;
        Ok(count)
    }

    pub async fn link_cross_repo_e2e_tests(&mut self) -> Result<usize> {
        self.graph.ensure_connected().await?;

        let tests = self.graph.find_nodes_by_type_async(NodeType::E2eTest).await;
        let functions = self
            .graph
            .find_nodes_by_type_async(NodeType::Function)
            .await;

        if tests.is_empty() || functions.is_empty() {
            return Ok(0);
        }
        let mut frontend_funcs: Vec<(NodeData, Vec<String>)> = Vec::new();
        for f in &functions {
            if let Ok(lang) = infer_lang(f) {
                if lang.is_frontend() {
                    let ids = extract_test_ids(&f.body, &lang).unwrap_or_default();
                    if !ids.is_empty() {
                        frontend_funcs.push((f.clone(), ids));
                    }
                }
            }
        }

        if frontend_funcs.is_empty() {
            return Ok(0);
        }

        let mut queries: Vec<(String, BoltMap)> = Vec::new();
        let mut count = 0;

        for t in &tests {
            let test_ids = if let Ok(lang) = infer_lang(t) {
                extract_test_ids(&t.body, &lang).unwrap_or_default()
            } else {
                Vec::new()
            };
            if test_ids.is_empty() {
                continue;
            }
            for (f, f_ids) in &frontend_funcs {
                if f_ids.iter().any(|id| test_ids.contains(id)) {
                    let edge = Edge::linked_e2e_test_call(t, f);
                    queries.push(add_edge_query(&edge));
                    count += 1;
                }
            }
        }

        if queries.is_empty() {
            return Ok(0);
        }
        self.graph.execute_simple(queries).await?;
        Ok(count)

    }
    pub async fn list_nodes_with_coverage(
        &mut self,
        node_type: NodeType,
        with_usage: bool,
        offset: usize,
        limit: usize,
        root: Option<&str>,
        tests_filter: Option<&str>,
        covered_only: Option<bool>,
    ) -> Result<(
        Vec<(NodeData, usize, bool, usize, String)>,
        Vec<(NodeData, usize, bool, usize, String)>,
    )> {
        self.graph.ensure_connected().await?;

        let results = self
            .graph
            .find_nodes_with_coverage_async(
                node_type.clone(),
                with_usage,
                offset,
                limit,
                root,
                tests_filter,
                covered_only,
            )
            .await;

        match node_type {
            NodeType::Function => Ok((results, vec![])),
            NodeType::Endpoint => Ok((vec![], results)),
            _ => Ok((vec![], vec![])),
        }
    }

    pub async fn query_nodes_simple(
        &mut self,
        node_type: NodeType,
        offset: usize,
        limit: usize,
        sort_by_test_count: bool,
        coverage_filter: Option<&str>,
        body_length: bool,
        line_count: bool,
    ) -> Result<Vec<(NodeData, usize, bool, usize, String)>> {
        self.graph.ensure_connected().await?;
        let results = self
            .graph
            .find_nodes_simple_async(
                node_type.clone(),
                offset,
                limit,
                sort_by_test_count,
                coverage_filter,
                body_length,
                line_count,
            )
            .await;
        Ok(results)
    }

    pub async fn count_nodes_simple(
        &mut self,
        node_type: NodeType,
        coverage_filter: Option<&str>,
    ) -> Result<usize> {
        self.graph.ensure_connected().await?;
        let count = self
            .graph
            .count_nodes_simple_async(node_type, coverage_filter)
            .await;
        Ok(count)
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
        let in_scope = |n: &NodeData| root.map_or(true, |r| n.file.starts_with(r));

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
}
