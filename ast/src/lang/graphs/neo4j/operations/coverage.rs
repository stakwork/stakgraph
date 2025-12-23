use std::collections::HashSet;
use shared::Result;

use crate::lang::{EdgeType, Neo4jGraph, Node, NodeData, NodeType, TestFilters, coverage::CoverageLanguage, executor::*, graph_ops::GraphOps, helpers::{MutedNodeIdentifier, *}, graphs::queries::*};
use crate::lang::graphs::utils::{tests_sources};
use crate::utils::create_node_key;


#[derive(Debug, Clone)]
pub struct GraphCoverage {
    pub language: Option<String>,
    pub unit_tests: Option<CoverageStat>,
    pub integration_tests: Option<CoverageStat>,
    pub e2e_tests: Option<CoverageStat>,
    pub mocks: Option<MockStat>,
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



impl Neo4jGraph{
        pub async fn get_muted_nodes_for_files_async(&self, files: &[String]) -> Result<Vec<MutedNodeIdentifier>> {
        let conn = match self.ensure_connected().await {
            Ok(conn) => conn,
            Err(e) => {
                println!("Failed to connect to graph database: {:?}", e);
                return Ok(Vec::new());
            }
        };

        let (query_str, params) = get_muted_nodes_for_files_query(files);
        Ok(execute_muted_nodes_query(&conn, query_str, params).await)
    }

    pub async fn restore_muted_nodes_async(&self, identifiers: &[MutedNodeIdentifier]) -> Result<usize> {
        if identifiers.is_empty() {
            return Ok(0);
        }

        let conn = match self.ensure_connected().await {
            Ok(conn) => conn,
            Err(e) => {
                println!("Failed to connect to graph database: {:?}", e);
                return Ok(0);
            }
        };

        let (query_str, params) = restore_muted_status_query(identifiers);
        Ok(execute_count_query(&conn, query_str, params).await)
    }

    pub async fn set_node_muted_async(&self, node_type: &NodeType, name: &str, file: &str, is_muted: bool) -> Result<usize> {
        let conn = match self.ensure_connected().await {
            Ok(conn) => conn,
            Err(_) => {
                return Ok(0);
            }
        };

        let (query_str, params) = set_node_muted_query(node_type, name, file, is_muted);
        let result = execute_count_query(&conn, query_str, params).await;
        Ok(result)
    }

    pub async fn is_node_muted_async(&self, node_type: &NodeType, name: &str, file: &str) -> Result<bool> {
        let conn = match self.ensure_connected().await {
            Ok(conn) => conn,
            Err(_) => {
                return Ok(false);
            }
        };
        let (query_str, params) = check_node_muted_query(node_type, name, file);
        let is_muted = execute_boolean_query(&conn, query_str, params).await;
        Ok(is_muted)
    }

}

impl GraphOps{
      pub async fn get_coverage(
        &mut self,
        repo: Option<&str>,
        test_filters: Option<TestFilters>,
        is_muted: Option<bool>,
    ) -> Result<GraphCoverage> {
        self.graph.ensure_connected().await?;
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
}