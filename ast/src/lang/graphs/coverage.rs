use super::{EdgeType, NodeData, NodeType};
use crate::lang::graphs::{
    operations::{CoverageStat, GraphCoverage, MockStat},
    Neo4jGraph,
};
use shared::Result;
use std::collections::HashSet;

#[derive(Debug, Clone)]
pub enum CoverageLanguage {
    Typescript,
    Ruby,
    Rust,
}

impl CoverageLanguage {
    pub async fn from_graph(graph: &Neo4jGraph) -> Self {
        let language_nodes = graph.find_nodes_by_type_async(NodeType::Language).await;

        for lang_node in language_nodes {
            let lang_name = lang_node.name.to_lowercase();
            if lang_name == "ruby" {
                return CoverageLanguage::Ruby;
            } else if lang_name == "rust" {
                return CoverageLanguage::Rust;
            }
        }

        CoverageLanguage::Typescript
    }

    pub fn language_name(&self) -> String {
        match self {
            CoverageLanguage::Typescript => "typescript".to_string(),
            CoverageLanguage::Ruby => "ruby".to_string(),
            CoverageLanguage::Rust => "rust".to_string(),
        }
    }

    pub async fn get_coverage(
        &self,
        graph: &Neo4jGraph,
        in_scope: impl Fn(&NodeData) -> bool,
    ) -> Result<GraphCoverage> {
        match self {
            CoverageLanguage::Typescript => self.get_typescript_coverage(graph, in_scope).await,
            CoverageLanguage::Ruby => self.get_ruby_coverage(graph, in_scope).await,
            CoverageLanguage::Rust => self.get_rust_coverage(graph, in_scope).await,
        }
    }

    async fn get_typescript_coverage(
        &self,
        graph: &Neo4jGraph,
        in_scope: impl Fn(&NodeData) -> bool,
    ) -> Result<GraphCoverage> {
        let unit_tests = graph.find_nodes_by_type_async(NodeType::UnitTest).await;
        let integration_tests = graph
            .find_nodes_by_type_async(NodeType::IntegrationTest)
            .await;
        let e2e_tests = graph.find_nodes_by_type_async(NodeType::E2eTest).await;

        let functions = graph.find_top_level_functions_async().await;
        let endpoints = graph.find_nodes_by_type_async(NodeType::Endpoint).await;
        let pages = graph.find_nodes_by_type_async(NodeType::Page).await;

        let unit_calls_funcs = graph
            .find_nodes_with_edge_type_async(
                NodeType::UnitTest,
                NodeType::Function,
                EdgeType::Calls,
            )
            .await;
        let integration_calls_endpoints = graph
            .find_nodes_with_edge_type_async(
                NodeType::IntegrationTest,
                NodeType::Endpoint,
                EdgeType::Calls,
            )
            .await;
        let e2e_calls_pages = graph
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
                let is_operand = n.meta.get("operand").is_some();
                !is_component && !is_operand
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

            let total_lines: usize = total_nodes
                .iter()
                .map(|n| n.end.saturating_sub(n.start) + 1)
                .sum();

            let covered_lines: usize = total_nodes
                .iter()
                .filter(|n| covered_set.contains(&format!("{}:{}:{}", n.name, n.file, n.start)))
                .map(|n| n.end.saturating_sub(n.start) + 1)
                .sum();

            let line_percent = if total_lines == 0 {
                0.0
            } else {
                (covered_lines as f64 / total_lines as f64) * 100.0
            };

            Some(CoverageStat {
                total: total_nodes.len(),
                total_tests: total_tests.len(),
                covered: covered_count,
                percent: (percent * 100.0).round() / 100.0,
                total_lines,
                covered_lines,
                line_percent: (line_percent * 100.0).round() / 100.0,
            })
        };

        let mocks = graph.find_nodes_by_type_async(NodeType::Mock).await;
        let mocks_in_scope: Vec<NodeData> = mocks.into_iter().filter(|n| in_scope(n)).collect();

        let mocked_count = mocks_in_scope
            .iter()
            .filter(|n| n.meta.get("mocked").map(|v| v == "true").unwrap_or(false))
            .count();

        let mock_stat = if mocks_in_scope.is_empty() {
            None
        } else {
            let percent = (mocked_count as f64 / mocks_in_scope.len() as f64) * 100.0;
            Some(MockStat {
                total: mocks_in_scope.len(),
                mocked: mocked_count,
                percent: (percent * 100.0).round() / 100.0,
            })
        };

        Ok(GraphCoverage {
            language: Some(self.language_name()),
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
            mocks: mock_stat,
        })
    }

    async fn get_ruby_coverage(
        &self,
        graph: &Neo4jGraph,
        in_scope: impl Fn(&NodeData) -> bool,
    ) -> Result<GraphCoverage> {
        let unit_tests = graph.find_nodes_by_type_async(NodeType::UnitTest).await;
        let integration_tests = graph
            .find_nodes_by_type_async(NodeType::IntegrationTest)
            .await;
        let e2e_tests = graph.find_nodes_by_type_async(NodeType::E2eTest).await;

        let classes = graph.find_nodes_by_type_async(NodeType::Class).await;
        let pages = graph.find_nodes_by_type_async(NodeType::Page).await;

        let unit_calls_classes = graph
            .find_nodes_with_edge_type_async(NodeType::UnitTest, NodeType::Class, EdgeType::Calls)
            .await;
        let integration_calls_classes = graph
            .find_nodes_with_edge_type_async(
                NodeType::IntegrationTest,
                NodeType::Class,
                EdgeType::Calls,
            )
            .await;
        let e2e_calls_pages = graph
            .find_nodes_with_edge_type_async(NodeType::E2eTest, NodeType::Page, EdgeType::Calls)
            .await;

        let collect_targets = |calls: &Vec<(NodeData, NodeData)>| -> HashSet<String> {
            calls
                .iter()
                .map(|(_, tgt)| format!("{}:{}:{}", tgt.name, tgt.file, tgt.start))
                .collect()
        };

        let unit_target_classes = collect_targets(&unit_calls_classes);
        let integration_target_classes = collect_targets(&integration_calls_classes);
        let e2e_target_pages = collect_targets(&e2e_calls_pages);

        let unit_classes_in_scope: Vec<NodeData> = classes
            .iter()
            .filter(|n| in_scope(n))
            .filter(|n| !n.file.ends_with("_controller.rb"))
            .cloned()
            .collect();

        let integration_classes_in_scope: Vec<NodeData> = classes
            .into_iter()
            .filter(|n| in_scope(n))
            .filter(|n| n.file.ends_with("_controller.rb"))
            .collect();

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

            let total_lines: usize = total_nodes
                .iter()
                .map(|n| n.end.saturating_sub(n.start) + 1)
                .sum();

            let covered_lines: usize = total_nodes
                .iter()
                .filter(|n| covered_set.contains(&format!("{}:{}:{}", n.name, n.file, n.start)))
                .map(|n| n.end.saturating_sub(n.start) + 1)
                .sum();

            let line_percent = if total_lines == 0 {
                0.0
            } else {
                (covered_lines as f64 / total_lines as f64) * 100.0
            };

            Some(CoverageStat {
                total: total_nodes.len(),
                total_tests: total_tests.len(),
                covered: covered_count,
                percent: (percent * 100.0).round() / 100.0,
                total_lines,
                covered_lines,
                line_percent: (line_percent * 100.0).round() / 100.0,
            })
        };

        let mocks = graph.find_nodes_by_type_async(NodeType::Mock).await;
        let mocks_in_scope: Vec<NodeData> = mocks.into_iter().filter(|n| in_scope(n)).collect();

        let mocked_count = mocks_in_scope
            .iter()
            .filter(|n| n.meta.get("mocked").map(|v| v == "true").unwrap_or(false))
            .count();

        let mock_stat = if mocks_in_scope.is_empty() {
            None
        } else {
            let percent = (mocked_count as f64 / mocks_in_scope.len() as f64) * 100.0;
            Some(MockStat {
                total: mocks_in_scope.len(),
                mocked: mocked_count,
                percent: (percent * 100.0).round() / 100.0,
            })
        };

        Ok(GraphCoverage {
            language: Some(self.language_name()),
            unit_tests: build_stat(
                &unit_classes_in_scope,
                &unit_tests_in_scope,
                &unit_target_classes,
            ),
            integration_tests: build_stat(
                &integration_classes_in_scope,
                &integration_tests_in_scope,
                &integration_target_classes,
            ),
            e2e_tests: build_stat(&e2e_pages_in_scope, &e2e_tests_in_scope, &e2e_target_pages),
            mocks: mock_stat,
        })
    }

    async fn get_rust_coverage(
        &self,
        graph: &Neo4jGraph,
        in_scope: impl Fn(&NodeData) -> bool,
    ) -> Result<GraphCoverage> {
        let unit_tests = graph.find_nodes_by_type_async(NodeType::UnitTest).await;
        let integration_tests = graph
            .find_nodes_by_type_async(NodeType::IntegrationTest)
            .await;
        let e2e_tests = graph.find_nodes_by_type_async(NodeType::E2eTest).await;

        let functions = graph.find_top_level_functions_async().await;
        let endpoints = graph.find_nodes_by_type_async(NodeType::Endpoint).await;

        let unit_calls_funcs = graph
            .find_nodes_with_edge_type_async(
                NodeType::UnitTest,
                NodeType::Function,
                EdgeType::Calls,
            )
            .await;
        let integration_calls_funcs = graph
            .find_nodes_with_edge_type_async(
                NodeType::IntegrationTest,
                NodeType::Function,
                EdgeType::Calls,
            )
            .await;
        let e2e_calls_endpoints = graph
            .find_nodes_with_edge_type_async(NodeType::E2eTest, NodeType::Endpoint, EdgeType::Calls)
            .await;

        let collect_targets = |calls: &Vec<(NodeData, NodeData)>| -> HashSet<String> {
            calls
                .iter()
                .map(|(_, tgt)| format!("{}:{}:{}", tgt.name, tgt.file, tgt.start))
                .collect()
        };

        let unit_target_functions = collect_targets(&unit_calls_funcs);
        let integration_target_functions = collect_targets(&integration_calls_funcs);
        let e2e_target_endpoints = collect_targets(&e2e_calls_endpoints);

        let unit_functions_in_scope: Vec<NodeData> = functions
            .clone()
            .into_iter()
            .filter(|n| in_scope(n))
            .filter(|n| {
                if n.body.trim().is_empty() {
                    return false;
                }
                true
            })
            .collect();

        let integration_functions_in_scope: Vec<NodeData> = functions
            .into_iter()
            .filter(|n| in_scope(n))
            .filter(|n| {
                if n.body.trim().is_empty() {
                    return false;
                }
                true
            })
            .collect();

        let endpoints_in_scope: Vec<NodeData> =
            endpoints.into_iter().filter(|n| in_scope(n)).collect();
        let unit_tests_in_scope: Vec<NodeData> =
            unit_tests.into_iter().filter(|n| in_scope(n)).collect();
        let integration_tests_in_scope: Vec<NodeData> = integration_tests
            .into_iter()
            .filter(|n| in_scope(n))
            .collect();
        let e2e_tests_in_scope: Vec<NodeData> =
            e2e_tests.into_iter().filter(|n| in_scope(n)).collect();

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

            let total_lines: usize = total_nodes
                .iter()
                .map(|n| n.end.saturating_sub(n.start) + 1)
                .sum();

            let covered_lines: usize = total_nodes
                .iter()
                .filter(|n| covered_set.contains(&format!("{}:{}:{}", n.name, n.file, n.start)))
                .map(|n| n.end.saturating_sub(n.start) + 1)
                .sum();

            let line_percent = if total_lines == 0 {
                0.0
            } else {
                (covered_lines as f64 / total_lines as f64) * 100.0
            };

            Some(CoverageStat {
                total: total_nodes.len(),
                total_tests: total_tests.len(),
                covered: covered_count,
                percent: (percent * 100.0).round() / 100.0,
                total_lines,
                covered_lines,
                line_percent: (line_percent * 100.0).round() / 100.0,
            })
        };

        let mocks = graph.find_nodes_by_type_async(NodeType::Mock).await;
        let mocks_in_scope: Vec<NodeData> = mocks.into_iter().filter(|n| in_scope(n)).collect();

        let mocked_count = mocks_in_scope
            .iter()
            .filter(|n| n.meta.get("mocked").map(|v| v == "true").unwrap_or(false))
            .count();

        let mock_stat = if mocks_in_scope.is_empty() {
            None
        } else {
            let percent = (mocked_count as f64 / mocks_in_scope.len() as f64) * 100.0;
            Some(MockStat {
                total: mocks_in_scope.len(),
                mocked: mocked_count,
                percent: (percent * 100.0).round() / 100.0,
            })
        };

        Ok(GraphCoverage {
            language: Some(self.language_name()),
            unit_tests: build_stat(
                &unit_functions_in_scope,
                &unit_tests_in_scope,
                &unit_target_functions,
            ),
            integration_tests: build_stat(
                &integration_functions_in_scope,
                &integration_tests_in_scope,
                &integration_target_functions,
            ),
            e2e_tests: build_stat(
                &endpoints_in_scope,
                &e2e_tests_in_scope,
                &e2e_target_endpoints,
            ),
            mocks: mock_stat,
        })
    }
}
