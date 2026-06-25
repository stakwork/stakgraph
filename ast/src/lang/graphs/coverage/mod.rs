mod python;
mod ruby;
mod rust;
mod typescript;

use super::{NodeData, NodeType};
use crate::lang::graphs::{
    operations::{CoverageStat, GraphCoverage, MockStat},
    Neo4jGraph,
};
use shared::Result;
use std::collections::HashSet;

#[derive(Debug, Clone)]
pub enum CoverageLanguage {
    Typescript,
    Python,
    Ruby,
    Rust,
}

pub(crate) struct CoverageTier {
    pub targets: Vec<NodeData>,
    pub tests: Vec<NodeData>,
    pub covered: HashSet<String>,
}

pub(crate) fn collect_targets(calls: &[(NodeData, NodeData)]) -> HashSet<String> {
    calls
        .iter()
        .map(|(_, tgt)| format!("{}:{}:{}", tgt.name, tgt.file, tgt.start))
        .collect()
}

fn build_stat(tier: &CoverageTier) -> Option<CoverageStat> {
    if tier.targets.is_empty() {
        return None;
    }

    let node_key = |n: &NodeData| format!("{}:{}:{}", n.name, n.file, n.start);

    let covered_count = tier
        .targets
        .iter()
        .filter(|n| tier.covered.contains(&node_key(n)))
        .count();

    let percent = (covered_count as f64 / tier.targets.len() as f64) * 100.0;

    let total_lines: usize = tier
        .targets
        .iter()
        .map(|n| n.end.saturating_sub(n.start) + 1)
        .sum();

    let covered_lines: usize = tier
        .targets
        .iter()
        .filter(|n| tier.covered.contains(&node_key(n)))
        .map(|n| n.end.saturating_sub(n.start) + 1)
        .sum();

    let line_percent = if total_lines == 0 {
        0.0
    } else {
        (covered_lines as f64 / total_lines as f64) * 100.0
    };

    Some(CoverageStat {
        total: tier.targets.len(),
        total_tests: tier.tests.len(),
        covered: covered_count,
        percent: (percent * 100.0).round() / 100.0,
        total_lines,
        covered_lines,
        line_percent: (line_percent * 100.0).round() / 100.0,
    })
}

fn build_mock_stat(mocks: &[NodeData]) -> Option<MockStat> {
    if mocks.is_empty() {
        return None;
    }
    let mocked_count = mocks
        .iter()
        .filter(|n| n.meta.get("mocked").map(|v| v == "true").unwrap_or(false))
        .count();
    let percent = (mocked_count as f64 / mocks.len() as f64) * 100.0;
    Some(MockStat {
        total: mocks.len(),
        mocked: mocked_count,
        percent: (percent * 100.0).round() / 100.0,
    })
}

pub(crate) fn assemble_coverage(
    language: String,
    unit: CoverageTier,
    integration: CoverageTier,
    e2e: CoverageTier,
    mocks: &[NodeData],
) -> GraphCoverage {
    GraphCoverage {
        language: Some(language),
        unit_tests: build_stat(&unit),
        integration_tests: build_stat(&integration),
        e2e_tests: build_stat(&e2e),
        mocks: build_mock_stat(mocks),
    }
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
            } else if lang_name == "python" {
                return CoverageLanguage::Python;
            }
        }

        CoverageLanguage::Typescript
    }

    pub fn language_name(&self) -> String {
        match self {
            CoverageLanguage::Typescript => "typescript".to_string(),
            CoverageLanguage::Python => "python".to_string(),
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
            CoverageLanguage::Typescript => typescript::get_coverage(self, graph, in_scope).await,
            CoverageLanguage::Python => python::get_coverage(self, graph, in_scope).await,
            CoverageLanguage::Ruby => ruby::get_coverage(self, graph, in_scope).await,
            CoverageLanguage::Rust => rust::get_coverage(self, graph, in_scope).await,
        }
    }
}
