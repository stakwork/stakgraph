use super::{assemble_coverage, collect_targets, CoverageLanguage, CoverageTier};
use crate::lang::graphs::{
    operations::GraphCoverage, EdgeType, Neo4jGraph, NodeData, NodeType,
};
use shared::Result;

pub(super) async fn get_coverage(
    lang: &CoverageLanguage,
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

    let unit_covered = collect_targets(
        &graph
            .find_nodes_with_edge_type_async(
                NodeType::UnitTest,
                NodeType::Function,
                EdgeType::Calls,
            )
            .await,
    );
    let integration_covered = collect_targets(
        &graph
            .find_nodes_with_edge_type_async(
                NodeType::IntegrationTest,
                NodeType::Endpoint,
                EdgeType::Calls,
            )
            .await,
    );
    let e2e_covered = collect_targets(
        &graph
            .find_nodes_with_edge_type_async(NodeType::E2eTest, NodeType::Page, EdgeType::Calls)
            .await,
    );

    let unit_targets: Vec<NodeData> = functions
        .into_iter()
        .filter(|n| in_scope(n) && !n.body.trim().is_empty())
        .collect();

    let mocks = graph.find_nodes_by_type_async(NodeType::Mock).await;
    let mocks_in_scope: Vec<NodeData> = mocks.into_iter().filter(|n| in_scope(n)).collect();

    Ok(assemble_coverage(
        lang.language_name(),
        CoverageTier {
            targets: unit_targets,
            tests: unit_tests.into_iter().filter(|n| in_scope(n)).collect(),
            covered: unit_covered,
        },
        CoverageTier {
            targets: endpoints.into_iter().filter(|n| in_scope(n)).collect(),
            tests: integration_tests
                .into_iter()
                .filter(|n| in_scope(n))
                .collect(),
            covered: integration_covered,
        },
        CoverageTier {
            targets: pages.into_iter().filter(|n| in_scope(n)).collect(),
            tests: e2e_tests.into_iter().filter(|n| in_scope(n)).collect(),
            covered: e2e_covered,
        },
        &mocks_in_scope,
    ))
}
