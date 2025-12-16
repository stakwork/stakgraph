use crate::lang::graphs::{BTreeMapGraph, EdgeType, Graph, NodeType};
use crate::lang::Lang;
use crate::repo::Repo;
use shared::error::Result;
use std::str::FromStr;

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_rust_coverage() -> Result<()> {
    let repo = Repo::new(
        "src/testing/rust",
        Lang::from_str("rust").unwrap(),
        false,
        Vec::new(),
        Vec::new(),
    )
    .unwrap();

    let graph = repo.build_graph_inner::<BTreeMapGraph>().await?;

    graph.analysis();

    let endpoints = graph.find_nodes_by_type(NodeType::Endpoint);
    assert_eq!(endpoints.len(), 18, "Expected exactly 18 endpoints");

    let functions = graph.find_nodes_by_type(NodeType::Function);
    assert_eq!(functions.len(), 74, "Expected 74 functions after test improvements");
    
    let unit_tests = graph.find_nodes_by_type(NodeType::UnitTest);
    assert_eq!(unit_tests.len(), 43, "Expected 43 unit tests");

    let integration_tests = graph.find_nodes_by_type(NodeType::IntegrationTest);
    assert_eq!(integration_tests.len(), 17, "Expected 17 integration tests");
    
    let e2e_tests = graph.find_nodes_by_type(NodeType::E2eTest);
    assert_eq!(e2e_tests.len(), 8, "Expected 8 e2e tests");

    let calls_edges = graph.count_edges_of_type(EdgeType::Calls);
    assert_eq!(calls_edges, 104, "Expected 104 Calls edges");

    let unit_test_to_function_edges = graph.find_nodes_with_edge_type(
        NodeType::UnitTest,
        NodeType::Function,
        EdgeType::Calls,
    );
    assert_eq!(unit_test_to_function_edges.len(), 32, "Expected 32 UnitTest → Function edges");

    let integration_test_to_function_edges = graph.find_nodes_with_edge_type(
        NodeType::IntegrationTest,
        NodeType::Function,
        EdgeType::Calls,
    );
    assert_eq!(integration_test_to_function_edges.len(), 31, "Expected 31 IntegrationTest → Function edges");

    let unique_functions_tested: std::collections::HashSet<String> = unit_test_to_function_edges
        .iter()
        .map(|(_, target)| format!("{}|{}|{}", target.name, target.file, target.start))
        .collect();
    assert_eq!(unique_functions_tested.len(), 19, "Expected 19 unique functions covered by unit tests");

    let total_test_coverage_edges = unit_test_to_function_edges.len() + integration_test_to_function_edges.len();
    assert_eq!(total_test_coverage_edges, 63, "Expected 63 total test coverage edges");

    Ok(())
}
