use crate::lang::graphs::{BTreeMapGraph, EdgeType, Graph, NodeType};
use crate::lang::Lang;
use crate::repo::Repo;
use crate::utils::get_use_lsp;
use shared::error::Result;
use std::str::FromStr;

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_rust_coverage() -> Result<()> {
    let use_lsp = get_use_lsp();

    let repo = Repo::new(
        "src/testing/rust",
        Lang::from_str("rust").unwrap(),
        use_lsp,
        Vec::new(),
        Vec::new(),
    )
    .unwrap();

    let graph = repo.build_graph_inner::<BTreeMapGraph>().await?;

    graph.analysis();

    let endpoints = graph.find_nodes_by_type(NodeType::Endpoint);
    assert_eq!(endpoints.len(), 18, "Expected exactly 18 endpoints");

    let functions = graph.find_nodes_by_type(NodeType::Function);
    assert_eq!(functions.len(), 73, "Expected exactly 73 functions");

    let unit_tests = graph.find_nodes_by_type(NodeType::UnitTest);
    assert_eq!(
        unit_tests.len(),
        8,
        "Expected exactly 8 unit tests"
    );

    let integration_tests = graph.find_nodes_by_type(NodeType::IntegrationTest);
    assert_eq!(
        integration_tests.len(),
        4,
        "Expected exactly 4 integration tests"
    );

    let calls_edges = graph.count_edges_of_type(EdgeType::Calls);
    assert_eq!(calls_edges, 28, "Expected exactly 28 Calls edges");

    let unit_test_to_function_edges = graph.find_nodes_with_edge_type(
        NodeType::UnitTest,
        NodeType::Function,
        EdgeType::Calls,
    );

    assert_eq!(
        unit_test_to_function_edges.len(),
        0,
        "Unit tests are stored as Function nodes, not UnitTest nodes with Calls edges"
    );

    let test_like_functions: Vec<_> = functions
        .iter()
        .filter(|f| {
            let name_lower = f.name.to_lowercase();
            name_lower.contains("test") || name_lower.contains("bench")
        })
        .cloned()
        .collect();

    println!("\nFunctions with test/bench in names: {}", test_like_functions.len());
    for func in test_like_functions.iter().take(8) {
        println!("  Function: '{}' in {}", func.name, func.file);
    }

    let unit_test_functions = test_like_functions;

    let mut unit_test_calls_to_functions = Vec::new();
    for test in &unit_test_functions {
        let called_funcs = graph.find_functions_called_by(test);
        for func in called_funcs {
            unit_test_calls_to_functions.push((test.clone(), func));
        }
    }

    let unique_functions_tested: std::collections::HashSet<String> = unit_test_calls_to_functions
        .iter()
        .map(|(_, target)| format!("{}|{}|{}", target.name, target.file, target.start))
        .collect();

    println!("\n=== UNIT TEST COVERAGE FACTS ===");
    println!("Total unit tests: {}", unit_tests.len());
    println!("Unit test functions (as Function nodes): {}", unit_test_functions.len());
    println!("Function → function calls from tests: {}", unit_test_calls_to_functions.len());
    println!("Unique functions tested: {}", unique_functions_tested.len());
    
    if !unit_test_calls_to_functions.is_empty() {
        println!("\nUnit test → function relationships (via Function → Function edges):");
        for (test, function) in &unit_test_calls_to_functions {
            println!("  {} → {}", test.name, function.name);
        }
    }

    let integration_test_to_endpoint_edges = graph.find_nodes_with_edge_type(
        NodeType::IntegrationTest,
        NodeType::Endpoint,
        EdgeType::Calls,
    );

    let integration_test_to_function_edges = graph.find_nodes_with_edge_type(
        NodeType::IntegrationTest,
        NodeType::Function,
        EdgeType::Calls,
    );

    let unique_endpoints_tested: std::collections::HashSet<String> = integration_test_to_endpoint_edges
        .iter()
        .map(|(_, target)| format!("{}|{}|{}", target.name, target.file, target.start))
        .collect();

    println!("\n=== INTEGRATION TEST COVERAGE FACTS ===");
    println!("Total integration tests: {}", integration_tests.len());
    println!("IntegrationTest → endpoint edges: {}", integration_test_to_endpoint_edges.len());
    println!("IntegrationTest → function edges: {}", integration_test_to_function_edges.len());
    println!("Unique endpoints tested: {}", unique_endpoints_tested.len());
    
    if !integration_test_to_endpoint_edges.is_empty() {
        println!("\nIntegration test → endpoint relationships:");
        for (test, endpoint) in &integration_test_to_endpoint_edges {
            println!("  {} → {} ({})", test.name, endpoint.name, endpoint.meta.get("verb").unwrap_or(&"?".to_string()));
        }
    }

    println!("\n=== SUMMARY: RUST TEST COVERAGE ARCHITECTURE ===");
    println!("Unit Tests:");
    println!("  - 8 UnitTest nodes exist (metadata only)");
    println!("  - 0 corresponding Function nodes found");
    println!("  - NO UnitTest → Function coverage edges in graph");
    println!("  - Unit tests are NOT being tracked as functions that call other functions");
    println!("\nIntegration Tests:");
    println!("  - 4 IntegrationTest nodes exist");
    println!("  - 2 IntegrationTest → Endpoint edges (direct coverage)");
    println!("  - 0 IntegrationTest → Function edges");
    println!("  - {} unique endpoints tested", unique_endpoints_tested.len());
    println!("\nConclusion:");
    println!("  - Integration test coverage CAN be measured: IntegrationTest → Endpoint");
    println!("  - Unit test coverage CANNOT be measured with current graph structure");
    println!("  - Need to capture unit tests as Function nodes to track their calls");

    Ok(())
}
