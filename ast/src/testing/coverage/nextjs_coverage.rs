use crate::lang::graphs::{EdgeType, Graph, NodeType};
use crate::lang::linker::normalize_backend_path;
use crate::lang::Lang;
use crate::repo::{Repo, Repos};
use crate::utils::get_use_lsp;
use shared::error::Result;
use std::str::FromStr;

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_nextjs_integration_coverage() -> Result<()> {
    let use_lsp = get_use_lsp();
    
    // Phase 1: Build BTreeMapGraph from nextjs test server
    let repo = Repo::new(
        "src/testing/nextjs",
        Lang::from_str("tsx").unwrap(),
        use_lsp,
        Vec::new(),
        Vec::new(),
    )
    .unwrap();

    let repos = Repos(vec![repo]);
    let btree_graph = repos.build_graphs_btree().await?;

    btree_graph.analysis();
    
    // Phase 2: Validate BTreeMapGraph - Count nodes
    let endpoints = btree_graph.find_nodes_by_type(NodeType::Endpoint);
    assert_eq!(endpoints.len(), 9, "Expected exactly 9 endpoints");
    
    let integration_tests = btree_graph.find_nodes_by_type(NodeType::IntegrationTest);
    assert_eq!(integration_tests.len(), 4, "Expected exactly 4 integration test describe blocks");
    
    let test_edges = btree_graph.find_nodes_with_edge_type(
        NodeType::IntegrationTest,
        NodeType::Endpoint,
        EdgeType::Calls
    );
    
    // We expect at least 6 edges for the 6 tested endpoints
    // Some endpoints might have multiple test edges from different test files
    let unique_tested_endpoints: std::collections::HashSet<String> = test_edges
        .iter()
        .map(|(_, target)| format!("{}|{}|{}", target.name, target.file, target.start))
        .collect();
    
    assert_eq!(
        unique_tested_endpoints.len(),
        6,
        "Expected exactly 6 unique tested endpoints"
    );
    
    // Phase 2: Validate specific tested endpoints have test edges
    // Note: paths are normalized ([id] -> :param)
    let tested_endpoints = vec![
        ("/api/items", "GET"),
        ("/api/items", "POST"),
        ("/api/person", "GET"),
        ("/api/person", "POST"),
        ("/api/person/:param", "GET"),    // normalized from [id]
        ("/api/person/:param", "DELETE"), // normalized from [id]
    ];
    
    for (expected_path, expected_verb) in &tested_endpoints {
        // Find the endpoint in the graph
        let matching_endpoints: Vec<_> = endpoints
            .iter()
            .filter(|e| {
                let normalized = normalize_backend_path(&e.name).unwrap_or(e.name.clone());
                normalized == *expected_path && e.meta.get("verb") == Some(&expected_verb.to_string())
            })
            .collect();
        
        assert_eq!(
            matching_endpoints.len(),
            1,
            "Expected exactly 1 endpoint matching {} {}, found {}",
            expected_verb,
            expected_path,
            matching_endpoints.len()
        );
        
        let endpoint = matching_endpoints[0];
        
        // Check if this endpoint has a test edge
        let has_test_edge = test_edges.iter().any(|(_, target)| {
            target.name == endpoint.name
                && target.file == endpoint.file
                && target.start == endpoint.start
        });
        
        assert_eq!(
            has_test_edge,
            true,
            "Endpoint {} {} should have integration test edge",
            expected_verb,
            expected_path
        );
    }
    
    // Phase 2: Validate specific untested endpoints have NO test edges
    let untested_endpoints = vec![
        ("/api/orders", "PUT"),
        ("/api/users", "POST"),
        ("/api/products/:param", "GET"), // normalized from [id]
    ];
    
    for (expected_path, expected_verb) in &untested_endpoints {
        // Find the endpoint in the graph
        let matching_endpoints: Vec<_> = endpoints
            .iter()
            .filter(|e| {
                let normalized = normalize_backend_path(&e.name).unwrap_or(e.name.clone());
                normalized == *expected_path && e.meta.get("verb") == Some(&expected_verb.to_string())
            })
            .collect();
        
        assert_eq!(
            matching_endpoints.len(),
            1,
            "Expected exactly 1 endpoint matching {} {}, found {}",
            expected_verb,
            expected_path,
            matching_endpoints.len()
        );
        
        let endpoint = matching_endpoints[0];
        
        // Check that this endpoint has NO test edge
        let has_test_edge = test_edges.iter().any(|(_, target)| {
            target.name == endpoint.name
                && target.file == endpoint.file
                && target.start == endpoint.start
        });
        
        assert_eq!(
            has_test_edge,
            false,
            "Endpoint {} {} should NOT have integration test edge",
            expected_verb,
            expected_path
        );
    }
    
    // Phase 3 & 4 & 5: Neo4j upload and query validation (feature gated)
    #[cfg(feature = "neo4j")]
    {
        use crate::lang::graphs::graph_ops::GraphOps;
        
        // Phase 3: Upload to Neo4j
        let mut graph_ops = GraphOps::new();
        graph_ops.connect().await?;
        graph_ops.graph.clear().await?;
        graph_ops.upload_btreemap_to_neo4j(&btree_graph, None).await?;
        
        // Phase 4: Query tested endpoints using query_nodes_with_count
        let (tested_count, tested_results) = graph_ops
            .query_nodes_with_count(
                NodeType::Endpoint,
                0,
                100,
                false,
                Some("tested"),
                false,
                false,
                None,
                None,
                None,
            )
            .await?;
        
        assert_eq!(
            tested_count,
            6,
            "Neo4j should report exactly 6 tested endpoints"
        );
        
        assert_eq!(
            tested_results.len(),
            6,
            "Neo4j should return exactly 6 tested endpoint results"
        );
        
        // Verify each tested result has test_count > 0 and is_covered = true
        for (node_data, _usage_count, is_covered, test_count, _ref_id, _body_length, _line_count) in &tested_results {
            assert_eq!(
                *is_covered,
                true,
                "Tested endpoint {} {} should have is_covered = true",
                node_data.meta.get("verb").unwrap_or(&"".to_string()),
                node_data.name
            );
            assert!(
                *test_count > 0,
                "Tested endpoint {} {} should have test_count > 0, got {}",
                node_data.meta.get("verb").unwrap_or(&"".to_string()),
                node_data.name,
                test_count
            );
        }
        
        // Phase 4: Query untested endpoints
        let (untested_count, untested_results) = graph_ops
            .query_nodes_with_count(
                NodeType::Endpoint,
                0,
                100,
                false,
                Some("untested"),
                false,
                false,
                None,
                None,
                None,
            )
            .await?;
        
        assert_eq!(
            untested_count,
            3,
            "Neo4j should report exactly 3 untested endpoints"
        );
        
        assert_eq!(
            untested_results.len(),
            3,
            "Neo4j should return exactly 3 untested endpoint results"
        );
        
        // Verify each untested result has test_count == 0 and is_covered = false
        for (node_data, _usage_count, is_covered, test_count, _ref_id, _body_length, _line_count) in &untested_results {
            assert_eq!(
                *is_covered,
                false,
                "Untested endpoint {} {} should have is_covered = false",
                node_data.meta.get("verb").unwrap_or(&"".to_string()),
                node_data.name
            );
            assert_eq!(
                *test_count,
                0,
                "Untested endpoint {} {} should have test_count == 0, got {}",
                node_data.meta.get("verb").unwrap_or(&"".to_string()),
                node_data.name,
                test_count
            );
        }
        
        // Phase 5: Validate coverage statistics
        let coverage = graph_ops.get_coverage(None, None).await?;
        
        let integration_stats = coverage
            .integration_tests
            .expect("Integration test stats should exist");
        
        // Validate integration test stats
        assert_eq!(
            integration_stats.total,
            9,
            "Integration coverage should report 9 total endpoints"
        );
        
        assert_eq!(
            integration_stats.covered,
            6,
            "Integration coverage should report 6 covered endpoints"
        );
        
        assert_eq!(
            integration_stats.total_tests,
            4,
            "Integration coverage should report 4 total integration tests"
        );
        
        // Calculate expected percentage: 6/9 * 100 = 66.666...
        let expected_percent = 66.67;
        let percent_tolerance = 0.01;
        let percent_diff = (integration_stats.percent - expected_percent).abs();
        
        assert!(
            percent_diff <= percent_tolerance,
            "Integration coverage percentage should be approximately {}%, got {}%",
            expected_percent,
            integration_stats.percent
        );
    }
    
    Ok(())
}
