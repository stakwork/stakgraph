use crate::lang::graphs::{BTreeMapGraph, EdgeType, Graph, NodeType, TestFilters};
use crate::lang::Lang;
use crate::repo::{Repo, Repos};
use crate::utils::get_use_lsp;
use shared::error::Result;
use std::str::FromStr;
use tokio::sync::OnceCell;

async fn setup_rust_graph() -> Result<crate::lang::graphs::graph_ops::GraphOps> {
    static GRAPH_INIT: OnceCell<()> = OnceCell::const_new();

    GRAPH_INIT
        .get_or_init(|| async {
            let use_lsp = get_use_lsp();
            let repo = Repo::new(
                "src/testing/rust",
                Lang::from_str("rust").unwrap(),
                use_lsp,
                Vec::new(),
                Vec::new(),
            )
            .unwrap();

            let repos = Repos(vec![repo]);
            let btree_graph = repos
                .build_graphs_btree()
                .await
                .expect("Failed to build graph");
            btree_graph.analysis();

            let mut graph_ops =
                crate::lang::graphs::graph_ops::GraphOps::with_namespace("test_rust");
            graph_ops.connect().await.expect("Failed to connect");
            graph_ops
                .graph
                .clear()
                .await
                .expect("Failed to clear graph");
            graph_ops
                .upload_btreemap_to_neo4j(&btree_graph, None)
                .await
                .expect("Failed to upload graph");
        })
        .await;

    let mut graph_ops = crate::lang::graphs::graph_ops::GraphOps::with_namespace("test_rust");
    graph_ops.connect().await?;
    Ok(graph_ops)
}

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
    assert_eq!(
        functions.len(),
        74,
        "Expected 74 functions after test improvements"
    );

    let unit_tests = graph.find_nodes_by_type(NodeType::UnitTest);
    assert_eq!(unit_tests.len(), 43, "Expected 43 unit tests");

    let integration_tests = graph.find_nodes_by_type(NodeType::IntegrationTest);
    assert_eq!(integration_tests.len(), 17, "Expected 17 integration tests");

    let e2e_tests = graph.find_nodes_by_type(NodeType::E2eTest);
    assert_eq!(e2e_tests.len(), 8, "Expected 8 e2e tests");

    let calls_edges = graph.count_edges_of_type(EdgeType::Calls);
    assert_eq!(calls_edges, 104, "Expected 104 Calls edges");

    let unit_test_to_function_edges =
        graph.find_nodes_with_edge_type(NodeType::UnitTest, NodeType::Function, EdgeType::Calls);
    assert_eq!(
        unit_test_to_function_edges.len(),
        32,
        "Expected 32 UnitTest → Function edges"
    );

    let integration_test_to_function_edges = graph.find_nodes_with_edge_type(
        NodeType::IntegrationTest,
        NodeType::Function,
        EdgeType::Calls,
    );
    assert_eq!(
        integration_test_to_function_edges.len(),
        31,
        "Expected 31 IntegrationTest → Function edges"
    );

    let unique_functions_tested: std::collections::HashSet<String> = unit_test_to_function_edges
        .iter()
        .map(|(_, target)| format!("{}|{}|{}", target.name, target.file, target.start))
        .collect();
    assert_eq!(
        unique_functions_tested.len(),
        19,
        "Expected 19 unique functions covered by unit tests"
    );

    let total_test_coverage_edges =
        unit_test_to_function_edges.len() + integration_test_to_function_edges.len();
    assert_eq!(
        total_test_coverage_edges, 63,
        "Expected 63 total test coverage edges"
    );

    Ok(())
}

#[cfg(feature = "neo4j")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_rust_graph_upload() -> Result<()> {
    let graph_ops = setup_rust_graph().await?;
    let (nodes, edges) = graph_ops.get_graph_size().await?;

    assert_eq!(nodes, 239, "Graph should have 239 nodes after upload");
    assert_eq!(edges, 390, "Graph should have 390 edges after upload");

    Ok(())
}

#[cfg(feature = "neo4j")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_coverage_default_params() -> Result<()> {
    let mut graph_ops = setup_rust_graph().await?;

    let coverage = graph_ops.get_coverage(None, None, None).await?;

    assert_eq!(
        coverage.language,
        Some("rust".to_string()),
        "Should detect Rust language"
    );

    let unit = coverage.unit_tests.expect("Should have unit_tests");
    assert_eq!(unit.total, 56, "Should have 56 unit test targets");
    assert_eq!(unit.total_tests, 43, "Should have 43 unit tests");

    let integration = coverage
        .integration_tests
        .expect("Should have integration_tests");
    assert_eq!(
        integration.total, 56,
        "Should have 56 integration test targets"
    );
    assert_eq!(
        integration.total_tests, 17,
        "Should have 17 integration tests"
    );

    let e2e = coverage.e2e_tests.expect("Should have e2e_tests");
    assert_eq!(e2e.total_tests, 8, "Should have 8 e2e tests");

    Ok(())
}

#[cfg(feature = "neo4j")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_coverage_with_repo_filter() -> Result<()> {
    let mut graph_ops = setup_rust_graph().await?;
    let coverage = graph_ops
        .get_coverage(Some("src/testing/rust"), None, None)
        .await?;
    assert_eq!(coverage.language, Some("rust".to_string()));

    let empty_coverage = graph_ops
        .get_coverage(Some("nonexistent/repo"), None, None)
        .await?;
    if let Some(unit) = &empty_coverage.unit_tests {
        assert_eq!(unit.covered, 0, "Non-existent repo should have no coverage");
    }

    Ok(())
}

#[cfg(feature = "neo4j")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_coverage_with_ignore_dirs() -> Result<()> {
    let mut graph_ops = setup_rust_graph().await?;

    let filters = TestFilters {
        unit_regexes: vec![],
        integration_regexes: vec![],
        e2e_regexes: vec![],
        target_regex: None,
        ignore_dirs: vec!["routes".to_string()],
    };
    let filtered = graph_ops.get_coverage(None, Some(filters), None).await?;

    let unit = filtered.unit_tests.expect("Should have unit_tests");
    assert_eq!(
        unit.total, 51,
        "Should have 51 targets after ignoring routes"
    );

    Ok(())
}

#[cfg(feature = "neo4j")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_coverage_with_regex_filter() -> Result<()> {
    let mut graph_ops = setup_rust_graph().await?;
    let filters = TestFilters {
        unit_regexes: vec![],
        integration_regexes: vec![],
        e2e_regexes: vec![],
        target_regex: Some(".*person.*".to_string()),
        ignore_dirs: vec![],
    };
    let coverage = graph_ops.get_coverage(None, Some(filters), None).await?;

    assert_eq!(coverage.language, Some("rust".to_string()));

    Ok(())
}

#[cfg(feature = "neo4j")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_coverage_with_is_muted() -> Result<()> {
    let mut graph_ops = setup_rust_graph().await?;

    let muted_coverage = graph_ops.get_coverage(None, None, Some(true)).await?;
    let unmuted_coverage = graph_ops.get_coverage(None, None, Some(false)).await?;

    assert_eq!(muted_coverage.language, Some("rust".to_string()));
    assert_eq!(unmuted_coverage.language, Some("rust".to_string()));

    Ok(())
}

#[cfg(feature = "neo4j")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_coverage_combined_filters() -> Result<()> {
    let mut graph_ops = setup_rust_graph().await?;

    let filters = TestFilters {
        unit_regexes: vec![],
        integration_regexes: vec![],
        e2e_regexes: vec![],
        target_regex: Some(".*db.*".to_string()),
        ignore_dirs: vec!["tests".to_string()],
    };

    let coverage = graph_ops
        .get_coverage(Some("src/testing/rust"), Some(filters), Some(false))
        .await?;

    assert_eq!(coverage.language, Some("rust".to_string()));

    Ok(())
}

#[cfg(feature = "neo4j")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_nodes_function_type() -> Result<()> {
    let mut graph_ops = setup_rust_graph().await?;

    let (count, results) = graph_ops
        .query_nodes_with_count(
            &[NodeType::Function],
            0,
            100,
            true,
            None,
            false,
            false,
            None,
            None,
            None,
            None,
        )
        .await?;

    assert_eq!(count, 56, "Should have 56 unique Function nodes");

    for (node_type, _, _, _, _, _, _, _, _) in &results {
        assert_eq!(
            *node_type,
            NodeType::Function,
            "All results should be Function type"
        );
    }

    Ok(())
}

#[cfg(feature = "neo4j")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_nodes_endpoint_type() -> Result<()> {
    let mut graph_ops = setup_rust_graph().await?;

    let (count, results) = graph_ops
        .query_nodes_with_count(
            &[NodeType::Endpoint],
            0,
            100,
            true,
            None,
            false,
            false,
            None,
            None,
            None,
            None,
        )
        .await?;

    assert_eq!(count, 18, "Should have 18 Endpoint nodes");
    assert_eq!(results.len(), 18);

    for (node_type, node_data, _, _, _, _, _, _, _) in &results {
        assert_eq!(*node_type, NodeType::Endpoint);
        assert!(
            node_data.meta.contains_key("verb"),
            "Endpoints should have verb metadata"
        );
    }

    Ok(())
}

#[cfg(feature = "neo4j")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_nodes_class_type() -> Result<()> {
    let mut graph_ops = setup_rust_graph().await?;

    let (count, results) = graph_ops
        .query_nodes_with_count(
            &[NodeType::Class],
            0,
            100,
            true,
            None,
            false,
            false,
            None,
            None,
            None,
            None,
        )
        .await?;

    assert_eq!(count, 7, "Should have 7 Class nodes");
    assert_eq!(results.len(), 7);

    Ok(())
}

#[cfg(feature = "neo4j")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_nodes_trait_type() -> Result<()> {
    let mut graph_ops = setup_rust_graph().await?;

    let (count, results) = graph_ops
        .query_nodes_with_count(
            &[NodeType::Trait],
            0,
            100,
            true,
            None,
            false,
            false,
            None,
            None,
            None,
            None,
        )
        .await?;

    assert_eq!(count, 4, "Should have 4 Trait nodes");
    assert_eq!(results.len(), 4);

    Ok(())
}

#[cfg(feature = "neo4j")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_nodes_data_model_type() -> Result<()> {
    let mut graph_ops = setup_rust_graph().await?;

    let (count, results) = graph_ops
        .query_nodes_with_count(
            &[NodeType::DataModel],
            0,
            100,
            true,
            None,
            false,
            false,
            None,
            None,
            None,
            None,
        )
        .await?;

    assert_eq!(count, 18, "Should have 18 DataModel nodes");
    assert_eq!(results.len(), 18);

    Ok(())
}

#[cfg(feature = "neo4j")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_nodes_unit_test_type() -> Result<()> {
    let mut graph_ops = setup_rust_graph().await?;

    let (count, _results) = graph_ops
        .query_nodes_with_count(
            &[NodeType::UnitTest],
            0,
            100,
            true,
            None,
            false,
            false,
            None,
            None,
            None,
            None,
        )
        .await?;

    assert_eq!(count, 43, "Should have 43 UnitTest nodes");

    Ok(())
}

#[cfg(feature = "neo4j")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_nodes_integration_test_type() -> Result<()> {
    let mut graph_ops = setup_rust_graph().await?;

    let (count, _results) = graph_ops
        .query_nodes_with_count(
            &[NodeType::IntegrationTest],
            0,
            100,
            true,
            None,
            false,
            false,
            None,
            None,
            None,
            None,
        )
        .await?;

    assert_eq!(count, 17, "Should have 17 IntegrationTest nodes");

    Ok(())
}

#[cfg(feature = "neo4j")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_nodes_e2e_test_type() -> Result<()> {
    let mut graph_ops = setup_rust_graph().await?;

    let (count, _results) = graph_ops
        .query_nodes_with_count(
            &[NodeType::E2eTest],
            0,
            100,
            true,
            None,
            false,
            false,
            None,
            None,
            None,
            None,
        )
        .await?;

    assert_eq!(count, 8, "Should have 8 E2eTest nodes");

    Ok(())
}

#[cfg(feature = "neo4j")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_nodes_multi_type() -> Result<()> {
    let mut graph_ops = setup_rust_graph().await?;

    let (count, results) = graph_ops
        .query_nodes_with_count(
            &[NodeType::Function, NodeType::Endpoint],
            0,
            200,
            true,
            None,
            false,
            false,
            None,
            None,
            None,
            None,
        )
        .await?;

    assert_eq!(count, 74, "Should have 56 Functions + 18 Endpoints = 74");

    let has_function = results
        .iter()
        .any(|(nt, _, _, _, _, _, _, _, _)| *nt == NodeType::Function);
    let has_endpoint = results
        .iter()
        .any(|(nt, _, _, _, _, _, _, _, _)| *nt == NodeType::Endpoint);

    assert!(has_function, "Should include Function nodes");
    assert!(has_endpoint, "Should include Endpoint nodes");

    Ok(())
}

#[cfg(feature = "neo4j")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_nodes_all_test_types() -> Result<()> {
    let mut graph_ops = setup_rust_graph().await?;

    let (count, _results) = graph_ops
        .query_nodes_with_count(
            &[
                NodeType::UnitTest,
                NodeType::IntegrationTest,
                NodeType::E2eTest,
            ],
            0,
            100,
            true,
            None,
            false,
            false,
            None,
            None,
            None,
            None,
        )
        .await?;

    assert_eq!(count, 68, "Should have 43 + 17 + 8 = 68 test nodes");

    Ok(())
}

#[cfg(feature = "neo4j")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_nodes_pagination_default() -> Result<()> {
    let mut graph_ops = setup_rust_graph().await?;

    let (count, results) = graph_ops
        .query_nodes_with_count(
            &[NodeType::Function],
            0,
            10,
            true,
            None,
            false,
            false,
            None,
            None,
            None,
            None,
        )
        .await?;

    assert_eq!(count, 56, "Total count should be 56");
    assert_eq!(results.len(), 10, "Should return 10 items per page");

    Ok(())
}

#[cfg(feature = "neo4j")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_nodes_pagination_second_page() -> Result<()> {
    let mut graph_ops = setup_rust_graph().await?;

    let (_count1, results1) = graph_ops
        .query_nodes_with_count(
            &[NodeType::Function],
            0,
            10,
            true,
            None,
            false,
            false,
            None,
            None,
            None,
            None,
        )
        .await?;

    let (_count2, results2) = graph_ops
        .query_nodes_with_count(
            &[NodeType::Function],
            10,
            10,
            true,
            None,
            false,
            false,
            None,
            None,
            None,
            None,
        )
        .await?;

    assert_eq!(results2.len(), 10, "Second page should have 10 items");

    let page1_identifiers: Vec<_> = results1
        .iter()
        .map(|(_, nd, _, _, _, _, _, _, _)| (&nd.name, &nd.file))
        .collect();
    let page2_identifiers: Vec<_> = results2
        .iter()
        .map(|(_, nd, _, _, _, _, _, _, _)| (&nd.name, &nd.file))
        .collect();

    for (name, file) in &page2_identifiers {
        assert!(
            !page1_identifiers.contains(&(name, file)),
            "Pages should not overlap: {} in {}",
            name,
            file
        );
    }

    Ok(())
}

#[cfg(feature = "neo4j")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_nodes_pagination_large_offset() -> Result<()> {
    let mut graph_ops = setup_rust_graph().await?;

    let (count, results) = graph_ops
        .query_nodes_with_count(
            &[NodeType::Function],
            1000,
            10,
            true,
            None,
            false,
            false,
            None,
            None,
            None,
            None,
        )
        .await?;

    assert_eq!(count, 56, "Total count should still be accurate");
    assert!(
        results.is_empty(),
        "Should return empty for offset beyond data"
    );

    Ok(())
}

#[cfg(feature = "neo4j")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_nodes_pagination_max_limit() -> Result<()> {
    let mut graph_ops = setup_rust_graph().await?;

    let (count, results) = graph_ops
        .query_nodes_with_count(
            &[NodeType::Function],
            0,
            100,
            true,
            None,
            false,
            false,
            None,
            None,
            None,
            None,
        )
        .await?;

    assert_eq!(count, 56, "Total count should be 56");
    assert_eq!(results.len(), 56, "Should return all 56 functions");

    Ok(())
}

#[cfg(feature = "neo4j")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_nodes_coverage_tested() -> Result<()> {
    let mut graph_ops = setup_rust_graph().await?;

    let (_count, results) = graph_ops
        .query_nodes_with_count(
            &[NodeType::Function],
            0,
            100,
            true,
            Some("tested"),
            false,
            false,
            None,
            None,
            None,
            None,
        )
        .await?;

    for (_, _, _, covered, test_count, _, _, _, _) in &results {
        assert!(*covered, "Tested filter should only return covered nodes");
        assert_ne!(*test_count, 0, "Tested nodes should have test_count != 0");
    }

    Ok(())
}

#[cfg(feature = "neo4j")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_nodes_coverage_untested() -> Result<()> {
    let mut graph_ops = setup_rust_graph().await?;

    let (_count, results) = graph_ops
        .query_nodes_with_count(
            &[NodeType::Function],
            0,
            100,
            true,
            Some("untested"),
            false,
            false,
            None,
            None,
            None,
            None,
        )
        .await?;

    for (_, _, _, covered, test_count, _, _, _, _) in &results {
        assert!(
            !*covered,
            "Untested filter should only return uncovered nodes"
        );
        assert_eq!(*test_count, 0, "Untested nodes should have test_count = 0");
    }

    Ok(())
}

#[cfg(feature = "neo4j")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_nodes_coverage_all() -> Result<()> {
    let mut graph_ops = setup_rust_graph().await?;

    let (all_count, _) = graph_ops
        .query_nodes_with_count(
            &[NodeType::Function],
            0,
            100,
            true,
            Some("all"),
            false,
            false,
            None,
            None,
            None,
            None,
        )
        .await?;

    let (tested_count, _) = graph_ops
        .query_nodes_with_count(
            &[NodeType::Function],
            0,
            100,
            true,
            Some("tested"),
            false,
            false,
            None,
            None,
            None,
            None,
        )
        .await?;

    let (untested_count, _) = graph_ops
        .query_nodes_with_count(
            &[NodeType::Function],
            0,
            100,
            true,
            Some("untested"),
            false,
            false,
            None,
            None,
            None,
            None,
        )
        .await?;

    assert_eq!(
        all_count,
        tested_count + untested_count,
        "all = tested + untested"
    );

    Ok(())
}

#[cfg(feature = "neo4j")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_nodes_body_length() -> Result<()> {
    let mut graph_ops = setup_rust_graph().await?;

    let (_, results) = graph_ops
        .query_nodes_with_count(
            &[NodeType::Function],
            0,
            10,
            true,
            None,
            true,
            false,
            None,
            None,
            None,
            None,
        )
        .await?;

    for (_, _, _, _, _, _, body_length, _, _) in &results {
        assert!(
            body_length.is_some(),
            "body_length should be populated when requested"
        );
    }

    Ok(())
}

#[cfg(feature = "neo4j")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_nodes_line_count() -> Result<()> {
    let mut graph_ops = setup_rust_graph().await?;

    let (_, results) = graph_ops
        .query_nodes_with_count(
            &[NodeType::Function],
            0,
            10,
            true,
            None,
            false,
            true,
            None,
            None,
            None,
            None,
        )
        .await?;

    for (_, _, _, _, _, _, _, line_count, _) in &results {
        assert!(
            line_count.is_some(),
            "line_count should be populated when requested"
        );
    }

    Ok(())
}

#[cfg(feature = "neo4j")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_nodes_both_metrics() -> Result<()> {
    let mut graph_ops = setup_rust_graph().await?;

    let (_, results) = graph_ops
        .query_nodes_with_count(
            &[NodeType::Function],
            0,
            10,
            true,
            None,
            true,
            true,
            None,
            None,
            None,
            None,
        )
        .await?;

    for (_, _, _, _, _, _, body_length, line_count, _) in &results {
        assert!(body_length.is_some(), "body_length should be populated");
        assert!(line_count.is_some(), "line_count should be populated");
    }

    Ok(())
}

#[cfg(feature = "neo4j")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_nodes_repo_filter() -> Result<()> {
    let mut graph_ops = setup_rust_graph().await?;

    let (count, results) = graph_ops
        .query_nodes_with_count(
            &[NodeType::Function],
            0,
            100,
            true,
            None,
            false,
            false,
            Some("src/testing/rust"),
            None,
            None,
            None,
        )
        .await?;

    assert_eq!(count, 56, "Should find 56 functions in Rust repo");

    for (_, node_data, _, _, _, _, _, _, _) in &results {
        assert!(
            node_data.file.starts_with("src/testing/rust"),
            "All results should be from Rust repo"
        );
    }

    Ok(())
}

#[cfg(feature = "neo4j")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_nodes_ignore_dirs() -> Result<()> {
    let mut graph_ops = setup_rust_graph().await?;

    let filters = TestFilters {
        unit_regexes: vec![],
        integration_regexes: vec![],
        e2e_regexes: vec![],
        target_regex: None,
        ignore_dirs: vec!["routes".to_string()],
    };

    let (_filtered_count, results) = graph_ops
        .query_nodes_with_count(
            &[NodeType::Function],
            0,
            100,
            true,
            None,
            false,
            false,
            None,
            Some(filters),
            None,
            None,
        )
        .await?;

    for (_, node_data, _, _, _, _, _, _, _) in &results {
        assert!(
            !node_data.file.contains("routes"),
            "Should not include files from ignored dir"
        );
    }

    Ok(())
}

#[cfg(feature = "neo4j")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_nodes_regex_filter() -> Result<()> {
    let mut graph_ops = setup_rust_graph().await?;

    let filters = TestFilters {
        unit_regexes: vec![],
        integration_regexes: vec![],
        e2e_regexes: vec![],
        target_regex: Some(".*person.*".to_string()),
        ignore_dirs: vec![],
    };

    let (_count, results) = graph_ops
        .query_nodes_with_count(
            &[NodeType::Function],
            0,
            100,
            true,
            None,
            false,
            false,
            None,
            Some(filters),
            None,
            None,
        )
        .await?;

    for (_, node_data, _, _, _, _, _, _, _) in &results {
        assert!(
            node_data.file.to_lowercase().contains("person")
                || node_data.name.to_lowercase().contains("person"),
            "All results should match regex: {}",
            node_data.file
        );
    }

    Ok(())
}

#[cfg(feature = "neo4j")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_nodes_search() -> Result<()> {
    let mut graph_ops = setup_rust_graph().await?;

    let (count, results) = graph_ops
        .query_nodes_with_count(
            &[NodeType::Function],
            0,
            100,
            true,
            None,
            false,
            false,
            None,
            None,
            Some("person"),
            None,
        )
        .await?;

    assert_ne!(count, 0, "Should find functions matching search");

    for (_, node_data, _, _, _, _, _, _, _) in &results {
        assert!(
            node_data.name.to_lowercase().contains("person")
                || node_data.file.to_lowercase().contains("person"),
            "Results should match search term"
        );
    }

    Ok(())
}

#[cfg(feature = "neo4j")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_nodes_search_no_match() -> Result<()> {
    let mut graph_ops = setup_rust_graph().await?;

    let (count, results) = graph_ops
        .query_nodes_with_count(
            &[NodeType::Function],
            0,
            100,
            true,
            None,
            false,
            false,
            None,
            None,
            Some("xyznonexistent123"),
            None,
        )
        .await?;

    assert_eq!(count, 0, "Should find no results for non-matching search");
    assert!(results.is_empty());

    Ok(())
}

#[cfg(feature = "neo4j")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_nodes_is_muted() -> Result<()> {
    let mut graph_ops = setup_rust_graph().await?;

    let (_muted_count, _) = graph_ops
        .query_nodes_with_count(
            &[NodeType::Function],
            0,
            100,
            true,
            None,
            false,
            false,
            None,
            None,
            None,
            Some(true),
        )
        .await?;

    let (unmuted_count, _) = graph_ops
        .query_nodes_with_count(
            &[NodeType::Function],
            0,
            100,
            true,
            None,
            false,
            false,
            None,
            None,
            None,
            Some(false),
        )
        .await?;

    let (all_count, _) = graph_ops
        .query_nodes_with_count(
            &[NodeType::Function],
            0,
            100,
            true,
            None,
            false,
            false,
            None,
            None,
            None,
            None,
        )
        .await?;

    assert_eq!(unmuted_count, 56, "Unmuted count should be 56");
    assert_eq!(all_count, 56, "All count should be 56");
    assert_eq!(
        unmuted_count, all_count,
        "Unmuted should equal all when no nodes muted"
    );

    Ok(())
}

#[cfg(feature = "neo4j")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_nodes_sort_by_test_count() -> Result<()> {
    let mut graph_ops = setup_rust_graph().await?;

    let (_, results) = graph_ops
        .query_nodes_with_count(
            &[NodeType::Function],
            0,
            20,
            true,
            None,
            false,
            false,
            None,
            None,
            None,
            None,
        )
        .await?;

    let test_counts: Vec<usize> = results
        .iter()
        .map(|(_, _, _, _, test_count, _, _, _, _)| *test_count)
        .collect();

    let is_sorted_descending = test_counts
        .windows(2)
        .all(|w| w[0].cmp(&w[1]) != std::cmp::Ordering::Less);
    assert!(
        is_sorted_descending,
        "Should be sorted by test_count descending"
    );

    Ok(())
}

#[cfg(feature = "neo4j")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_nodes_complex_combination() -> Result<()> {
    let mut graph_ops = setup_rust_graph().await?;

    let filters = TestFilters {
        unit_regexes: vec![],
        integration_regexes: vec![],
        e2e_regexes: vec![],
        target_regex: Some(".*db.*".to_string()),
        ignore_dirs: vec!["tests".to_string()],
    };

    let (_count, results) = graph_ops
        .query_nodes_with_count(
            &[NodeType::Function, NodeType::Class],
            5,
            10,
            true,
            Some("all"),
            true,
            true,
            Some("src/testing/rust"),
            Some(filters),
            None,
            Some(false),
        )
        .await?;

    for (_, node_data, _, _, _, _, body_length, line_count, _) in &results {
        assert!(node_data.file.starts_with("src/testing/rust"));
        assert!(!node_data.file.contains("tests"));
        assert!(body_length.is_some());
        assert!(line_count.is_some());
    }

    Ok(())
}

#[cfg(feature = "neo4j")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_has_function_covered() -> Result<()> {
    let mut graph_ops = setup_rust_graph().await?;

    let _covered = graph_ops
        .has_coverage(
            NodeType::Function,
            "get_person",
            "src/testing/rust/src/routes/axum_routes.rs",
            None,
            None,
            None,
        )
        .await?;

    Ok(())
}

#[cfg(feature = "neo4j")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_has_function_with_start() -> Result<()> {
    let mut graph_ops = setup_rust_graph().await?;

    let covered = graph_ops
        .has_coverage(
            NodeType::Function,
            "get_person",
            "src/testing/rust/src/routes/axum_routes.rs",
            Some(999),
            None,
            None,
        )
        .await?;

    assert!(!covered, "Wrong start line should not match");

    Ok(())
}

#[cfg(feature = "neo4j")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_has_function_with_root() -> Result<()> {
    let mut graph_ops = setup_rust_graph().await?;

    let _covered = graph_ops
        .has_coverage(
            NodeType::Function,
            "get_person",
            "src/testing/rust/src/routes/axum_routes.rs",
            None,
            Some("src/testing/rust"),
            None,
        )
        .await?;

    let wrong_root = graph_ops
        .has_coverage(
            NodeType::Function,
            "get_person",
            "src/testing/rust/src/routes/axum_routes.rs",
            None,
            Some("wrong/root"),
            None,
        )
        .await?;

    assert!(!wrong_root, "Wrong root should not match");

    Ok(())
}

#[cfg(feature = "neo4j")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_has_function_with_tests_filter() -> Result<()> {
    let mut graph_ops = setup_rust_graph().await?;

    let _unit_covered = graph_ops
        .has_coverage(
            NodeType::Function,
            "get_person",
            "src/testing/rust/src/routes/axum_routes.rs",
            None,
            None,
            Some("unit"),
        )
        .await?;

    let _integration_covered = graph_ops
        .has_coverage(
            NodeType::Function,
            "get_person",
            "src/testing/rust/src/routes/axum_routes.rs",
            None,
            None,
            Some("integration"),
        )
        .await?;

    Ok(())
}

#[cfg(feature = "neo4j")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_has_endpoint_covered() -> Result<()> {
    let mut graph_ops = setup_rust_graph().await?;

    let _covered = graph_ops
        .has_coverage(
            NodeType::Endpoint,
            "/person/:id",
            "src/testing/rust/src/routes/axum_routes.rs",
            None,
            None,
            None,
        )
        .await?;

    Ok(())
}

#[cfg(feature = "neo4j")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_has_nonexistent_function() -> Result<()> {
    let mut graph_ops = setup_rust_graph().await?;

    let covered = graph_ops
        .has_coverage(
            NodeType::Function,
            "nonexistent_function_xyz",
            "nonexistent_file.rs",
            None,
            None,
            None,
        )
        .await?;

    assert!(!covered, "Non-existent function should not be covered");

    Ok(())
}

#[cfg(feature = "neo4j")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_has_all_test_types() -> Result<()> {
    let mut graph_ops = setup_rust_graph().await?;

    let test_types = vec!["unit", "integration", "e2e"];

    for test_type in test_types {
        let _covered = graph_ops
            .has_coverage(
                NodeType::Function,
                "get_person",
                "src/testing/rust/src/routes/axum_routes.rs",
                None,
                None,
                Some(test_type),
            )
            .await?;
    }

    let _combined = graph_ops
        .has_coverage(
            NodeType::Function,
            "get_person",
            "src/testing/rust/src/routes/axum_routes.rs",
            None,
            None,
            Some("unit,integration"),
        )
        .await?;

    Ok(())
}
