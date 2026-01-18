use crate::lang::graphs::{Graph, NodeType, TestFilters};
use crate::lang::Lang;
use crate::repo::{Repo, Repos};
use crate::utils::get_use_lsp;
use shared::error::Result;
use std::str::FromStr;
use tokio::sync::OnceCell;

/// Helper to build the Ruby test graph and upload to Neo4j
async fn setup_ruby_graph() -> Result<crate::lang::graphs::graph_ops::GraphOps> {
    static GRAPH_INIT: OnceCell<()> = OnceCell::const_new();

    GRAPH_INIT
        .get_or_init(|| async {
            let use_lsp = get_use_lsp();
            let repo = Repo::new(
                "src/testing/ruby",
                Lang::from_str("ruby").unwrap(),
                use_lsp,
                Vec::new(),
                Vec::new(),
            )
            .unwrap();

            let repos = Repos(vec![repo], Vec::new(), None);
            let btree_graph = repos
                .build_graphs_btree()
                .await
                .expect("Failed to build graph");
            btree_graph.analysis();

            let mut graph_ops = crate::lang::graphs::graph_ops::GraphOps::new();
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

    let mut graph_ops = crate::lang::graphs::graph_ops::GraphOps::new();
    graph_ops.connect().await?;
    Ok(graph_ops)
}

#[cfg(feature = "neo4j")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_ruby_graph_upload() -> Result<()> {
    let graph_ops = setup_ruby_graph().await?;
    let (nodes, edges) = graph_ops.get_graph_size().await?;

    assert!(nodes > 0, "Graph should have nodes after upload");
    assert!(edges > 0, "Graph should have edges after upload");

    Ok(())
}

#[cfg(feature = "neo4j")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_coverage_default_params() -> Result<()> {
    let mut graph_ops = setup_ruby_graph().await?;

    let coverage = graph_ops.get_coverage(None, None, None).await?;

    assert_eq!(
        coverage.language,
        Some("ruby".to_string()),
        "Should detect Ruby language"
    );

    if let Some(unit) = &coverage.unit_tests {
        assert!(unit.total > 0, "Should have unit test targets");
        assert!(unit.total_tests > 0, "Should have unit tests");
        assert!(
            unit.percent >= 0.0 && unit.percent <= 100.0,
            "Percent should be valid"
        );
        assert!(
            unit.line_percent >= 0.0 && unit.line_percent <= 100.0,
            "Line percent should be valid"
        );
        if unit.total > 0 {
            let expected_percent = (unit.covered as f64 / unit.total as f64) * 100.0;
            let diff = (unit.percent - expected_percent).abs();
            assert!(diff < 1.0, "Percent calculation should be consistent");
        }
    }

    if let Some(integration) = &coverage.integration_tests {
        assert!(
            integration.total > 0,
            "Should have integration test targets"
        );
        assert!(integration.total_tests > 0, "Should have integration tests");
    }

    if let Some(e2e) = &coverage.e2e_tests {
        assert!(e2e.total_tests > 0, "Should have e2e tests");
    }

    Ok(())
}

#[cfg(feature = "neo4j")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_coverage_with_repo_filter() -> Result<()> {
    let mut graph_ops = setup_ruby_graph().await?;
    let coverage = graph_ops
        .get_coverage(Some("src/testing/ruby"), None, None)
        .await?;
    assert_eq!(coverage.language, Some("ruby".to_string()));

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
    let mut graph_ops = setup_ruby_graph().await?;

    let baseline = graph_ops.get_coverage(None, None, None).await?;

    let filters = TestFilters {
        unit_regexes: vec![],
        integration_regexes: vec![],
        e2e_regexes: vec![],
        target_regex: None,
        ignore_dirs: vec!["spec/support".to_string()],
    };
    let filtered = graph_ops.get_coverage(None, Some(filters), None).await?;

    if let (Some(base_unit), Some(filt_unit)) = (&baseline.unit_tests, &filtered.unit_tests) {
        assert!(
            filt_unit.total <= base_unit.total,
            "Ignoring dirs should not increase targets"
        );
    }

    Ok(())
}

#[cfg(feature = "neo4j")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_coverage_with_regex_filter() -> Result<()> {
    let mut graph_ops = setup_ruby_graph().await?;
    let filters = TestFilters {
        unit_regexes: vec![],
        integration_regexes: vec![],
        e2e_regexes: vec![],
        target_regex: Some(".*person.*".to_string()),
        ignore_dirs: vec![],
    };
    let coverage = graph_ops.get_coverage(None, Some(filters), None).await?;

    assert_eq!(coverage.language, Some("ruby".to_string()));

    Ok(())
}

#[cfg(feature = "neo4j")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_coverage_with_is_muted() -> Result<()> {
    let mut graph_ops = setup_ruby_graph().await?;

    let muted_coverage = graph_ops.get_coverage(None, None, Some(true)).await?;
    let unmuted_coverage = graph_ops.get_coverage(None, None, Some(false)).await?;

    assert_eq!(muted_coverage.language, Some("ruby".to_string()));
    assert_eq!(unmuted_coverage.language, Some("ruby".to_string()));

    Ok(())
}

#[cfg(feature = "neo4j")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_coverage_combined_filters() -> Result<()> {
    let mut graph_ops = setup_ruby_graph().await?;

    let filters = TestFilters {
        unit_regexes: vec![],
        integration_regexes: vec![],
        e2e_regexes: vec![],
        target_regex: Some(".*Controller.*".to_string()),
        ignore_dirs: vec!["spec/support".to_string()],
    };

    let coverage = graph_ops
        .get_coverage(Some("src/testing/ruby"), Some(filters), Some(false))
        .await?;

    assert_eq!(coverage.language, Some("ruby".to_string()));

    Ok(())
}

#[cfg(feature = "neo4j")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_nodes_function_type() -> Result<()> {
    let mut graph_ops = setup_ruby_graph().await?;

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

    assert!(count > 0, "Should have Function nodes");
    assert!(!results.is_empty(), "Should return Function results");

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
    let mut graph_ops = setup_ruby_graph().await?;

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

    assert_eq!(count, 23, "Should have 23 Endpoint nodes");

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
    let mut graph_ops = setup_ruby_graph().await?;

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

    assert_eq!(count, 35, "Should have 35 Class nodes");
    assert_eq!(results.len(), 35);

    Ok(())
}

#[cfg(feature = "neo4j")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_nodes_unit_test_type() -> Result<()> {
    let mut graph_ops = setup_ruby_graph().await?;

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

    assert_eq!(count, 21, "Should have 21 UnitTest nodes");

    Ok(())
}

#[cfg(feature = "neo4j")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_nodes_integration_test_type() -> Result<()> {
    let mut graph_ops = setup_ruby_graph().await?;

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

    assert_eq!(count, 22, "Should have 22 IntegrationTest nodes");

    Ok(())
}

#[cfg(feature = "neo4j")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_nodes_e2e_test_type() -> Result<()> {
    let mut graph_ops = setup_ruby_graph().await?;

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

    assert_eq!(count, 10, "Should have 10 E2eTest nodes");

    Ok(())
}

#[cfg(feature = "neo4j")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_nodes_multi_type() -> Result<()> {
    let mut graph_ops = setup_ruby_graph().await?;

    let (count, results) = graph_ops
        .query_nodes_with_count(
            &[NodeType::Function, NodeType::Endpoint],
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

    assert!(count > 23, "Should have more than just endpoints");

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
    let mut graph_ops = setup_ruby_graph().await?;

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

    // 21 + 22 + 10 = 53
    assert_eq!(count, 53, "Should have all test types combined");

    Ok(())
}

#[cfg(feature = "neo4j")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_nodes_pagination_default() -> Result<()> {
    let mut graph_ops = setup_ruby_graph().await?;

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

    assert_eq!(count, 7, "Should have exactly 7 function items");
    assert_eq!(results.len(), 7, "Should return all function items");

    Ok(())
}

#[cfg(feature = "neo4j")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_nodes_pagination_second_page() -> Result<()> {
    let mut graph_ops = setup_ruby_graph().await?;

    // Get first page
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

    // Get second page
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

    assert_eq!(
        results2.len(),
        0,
        "Second page should be empty with only 7 functions"
    );

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
    let mut graph_ops = setup_ruby_graph().await?;

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

    assert!(count > 0, "Total count should still be accurate");
    assert!(
        results.is_empty(),
        "Should return empty for offset beyond data"
    );

    Ok(())
}

#[cfg(feature = "neo4j")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_nodes_pagination_max_limit() -> Result<()> {
    let mut graph_ops = setup_ruby_graph().await?;

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

    assert!(results.len() <= 100, "Should respect max limit");
    assert_eq!(
        results.len(),
        count.min(100),
        "Should return min(count, limit)"
    );

    Ok(())
}

#[cfg(feature = "neo4j")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_nodes_coverage_tested() -> Result<()> {
    let mut graph_ops = setup_ruby_graph().await?;

    let (_count, results) = graph_ops
        .query_nodes_with_count(
            &[NodeType::Class],
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

    // All results should be covered
    for (_, _, _, covered, test_count, _, _, _, _) in &results {
        assert!(*covered, "Tested filter should only return covered nodes");
        assert!(*test_count > 0, "Tested nodes should have test_count > 0");
    }

    Ok(())
}

#[cfg(feature = "neo4j")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_nodes_coverage_untested() -> Result<()> {
    let mut graph_ops = setup_ruby_graph().await?;

    let (_count, results) = graph_ops
        .query_nodes_with_count(
            &[NodeType::Class],
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

    // All results should be uncovered
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
    let mut graph_ops = setup_ruby_graph().await?;

    let (all_count, _) = graph_ops
        .query_nodes_with_count(
            &[NodeType::Class],
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
            &[NodeType::Class],
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
            &[NodeType::Class],
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
    let mut graph_ops = setup_ruby_graph().await?;

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
        if let Some(len) = body_length {
            assert!(*len >= 0, "body_length should be non-negative");
        }
    }

    Ok(())
}

#[cfg(feature = "neo4j")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_nodes_line_count() -> Result<()> {
    let mut graph_ops = setup_ruby_graph().await?;

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
        if let Some(count) = line_count {
            assert!(*count >= 0, "line_count should be non-negative");
        }
    }

    Ok(())
}

#[cfg(feature = "neo4j")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_nodes_both_metrics() -> Result<()> {
    let mut graph_ops = setup_ruby_graph().await?;

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
    let mut graph_ops = setup_ruby_graph().await?;

    let (count, results) = graph_ops
        .query_nodes_with_count(
            &[NodeType::Function],
            0,
            100,
            true,
            None,
            false,
            false,
            Some("src/testing/ruby"),
            None,
            None,
            None,
        )
        .await?;

    assert!(count > 0, "Should find functions in Ruby repo");

    for (_, node_data, _, _, _, _, _, _, _) in &results {
        assert!(
            node_data.file.starts_with("src/testing/ruby"),
            "All results should be from Ruby repo"
        );
    }

    Ok(())
}

#[cfg(feature = "neo4j")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_nodes_ignore_dirs() -> Result<()> {
    let mut graph_ops = setup_ruby_graph().await?;

    // Get baseline
    let (_baseline_count, _) = graph_ops
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

    // Get with ignore
    let filters = TestFilters {
        unit_regexes: vec![],
        integration_regexes: vec![],
        e2e_regexes: vec![],
        target_regex: None,
        ignore_dirs: vec!["spec/unit".to_string()],
    };

    let (_filtered_count, results) = graph_ops
        .query_nodes_with_count(
            &[NodeType::UnitTest],
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

    // Verify no results from ignored dir
    for (_, node_data, _, _, _, _, _, _, _) in &results {
        assert!(
            !node_data.file.contains("spec/unit"),
            "Should not include files from ignored dir"
        );
    }

    Ok(())
}

#[cfg(feature = "neo4j")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_nodes_regex_filter() -> Result<()> {
    let mut graph_ops = setup_ruby_graph().await?;

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

    // All results should match the regex
    for (_, node_data, _, _, _, _, _, _, _) in &results {
        assert!(
            node_data.file.to_lowercase().contains("person"),
            "All results should match regex: {}",
            node_data.file
        );
    }

    Ok(())
}

#[cfg(feature = "neo4j")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_nodes_search() -> Result<()> {
    let mut graph_ops = setup_ruby_graph().await?;

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
            Some("Person"),
            None,
        )
        .await?;

    assert_eq!(count, 6, "Should find exactly 6 classes matching search");

    for (_, node_data, _, _, _, _, _, _, _) in &results {
        assert!(
            node_data.name.contains("Person") || node_data.file.contains("person"),
            "Results should match search term"
        );
    }

    Ok(())
}

#[cfg(feature = "neo4j")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_nodes_search_no_match() -> Result<()> {
    let mut graph_ops = setup_ruby_graph().await?;

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
    let mut graph_ops = setup_ruby_graph().await?;

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

    assert!(
        unmuted_count <= all_count,
        "Unmuted count should be <= all count"
    );

    Ok(())
}

#[cfg(feature = "neo4j")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_nodes_sort_by_test_count() -> Result<()> {
    let mut graph_ops = setup_ruby_graph().await?;

    let (_, results) = graph_ops
        .query_nodes_with_count(
            &[NodeType::Class],
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

    for i in 1..test_counts.len() {
        assert!(
            test_counts[i - 1] >= test_counts[i],
            "Should be sorted by test_count descending: {} >= {}",
            test_counts[i - 1],
            test_counts[i]
        );
    }

    Ok(())
}

#[cfg(feature = "neo4j")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_nodes_complex_combination() -> Result<()> {
    let mut graph_ops = setup_ruby_graph().await?;

    let filters = TestFilters {
        unit_regexes: vec![],
        integration_regexes: vec![],
        e2e_regexes: vec![],
        target_regex: Some(".*service.*".to_string()),
        ignore_dirs: vec!["spec/support".to_string()],
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
            Some("src/testing/ruby"),
            Some(filters),
            None,
            Some(false),
        )
        .await?;

    for (_, node_data, _, _, _, _, body_length, line_count, _) in &results {
        assert!(node_data.file.starts_with("src/testing/ruby"));
        assert!(!node_data.file.contains("spec/support"));
        assert!(body_length.is_some());
        assert!(line_count.is_some());
    }

    Ok(())
}

#[cfg(feature = "neo4j")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_has_function_covered() -> Result<()> {
    let mut graph_ops = setup_ruby_graph().await?;

    let covered = graph_ops
        .has_coverage(
            NodeType::Function,
            "get_person_by_id",
            "src/testing/ruby/app/services/person_service.rb",
            None,
            None,
            None,
        )
        .await?;

    // Currently returns false as it's not covered by direct tests
    assert!(!covered, "get_person_by_id should be found but not covered");

    Ok(())
}

#[cfg(feature = "neo4j")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_has_function_with_start() -> Result<()> {
    let mut graph_ops = setup_ruby_graph().await?;

    let covered = graph_ops
        .has_coverage(
            NodeType::Function,
            "get_person_by_id",
            "src/testing/ruby/app/services/person_service.rb",
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
    let mut graph_ops = setup_ruby_graph().await?;

    // Test with correct root
    let _covered = graph_ops
        .has_coverage(
            NodeType::Function,
            "get_person_by_id",
            "src/testing/ruby/app/services/person_service.rb",
            None,
            Some("src/testing/ruby"),
            None,
        )
        .await?;

    // Test with wrong root
    let wrong_root = graph_ops
        .has_coverage(
            NodeType::Function,
            "get_person_by_id",
            "src/testing/ruby/app/services/person_service.rb",
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
    let mut graph_ops = setup_ruby_graph().await?;

    let unit_covered = graph_ops
        .has_coverage(
            NodeType::Class,
            "PersonService",
            "src/testing/ruby/app/services/person_service.rb",
            None,
            None,
            Some("unit"),
        )
        .await?;

    let integration_covered = graph_ops
        .has_coverage(
            NodeType::Class,
            "PersonService",
            "src/testing/ruby/app/services/person_service.rb",
            None,
            None,
            Some("integration"),
        )
        .await?;

    let _any_covered = unit_covered || integration_covered;

    Ok(())
}

#[cfg(feature = "neo4j")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_has_endpoint_covered() -> Result<()> {
    let mut graph_ops = setup_ruby_graph().await?;

    let _covered = graph_ops
        .has_coverage(
            NodeType::Endpoint,
            "/person/:id",
            "src/testing/ruby/config/routes.rb",
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
    let mut graph_ops = setup_ruby_graph().await?;

    let covered = graph_ops
        .has_coverage(
            NodeType::Function,
            "nonexistent_function_xyz",
            "nonexistent_file.rb",
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
    let mut graph_ops = setup_ruby_graph().await?;

    let test_types = vec!["unit", "integration", "e2e"];

    for test_type in test_types {
        let _covered = graph_ops
            .has_coverage(
                NodeType::Function,
                "get_person_by_id",
                "src/testing/ruby/app/services/person_service.rb",
                None,
                None,
                Some(test_type),
            )
            .await?;
    }

    let _combined = graph_ops
        .has_coverage(
            NodeType::Function,
            "get_person_by_id",
            "src/testing/ruby/app/services/person_service.rb",
            None,
            None,
            Some("unit,integration"),
        )
        .await?;

    Ok(())
}
