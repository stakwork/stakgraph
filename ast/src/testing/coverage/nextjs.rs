use crate::lang::graphs::{EdgeType, Graph, NodeType, TestFilters};
use crate::lang::linker::normalize_backend_path;
use crate::lang::Lang;
use crate::repo::{Repo, Repos};
use crate::utils::get_use_lsp;
use shared::error::Result;
use std::str::FromStr;
use tokio::sync::OnceCell;

async fn setup_nextjs_graph() -> Result<crate::lang::graphs::graph_ops::GraphOps> {
    static GRAPH_INIT: OnceCell<()> = OnceCell::const_new();

    GRAPH_INIT
        .get_or_init(|| async {
            // Explicitly disable LSP for these tests
            std::env::set_var("USE_LSP", "false");

            let use_lsp = false;
            let repo = Repo::new(
                "src/testing/nextjs",
                Lang::from_str("tsx").unwrap(),
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

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_btreemap_graph_structure() -> Result<()> {
    let use_lsp = false;

    let repo = Repo::new(
        "src/testing/nextjs",
        Lang::from_str("tsx").unwrap(),
        use_lsp,
        Vec::new(),
        Vec::new(),
    )
    .unwrap();
    // ... rest of test ...

    let repos = Repos(vec![repo], Vec::new(), None);
    let btree_graph = repos.build_graphs_btree().await?;

    btree_graph.analysis();

    let endpoints = btree_graph.find_nodes_by_type(NodeType::Endpoint);
    assert_eq!(endpoints.len(), 21);

    let integration_tests = btree_graph.find_nodes_by_type(NodeType::IntegrationTest);
    assert_eq!(integration_tests.len(), 19);

    let test_edges = btree_graph.find_nodes_with_edge_type(
        NodeType::IntegrationTest,
        NodeType::Endpoint,
        EdgeType::Calls,
    );

    let unique_tested_endpoints: std::collections::HashSet<String> = test_edges
        .iter()
        .map(|(_, target)| format!("{}|{}|{}", target.name, target.file, target.start))
        .collect();

    assert_eq!(unique_tested_endpoints.len(), 11);

    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_btreemap_indirect_test_metadata() -> Result<()> {
    let use_lsp = get_use_lsp();

    let repo = Repo::new(
        "src/testing/nextjs",
        Lang::from_str("tsx").unwrap(),
        use_lsp,
        Vec::new(),
        Vec::new(),
    )
    .unwrap();

    let repos = Repos(vec![repo], Vec::new(), None);
    let btree_graph = repos.build_graphs_btree().await?;
    btree_graph.analysis();

    let endpoints = btree_graph.find_nodes_by_type(NodeType::Endpoint);

    let indirect_tested: Vec<_> = endpoints
        .iter()
        .filter(|e| e.meta.contains_key("indirect_test"))
        .collect();

    assert_eq!(indirect_tested.len(), 4);

    let expected_indirect = vec![
        ("/api/items", "POST", "createItem"),
        ("/api/items", "GET", "fetchItems"),
        ("/api/products", "POST", "createProduct"),
        ("/api/products", "GET", "listProducts"),
    ];

    for (path, verb, helper_name) in &expected_indirect {
        let matching: Vec<_> = indirect_tested
            .iter()
            .filter(|e| {
                e.name.contains(path)
                    && e.meta.get("verb") == Some(&verb.to_string())
                    && !e.name.contains("[")
            })
            .collect();

        assert_eq!(matching.len(), 1);

        let endpoint = matching[0];
        assert!(endpoint.meta.contains_key("test_helper"));

        let helper = endpoint.meta.get("test_helper").unwrap();
        assert!(helper.contains(helper_name));
    }

    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_btreemap_tested_endpoints() -> Result<()> {
    let use_lsp = get_use_lsp();

    let repo = Repo::new(
        "src/testing/nextjs",
        Lang::from_str("tsx").unwrap(),
        use_lsp,
        Vec::new(),
        Vec::new(),
    )
    .unwrap();

    let repos = Repos(vec![repo], Vec::new(), None);
    let btree_graph = repos.build_graphs_btree().await?;
    btree_graph.analysis();

    let endpoints = btree_graph.find_nodes_by_type(NodeType::Endpoint);
    let test_edges = btree_graph.find_nodes_with_edge_type(
        NodeType::IntegrationTest,
        NodeType::Endpoint,
        EdgeType::Calls,
    );

    let tested_endpoints = vec![
        ("/api/items", "GET"),
        ("/api/items", "POST"),
        ("/api/person", "GET"),
        ("/api/person", "POST"),
        ("/api/person/:param", "GET"),
        ("/api/person/:param", "DELETE"),
        ("/api/orders", "PUT"),
    ];

    for (expected_path, expected_verb) in &tested_endpoints {
        let matching_endpoints: Vec<_> = endpoints
            .iter()
            .filter(|e| {
                let normalized = normalize_backend_path(&e.name).unwrap_or(e.name.clone());
                normalized == *expected_path
                    && e.meta.get("verb") == Some(&expected_verb.to_string())
            })
            .collect();

        assert_eq!(matching_endpoints.len(), 1);

        let endpoint = matching_endpoints[0];

        let has_test_edge = test_edges.iter().any(|(_, target)| {
            target.name == endpoint.name
                && target.file == endpoint.file
                && target.start == endpoint.start
        });

        assert!(has_test_edge);
    }

    Ok(())
}

#[cfg(feature = "neo4j")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_nextjs_graph_upload() -> Result<()> {
    let graph_ops = setup_nextjs_graph().await?;
    let (nodes, edges) = graph_ops.get_graph_size().await?;

    assert_eq!(nodes, 564);
    assert_eq!(edges, 962);

    Ok(())
}

#[cfg(feature = "neo4j")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_coverage_default_params() -> Result<()> {
    let mut graph_ops = setup_nextjs_graph().await?;

    let coverage = graph_ops.get_coverage(None, None, None).await?;

    assert_eq!(coverage.language, Some("typescript".to_string()));

    if let Some(integration) = &coverage.integration_tests {
        assert_eq!(integration.total, 21);
        assert_eq!(integration.total_tests, 19);
    }

    Ok(())
}

#[cfg(feature = "neo4j")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_coverage_with_repo_filter() -> Result<()> {
    let mut graph_ops = setup_nextjs_graph().await?;
    let coverage = graph_ops
        .get_coverage(Some("src/testing/nextjs"), None, None)
        .await?;

    assert_eq!(coverage.language, Some("typescript".to_string()));

    let empty_coverage = graph_ops
        .get_coverage(Some("nonexistent/repo"), None, None)
        .await?;
    if let Some(integration) = &empty_coverage.integration_tests {
        assert_eq!(integration.covered, 0);
    }

    Ok(())
}

#[cfg(feature = "neo4j")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_coverage_with_ignore_dirs() -> Result<()> {
    let mut graph_ops = setup_nextjs_graph().await?;

    let baseline = graph_ops.get_coverage(None, None, None).await?;

    let filters = TestFilters {
        unit_regexes: vec![],
        integration_regexes: vec![],
        e2e_regexes: vec![],
        target_regex: None,
        ignore_dirs: vec!["lib".to_string()],
    };
    let filtered = graph_ops.get_coverage(None, Some(filters), None).await?;

    if let (Some(_base_int), Some(filt_int)) =
        (&baseline.integration_tests, &filtered.integration_tests)
    {
        assert_eq!(filt_int.total, 21);
    }

    Ok(())
}

#[cfg(feature = "neo4j")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_coverage_with_regex_filter() -> Result<()> {
    let mut graph_ops = setup_nextjs_graph().await?;
    let filters = TestFilters {
        unit_regexes: vec![],
        integration_regexes: vec![],
        e2e_regexes: vec![],
        target_regex: Some(".*api.*".to_string()),
        ignore_dirs: vec![],
    };
    let coverage = graph_ops.get_coverage(None, Some(filters), None).await?;

    assert_eq!(coverage.language, Some("typescript".to_string()));

    Ok(())
}

#[cfg(feature = "neo4j")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_coverage_with_is_muted() -> Result<()> {
    let mut graph_ops = setup_nextjs_graph().await?;

    let muted_coverage = graph_ops.get_coverage(None, None, Some(true)).await?;
    let unmuted_coverage = graph_ops.get_coverage(None, None, Some(false)).await?;

    assert_eq!(muted_coverage.language, Some("typescript".to_string()));
    assert_eq!(unmuted_coverage.language, Some("typescript".to_string()));

    Ok(())
}

#[cfg(feature = "neo4j")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_coverage_combined_filters() -> Result<()> {
    let mut graph_ops = setup_nextjs_graph().await?;

    let filters = TestFilters {
        unit_regexes: vec![],
        integration_regexes: vec![],
        e2e_regexes: vec![],
        target_regex: Some(".*person.*".to_string()),
        ignore_dirs: vec!["components/".to_string()],
    };

    let coverage = graph_ops
        .get_coverage(Some("src/testing/nextjs"), Some(filters), Some(false))
        .await?;

    assert_eq!(coverage.language, Some("typescript".to_string()));

    Ok(())
}

#[cfg(feature = "neo4j")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_nodes_endpoint_type() -> Result<()> {
    let mut graph_ops = setup_nextjs_graph().await?;

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

    assert_eq!(count, 21);
    assert_eq!(results.len(), 21);

    for (node_type, node_data, _, _, _, _, _, _, _) in &results {
        assert_eq!(*node_type, NodeType::Endpoint);
        assert!(node_data.meta.contains_key("verb"));
    }

    Ok(())
}

#[cfg(feature = "neo4j")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_nodes_function_type() -> Result<()> {
    let mut graph_ops = setup_nextjs_graph().await?;

    let (count, results) = graph_ops
        .query_nodes_with_count(
            &[NodeType::Function],
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

    assert_eq!(count, 49);
    assert_eq!(results.len(), 49);

    for (node_type, _, _, _, _, _, _, _, _) in &results {
        assert_eq!(*node_type, NodeType::Function);
    }

    Ok(())
}

#[cfg(feature = "neo4j")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_nodes_page_type() -> Result<()> {
    let mut graph_ops = setup_nextjs_graph().await?;

    let (count, results) = graph_ops
        .query_nodes_with_count(
            &[NodeType::Page],
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

    assert_eq!(count, 10);
    assert_eq!(results.len(), 10);

    Ok(())
}

#[cfg(feature = "neo4j")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_nodes_integration_test_type() -> Result<()> {
    let mut graph_ops = setup_nextjs_graph().await?;

    let (count, results) = graph_ops
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

    assert_eq!(count, 19);
    assert_eq!(results.len(), 19);

    Ok(())
}

#[cfg(feature = "neo4j")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_nodes_unit_test_type() -> Result<()> {
    let mut graph_ops = setup_nextjs_graph().await?;

    let (count, results) = graph_ops
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

    assert_eq!(count, 27);
    assert_eq!(results.len(), 27);

    Ok(())
}

#[cfg(feature = "neo4j")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_nodes_e2e_test_type() -> Result<()> {
    let mut graph_ops = setup_nextjs_graph().await?;

    let (count, results) = graph_ops
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

    assert_eq!(count, 5);
    assert_eq!(results.len(), 5);

    Ok(())
}

#[cfg(feature = "neo4j")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_nodes_multi_type() -> Result<()> {
    let mut graph_ops = setup_nextjs_graph().await?;

    let (count, results) = graph_ops
        .query_nodes_with_count(
            &[NodeType::Function, NodeType::Endpoint],
            0,
            250,
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

    assert_eq!(count, 70);

    let has_function = results
        .iter()
        .any(|(nt, _, _, _, _, _, _, _, _)| *nt == NodeType::Function);
    let has_endpoint = results
        .iter()
        .any(|(nt, _, _, _, _, _, _, _, _)| *nt == NodeType::Endpoint);

    assert!(has_function);
    assert!(has_endpoint);

    Ok(())
}

#[cfg(feature = "neo4j")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_nodes_all_test_types() -> Result<()> {
    let mut graph_ops = setup_nextjs_graph().await?;

    let (count, results) = graph_ops
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

    assert_eq!(count, 51);
    assert_eq!(results.len(), 51);

    Ok(())
}

#[cfg(feature = "neo4j")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_nodes_pagination_default() -> Result<()> {
    let mut graph_ops = setup_nextjs_graph().await?;

    let (count, results) = graph_ops
        .query_nodes_with_count(
            &[NodeType::Endpoint],
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

    assert_eq!(count, 21);
    assert_eq!(results.len(), 10);

    Ok(())
}

#[cfg(feature = "neo4j")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_nodes_pagination_second_page() -> Result<()> {
    let mut graph_ops = setup_nextjs_graph().await?;

    let (_count1, results1) = graph_ops
        .query_nodes_with_count(
            &[NodeType::Endpoint],
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
            &[NodeType::Endpoint],
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

    assert_eq!(results2.len(), 10);

    let page1_identifiers: Vec<_> = results1
        .iter()
        .map(|(_, nd, _, _, _, _, _, _, _)| (&nd.name, &nd.file))
        .collect();
    let page2_identifiers: Vec<_> = results2
        .iter()
        .map(|(_, nd, _, _, _, _, _, _, _)| (&nd.name, &nd.file))
        .collect();

    for (name, file) in &page2_identifiers {
        assert!(!page1_identifiers.contains(&(name, file)));
    }

    Ok(())
}

#[cfg(feature = "neo4j")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_nodes_pagination_third_page() -> Result<()> {
    let mut graph_ops = setup_nextjs_graph().await?;

    let (count, results) = graph_ops
        .query_nodes_with_count(
            &[NodeType::Endpoint],
            20,
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

    assert_eq!(count, 21);
    assert_eq!(results.len(), 1);

    Ok(())
}

#[cfg(feature = "neo4j")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_nodes_pagination_large_offset() -> Result<()> {
    let mut graph_ops = setup_nextjs_graph().await?;

    let (count, results) = graph_ops
        .query_nodes_with_count(
            &[NodeType::Endpoint],
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

    assert_eq!(count, 21);
    assert_eq!(results.len(), 0);

    Ok(())
}

#[cfg(feature = "neo4j")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_nodes_coverage_tested() -> Result<()> {
    let mut graph_ops = setup_nextjs_graph().await?;

    let (count, results) = graph_ops
        .query_nodes_with_count(
            &[NodeType::Endpoint],
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

    assert_eq!(count, 13);
    assert_eq!(results.len(), 13);

    for (_, _, _, covered, _test_count, _, _, _, _) in &results {
        assert!(*covered);
    }

    Ok(())
}

#[cfg(feature = "neo4j")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_nodes_coverage_untested() -> Result<()> {
    let mut graph_ops = setup_nextjs_graph().await?;

    let (count, results) = graph_ops
        .query_nodes_with_count(
            &[NodeType::Endpoint],
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

    assert_eq!(count, 8);
    assert_eq!(results.len(), 8);

    for (_, _, _, covered, test_count, _, _, _, _) in &results {
        assert!(!*covered);
        assert_eq!(*test_count, 0);
    }

    Ok(())
}

#[cfg(feature = "neo4j")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_nodes_coverage_all() -> Result<()> {
    let mut graph_ops = setup_nextjs_graph().await?;

    let (all_count, _) = graph_ops
        .query_nodes_with_count(
            &[NodeType::Endpoint],
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
            &[NodeType::Endpoint],
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
            &[NodeType::Endpoint],
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

    assert_eq!(all_count, 21);
    assert_eq!(tested_count, 13);
    assert_eq!(untested_count, 8);
    assert_eq!(all_count, tested_count + untested_count);

    Ok(())
}

#[cfg(feature = "neo4j")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_nodes_body_length() -> Result<()> {
    let mut graph_ops = setup_nextjs_graph().await?;

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
        assert!(body_length.is_some());
    }

    Ok(())
}

#[cfg(feature = "neo4j")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_nodes_line_count() -> Result<()> {
    let mut graph_ops = setup_nextjs_graph().await?;

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
        assert!(line_count.is_some());
    }

    Ok(())
}

#[cfg(feature = "neo4j")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_nodes_both_metrics() -> Result<()> {
    let mut graph_ops = setup_nextjs_graph().await?;

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
        assert!(body_length.is_some());
        assert!(line_count.is_some());
    }

    Ok(())
}

#[cfg(feature = "neo4j")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_nodes_repo_filter() -> Result<()> {
    let mut graph_ops = setup_nextjs_graph().await?;

    let (count, results) = graph_ops
        .query_nodes_with_count(
            &[NodeType::Endpoint],
            0,
            100,
            true,
            None,
            false,
            false,
            Some("src/testing/nextjs"),
            None,
            None,
            None,
        )
        .await?;

    assert_eq!(count, 21);

    for (_, node_data, _, _, _, _, _, _, _) in &results {
        assert!(node_data.file.starts_with("src/testing/nextjs"));
    }

    Ok(())
}

#[cfg(feature = "neo4j")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_nodes_ignore_dirs() -> Result<()> {
    let mut graph_ops = setup_nextjs_graph().await?;

    let filters = TestFilters {
        unit_regexes: vec![],
        integration_regexes: vec![],
        e2e_regexes: vec![],
        target_regex: None,
        ignore_dirs: vec!["lib".to_string()],
    };

    let (_filtered_count, results) = graph_ops
        .query_nodes_with_count(
            &[NodeType::Function],
            0,
            200,
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
        assert!(!node_data.file.contains("lib"));
    }

    Ok(())
}

#[cfg(feature = "neo4j")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_nodes_regex_filter() -> Result<()> {
    let mut graph_ops = setup_nextjs_graph().await?;

    let filters = TestFilters {
        unit_regexes: vec![],
        integration_regexes: vec![],
        e2e_regexes: vec![],
        target_regex: Some(".*person.*".to_string()),
        ignore_dirs: vec![],
    };

    let (_count, results) = graph_ops
        .query_nodes_with_count(
            &[NodeType::Endpoint],
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
                || node_data.name.to_lowercase().contains("person")
        );
    }

    Ok(())
}

#[cfg(feature = "neo4j")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_nodes_search() -> Result<()> {
    let mut graph_ops = setup_nextjs_graph().await?;

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
            Some("person"),
            None,
        )
        .await?;

    assert_eq!(count, 4);

    for (_, node_data, _, _, _, _, _, _, _) in &results {
        assert!(
            node_data.name.to_lowercase().contains("person")
                || node_data.file.to_lowercase().contains("person")
        );
    }

    Ok(())
}

#[cfg(feature = "neo4j")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_nodes_search_no_match() -> Result<()> {
    let mut graph_ops = setup_nextjs_graph().await?;

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
            Some("xyznonexistent123"),
            None,
        )
        .await?;

    assert_eq!(count, 0);
    assert_eq!(results.len(), 0);

    Ok(())
}

#[cfg(feature = "neo4j")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_nodes_is_muted() -> Result<()> {
    let mut graph_ops = setup_nextjs_graph().await?;

    let (_muted_count, _) = graph_ops
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
            Some(true),
        )
        .await?;

    let (unmuted_count, _) = graph_ops
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
            Some(false),
        )
        .await?;

    let (all_count, _) = graph_ops
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

    assert_eq!(all_count, 21);
    assert!(unmuted_count <= all_count);

    Ok(())
}

#[cfg(feature = "neo4j")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_nodes_sort_by_test_count() -> Result<()> {
    let mut graph_ops = setup_nextjs_graph().await?;

    let (_, results) = graph_ops
        .query_nodes_with_count(
            &[NodeType::Endpoint],
            0,
            21,
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
        assert!(test_counts[i - 1] >= test_counts[i]);
    }

    Ok(())
}

#[cfg(feature = "neo4j")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_nodes_complex_combination() -> Result<()> {
    let mut graph_ops = setup_nextjs_graph().await?;

    let filters = TestFilters {
        unit_regexes: vec![],
        integration_regexes: vec![],
        e2e_regexes: vec![],
        target_regex: Some(".*api.*".to_string()),
        ignore_dirs: vec!["components/".to_string()],
    };

    let (_count, results) = graph_ops
        .query_nodes_with_count(
            &[NodeType::Function, NodeType::Endpoint],
            5,
            10,
            true,
            Some("all"),
            true,
            true,
            Some("src/testing/nextjs"),
            Some(filters),
            None,
            Some(false),
        )
        .await?;

    for (_, node_data, _, _, _, _, body_length, line_count, _) in &results {
        assert!(node_data.file.starts_with("src/testing/nextjs"));
        assert!(!node_data.file.contains("components/"));
        assert!(body_length.is_some());
        assert!(line_count.is_some());
    }

    Ok(())
}

#[cfg(feature = "neo4j")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_has_function_covered() -> Result<()> {
    let mut graph_ops = setup_nextjs_graph().await?;

    let _covered = graph_ops
        .has_coverage(
            NodeType::Function,
            "fetchItems",
            "src/testing/nextjs/lib/api-helpers.ts",
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
    let mut graph_ops = setup_nextjs_graph().await?;

    let covered = graph_ops
        .has_coverage(
            NodeType::Function,
            "fetchItems",
            "src/testing/nextjs/lib/api-helpers.ts",
            Some(999),
            None,
            None,
        )
        .await?;

    assert!(!covered);

    Ok(())
}

#[cfg(feature = "neo4j")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_has_function_with_root() -> Result<()> {
    let mut graph_ops = setup_nextjs_graph().await?;

    let _covered = graph_ops
        .has_coverage(
            NodeType::Function,
            "fetchItems",
            "src/testing/nextjs/lib/api-helpers.ts",
            None,
            Some("src/testing/nextjs"),
            None,
        )
        .await?;

    let wrong_root = graph_ops
        .has_coverage(
            NodeType::Function,
            "fetchItems",
            "src/testing/nextjs/lib/api-helpers.ts",
            None,
            Some("wrong/root"),
            None,
        )
        .await?;

    assert!(!wrong_root);

    Ok(())
}

#[cfg(feature = "neo4j")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_has_function_with_tests_filter() -> Result<()> {
    let mut graph_ops = setup_nextjs_graph().await?;

    let _unit_covered = graph_ops
        .has_coverage(
            NodeType::Function,
            "fetchItems",
            "src/testing/nextjs/lib/api-helpers.ts",
            None,
            None,
            Some("unit"),
        )
        .await?;

    let _integration_covered = graph_ops
        .has_coverage(
            NodeType::Function,
            "fetchItems",
            "src/testing/nextjs/lib/api-helpers.ts",
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
    let mut graph_ops = setup_nextjs_graph().await?;

    let _covered = graph_ops
        .has_coverage(
            NodeType::Endpoint,
            "/api/person",
            "src/testing/nextjs/app/api/person/route.ts",
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
    let mut graph_ops = setup_nextjs_graph().await?;

    let covered = graph_ops
        .has_coverage(
            NodeType::Function,
            "nonexistent_function_xyz",
            "nonexistent_file.ts",
            None,
            None,
            None,
        )
        .await?;

    assert!(!covered);

    Ok(())
}

#[cfg(feature = "neo4j")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_has_all_test_types() -> Result<()> {
    let mut graph_ops = setup_nextjs_graph().await?;

    let test_types = vec!["unit", "integration"];

    for test_type in test_types {
        let _covered = graph_ops
            .has_coverage(
                NodeType::Function,
                "fetchItems",
                "src/testing/nextjs/lib/api-helpers.ts",
                None,
                None,
                Some(test_type),
            )
            .await?;
    }

    let _combined = graph_ops
        .has_coverage(
            NodeType::Function,
            "fetchItems",
            "src/testing/nextjs/lib/api-helpers.ts",
            None,
            None,
            Some("unit,integration"),
        )
        .await?;

    Ok(())
}
