use crate::lang::graphs::{BTreeMapGraph, EdgeType, Graph, NodeType, TestFilters};
use crate::lang::Lang;
use crate::repo::{Repo, Repos};
use shared::error::Result;
use std::str::FromStr;
use tokio::sync::OnceCell;

async fn setup_typescript_graph() -> Result<crate::lang::graphs::graph_ops::GraphOps> {
    static GRAPH_INIT: OnceCell<()> = OnceCell::const_new();

    GRAPH_INIT
        .get_or_init(|| async {
            std::env::set_var("USE_LSP", "false");

            let use_lsp = false;
            let repo = Repo::new(
                "src/testing/typescript",
                Lang::from_str("ts").unwrap(),
                use_lsp,
                Vec::new(),
                Vec::new(),
            )
            .unwrap();

            let repos = Repos { repos: vec![repo], packages: Vec::new(), workspace_root: None };
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
        "src/testing/typescript",
        Lang::from_str("ts").unwrap(),
        use_lsp,
        Vec::new(),
        Vec::new(),
    )
    .unwrap();

    let graph = repo.build_graph_inner::<BTreeMapGraph>().await?;

    let endpoints = graph.find_nodes_by_type(NodeType::Endpoint);
    assert_eq!(endpoints.len(), 22);

    let functions = graph.find_nodes_by_type(NodeType::Function);
    assert_eq!(functions.len(), 60);

    let unit_tests = graph.find_nodes_by_type(NodeType::UnitTest);
    assert_eq!(unit_tests.len(), 8);

    let integration_tests = graph.find_nodes_by_type(NodeType::IntegrationTest);
    assert_eq!(integration_tests.len(), 3);

    let e2e_tests = graph.find_nodes_by_type(NodeType::E2eTest);
    assert_eq!(e2e_tests.len(), 3);

    let classes = graph.find_nodes_by_type(NodeType::Class);
    assert_eq!(classes.len(), 9);

    let data_models = graph.find_nodes_by_type(NodeType::DataModel);
    assert_eq!(data_models.len(), 27);

    let traits = graph.find_nodes_by_type(NodeType::Trait);
    assert_eq!(traits.len(), 6);

    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_btreemap_test_to_function_edges() -> Result<()> {
    let use_lsp = false;

    let repo = Repo::new(
        "src/testing/typescript",
        Lang::from_str("ts").unwrap(),
        use_lsp,
        Vec::new(),
        Vec::new(),
    )
    .unwrap();

    let graph = repo.build_graph_inner::<BTreeMapGraph>().await?;

    let calls_edges = graph.count_edges_of_type(EdgeType::Calls);
    assert_eq!(calls_edges, 14);

    let contains_edges = graph.count_edges_of_type(EdgeType::Contains);
    assert_eq!(contains_edges, 215);

    let handler_edges = graph.count_edges_of_type(EdgeType::Handler);
    assert_eq!(handler_edges, 22);

    let implements_edges = graph.count_edges_of_type(EdgeType::Implements);
    assert_eq!(implements_edges, 3);

    Ok(())
}

#[cfg(feature = "neo4j")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_typescript_graph_upload() -> Result<()> {
    let graph_ops = setup_typescript_graph().await?;
    let (nodes, edges) = graph_ops.get_graph_size().await?;

    assert_eq!(nodes, 215);
    assert_eq!(edges, 297);

    Ok(())
}

#[cfg(feature = "neo4j")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_coverage_default_params() -> Result<()> {
    let mut graph_ops = setup_typescript_graph().await?;

    let coverage = graph_ops.get_coverage(None, None, None).await?;

    assert_eq!(coverage.language, Some("typescript".to_string()));

    if let Some(unit) = &coverage.unit_tests {
        assert_eq!(unit.total_tests, 8);
    }

    if let Some(integration) = &coverage.integration_tests {
        assert_eq!(integration.total, 22);
        assert_eq!(integration.total_tests, 3);
    }

    if let Some(e2e) = &coverage.e2e_tests {
        assert_eq!(e2e.total_tests, 4);
    }

    Ok(())
}

#[cfg(feature = "neo4j")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_coverage_with_repo_filter() -> Result<()> {
    let mut graph_ops = setup_typescript_graph().await?;
    let coverage = graph_ops
        .get_coverage(Some("src/testing/typescript"), None, None)
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
    let mut graph_ops = setup_typescript_graph().await?;

    let filters = TestFilters {
        unit_regexes: vec![],
        integration_regexes: vec![],
        e2e_regexes: vec![],
        target_regex: None,
        ignore_dirs: vec!["routers".to_string()],
    };
    let filtered = graph_ops.get_coverage(None, Some(filters), None).await?;

    assert_eq!(filtered.language, Some("typescript".to_string()));

    Ok(())
}

#[cfg(feature = "neo4j")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_coverage_with_regex_filter() -> Result<()> {
    let mut graph_ops = setup_typescript_graph().await?;
    let filters = TestFilters {
        unit_regexes: vec![],
        integration_regexes: vec![],
        e2e_regexes: vec![],
        target_regex: Some(".*person.*".to_string()),
        ignore_dirs: vec![],
    };
    let coverage = graph_ops.get_coverage(None, Some(filters), None).await?;

    assert_eq!(coverage.language, Some("typescript".to_string()));

    Ok(())
}

#[cfg(feature = "neo4j")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_coverage_with_is_muted() -> Result<()> {
    let mut graph_ops = setup_typescript_graph().await?;

    let muted_coverage = graph_ops.get_coverage(None, None, Some(true)).await?;
    let unmuted_coverage = graph_ops.get_coverage(None, None, Some(false)).await?;

    assert_eq!(muted_coverage.language, Some("typescript".to_string()));
    assert_eq!(unmuted_coverage.language, Some("typescript".to_string()));

    Ok(())
}

#[cfg(feature = "neo4j")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_coverage_combined_filters() -> Result<()> {
    let mut graph_ops = setup_typescript_graph().await?;

    let filters = TestFilters {
        unit_regexes: vec![],
        integration_regexes: vec![],
        e2e_regexes: vec![],
        target_regex: Some(".*api.*".to_string()),
        ignore_dirs: vec!["test".to_string()],
    };

    let coverage = graph_ops
        .get_coverage(Some("src/testing/typescript"), Some(filters), Some(false))
        .await?;

    assert_eq!(coverage.language, Some("typescript".to_string()));

    Ok(())
}

#[cfg(feature = "neo4j")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_nodes_endpoint_type() -> Result<()> {
    let mut graph_ops = setup_typescript_graph().await?;

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

    assert_eq!(count, 22);
    assert_eq!(results.len(), 22);

    for (node_type, node_data, _, _, _, _, _, _, _) in &results {
        assert_eq!(*node_type, NodeType::Endpoint);
        assert!(node_data.meta.contains_key("verb"));
    }

    Ok(())
}

#[cfg(feature = "neo4j")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_nodes_function_type() -> Result<()> {
    let mut graph_ops = setup_typescript_graph().await?;

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

    assert_eq!(count, 16);
    assert_eq!(results.len(), 16);

    for (node_type, _, _, _, _, _, _, _, _) in &results {
        assert_eq!(*node_type, NodeType::Function);
    }

    Ok(())
}

#[cfg(feature = "neo4j")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_nodes_class_type() -> Result<()> {
    let mut graph_ops = setup_typescript_graph().await?;

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

    assert_eq!(count, 9);
    assert_eq!(results.len(), 9);

    Ok(())
}

#[cfg(feature = "neo4j")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_nodes_data_model_type() -> Result<()> {
    let mut graph_ops = setup_typescript_graph().await?;

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

    assert_eq!(count, 27);
    assert_eq!(results.len(), 27);

    Ok(())
}

#[cfg(feature = "neo4j")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_nodes_trait_type() -> Result<()> {
    let mut graph_ops = setup_typescript_graph().await?;

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

    assert_eq!(count, 6);
    assert_eq!(results.len(), 6);

    Ok(())
}

#[cfg(feature = "neo4j")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_nodes_unit_test_type() -> Result<()> {
    let mut graph_ops = setup_typescript_graph().await?;

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

    assert_eq!(count, 8);
    assert_eq!(results.len(), 8);

    Ok(())
}

#[cfg(feature = "neo4j")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_nodes_integration_test_type() -> Result<()> {
    let mut graph_ops = setup_typescript_graph().await?;

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

    assert_eq!(count, 3);
    assert_eq!(results.len(), 3);

    Ok(())
}

#[cfg(feature = "neo4j")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_nodes_e2e_test_type() -> Result<()> {
    let mut graph_ops = setup_typescript_graph().await?;

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

    assert_eq!(count, 3);
    assert_eq!(results.len(), 3);

    Ok(())
}

#[cfg(feature = "neo4j")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_nodes_multi_type() -> Result<()> {
    let mut graph_ops = setup_typescript_graph().await?;

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

    assert_eq!(count, 38);

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
    let mut graph_ops = setup_typescript_graph().await?;

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

    assert_eq!(count, 14);
    assert_eq!(results.len(), 14);

    Ok(())
}

#[cfg(feature = "neo4j")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_nodes_pagination_default() -> Result<()> {
    let mut graph_ops = setup_typescript_graph().await?;

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

    assert_eq!(count, 22);
    assert_eq!(results.len(), 10);

    Ok(())
}

#[cfg(feature = "neo4j")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_nodes_pagination_second_page() -> Result<()> {
    let mut graph_ops = setup_typescript_graph().await?;

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
    let mut graph_ops = setup_typescript_graph().await?;

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

    assert_eq!(count, 22);
    assert_eq!(results.len(), 2);

    Ok(())
}

#[cfg(feature = "neo4j")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_nodes_pagination_large_offset() -> Result<()> {
    let mut graph_ops = setup_typescript_graph().await?;

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

    assert_eq!(count, 22);
    assert_eq!(results.len(), 0);

    Ok(())
}

#[cfg(feature = "neo4j")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_nodes_coverage_tested() -> Result<()> {
    let mut graph_ops = setup_typescript_graph().await?;

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
        assert!(*covered);
        assert_ne!(*test_count, 0);
    }

    Ok(())
}

#[cfg(feature = "neo4j")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_nodes_coverage_untested() -> Result<()> {
    let mut graph_ops = setup_typescript_graph().await?;

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
        assert!(!*covered);
        assert_eq!(*test_count, 0);
    }

    Ok(())
}

#[cfg(feature = "neo4j")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_nodes_coverage_all() -> Result<()> {
    let mut graph_ops = setup_typescript_graph().await?;

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

    assert_eq!(all_count, tested_count + untested_count);

    Ok(())
}

#[cfg(feature = "neo4j")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_nodes_body_length() -> Result<()> {
    let mut graph_ops = setup_typescript_graph().await?;

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
    let mut graph_ops = setup_typescript_graph().await?;

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
    let mut graph_ops = setup_typescript_graph().await?;

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
async fn test_nodes_search() -> Result<()> {
    let mut graph_ops = setup_typescript_graph().await?;

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

    assert_ne!(count, 0);

    for (_, node_data, _, _, _, _, _, _, _) in &results {
        let name_lower = node_data.name.to_lowercase();
        let file_lower = node_data.file.to_lowercase();
        assert!(name_lower.contains("person") || file_lower.contains("person"));
    }

    Ok(())
}

#[cfg(feature = "neo4j")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_nodes_with_repo_filter() -> Result<()> {
    let mut graph_ops = setup_typescript_graph().await?;

    let (count, _) = graph_ops
        .query_nodes_with_count(
            &[NodeType::Function],
            0,
            100,
            true,
            None,
            false,
            false,
            Some("src/testing/typescript"),
            None,
            None,
            None,
        )
        .await?;

    assert_eq!(count, 16);

    let (empty_count, _) = graph_ops
        .query_nodes_with_count(
            &[NodeType::Function],
            0,
            100,
            true,
            None,
            false,
            false,
            Some("nonexistent/repo"),
            None,
            None,
            None,
        )
        .await?;

    assert_eq!(empty_count, 0);

    Ok(())
}

#[cfg(feature = "neo4j")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_nodes_with_ignore_dirs() -> Result<()> {
    let mut graph_ops = setup_typescript_graph().await?;

    let filters = TestFilters {
        unit_regexes: vec![],
        integration_regexes: vec![],
        e2e_regexes: vec![],
        target_regex: None,
        ignore_dirs: vec!["routers".to_string()],
    };

    let (count, _) = graph_ops
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

    assert_ne!(count, 33);

    Ok(())
}

#[cfg(feature = "neo4j")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_has_coverage_tested_endpoint() -> Result<()> {
    let mut graph_ops = setup_typescript_graph().await?;

    let endpoints = graph_ops
        .graph
        .find_nodes_by_type_async(NodeType::Endpoint)
        .await;

    let post_endpoint = endpoints
        .iter()
        .find(|e| e.name == "/person" && e.meta.get("verb") == Some(&"POST".to_string()));

    if let Some(endpoint) = post_endpoint {
        let covered = graph_ops
            .has_coverage(
                NodeType::Endpoint,
                &endpoint.name,
                &endpoint.file,
                Some(endpoint.start),
                None,
                None,
            )
            .await?;

        assert!(covered || !covered);
    }

    Ok(())
}

#[cfg(feature = "neo4j")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_has_coverage_with_root_filter() -> Result<()> {
    let mut graph_ops = setup_typescript_graph().await?;

    let functions = graph_ops
        .graph
        .find_nodes_by_type_async(NodeType::Function)
        .await;

    if let Some(func) = functions.first() {
        let covered = graph_ops
            .has_coverage(
                NodeType::Function,
                &func.name,
                &func.file,
                Some(func.start),
                Some("src/testing/typescript"),
                None,
            )
            .await?;

        assert!(covered || !covered);
    }

    Ok(())
}

#[cfg(feature = "neo4j")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_has_coverage_with_tests_filter() -> Result<()> {
    let mut graph_ops = setup_typescript_graph().await?;

    let functions = graph_ops
        .graph
        .find_nodes_by_type_async(NodeType::Function)
        .await;

    if let Some(func) = functions.first() {
        let unit_covered = graph_ops
            .has_coverage(
                NodeType::Function,
                &func.name,
                &func.file,
                Some(func.start),
                None,
                Some("unit"),
            )
            .await?;

        let integration_covered = graph_ops
            .has_coverage(
                NodeType::Function,
                &func.name,
                &func.file,
                Some(func.start),
                None,
                Some("integration"),
            )
            .await?;

        assert!(unit_covered || !unit_covered);
        assert!(integration_covered || !integration_covered);
    }

    Ok(())
}
