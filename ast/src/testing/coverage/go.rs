use crate::lang::graphs::{BTreeMapGraph, EdgeType, Graph, NodeType};
use crate::lang::Lang;
use crate::repo::Repo;
use shared::error::Result;
use std::str::FromStr;

#[cfg(feature = "neo4j")]
use crate::lang::graphs::TestFilters;
#[cfg(feature = "neo4j")]
use crate::repo::Repos;
#[cfg(feature = "neo4j")]
use tokio::sync::OnceCell;

#[cfg(feature = "neo4j")]
async fn setup_go_graph() -> Result<crate::lang::graphs::graph_ops::GraphOps> {
    static GRAPH_INIT: OnceCell<()> = OnceCell::const_new();

    GRAPH_INIT
        .get_or_init(|| async {
            unsafe { std::env::set_var("USE_LSP", "false") };

            let repo = Repo::new(
                "src/testing/go",
                Lang::from_str("go").unwrap(),
                false,
                Vec::new(),
                Vec::new(),
            )
            .unwrap();

            let repos = Repos(vec![repo]);
            let btree_graph = repos
                .build_graphs_local()
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
    let repo = Repo::new(
        "src/testing/go",
        Lang::from_str("go").unwrap(),
        false,
        Vec::new(),
        Vec::new(),
    )
    .unwrap();

    let graph = repo.build_graph_inner::<BTreeMapGraph>().await?;

    let endpoints = graph.find_nodes_by_type(NodeType::Endpoint);
    assert_eq!(endpoints.len(), 5);

    let functions = graph.find_nodes_by_type(NodeType::Function);
    assert_eq!(functions.len(), 40);

    let unit_tests = graph.find_nodes_by_type(NodeType::UnitTest);
    assert_eq!(unit_tests.len(), 2);

    let integration_tests = graph.find_nodes_by_type(NodeType::IntegrationTest);
    assert_eq!(integration_tests.len(), 1);

    let e2e_tests = graph.find_nodes_by_type(NodeType::E2eTest);
    assert_eq!(e2e_tests.len(), 2);

    let classes = graph.find_nodes_by_type(NodeType::Class);
    assert_eq!(classes.len(), 11);

    let data_models = graph.find_nodes_by_type(NodeType::DataModel);
    assert_eq!(data_models.len(), 15);

    let traits = graph.find_nodes_by_type(NodeType::Trait);
    assert_eq!(traits.len(), 1);

    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_btreemap_test_to_function_edges() -> Result<()> {
    let repo = Repo::new(
        "src/testing/go",
        Lang::from_str("go").unwrap(),
        false,
        Vec::new(),
        Vec::new(),
    )
    .unwrap();

    let graph = repo.build_graph_inner::<BTreeMapGraph>().await?;

    let calls_edges = graph.count_edges_of_type(EdgeType::Calls);
    assert_eq!(calls_edges, 22);

    let contains_edges = graph.count_edges_of_type(EdgeType::Contains);
    assert_eq!(contains_edges, 136);

    let handler_edges = graph.count_edges_of_type(EdgeType::Handler);
    assert_eq!(handler_edges, 5);

    let implements_edges = graph.count_edges_of_type(EdgeType::Implements);
    assert_eq!(implements_edges, 0);

    let unit_test_to_function_edges =
        graph.find_nodes_with_edge_type(NodeType::UnitTest, NodeType::Function, EdgeType::Calls);
    assert_eq!(unit_test_to_function_edges.len(), 2);

    let integration_test_to_function_edges = graph.find_nodes_with_edge_type(
        NodeType::IntegrationTest,
        NodeType::Function,
        EdgeType::Calls,
    );
    assert_eq!(integration_test_to_function_edges.len(), 0);

    Ok(())
}

#[cfg(feature = "neo4j")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_coverage_default_params() -> Result<()> {
    let mut graph_ops = setup_go_graph().await?;

    let coverage = graph_ops.get_coverage(None, None, None).await?;

    assert_eq!(coverage.language, Some("go".to_string()));

    if let Some(unit) = &coverage.unit_tests {
        assert_eq!(unit.total_tests, 2);
    }

    if let Some(integration) = &coverage.integration_tests {
        assert_eq!(integration.total_tests, 1);
    }

    if let Some(e2e) = &coverage.e2e_tests {
        assert_eq!(e2e.total_tests, 2);
    }

    Ok(())
}

#[cfg(feature = "neo4j")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_coverage_with_repo_filter() -> Result<()> {
    let mut graph_ops = setup_go_graph().await?;
    let coverage = graph_ops
        .get_coverage(Some("src/testing/go"), None, None)
        .await?;

    assert_eq!(coverage.language, Some("go".to_string()));

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
    let mut graph_ops = setup_go_graph().await?;

    let filters = TestFilters {
        unit_regexes: vec![],
        integration_regexes: vec![],
        e2e_regexes: vec![],
        target_regex: None,
        ignore_dirs: vec!["tests".to_string()],
    };
    let filtered = graph_ops.get_coverage(None, Some(filters), None).await?;

    assert_eq!(filtered.language, Some("go".to_string()));

    Ok(())
}

#[cfg(feature = "neo4j")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_coverage_with_regex_filter() -> Result<()> {
    let mut graph_ops = setup_go_graph().await?;
    let filters = TestFilters {
        unit_regexes: vec![],
        integration_regexes: vec![],
        e2e_regexes: vec![],
        target_regex: Some(".*Handler.*".to_string()),
        ignore_dirs: vec![],
    };
    let coverage = graph_ops.get_coverage(None, Some(filters), None).await?;

    assert_eq!(coverage.language, Some("go".to_string()));

    Ok(())
}

#[cfg(feature = "neo4j")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_coverage_with_is_muted() -> Result<()> {
    let mut graph_ops = setup_go_graph().await?;

    let muted_coverage = graph_ops.get_coverage(None, None, Some(true)).await?;
    let unmuted_coverage = graph_ops.get_coverage(None, None, Some(false)).await?;

    assert_eq!(muted_coverage.language, Some("go".to_string()));
    assert_eq!(unmuted_coverage.language, Some("go".to_string()));

    Ok(())
}

#[cfg(feature = "neo4j")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_coverage_combined_filters() -> Result<()> {
    let mut graph_ops = setup_go_graph().await?;

    let filters = TestFilters {
        unit_regexes: vec![],
        integration_regexes: vec![],
        e2e_regexes: vec![],
        target_regex: Some(".*db.*".to_string()),
        ignore_dirs: vec!["tests".to_string()],
    };

    let coverage = graph_ops
        .get_coverage(Some("src/testing/go"), Some(filters), Some(false))
        .await?;

    assert_eq!(coverage.language, Some("go".to_string()));

    Ok(())
}
