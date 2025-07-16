#[cfg(feature = "neo4j")]
use ast::lang::graphs::graph_ops::GraphOps;

use ast::lang::linker::{normalize_backend_path, normalize_frontend_path};

fn standardized_request_name(name: &str) -> String {
    normalize_frontend_path(name).unwrap_or_else(|| name.to_string())
}

fn standardized_endpoint_name(name: &str) -> String {
    normalize_backend_path(name).unwrap_or_else(|| name.to_string())
}

#[cfg(feature = "neo4j")]
async fn assert_edge_exists(graph: &mut GraphOps, src: &str, tgt: &str) -> bool {
    use ast::lang::{graphs::EdgeType, Graph, NodeType};

    let pairs = graph.graph.find_nodes_with_edge_type(
        NodeType::Function,
        NodeType::Function,
        EdgeType::Calls,
    );
    pairs.iter().any(|(s, t)| s.name == src && t.name == tgt)
}

#[cfg(feature = "neo4j")]
async fn assert_request_calls_endpoint(
    graph: &mut GraphOps,
    request_name: &str,
    endpoint_name: &str,
) -> bool {
    use ast::lang::{graphs::EdgeType, Graph, NodeType};
    let pairs = graph.graph.find_nodes_with_edge_type(
        NodeType::Request,
        NodeType::Endpoint,
        EdgeType::Calls,
    );
    pairs.iter().any(|(req, ep)| {
        standardized_request_name(&req.name) == request_name
            && standardized_endpoint_name(&ep.name) == endpoint_name
    })
}

#[cfg(feature = "neo4j")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_graph_update() {
    use ast::lang::Graph;
    use ast::repo::{clone_repo, Repo};
    use lsp::git::get_changed_files_between;
    use tracing::info;

    let repo_url = "https://github.com/fayekelmith/graph-update";
    let repo_path = Repo::get_path_from_url(repo_url).unwrap();
    let before_commit = "f31f8371936097c20a4384dbf8620ae7776198c4";
    let after_commit = "f427783e90338f55fb21eec582dd0bb0735991d8";
    let use_lsp = Some(false);

    let mut graph_ops = GraphOps::new();
    graph_ops.connect().await.unwrap();
    graph_ops.clear().await.unwrap();

    let (nodes_before, edges_before) = graph_ops
        .update_full(
            repo_url,
            None,
            None,
            before_commit,
            Some(before_commit),
            use_lsp,
        )
        .await
        .unwrap();

    info!("Before: {} nodes and {} edges", nodes_before, edges_before);

    graph_ops.graph.analysis();

    // --- Assert initial state ---
    assert!(assert_edge_exists(&mut graph_ops, "Alpha", "Beta").await);
    assert!(assert_edge_exists(&mut graph_ops, "Alpha", "Gamma").await);
    assert!(assert_edge_exists(&mut graph_ops, "Beta", "Alpha").await);
    assert!(assert_edge_exists(&mut graph_ops, "GammaExtra", "Gamma").await);
    assert!(assert_node_exists(&mut graph_ops, "AlphaHelper").await);
    assert!(assert_node_exists(&mut graph_ops, "Gamma").await);
    assert!(!assert_node_exists(&mut graph_ops, "Delta").await);

    // REQUEST - CALLS -> ENDPOINT
    assert!(
        assert_request_calls_endpoint(
            &mut graph_ops,
            &standardized_request_name("${api.host}/people"),
            &standardized_endpoint_name("/people")
        )
        .await
    );

    assert!(
        assert_request_calls_endpoint(
            &mut graph_ops,
            &standardized_request_name("${api.host}/person"),
            &standardized_endpoint_name("/person")
        )
        .await
    );
    // This one should NOT exist before update
    assert!(!assert_request_calls_endpoint(&mut graph_ops, "/updateAlpha", "/updateAlpha").await);

    let changed_files = get_changed_files_between(&repo_path, before_commit, after_commit)
        .await
        .unwrap();
    info!("==>>Changed files: {:?}", changed_files);

    clone_repo(&repo_url, &repo_path, None, None, Some(after_commit))
        .await
        .unwrap();

    let (nodes_after, edges_after) = graph_ops
        .update_incremental(
            repo_url,
            None,
            None,
            after_commit,
            before_commit,
            Some(after_commit),
            use_lsp,
        )
        .await
        .unwrap();

    info!("==>>After: {} nodes and {} edges", nodes_after, edges_after);

    graph_ops.graph.analysis();

    // New relationships
    assert!(assert_edge_exists(&mut graph_ops, "Alpha", "Delta").await);
    assert!(assert_edge_exists(&mut graph_ops, "Alpha", "NewHelper").await);
    assert!(assert_edge_exists(&mut graph_ops, "Delta", "Alpha").await);

    // Removed relationships
    assert!(!assert_edge_exists(&mut graph_ops, "Alpha", "Beta").await);
    assert!(!assert_edge_exists(&mut graph_ops, "Alpha", "Gamma").await);
    assert!(!assert_edge_exists(&mut graph_ops, "Beta", "Alpha").await);

    // Deleted nodes (from deleted file and functions)
    assert!(!assert_node_exists(&mut graph_ops, "Gamma").await);
    assert!(!assert_node_exists(&mut graph_ops, "GammaExtra").await);
    assert!(!assert_node_exists(&mut graph_ops, "AlphaHelper").await);

    // New nodes
    assert!(assert_node_exists(&mut graph_ops, "Delta").await);
    assert!(assert_node_exists(&mut graph_ops, "DeltaHelper").await);
    assert!(assert_node_exists(&mut graph_ops, "NewHelper").await);
    assert!(assert_node_exists(&mut graph_ops, "NewBetaFunction").await);
    assert!(assert_node_exists(&mut graph_ops, "CrossFileFunc").await);

    // Stable elements
    assert!(assert_node_exists(&mut graph_ops, "main").await);
    assert!(assert_node_exists(&mut graph_ops, "ExistingUtil").await);

    // REQUEST - CALLS -> ENDPOINT
    assert!(assert_request_calls_endpoint(&mut graph_ops, "/updateAlpha", "/updateAlpha").await);
    assert!(assert_request_calls_endpoint(&mut graph_ops, "/fetchAlpha", "/fetchAlpha").await);
    // Stable requests
    assert!(
        assert_request_calls_endpoint(
            &mut graph_ops,
            &standardized_request_name("${api.host}/people"),
            &standardized_endpoint_name("/people")
        )
        .await
    );

    assert!(
        assert_request_calls_endpoint(
            &mut graph_ops,
            &standardized_request_name("${api.host}/person"),
            &standardized_endpoint_name("/person")
        )
        .await
    );
}

#[cfg(feature = "neo4j")]
async fn assert_node_exists(graph: &mut GraphOps, node_name: &str) -> bool {
    use ast::lang::NodeType;
    let nodes = graph
        .graph
        .find_nodes_by_name_any_language(NodeType::Function, node_name)
        .await;
    !nodes.is_empty()
}
