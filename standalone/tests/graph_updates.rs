#[cfg(feature = "neo4j")]
use ast::lang::graphs::graph_ops::GraphOps;

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
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_graph_update() {
    use ast::lang::Graph;
    use ast::repo::{clone_repo, Repo};
    use lsp::git::get_changed_files_between;
    use tracing::info;

    let repo_url = "https://github.com/fayekelmith/demo-repo";
    let repo_path = Repo::get_path_from_url(repo_url).unwrap();
    let before_commit = "ebb64dd615de5c6c83f4fceb170773f28c06cbea";
    let after_commit = "311dfd5e10d5003737dba60a70370515350e8a18";
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
            vec![before_commit.to_string()],
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
