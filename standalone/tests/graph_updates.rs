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
    use ast::repo::Repo;
    use lsp::git::get_changed_files_between;
    use tracing::info;

    let repo_url = "https://github.com/fayekelmith/demorepo.git";
    let repo_path = Repo::get_path_from_url(repo_url).unwrap();
    let before_commit = "3a2bd5cc2e0a38ce80214a32ed06b2fb9430ab73";
    let after_commit = "778b5202fca04a2cd5daed377c0063e9af52b24c";
    let use_lsp = Some(false);

    let mut graph_ops = GraphOps::new();
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

    let _ = graph_ops.graph.analysis();

    // --- Assert initial state ---
    assert!(
        assert_edge_exists(&mut graph_ops, "Alpha", "Beta").await,
        "Before: Alpha should call Beta"
    );
    assert!(
        assert_edge_exists(&mut graph_ops, "Alpha", "Gamma").await,
        "Before: Alpha should call Gamma"
    );
    assert!(
        assert_edge_exists(&mut graph_ops, "Beta", "Alpha").await,
        "Before: Beta should call Alpha"
    );

    let (nodes_before_inc, edges_before_inc) = graph_ops.graph.get_graph_size();
    info!(
        "[DEBUG] GraphOps state BEFORE incremental update: {} nodes, {} edges",
        nodes_before_inc, edges_before_inc
    );

    let changed_files = get_changed_files_between(&repo_path, before_commit, after_commit)
        .await
        .unwrap();
    info!("==>>Changed files: {:?}", changed_files);

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

    let _ = graph_ops.graph.analysis();
    // --- Assert updated state ---
    // Alpha should now call Delta, not Beta or Gamma
    assert!(
        assert_edge_exists(&mut graph_ops, "Alpha", "Delta").await,
        "After: Alpha should call Delta"
    );
    assert!(
        assert_edge_exists(&mut graph_ops, "Delta", "Alpha").await,
        "After: Delta should call Alpha"
    );
    assert!(
        !assert_edge_exists(&mut graph_ops, "Alpha", "Beta").await,
        "After: Alpha should NOT call Beta"
    );
    assert!(
        !assert_edge_exists(&mut graph_ops, "Alpha", "Gamma").await,
        "After: Alpha should NOT call Gamma"
    );
    assert!(
        !assert_edge_exists(&mut graph_ops, "Beta", "Alpha").await,
        "After: Beta should NOT call Alpha"
    );
}
