#[cfg(feature = "neo4j")]
use crate::lang::graphs::graph_ops::GraphOps;
use crate::lang::{graphs::EdgeType, NodeType};
use lsp::git::{checkout_commit, get_changed_files_between, git_pull_or_clone};
use tracing::info;

async fn clear_neo4j() {
    let mut graph_ops = GraphOps::new();
    graph_ops.connect().await.unwrap();
    graph_ops.clear().await.unwrap();
}

async fn assert_edge_exists(graph: &mut GraphOps, src: &str, tgt: &str) -> bool {
    match graph
        .graph
        .find_nodes_with_edge_type(NodeType::Function, NodeType::Function, EdgeType::Calls)
        .await
    {
        Ok(pairs) => pairs.iter().any(|(s, t)| s.name == src && t.name == tgt),
        Err(_) => false,
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_graph_update() {
    let repo_url = "https://github.com/fayekelmith/demorepo.git";
    let repo_path = "/tmp/demorepo";
    let before_commit = "3a2bd5cc2e0a38ce80214a32ed06b2fb9430ab73";
    let after_commit = "778b5202fca04a2cd5daed377c0063e9af52b24c";

    git_pull_or_clone(repo_url, repo_path, None, None)
        .await
        .unwrap();
    clear_neo4j().await;

    // --- BEFORE UPDATE ---
    checkout_commit(repo_path, before_commit).await.unwrap();

    let mut graph_ops = GraphOps::new();
    graph_ops.connect().await.unwrap();
    graph_ops.clear().await.unwrap();

    let (nodes_before, edges_before) = graph_ops
        .update_full(repo_url, repo_path, before_commit)
        .await
        .unwrap();

    println!("Before: {} nodes and {} edges", nodes_before, edges_before);

    let _ = graph_ops.graph.analysis().await;

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

    // --- AFTER UPDATE ---
    checkout_commit(repo_path, after_commit).await.unwrap();

    let changed_files = get_changed_files_between(repo_path, before_commit, after_commit)
        .await
        .unwrap();
    info!("==>>Changed files: {:?}", changed_files);

    let (nodes_after, edges_after) = graph_ops
        .update_incremental(repo_url, repo_path, after_commit, before_commit)
        .await
        .unwrap();

    info!("==>>After: {} nodes and {} edges", nodes_after, edges_after);

    let _ = graph_ops.graph.analysis().await;
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
