use crate::lang::graphs::graph::Graph;
#[cfg(feature = "neo4j")]
use crate::lang::graphs::graph_ops::GraphOps;
use crate::lang::{graphs::EdgeType, NodeType};
use lsp::git::{checkout_commit, get_changed_files_between, git_pull_or_clone};
use tracing::{debug, info};
#[cfg(feature = "neo4j")]
fn clear_neo4j() {
    let mut graph_ops = GraphOps::new();
    graph_ops.connect().unwrap();
    graph_ops.clear().unwrap();
}
#[cfg(feature = "neo4j")]
fn assert_edge_exists(graph: &GraphOps, src: &str, tgt: &str) -> bool {
    graph
        .graph
        .find_nodes_with_edge_type(
            NodeType::Function,
            NodeType::Function,
            EdgeType::Calls(Default::default()),
        )
        .iter()
        .any(|(s, t)| s.name == src && t.name == tgt)
}

#[cfg(feature = "neo4j")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_graph_update() {
    use crate::lang::graph;

    let repo_url = "https://github.com/fayekelmith/demorepo.git";
    let repo_path = "/tmp/demorepo";
    let before_commit = "3a2bd5cc2e0a38ce80214a32ed06b2fb9430ab73";
    let after_commit = "778b5202fca04a2cd5daed377c0063e9af52b24c";

    git_pull_or_clone(repo_url, repo_path, None, None)
        .await
        .unwrap();
    clear_neo4j();

    // --- BEFORE UPDATE ---
    checkout_commit(repo_path, before_commit).await.unwrap();

    let mut graph_ops = GraphOps::new();
    graph_ops.connect().unwrap();
    graph_ops.clear().unwrap();

    let (nodes_before, edges_before) = graph_ops
        .update_full(repo_url, repo_path, before_commit)
        .unwrap();

    info!("Before: {} nodes and {} edges", nodes_before, edges_before);

    graph_ops.graph.analysis();

    // --- Assert initial state ---
    assert!(
        assert_edge_exists(&graph_ops, "Alpha", "Beta"),
        "Before: Alpha should call Beta"
    );
    assert!(
        assert_edge_exists(&graph_ops, "Alpha", "Gamma"),
        "Before: Alpha should call Gamma"
    );
    assert!(
        assert_edge_exists(&graph_ops, "Beta", "Alpha"),
        "Before: Beta should call Alpha"
    );

    // --- AFTER UPDATE ---
    checkout_commit(repo_path, after_commit).await.unwrap();

    let changed_files = get_changed_files_between(repo_path, before_commit, after_commit)
        .await
        .unwrap();
    println!("Changed files: {:?}", changed_files);

    let (nodes_after, edges_after) = graph_ops
        .update_incremental(repo_url, repo_path, after_commit, before_commit)
        .unwrap();

    info!("After: {} nodes and {} edges", nodes_after, edges_after);

    graph_ops.graph.analysis();
    // --- Assert updated state ---
    // Alpha should now call Delta, not Beta or Gamma
    assert!(
        assert_edge_exists(&graph_ops, "Alpha", "Delta"),
        "After: Alpha should call Delta"
    );
    assert!(
        assert_edge_exists(&graph_ops, "Delta", "Alpha"),
        "After: Delta should call Alpha"
    );
    assert!(
        !assert_edge_exists(&graph_ops, "Alpha", "Beta"),
        "After: Alpha should NOT call Beta"
    );
    assert!(
        !assert_edge_exists(&graph_ops, "Alpha", "Gamma"),
        "After: Alpha should NOT call Gamma"
    );
    assert!(
        !assert_edge_exists(&graph_ops, "Beta", "Alpha"),
        "After: Beta should NOT call Alpha"
    );
}
