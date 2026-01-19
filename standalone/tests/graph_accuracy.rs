#[cfg(feature = "neo4j")]
use ast::lang::graphs::graph_ops::GraphOps;

use ast::lang::graphs::BTreeMapGraph;
use ast::lang::{Graph, NodeType};
use ast::repo::{clone_repo, Repo};
use lsp::git::get_changed_files_between;
use tracing::info;

const REPO_URL: &str = "https://github.com/fayekelmith/demorepo.git";
const BEFORE_COMMIT: &str = "3a2bd5cc2e0a38ce80214a32ed06b2fb9430ab73";
const AFTER_COMMIT: &str = "778b5202fca04a2cd5daed377c0063e9af52b24c";
const USE_LSP: Option<bool> = Some(false);

async fn assert_graph_accuracy<G: Graph>(graph: &G, phase: &str) {
    let (num_nodes, num_edges) = graph.get_graph_size();
    info!("[{}] Nodes: {}, Edges: {}", phase, num_nodes, num_edges);

    let classes = graph.find_nodes_by_type(NodeType::Class);
    assert!(
        classes.iter().any(|c| c.name == "database"),
        "[{}] Missing class ",
        phase
    );

    let functions = graph.find_nodes_by_type(NodeType::Function);
    assert!(
        functions.iter().any(|f| f.name == "main"),
        "[{}] Missing function main",
        phase
    );
    assert!(
        functions.iter().any(|c| c.name == "Alpha"),
        "[{}] Missing class Alpha",
        phase
    );
    // TODO: Add more detailed assertions for endpoints, data models, etc.
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_graph_accuracy() {
    let repo_path = Repo::get_path_from_url(REPO_URL).unwrap();

    clone_repo(REPO_URL, &repo_path, None, None, Some(BEFORE_COMMIT), None)
        .await
        .unwrap();

    let repos = Repo::new_multi_detect(
        &repo_path,
        Some(REPO_URL.to_string()),
        Vec::new(),
        Vec::new(),
        USE_LSP,
    )
    .await
    .unwrap();

    let btree_graph = repos.build_graphs_inner::<BTreeMapGraph>().await.unwrap();
    assert_graph_accuracy(&btree_graph, "BTreeMapGraph BEFORE").await;

    #[cfg(feature = "neo4j")]
    {
        let mut graph_ops = GraphOps::new();
        graph_ops.clear().await.unwrap();
        graph_ops
            .update_full(
                REPO_URL,
                None,
                None,
                BEFORE_COMMIT,
                Some(BEFORE_COMMIT),
                None,
                USE_LSP,
                Some(false),
            )
            .await
            .unwrap();
        assert_graph_accuracy(&graph_ops.graph, "Neo4jGraph BEFORE").await;
    }

    clone_repo(REPO_URL, &repo_path, None, None, Some(AFTER_COMMIT), None)
        .await
        .unwrap();

    let changed_files = get_changed_files_between(&repo_path, BEFORE_COMMIT, AFTER_COMMIT)
        .await
        .unwrap();

    let expected_files = ["alpha.go", "beta.go", "delta.go"];
    for file in expected_files {
        assert!(
            changed_files.contains(&file.to_string()),
            "Expected changed file {} not found",
            file
        );
    }

    let new_repos = Repo::new_multi_detect(
        &repo_path,
        Some(REPO_URL.to_string()),
        Vec::new(),
        Vec::new(),
        USE_LSP,
    )
    .await
    .unwrap();

    let new_btree_graph = new_repos
        .build_graphs_inner::<BTreeMapGraph>()
        .await
        .unwrap();

    assert_graph_accuracy(&new_btree_graph, "BTreeMapGraph AFTER").await;

    #[cfg(feature = "neo4j")]
    {
        let mut graph_ops = GraphOps::new();
        graph_ops
            .update_incremental(
                REPO_URL,
                None,
                None,
                AFTER_COMMIT,
                BEFORE_COMMIT,
                Some(AFTER_COMMIT),
                None,
                USE_LSP,
                None,
            )
            .await
            .unwrap();

        graph_ops.graph.analysis();
        assert_graph_accuracy(&graph_ops.graph, "Neo4jGraph AFTER").await;

        let (btree_nodes, btree_edges) = new_btree_graph.get_graph_size();
        let (neo4j_nodes, neo4j_edges) = graph_ops.graph.get_graph_size();

        let (btree_node_keys, btree_edge_keys) = new_btree_graph.get_graph_keys();
        let (neo4j_node_keys, neo4j_edge_keys) = graph_ops.graph.get_graph_keys();

        let btree_nodes_only: Vec<_> = btree_node_keys.difference(&neo4j_node_keys).collect();
        let neo4j_nodes_only: Vec<_> = neo4j_node_keys.difference(&btree_node_keys).collect();

        let btree_only: Vec<_> = btree_edge_keys.difference(&neo4j_edge_keys).collect();
        let neo4j_only: Vec<_> = neo4j_edge_keys.difference(&btree_edge_keys).collect();

        assert!(
            btree_only.is_empty() && neo4j_only.is_empty(),
            "Edge key mismatch between BTreeMapGraph and Neo4jGraph"
        );

        assert!(
            btree_nodes_only.is_empty() && neo4j_nodes_only.is_empty(),
            "Node key mismatch between BTreeMapGraph and Neo4jGraph"
        );

        assert_eq!(
            btree_nodes, neo4j_nodes,
            "BTreeMapGraph and Neo4jGraph node count mismatch. BTreeMap only: {:?}, Neo4j only: {:?}",
            btree_nodes_only, neo4j_nodes_only
        );

        assert_eq!(
            btree_edges, neo4j_edges,
            "BTreeMapGraph and Neo4jGraph edge count mismatch. BTreeMap only: {:?}, Neo4j only: {:?}",
            btree_only, neo4j_only
        );
    }
}
