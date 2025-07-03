#[cfg(feature = "neo4j")]
use ast::lang::graphs::graph_ops::GraphOps;
use test_log::test;

#[cfg(feature = "neo4j")]
async fn clear_neo4j() {
    let mut graph_ops = GraphOps::new();
    graph_ops.connect().await.unwrap();
    graph_ops.clear().await.unwrap();
}

#[cfg(feature = "neo4j")]
#[test(tokio::test(flavor = "multi_thread", worker_threads = 2))]
async fn test_graph_consistency() {
    use ast::lang::graphs::{BTreeMapGraph, EdgeType};
    use ast::lang::Graph;
    use ast::repo::Repo;
    use tracing::info;

    let repo_url = "https://github.com/stakwork/demo-repo.git";

    clear_neo4j().await;

    info!("Building BTreeMapGraph...");
    let repos = Repo::new_clone_multi_detect(repo_url, None, None, Vec::new(), Vec::new(), None)
        .await
        .unwrap();

    let btree_graph = repos.build_graphs_inner::<BTreeMapGraph>().await.unwrap();

    btree_graph.analysis();

    let btree_node_count = btree_graph.nodes.len();
    let btree_edge_count = btree_graph.to_array_graph_edges().len();

    info!(
        "BTreeMapGraph: {} nodes, {} edges (formatted)",
        btree_node_count, btree_edge_count
    );

    let mut graph_ops = GraphOps::new();
    graph_ops.connect().await.unwrap();
    let (neo4j_nodes, neo4j_edges) = graph_ops
        .upload_btreemap_to_neo4j(&btree_graph)
        .await
        .unwrap();

    info!(
        "Neo4j upload result: {} nodes, {} edges",
        neo4j_nodes, neo4j_edges
    );

    assert_eq!(
        btree_node_count, neo4j_nodes as usize,
        "Node count mismatch: BTreeMapGraph={} Neo4j={}",
        btree_node_count, neo4j_nodes
    );
    assert_eq!(
        btree_edge_count, neo4j_edges as usize,
        "Edge count mismatch: BTreeMapGraph={} Neo4j={}",
        btree_edge_count, neo4j_edges
    );

    for edge_type in [
        EdgeType::Calls,
        EdgeType::Contains,
        EdgeType::Imports,
        EdgeType::Operand,
        EdgeType::Uses,
        EdgeType::ParentOf,
        EdgeType::Handler,
        EdgeType::Renders,
        EdgeType::ArgOf,
        EdgeType::Of,
    ] {
        let btree_count = btree_graph.count_edges_of_type(edge_type.clone());
        let neo4j_count = graph_ops.graph.count_edges_of_type(edge_type.clone());
        assert_eq!(
            btree_count, neo4j_count,
            "Edge count mismatch for {:?}: BTreeMapGraph={} Neo4j={}",
            edge_type, btree_count, neo4j_count
        );
        info!(
            "✅ EdgeType {:?}: BTreeMapGraph={} Neo4j={}",
            edge_type, btree_count, neo4j_count
        );
    }

    info!("✅ BTreeMapGraph and Neo4j upload counts are consistent!");
}
