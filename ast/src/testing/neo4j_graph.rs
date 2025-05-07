#[cfg(feature = "neo4j")]
use crate::lang::graphs::Neo4jGraph;
use crate::lang::Graph;
use crate::{lang::Lang, repo::Repo};
use anyhow::Result;
use std::str::FromStr;
use tracing_test::traced_test;

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[traced_test]
#[ignore]
pub async fn test_neo4j() -> Result<()> {
    let mut graph = Neo4jGraph::default();
    graph.connect().await?;

    assert!(graph.is_connected(), "Neo4j graph should be connected");

    graph.clear();

    let (nodes, edges) = graph.get_graph_size();
    graph.analysis();
    // Fix contradictory assertion
    assert_eq!(nodes, 0, "Graph should have 0 nodes after clearing");
    assert_eq!(edges, 0, "Graph should have 0 edges after clearing");

    let use_lsp = false;
    let repo = Repo::new(
        "src/testing/go",
        Lang::from_str("go").unwrap(),
        use_lsp,
        Vec::new(),
        Vec::new(),
    )?;

    let mut graph = repo.build_graph_inner::<Neo4jGraph>().await?;
    let (num_nodes, num_edges) = graph.get_graph_size();

    graph.analysis();

    assert_eq!(num_nodes, 30, "Expected 30 nodes");
    assert_eq!(num_edges, 48, "Expected 48 edges");

    graph.disconnect()?;
    assert!(!graph.is_connected(), "Neo4j graph should be disconnected");
    Ok(())
}
