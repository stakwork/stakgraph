use crate::lang::graphs::{BTreeMapGraph, EdgeType, Graph, NodeType};
use crate::lang::Lang;
use crate::repo::Repo;
use anyhow::Result;
use std::str::FromStr;
use test_log::test;

pub async fn test_go_graph_generic<G: Graph>() -> Result<()> {
    let repo = Repo::new(
        "src/testing/go",
        Lang::from_str("go").unwrap(),
        false,
        Vec::new(),
        Vec::new(),
    )
    .unwrap();

    let graph = repo.build_graph_inner::<G>().await?;

    graph.analysis();

    let (num_nodes, num_edges) = graph.get_graph_size();
    assert_eq!(num_nodes, 30, "Expected 30 nodes");
    assert_eq!(num_edges, 48, "Expected 48 edges");

    Ok(())
}

#[test(tokio::test)]
async fn test_go_graph() -> Result<()> {
    test_go_graph_generic::<BTreeMapGraph>().await
}
