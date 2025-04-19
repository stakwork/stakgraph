use crate::lang::asg::NodeKeys;
use crate::lang::graphs::{BTreeMapGraph, Edge, Node, NodeRef};
use crate::lang::{graph, ArrayGraph, Lang};
use crate::repo::Repo;
use anyhow::Result;
use std::str::FromStr;
use test_log::test;

#[test(tokio::test)]
async fn test_go_graphs() -> Result<()> {
    let repo_path = "src/testing/go";
    let lang = Lang::from_str("go").unwrap();

    let repo_a = Repo::new(repo_path, lang, false, Vec::new(), Vec::new()).unwrap();
    let graph_a = repo_a.build_graph_inner::<ArrayGraph>().await?;

    let lang = Lang::from_str("go").unwrap();
    let repo_b = Repo::new(repo_path, lang, false, Vec::new(), Vec::new()).unwrap();
    let graph_b = repo_b.build_graph_inner::<BTreeMapGraph>().await?;

    let mut nodes_a = graph_a.nodes; // Takes ownership
    let mut nodes_b: Vec<Node> = graph_b.nodes.values().cloned().collect();

    nodes_a.sort();
    nodes_b.sort();

    assert_eq!(nodes_a.len(), 30, "ArrayGraph node count mismatch");
    assert_eq!(nodes_b.len(), 30, "BTreeMapGraph node count mismatch");

    assert_eq!(nodes_a, nodes_b, "Node sets are not identical");

    let edges_a = graph_a.edges;

    let edges_b = graph_b.edges;

    println!("ArrayGraph edges: {:#?}", edges_a);
    println!("BTreeMapGraph edges: {:#?}", edges_b);

    assert_eq!(edges_a.len(), 48, "ArrayGraph edge count mismatch");
    assert_eq!(edges_b.len(), 48, "BTreeMapGraph edge count mismatch");

    Ok(())
}
