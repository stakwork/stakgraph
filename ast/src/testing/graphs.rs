use crate::lang::graphs::{BTreeMapGraph, Node, NodeType};
use crate::lang::{ArrayGraph, Lang};
use crate::repo::Repo;
use anyhow::Result;
use std::str::FromStr;

type CriticalEdgeCheck = (NodeType, &'static str);
pub struct GraphTestExpectations {
    pub lang_id: &'static str,
    pub repo_path: &'static str,
    pub expected_nodes: usize,
    pub expected_edges: usize,
    pub critical_edges: Vec<CriticalEdgeCheck>,
}

pub async fn run_graph_similarity_test(expectations: &GraphTestExpectations) -> Result<()> {
    let lang = Lang::from_str(expectations.lang_id).unwrap();

    let repo_a = Repo::new(expectations.repo_path, lang, false, Vec::new(), Vec::new()).unwrap();
    let graph_a = repo_a.build_graph_inner::<ArrayGraph>().await?;

    let lang = Lang::from_str(expectations.lang_id).unwrap();
    let repo_b = Repo::new(expectations.repo_path, lang, false, Vec::new(), Vec::new()).unwrap();
    let graph_b = repo_b.build_graph_inner::<BTreeMapGraph>().await?;

    let mut nodes_a = graph_a.nodes; // Takes ownership
    let mut nodes_b: Vec<Node> = graph_b.nodes.values().cloned().collect();

    nodes_a.sort();
    nodes_b.sort();

    assert_eq!(
        nodes_a.len(),
        expectations.expected_nodes,
        "ArrayGraph node count mismatch"
    );
    assert_eq!(
        nodes_b.len(),
        expectations.expected_nodes,
        "BTreeMapGraph node count mismatch"
    );

    assert_eq!(nodes_a, nodes_b, "Node sets are not identical");

    let edges_a = graph_a.edges;

    let edges_b = graph_b.edges;

    assert_eq!(
        edges_a.len(),
        expectations.expected_edges,
        "ArrayGraph edge count mismatch"
    );
    assert_eq!(
        edges_b.len(),
        expectations.expected_edges,
        "BTreeMapGraph edge count mismatch"
    );

    Ok(())
}
