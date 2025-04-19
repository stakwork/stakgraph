use crate::lang::graphs::{BTreeMapGraph, Node, NodeType};
use crate::lang::{ArrayGraph, Graph, Lang};
use crate::repo::Repo;
use anyhow::Result;
use std::str::FromStr;

type CriticalEdgeCheck = (NodeType, &'static str);
pub struct GraphTestExpectations {
    pub lang_id: &'static str,
    pub repo_path: &'static str,
    pub expected_nodes: u32,
    pub expected_edges: u32,
    pub expected_nodes_lsp: Option<u32>,
    pub expected_edges_lsp: Option<u32>,
    pub critical_edges: Vec<CriticalEdgeCheck>,
    pub critical_edges_lsp: Option<Vec<CriticalEdgeCheck>>,
}

pub async fn run_graph_similarity_test(
    expectations: &GraphTestExpectations,
    use_lsp: bool,
) -> Result<()> {
    let lang = Lang::from_str(expectations.lang_id).unwrap();

    let repo_a = Repo::new(
        expectations.repo_path,
        lang,
        use_lsp,
        Vec::new(),
        Vec::new(),
    )
    .unwrap();
    let graph_a = repo_a.build_graph_inner::<ArrayGraph>().await?;
    println!("ArrayGraph Analysis");
    graph_a.analysis();

    let lang = Lang::from_str(expectations.lang_id).unwrap();
    let repo_b = Repo::new(
        expectations.repo_path,
        lang,
        use_lsp,
        Vec::new(),
        Vec::new(),
    )
    .unwrap();
    let graph_b = repo_b.build_graph_inner::<BTreeMapGraph>().await?;

    println!("BTreeMapGraph Analysis");
    graph_b.analysis();

    let mut nodes_a = graph_a.nodes; // Takes ownership
    let mut nodes_b: Vec<Node> = graph_b.nodes.values().cloned().collect();

    nodes_a.sort();
    nodes_b.sort();

    if !use_lsp {
        assert_eq!(
            nodes_a.len() as u32,
            expectations.expected_nodes,
            "ArrayGraph node count mismatch"
        );
        assert_eq!(
            nodes_b.len() as u32,
            expectations.expected_nodes,
            "BTreeMapGraph node count mismatch"
        );
    } else {
        assert_eq!(
            nodes_a.len() as u32,
            expectations.expected_nodes_lsp.unwrap(),
            "ArrayGraph node count mismatch"
        );
        assert_eq!(
            nodes_b.len() as u32,
            expectations.expected_nodes_lsp.unwrap(),
            "BTreeMapGraph node count mismatch"
        );
    }

    assert_eq!(nodes_a, nodes_b, "Node sets are not identical");

    let edges_a = graph_a.edges;

    let edges_b = graph_b.edges;

    if !use_lsp {
        assert_eq!(
            edges_a.len() as u32,
            expectations.expected_edges,
            "ArrayGraph edge count mismatch"
        );
        assert_eq!(
            edges_b.len() as u32,
            expectations.expected_edges,
            "BTreeMapGraph edge count mismatch"
        );
    } else {
        assert_eq!(
            edges_a.len() as u32,
            expectations.expected_edges_lsp.unwrap(),
            "ArrayGraph edge count mismatch"
        );
        assert_eq!(
            edges_b.len() as u32,
            expectations.expected_edges_lsp.unwrap(),
            "BTreeMapGraph edge count mismatch"
        );
    }

    Ok(())
}
