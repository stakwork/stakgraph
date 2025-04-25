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
    pub _critical_edges: Vec<CriticalEdgeCheck>,
    pub _critical_edges_lsp: Option<Vec<CriticalEdgeCheck>>,
}

impl Default for GraphTestExpectations {
    fn default() -> Self {
        GraphTestExpectations {
            lang_id: "",
            repo_path: "",
            expected_nodes: 0,
            expected_edges: 0,
            expected_nodes_lsp: None,
            expected_edges_lsp: None,
            _critical_edges: Vec::new(),
            _critical_edges_lsp: None,
        }
    }
}
pub async fn run_graph_similarity_test(
    expectations: &GraphTestExpectations,
    use_lsp: bool,
) -> Result<()> {
    let use_lsp_to_test = use_lsp
        && expectations.expected_edges_lsp.is_some()
        && expectations.expected_nodes_lsp.is_some();

    let lang = Lang::from_str(expectations.lang_id).unwrap();
    let repo_a = Repo::new(
        expectations.repo_path,
        lang,
        use_lsp_to_test,
        Vec::new(),
        Vec::new(),
    )
    .unwrap();
    let graph_a = repo_a.build_graph_inner::<ArrayGraph>().await?;
    println!("ArrayGraph Analysis for {}", expectations.lang_id);
    //graph_a.analysis();

    let lang = Lang::from_str(expectations.lang_id).unwrap();
    let repo_b = Repo::new(
        expectations.repo_path,
        lang,
        use_lsp_to_test,
        Vec::new(),
        Vec::new(),
    )
    .unwrap();
    let graph_b = repo_b.build_graph_inner::<BTreeMapGraph>().await?;

    println!("BTreeMapGraph Analysis for {}", expectations.lang_id);
    //graph_b.analysis();

    let nodes_a = graph_a.nodes;
    let nodes_b: Vec<Node> = graph_b.nodes.values().cloned().collect();

    if use_lsp_to_test {
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
    } else {
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
    }

    let edges_a = graph_a.edges;

    let edges_b = graph_b.edges;

    if use_lsp_to_test {
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
    } else {
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
    }

    Ok(())
}

pub fn get_test_expectations() -> Vec<GraphTestExpectations> {
    vec![
        GraphTestExpectations {
            lang_id: "go",
            repo_path: "src/testing/go",
            expected_nodes: 30,
            expected_edges: 48,
            expected_nodes_lsp: Some(64),
            expected_edges_lsp: Some(108),
            ..Default::default()
        },
        GraphTestExpectations {
            lang_id: "react",
            repo_path: "src/testing/react",
            expected_nodes: 49,
            expected_edges: 61,
            expected_nodes_lsp: Some(55),
            expected_edges_lsp: Some(77),
            ..Default::default()
        },
        GraphTestExpectations {
            lang_id: "angular",
            repo_path: "src/testing/angular",
            expected_nodes: 76,
            expected_edges: 78,
            ..Default::default()
        },
        GraphTestExpectations {
            lang_id: "kotlin",
            repo_path: "src/testing/kotlin",
            expected_nodes: 115,
            expected_edges: 125,
            ..Default::default()
        },
        GraphTestExpectations {
            lang_id: "swift",
            repo_path: "src/testing/swift",
            expected_nodes: 55,
            expected_edges: 81,
            ..Default::default()
        },
        GraphTestExpectations {
            lang_id: "python",
            repo_path: "src/testing/python",
            expected_nodes: 61,
            expected_edges: 78,
            ..Default::default()
        },
        GraphTestExpectations {
            lang_id: "svelte",
            repo_path: "src/testing/svelte",
            expected_nodes: 43,
            expected_edges: 42,
            ..Default::default()
        },
        GraphTestExpectations {
            lang_id: "ruby",
            repo_path: "src/testing/ruby",
            expected_nodes: 55,
            expected_edges: 79,
            ..Default::default()
        },
        GraphTestExpectations {
            lang_id: "java",
            repo_path: "src/testing/java",
            expected_nodes: 37,
            expected_edges: 42,
            ..Default::default()
        },
        GraphTestExpectations {
            lang_id: "typescript",
            repo_path: "src/testing/typescript",
            expected_nodes: 42,
            expected_edges: 47,
            expected_nodes_lsp: Some(45),
            expected_edges_lsp: Some(52),
            ..Default::default()
        },
        GraphTestExpectations {
            lang_id: "rust",
            repo_path: "src/testing/rust",
            expected_nodes: 44,
            expected_edges: 56,
            ..Default::default()
        },
    ]
}
