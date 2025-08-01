use crate::lang::graphs::BTreeMapGraph;
use crate::lang::{ArrayGraph, Graph, Lang};
use crate::repo::Repo;
use crate::utils::get_use_lsp;
use shared::error::Result;
use std::collections::HashSet;
use std::str::FromStr;
use test_log::test;
use tracing::{debug, info};

const PROGRAMMING_LANGUAGES: [&str; 12] = [
    "angular",
    "go",
    "kotlin",
    "python",
    "react",
    "ruby",
    "svelte",
    "swift",
    "typescript",
    "java",
    "rust",
    "cpp",
];

#[test(tokio::test(flavor = "multi_thread", worker_threads = 2))]
async fn compare_graphs() {
    for lang in PROGRAMMING_LANGUAGES.iter() {
        let repo_path = format!("src/testing/{}", lang);
        info!("Comparing graphs for {}", lang);
        if let Err(e) = compare_graphs_inner(lang, &repo_path).await {
            debug!("Error comparing graphs for {}: {:?}", lang, e);
        };
    }
}

async fn compare_graphs_inner(lang_id: &str, repo_path: &str) -> Result<()> {
    let lang = Lang::from_str(lang_id).unwrap();
    let use_lsp = get_use_lsp() && lang.kind.default_do_lsp();
    let repo = Repo::new(repo_path, lang, use_lsp, Vec::new(), Vec::new()).unwrap();

    info!("ArrayGraph Analysis for {}", lang_id);
    let array_graph = repo.build_graph_inner::<ArrayGraph>().await?;

    info!("BTreeMapGraph Analysis for {}", lang_id);
    let btree_map_graph = repo.build_graph_inner::<BTreeMapGraph>().await?;

    let (array_graph_nodes, array_graph_edges) = array_graph.get_graph_keys();
    let (btree_map_graph_nodes, btree_map_graph_edges) = btree_map_graph.get_graph_keys();
    let nodes_only_in_array_graph: HashSet<_> = array_graph_nodes
        .difference(&btree_map_graph_nodes)
        .collect();
    let nodes_only_in_btree_map_graph: HashSet<_> = btree_map_graph_nodes
        .difference(&array_graph_nodes)
        .collect();

    let edges_only_in_array_graph: HashSet<_> = array_graph_edges
        .difference(&btree_map_graph_edges)
        .collect();
    let edges_only_in_btree_map_graph: HashSet<_> = btree_map_graph_edges
        .difference(&array_graph_edges)
        .collect();

    if !nodes_only_in_array_graph.is_empty() {
        debug!("Nodes only in ArrayGraph: {:#?}", nodes_only_in_array_graph);
        debug!(
            "Nodes only in BTreeMapGraph: {:#?}",
            nodes_only_in_btree_map_graph
        );
    }
    if !edges_only_in_array_graph.is_empty() {
        debug!("Edges only in ArrayGraph: {:#?}", edges_only_in_array_graph);
        debug!(
            "Edges only in BTreeMapGraph: {:#?}",
            edges_only_in_btree_map_graph
        );
    }

    assert_eq!(
        array_graph.nodes.len(),
        btree_map_graph.nodes.len(),
        "Node counts do not match: ArrayGraph has {}, BTreeMapGraph has {}",
        array_graph.nodes.len(),
        btree_map_graph.nodes.len()
    );

    if use_lsp {
        assert!(
            (array_graph.edges.len() as i32 - btree_map_graph.edges.len() as i32).abs() <= 2,
            "Edge counts differ by more than 2: ArrayGraph has {}, BTreeMapGraph has {}",
            array_graph.edges.len(),
            btree_map_graph.edges.len()
        );
    } else {
        assert_eq!(
            array_graph.edges.len(),
            btree_map_graph.edges.len(),
            "Edge counts do not match: ArrayGraph has {}, BTreeMapGraph has {}",
            array_graph.edges.len(),
            btree_map_graph.edges.len()
        );
    }

    Ok(())
}
