#[cfg(feature = "neo4j")]
use ast::lang::graphs::graph::Graph;
#[cfg(feature = "neo4j")]
use test_log::test;

#[cfg(feature = "neo4j")]
async fn clear_neo4j() {
    use ast::lang::graphs::graph_ops::GraphOps;
    let mut graph_ops = GraphOps::new();
    graph_ops.connect().await.unwrap();
    graph_ops.clear().await.unwrap();
}

#[cfg(feature = "neo4j")]
fn diff_sets<'a, T: std::cmp::Ord + std::hash::Hash + std::fmt::Debug + Clone>(
    local: impl IntoIterator<Item = T>,
    remote: impl IntoIterator<Item = T>,
) -> (Vec<T>, Vec<T>) {
    use std::collections::BTreeSet;
    let l: BTreeSet<T> = local.into_iter().collect();
    let r: BTreeSet<T> = remote.into_iter().collect();
    let missing_remote: Vec<T> = l.difference(&r).cloned().collect();
    let extra_remote: Vec<T> = r.difference(&l).cloned().collect();
    (missing_remote, extra_remote)
}

#[cfg(feature = "neo4j")]
fn format_sample<T: std::fmt::Debug>(v: &[T]) -> String {
    const MAX: usize = 5;
    let mut out = String::new();
    for (i, item) in v.iter().take(MAX).enumerate() {
        out.push_str(&format!("#{i}: {:?}\n", item));
    }
    out
}

#[cfg(feature = "neo4j")]
#[test(tokio::test(flavor = "multi_thread", worker_threads = 2))]
async fn graph_streaming_consistency() {
    use ast::lang::graphs::graph_ops::GraphOps;
    use ast::lang::graphs::{BTreeMapGraph, EdgeType};
    use ast::repo::Repo;
    use tracing::info;

    let repo_url = "https://github.com/stakwork/demo-repo";

    clear_neo4j().await;

    info!("Building graph with realtime streaming enabled");
    let repos = Repo::new_clone_multi_detect(
        repo_url,
        None,
        None,
        Vec::new(),
        Vec::new(),
        None,
        None,
        Some(false),
    )
    .await
    .unwrap();

    // Build with streaming enabled
    let _streaming_graph = repos
        .build_graphs_inner_with_streaming::<BTreeMapGraph>(true)
        .await
        .unwrap();

    // Also build local graph for comparison
    let repos_local = Repo::new_clone_multi_detect(
        repo_url,
        None,
        None,
        Vec::new(),
        Vec::new(),
        None,
        None,
        Some(false),
    )
    .await
    .unwrap();
    let local_graph = repos_local
        .build_graphs_inner::<BTreeMapGraph>()
        .await
        .unwrap();

    // Local sizes
    let local_node_count = local_graph.nodes.len();
    let local_edge_vec = local_graph.to_array_graph_edges();
    let local_edge_count = local_edge_vec.len();

    info!(
        "Local baseline: nodes={}, edges={}",
        local_node_count, local_edge_count
    );

    info!(">>>>> Local BTreeMapGraph analysis");
    local_graph.analysis();

    let mut graph_ops = GraphOps::new();
    graph_ops.connect().await.unwrap();
    let (neo_nodes, neo_edges) = graph_ops.get_graph_size().await.unwrap();
    info!(
        "Neo4j streamed result: nodes={}, edges={}",
        neo_nodes, neo_edges
    );

    info!(">>>>> Remote Neo4jGraph analysis");
    graph_ops.graph.analysis();

    let local_node_keys: Vec<String> = local_graph.nodes.keys().cloned().collect();
    let remote_node_keys = graph_ops.fetch_all_node_keys().await.unwrap_or_default();

    let local_edge_triples: Vec<(String, String, EdgeType)> = local_edge_vec
        .iter()
        .map(|e| {
            let s = ast::utils::create_node_key_from_ref(&e.source);
            let t = ast::utils::create_node_key_from_ref(&e.target);
            (s, t, e.edge.clone())
        })
        .collect();
    let remote_edge_triples = graph_ops.fetch_all_edge_triples().await.unwrap_or_default();

    let (missing_nodes, extra_nodes) = diff_sets(local_node_keys.clone(), remote_node_keys.clone());
    let (missing_edges, extra_edges) =
        diff_sets(local_edge_triples.clone(), remote_edge_triples.clone());

    let allowed_extra_node = extra_nodes.len() == 1 && extra_nodes[0].contains("person");
    let allowed_extra_edge = extra_edges.len() <= 1
        && extra_edges
            .iter()
            .all(|e| format!("{:?}", e).contains("person"));

    if !missing_nodes.is_empty()
        || !missing_edges.is_empty()
        || (!allowed_extra_node && !extra_nodes.is_empty())
        || (!allowed_extra_edge && !extra_edges.is_empty())
    {
        info!("Streaming differences detected. Missing nodes: {} extra nodes: {} missing edges: {} extra edges: {}", missing_nodes.len(), extra_nodes.len(), missing_edges.len(), extra_edges.len());
        info!(
            "Sample missing node keys:\n{}",
            format_sample(&missing_nodes)
        );
        info!("Sample missing edges:\n{}", format_sample(&missing_edges));
        info!("Sample extra node keys:\n{}", format_sample(&extra_nodes));
        info!("Sample extra edges:\n{}", format_sample(&extra_edges));
    }

    assert!(
        missing_nodes.is_empty(),
        "Missing streamed nodes: {}",
        missing_nodes.len()
    );
    assert!(
        missing_edges.is_empty(),
        "Missing streamed edges: {}",
        missing_edges.len()
    );
    assert!(
        allowed_extra_node || extra_nodes.is_empty(),
        "Unexpected extra nodes: {:?}",
        extra_nodes
    );
    assert!(
        allowed_extra_edge || extra_edges.is_empty(),
        "Unexpected extra edges: {:?}",
        extra_edges
    );
}
