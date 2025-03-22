use crate::lang::graph::{EdgeType, Node};
use crate::{lang::Lang, repo::Repo};
use std::str::FromStr;
use test_log::test;
use tracing::{info, debug, error};
use tracing_subscriber::{FmtSubscriber, EnvFilter};


fn setup_tracing_logger() {
    let subscriber = FmtSubscriber::builder()
        .with_env_filter(EnvFilter::new("info"))
        .finish();
    tracing::subscriber::set_global_default(subscriber).expect("Failed to set global subscriber");
}

#[test(tokio::test)]
async fn test_swift() {
    setup_tracing_logger();
    let repo = Repo::new(
        "src/testing/swift",
        Lang::from_str("swift").unwrap(),
        false,
        Vec::new(),
        Vec::new(),
    )
    .unwrap();


    let graph = repo.build_graph().await.unwrap();


    assert_eq!(graph.nodes.len(), 34);
    assert_eq!(graph.edges.len(), 33);


    fn normalize_path(path: &str) -> String {
        path.replace("\\", "/")
    }


    let language_nodes = graph
        .nodes
        .iter()
        .filter(|n| matches!(n, Node::Language(_)))
        .collect::<Vec<_>>();
    assert_eq!(language_nodes.len(), 1);
    let language_node = language_nodes[0].into_data();
    assert_eq!(language_node.name, "swift");
    assert_eq!(normalize_path(&language_node.file), "src/testing/swift/");


    let podfile_nodes = graph
        .nodes
        .iter()
        .filter(|n| matches!(n, Node::File(_)) && n.into_data().name == "Podfile")
        .collect::<Vec<_>>();
    assert_eq!(podfile_nodes.len(), 1);
    let podfile_node = podfile_nodes[0].into_data();
    assert_eq!(podfile_node.name, "Podfile");


    let imports = graph
        .nodes
        .iter()
        .filter(|n| matches!(n, Node::Library(_)))
        .collect::<Vec<_>>();
    assert_eq!(imports.len(), 0);


    let imports = graph
        .nodes
        .iter()
        .filter(|n| matches!(n, Node::Import(_)))
        .collect::<Vec<_>>();
    assert_eq!(imports.len(), 7);


    let classes = graph
        .nodes
        .iter()
        .filter(|n| matches!(n, Node::Class(_)))
        .collect::<Vec<_>>();
    assert_eq!(classes.len(), 7);

    // Ensure the first class node is what you expect
    let example_class = classes[0].into_data();
    assert_eq!(example_class.name, "API");
    assert_eq!(
        normalize_path(&example_class.file),
        "src/testing/swift/SphinxTestApp/API.swift"
    );


    let functions = graph
        .nodes
        .iter()
        .filter(|n| matches!(n, Node::Function(_)))
        .collect::<Vec<_>>();
    assert_eq!(functions.len(), 0);


    let endpoints = graph
        .nodes
        .iter()
        .filter(|n| matches!(n, Node::Endpoint(_)))
        .collect::<Vec<_>>();
    assert_eq!(endpoints.len(), 4);


    let calls_edges = graph
        .edges
        .iter()
        .filter(|e| matches!(e.edge, EdgeType::Calls(_)))
        .collect::<Vec<_>>();
    assert_eq!(calls_edges.len(), 0);

    // Optionally, assertions for other types of edges or relationships
}
