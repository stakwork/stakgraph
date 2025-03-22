use crate::lang::graph::{EdgeType, Node, NodeType};
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
    assert_eq!(graph.nodes.len(), 54);
    assert_eq!(graph.edges.len(), 79);

    let languages = graph
        .nodes
        .iter()
        .filter(|n| matches!(n, Node::Language(_)))
        .collect::<Vec<_>>();
    assert_eq!(languages.len(), 1);

    let language = languages[0].into_data();
    assert_eq!(language.name, "swift");
    assert_eq!(language.file, "src/testing/swift/");

    let files = graph
        .nodes
        .iter()
        .filter(|n| matches!(n, Node::File(_)))
        .collect::<Vec<_>>();

    assert_eq!(files.len(), 9, "wrong file count");

    let imports = graph
        .nodes
        .iter()
        .filter(|n| matches!(n, Node::Import(_)))
        .collect::<Vec<_>>();

    assert_eq!(imports.len(), 7, "wrong import count");

    let classes = graph
        .nodes
        .iter()
        .filter(|n| matches!(n, Node::Class(_)))
        .collect::<Vec<_>>();

    assert_eq!(classes.len(), 7);

    let class = classes[0].into_data();
    assert_eq!(class.name, "API");

    let functions = graph
        .nodes
        .iter()
        .filter(|n| matches!(n, Node::Function(_)))
        .collect::<Vec<_>>();
    assert_eq!(functions.len(), 26);
    let func = functions[0].into_data();
    assert_eq!(func.name, "createRequest");


    let data_models = graph
        .nodes
        .iter()
        .filter(|n| matches!(n, Node::DataModel(_)))
        .collect::<Vec<_>>();

    assert_eq!(data_models.len(), 1);

    let totalRequests = graph
        .nodes
        .iter()
        .filter(|n| matches!(n, Request))
        .collect::<Vec<_>>();
    let request = totalRequests[0].into_data();
    assert_eq!(request.name, "/Swift");

    assert_eq!(totalRequests.len(), 54, "wrong endpoint count");
}
