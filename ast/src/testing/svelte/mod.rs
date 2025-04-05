use crate::lang::graph::NodeType;
use crate::lang::ArrayGraph;
use crate::{lang::Lang, repo::Repo};
use std::str::FromStr;
use test_log::test;

#[test(tokio::test)]
async fn test_svelte() {
    let repo = Repo::new(
        "src/testing/svelte",
        Lang::from_str("svelte").unwrap(),
        false,
        Vec::new(),
        Vec::new(),
    )
    .unwrap();

    let graph = repo.build_graph::<ArrayGraph>().await.unwrap();
    assert_eq!(graph.nodes.len(), 43);
    assert_eq!(graph.edges.len(), 42);

    let languages = graph
        .nodes
        .iter()
        .filter(|n| matches!(n.node_type, NodeType::Language))
        .collect::<Vec<_>>();
    assert_eq!(languages.len(), 1);

    let language = languages[0].into_data();
    assert_eq!(language.name, "svelte");
    assert_eq!(language.file, "src/testing/svelte/");

    let files = graph
        .nodes
        .iter()
        .filter(|n| matches!(n.node_type, NodeType::File))
        .collect::<Vec<_>>();

    assert_eq!(files.len(), 7, "wrong file count");

    let imports = graph
        .nodes
        .iter()
        .filter(|n| matches!(n.node_type, NodeType::Import))
        .collect::<Vec<_>>();

    assert_eq!(imports.len(), 7, "wrong import count");

    let classes = graph
        .nodes
        .iter()
        .filter(|n| matches!(n.node_type, NodeType::Class))
        .collect::<Vec<_>>();

    assert_eq!(classes.len(), 3);
    let class = classes[0].into_data();
    assert_eq!(class.body, "");

    let functions = graph
        .nodes
        .iter()
        .filter(|n| matches!(n.node_type, NodeType::Function))
        .collect::<Vec<_>>();
    assert_eq!(functions.len(), 6);
    let func = functions[0].into_data();
    assert_eq!(func.name, "addPerson");

    let data_models = graph
        .nodes
        .iter()
        .filter(|n| matches!(n.node_type, NodeType::DataModel))
        .collect::<Vec<_>>();

    assert_eq!(data_models.len(), 13);

    let total_requests = graph
        .nodes
        .iter()
        .filter(|n| matches!(n.node_type, NodeType::Request))
        .collect::<Vec<_>>();
    let request = total_requests[0].into_data();
    assert_eq!(request.name, "fetchPeople");

    assert_eq!(total_requests.len(), 1, "wrong request count");
}
