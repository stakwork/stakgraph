use crate::lang::graph::NodeType;
use crate::lang::ArrayGraph;
use crate::{lang::Lang, repo::Repo};
use std::str::FromStr;
use test_log::test;

#[test(tokio::test)]
async fn test_swift() {
    let repo = Repo::new(
        "src/testing/swift",
        Lang::from_str("swift").unwrap(),
        false,
        Vec::new(),
        Vec::new(),
    )
    .unwrap();

    let graph = repo.build_graph::<ArrayGraph>().await.unwrap();
    assert_eq!(graph.nodes.len(), 55);
    assert_eq!(graph.edges.len(), 81);

    let languages = graph
        .nodes
        .iter()
        .filter(|n| matches!(n.node_type, NodeType::Language))
        .collect::<Vec<_>>();
    assert_eq!(languages.len(), 1);

    let language = languages[0].into_data();
    assert_eq!(language.name, "swift");
    assert_eq!(language.file, "src/testing/swift/");

    let files = graph
        .nodes
        .iter()
        .filter(|n| matches!(n.node_type, NodeType::File))
        .collect::<Vec<_>>();

    assert_eq!(files.len(), 8, "wrong file count");

    let imports = graph
        .nodes
        .iter()
        .filter(|n| matches!(n.node_type, NodeType::Import))
        .collect::<Vec<_>>();

    assert_eq!(imports.len(), 7, "wrong import count");

    let mut classes = graph
        .nodes
        .iter()
        .filter(|n| matches!(n.node_type, NodeType::Class))
        .collect::<Vec<_>>();

    assert_eq!(classes.len(), 7);

    classes.sort_by(|a, b| a.into_data().name.cmp(&b.into_data().name));

    let class = classes[0].into_data();
    assert_eq!(class.name, "API");

    let mut functions = graph
        .nodes
        .iter()
        .filter(|n| matches!(n.node_type, NodeType::Function))
        .collect::<Vec<_>>();
    assert_eq!(functions.len(), 26);

    functions.sort_by(|a, b| a.into_data().name.cmp(&b.into_data().name));

    let func = functions[0].into_data();

    assert_eq!(func.name, "application");

    let data_models = graph
        .nodes
        .iter()
        .filter(|n| matches!(n.node_type, NodeType::DataModel))
        .collect::<Vec<_>>();

    assert_eq!(data_models.len(), 1);

    let mut total_requests = graph
        .nodes
        .iter()
        .filter(|n| matches!(n.node_type, NodeType::Request))
        .collect::<Vec<_>>();
    let request = total_requests[0].into_data();

    total_requests.sort_by(|a, b| a.into_data().name.cmp(&b.into_data().name));

    assert_eq!(request.name, "/people");

    assert_eq!(total_requests.len(), 2, "wrong endpoint count");
}
