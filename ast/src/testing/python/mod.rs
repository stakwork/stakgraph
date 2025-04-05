use crate::lang::graph::{EdgeType, NodeType};
use crate::lang::ArrayGraph;
use crate::{lang::Lang, repo::Repo};
use std::str::FromStr;
use test_log::test;

#[test(tokio::test)]
async fn test_python() {
    let repo = Repo::new(
        "src/testing/python",
        Lang::from_str("python").unwrap(),
        false,
        Vec::new(),
        Vec::new(),
    )
    .unwrap();

    let graph = repo.build_graph::<ArrayGraph>().await.unwrap();
    assert_eq!(graph.nodes.len(), 60);
    assert_eq!(graph.edges.len(), 79);

    let languages = graph
        .nodes
        .iter()
        .filter(|n| matches!(n.node_type, NodeType::Language))
        .collect::<Vec<_>>();
    assert_eq!(languages.len(), 1);

    let language = languages[0].into_data();
    assert_eq!(language.name, "python");
    assert_eq!(language.file, "src/testing/python/");

    let files = graph
        .nodes
        .iter()
        .filter(|n| matches!(n.node_type, NodeType::File))
        .collect::<Vec<_>>();

    assert_eq!(files.len(), 17, "wrong file count");

    let imports = graph
        .nodes
        .iter()
        .filter(|n| matches!(n.node_type, NodeType::Import))
        .collect::<Vec<_>>();

    assert_eq!(imports.len(), 12, "wrong import count");

    let classes = graph
        .nodes
        .iter()
        .filter(|n| matches!(n.node_type, NodeType::Class))
        .collect::<Vec<_>>();

    assert_eq!(classes.len(), 3);

    let class = classes[0].into_data();
    assert_eq!(class.name, "Person");
    assert_eq!(class.file, "src/testing/python/model.py");

    let methods = graph
        .edges
        .iter()
        .filter(|e| matches!(e.edge, EdgeType::Operand) && e.source.node_type == NodeType::Class)
        .collect::<Vec<_>>();
    assert_eq!(methods.len(), 2);

    let data_models = graph
        .nodes
        .iter()
        .filter(|n| matches!(n.node_type, NodeType::DataModel))
        .collect::<Vec<_>>();
    //Data models are zero because they are just classes in python
    assert_eq!(data_models.len(), 3);

    let endpoints = graph
        .nodes
        .iter()
        .filter(|n| matches!(n.node_type, NodeType::Endpoint))
        .collect::<Vec<_>>();

    assert_eq!(endpoints.len(), 4, "wrong endpoint count");
}
