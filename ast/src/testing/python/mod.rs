use crate::lang::graph::{EdgeType, Node, NodeType};
use crate::{lang::Lang, repo::Repo};
use std::str::FromStr;

#[tokio::test]
async fn test_python() {
    crate::utils::logger();
    let repo = Repo::new(
        "src/testing/python",
        Lang::from_str("python").unwrap(),
        false,
        Vec::new(),
        Vec::new(),
    )
    .unwrap();
    let graph = repo.build_graph().await.unwrap();


    let l = graph
        .nodes
        .iter()
        .filter(|n| matches!(n, Node::Language(_)))
        .collect::<Vec<_>>();
    assert_eq!(l.len(), 1);
    let l = l[0].into_data();
    assert_eq!(l.name, "python");
    assert_eq!(l.file, "src/testing/python/");

    let files = graph
        .nodes
        .iter()
        .filter(|n| matches!(n, Node::File(_)))
        .collect::<Vec<_>>();
    for f in files {
        println!("file: {:?}", f.into_data().name);
    }

    let imports = graph
        .nodes
        .iter()
        .filter(|n| matches!(n, Node::Import(_)))
        .collect::<Vec<_>>();


    let classes = graph
        .nodes
        .iter()
        .filter(|n| matches!(n, Node::Class(_)))
        .collect::<Vec<_>>();

    let class = classes[0].into_data();
    assert_eq!(class.name, "Database");
    assert_eq!(class.file, "src/testing/python/db.py");

    let methods = graph
        .edges
        .iter()
        .filter(|e| matches!(e.edge, EdgeType::Operand) && e.source.node_type == NodeType::Class)
        .collect::<Vec<_>>();


    let dms = graph
        .nodes
        .iter()
        .filter(|n| matches!(n, Node::DataModel(_)))
        .collect::<Vec<_>>();

    let dm = dms[0].into_data();
    assert_eq!(dm.name, "Database");
    assert_eq!(dm.file, "src/testing/python/db.py");

    let endpoints = graph
        .nodes
        .iter()
        .filter(|n| matches!(n, Node::Endpoint(_)))
        .collect::<Vec<_>>();

    let end = endpoints[0].into_data();
    assert_eq!(end.name, "/persons");
    assert_eq!(end.file, "src/testing/python/routes.py");
    assert_eq!(end.meta.get("verb").unwrap(), "GET");

    let end = endpoints[1].into_data();
    assert_eq!(end.name, "/greet/{name}");
    assert_eq!(end.file, "src/testing/python/routes.py");
    assert_eq!(end.meta.get("verb").unwrap(), "POST");
}
