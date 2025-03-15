use crate::lang::graph::{EdgeType, Node, NodeType};
use crate::{lang::Lang, repo::Repo};
use std::str::FromStr;
use test_log::test;

#[test(tokio::test)]
async fn test_go() {
    let repo = Repo::new(
        "src/testing/go",
        Lang::from_str("go").unwrap(),
        false,
        Vec::new(),
        Vec::new(),
    )
    .unwrap();
    let graph = repo.build_graph().await.unwrap();
    // println!("graph: {:?}", graph);
    assert!(graph.nodes.len() == 40);
    assert!(graph.edges.len() == 58);

    let l = graph
        .nodes
        .iter()
        .filter(|n| matches!(n, Node::Language(_)))
        .collect::<Vec<_>>();
    assert_eq!(l.len(), 1);
    let l = l[0].into_data();
    assert_eq!(l.name, "go");
    assert_eq!(l.file, "src/testing/go/");

    let files = graph
        .nodes
        .iter()
        .filter(|n| matches!(n, Node::File(_)))
        .collect::<Vec<_>>();
    for f in files {
        println!("file: {:?}", f.into_data().name);
    }
    // FIXME go.mod is counted twice
    // assert_eq!(files.len(), 5);

    let imports = graph
        .nodes
        .iter()
        .filter(|n| matches!(n, Node::Import(_)))
        .collect::<Vec<_>>();
    assert_eq!(imports.len(), 3);

    let cls = graph
        .nodes
        .iter()
        .filter(|n| matches!(n, Node::Class(_)))
        .collect::<Vec<_>>();
    assert_eq!(cls.len(), 1);
    let cls = cls[0].into_data();
    assert_eq!(cls.name, "database");
    assert_eq!(cls.file, "src/testing/go/db.go");

    let methods = graph
        .edges
        .iter()
        .filter(|e| matches!(e.edge, EdgeType::Operand) && e.source.node_type == NodeType::Class)
        .collect::<Vec<_>>();
    assert_eq!(methods.len(), 4);

    let dms = graph
        .nodes
        .iter()
        .filter(|n| matches!(n, Node::DataModel(_)))
        .collect::<Vec<_>>();
    assert_eq!(dms.len(), 2);

    let dm = dms[1].into_data();
    assert_eq!(dm.name, "Person");
    assert_eq!(dm.file, "src/testing/go/db.go");

    let ends = graph
        .nodes
        .iter()
        .filter(|n| matches!(n, Node::Endpoint(_)))
        .collect::<Vec<_>>();
    assert_eq!(ends.len(), 2);

    let end = ends[0].into_data();
    assert_eq!(end.name, "/person/{id}");
    assert_eq!(end.file, "src/testing/go/routes.go");
    assert_eq!(end.meta.get("verb").unwrap(), "GET");

    let end = ends[1].into_data();
    assert_eq!(end.name, "/person");
    assert_eq!(end.file, "src/testing/go/routes.go");
    assert_eq!(end.meta.get("verb").unwrap(), "POST");

    // get handler edges
    let edges = graph
        .edges
        .iter()
        .filter(|e| matches!(e.edge, EdgeType::Handler))
        .collect::<Vec<_>>();
    assert_eq!(edges.len(), 2);
}
