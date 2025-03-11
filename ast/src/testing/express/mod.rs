use crate::lang::graph::{EdgeType, Node, NodeType};
use crate::{lang::Lang, repo::Repo};
use std::str::FromStr;

#[tokio::test]
async fn test_express() {

    crate::utils::logger();

    let repo = Repo::new(
        "src/testing/express",
        Lang::from_str("ts").unwrap(),
        false,
        Vec::new(),
        Vec::new(),
    )
    .unwrap();

    let graph = repo.build_graph().await.unwrap();

    assert!(graph.nodes.len() == 31);
    assert!(graph.edges.len() == 31);


    fn normalize_path(path: &str) -> String {
        path.replace("\\", "/")
    }

    let l = graph
        .nodes
        .iter()
        .filter(|n| matches!(n, Node::Language(_)))
        .collect::<Vec<_>>();
    assert_eq!(l.len(), 1);
    let l = l[0].into_data();
    assert_eq!(l.name, "typescript");
    assert_eq!(normalize_path(&l.file), "src/testing/express/");


    let pkg_file = graph
        .nodes
        .iter()
        .filter(|n| matches!(n, Node::File(_)) && n.into_data().name == "package.json")
        .collect::<Vec<_>>();
    assert_eq!(pkg_file.len(), 2);
    let pkg_file = pkg_file[0].into_data();
    assert_eq!(pkg_file.name, "package.json");


    let imports = graph
        .nodes
        .iter()
        .filter(|n| matches!(n, Node::Import(_)))
        .collect::<Vec<_>>();
    assert_eq!(imports.len(), 3);


    let functions = graph
        .nodes
        .iter()
        .filter(|n| matches!(n, Node::Function(_)))
        .collect::<Vec<_>>();
    assert_eq!(functions.len(), 1);

    let home_route_handler = functions[0].into_data();
    assert_eq!(home_route_handler.name, "constructor");
    assert_eq!(normalize_path(&home_route_handler.file), "src/testing/express/src/models/TypeORMModel.ts");


    let requests = graph
        .nodes
        .iter()
        .filter(|n| matches!(n, Node::Request(_)))
        .collect::<Vec<_>>();
    assert_eq!(requests.len(), 0);


    let calls_edges = graph
        .edges
        .iter()
        .filter(|e| matches!(e.edge, EdgeType::Calls(_)))
        .collect::<Vec<_>>();
    assert_eq!(calls_edges.len(), 0);

}
