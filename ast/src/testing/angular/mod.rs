use crate::lang::graph::{EdgeType, Node};
use crate::lang::ArrayGraph;
use crate::{lang::Lang, repo::Repo};
use std::str::FromStr;
use test_log::test;

#[test(tokio::test)]
async fn test_angular() {
    let repo = Repo::new(
        "src/testing/angular",
        Lang::from_str("angular").unwrap(),
        false,
        Vec::new(),
        Vec::new(),
    )
    .unwrap();
    let graph = repo.build_graph::<ArrayGraph>().await.unwrap();
    assert!(graph.nodes.len() == 77);
    assert!(graph.edges.len() == 78);

    let l = graph
        .nodes
        .iter()
        .filter(|n| matches!(n, Node::Language(_)))
        .collect::<Vec<_>>();
    assert_eq!(l.len(), 1);
    let l = l[0].into_data();
    assert_eq!(l.name, "angular");
    assert_eq!(l.file, "src/testing/angular/");

    let imports = graph
        .nodes
        .iter()
        .filter(|n| matches!(n, Node::Import(_)))
        .collect::<Vec<_>>();
    assert_eq!(imports.len(), 10);

    let cls = graph
        .nodes
        .iter()
        .filter(|n| matches!(n, Node::Class(_)))
        .collect::<Vec<_>>();
    assert_eq!(cls.len(), 5);

    let models = graph
        .nodes
        .iter()
        .filter(|n| matches!(n, Node::DataModel(_)))
        .collect::<Vec<_>>();
    assert_eq!(models.len(), 1);
    let models = models[0].into_data();
    assert_eq!(models.name, "Person");

    let functions = graph
        .nodes
        .iter()
        .filter(|n| matches!(n, Node::Function(_)))
        .collect::<Vec<_>>();
    assert_eq!(functions.len(), 8);
    let functions = functions[0].into_data();
    assert_eq!(functions.name, "constructor");

    let reqs = graph
        .nodes
        .iter()
        .filter(|n| matches!(n, Node::Request(_)))
        .collect::<Vec<_>>();
    assert_eq!(reqs.len(), 8);
    let reqs = reqs[0].into_data();
    assert_eq!(reqs.name, "this.people.push");

    let calls_edges = graph
        .edges
        .iter()
        .filter(|e| matches!(e.edge, EdgeType::Calls(_)))
        .collect::<Vec<_>>();
    assert_eq!(calls_edges.len(), 8);
}
