use crate::lang::graph::{EdgeType, Node};
use crate::{lang::Lang, repo::Repo};
use std::str::FromStr;
use test_log::test;

#[test(tokio::test)]
async fn test_react_typescript() {
    let repo = Repo::new(
        "src/testing/react",
        Lang::from_str("tsx").unwrap(),
        false,
        Vec::new(),
        Vec::new(),
    )
    .unwrap();

    let graph = repo.build_graph().await.unwrap();

    assert!(graph.nodes.len() == 69);
    assert!(graph.edges.len() == 78);

    // Function to normalize paths and replace backslashes with forward slashes
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
    assert_eq!(l.name, "react");
    assert_eq!(normalize_path(&l.file), "src/testing/react/");

    let pkg_file = graph
        .nodes
        .iter()
        .filter(|n| matches!(n, Node::File(_)) && n.into_data().name == "package.json")
        .collect::<Vec<_>>();
    assert_eq!(pkg_file.len(), 1);
    let pkg_file = pkg_file[0].into_data();
    assert_eq!(pkg_file.name, "package.json");

    let imports = graph
        .nodes
        .iter()
        .filter(|n| matches!(n, Node::Import(_)))
        .collect::<Vec<_>>();
    assert_eq!(imports.len(), 4);

    let mut functions = graph
        .nodes
        .iter()
        .filter(|n| matches!(n, Node::Function(_)))
        .collect::<Vec<_>>();

    functions.sort_by(|a, b| a.into_data().name.cmp(&b.into_data().name));

    assert_eq!(functions.len(), 11);

    let people_component = functions[0].into_data();
    assert_eq!(people_component.name, "App");
    assert_eq!(
        normalize_path(&people_component.file),
        "src/testing/react/src/App.tsx"
    );

    let new_person_component = functions[1].into_data();
    assert_eq!(new_person_component.name, "FormContainer");
    assert_eq!(
        normalize_path(&new_person_component.file),
        "src/testing/react/src/components/NewPerson.tsx"
    );

    let styled_components = graph
        .nodes
        .iter()
        .filter(|n| matches!(n, Node::Function(_)) && n.into_data().name == "SubmitButton")
        .collect::<Vec<_>>();

    assert_eq!(styled_components.len(), 1);

    let styled_component = styled_components[0].into_data();
    assert_eq!(styled_component.name, "SubmitButton");
    assert_eq!(
        normalize_path(&styled_component.file),
        "src/testing/react/src/components/NewPerson.tsx"
    );

    let requests = graph
        .nodes
        .iter()
        .filter(|n| matches!(n, Node::Request(_)))
        .collect::<Vec<_>>();
    assert_eq!(requests.len(), 3);

    let calls_edges = graph
        .edges
        .iter()
        .filter(|e| matches!(e.edge, EdgeType::Calls(_)))
        .collect::<Vec<_>>();
    assert_eq!(calls_edges.len(), 14);

    let page_node = graph
        .nodes
        .iter()
        .filter(|n| matches!(n, Node::Page(_)))
        .collect::<Vec<_>>();
    assert_eq!(page_node.len(), 2);

    let page = page_node[0].into_data();
    assert_eq!(page.name, "/");
    assert_eq!(normalize_path(&page.file), "src/testing/react/src/App.tsx");
}
