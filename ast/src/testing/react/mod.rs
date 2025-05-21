use crate::lang::graphs::{EdgeType, NodeType};
use crate::lang::Graph;
use crate::utils::get_use_lsp;
use crate::{lang::Lang, repo::Repo};
use std::str::FromStr;
pub async fn test_react_typescript_generic<G: Graph>() -> Result<(), anyhow::Error> {
    let use_lsp = get_use_lsp();
    let repo = Repo::new(
        "src/testing/react",
        Lang::from_str("tsx").unwrap(),
        use_lsp,
        Vec::new(),
        Vec::new(),
    )
    .unwrap();

    let graph = repo.build_graph_inner::<G>().await?;

    // graph.analysis();

    let (num_nodes, num_edges) = graph.get_graph_size();
    if use_lsp == true {
        assert_eq!(num_nodes, 63, "Expected 63 nodes");
        assert_eq!(num_edges, 85, "Expected 85 edges");
    } else {
        assert_eq!(num_nodes, 57, "Expected 57 nodes");
        assert_eq!(num_edges, 69, "Expected 69 edges");
    }

    fn normalize_path(path: &str) -> String {
        path.replace("\\", "/")
    }

    let language_nodes = graph.find_nodes_by_type(NodeType::Language);
    assert_eq!(language_nodes.len(), 1, "Expected 1 language node");
    assert_eq!(
        language_nodes[0].name, "react",
        "Language node name should be 'react'"
    );
    assert_eq!(
        normalize_path(&language_nodes[0].file),
        "src/testing/react/",
        "Language node file path is incorrect"
    );

    let pkg_files = graph.find_nodes_by_name(NodeType::File, "package.json");
    assert_eq!(pkg_files.len(), 1, "Expected 1 package.json file");
    assert_eq!(
        pkg_files[0].name, "package.json",
        "Package file name is incorrect"
    );

    let imports = graph.find_nodes_by_type(NodeType::Import);
    assert_eq!(imports.len(), 5, "Expected 5 imports");

    let functions = graph.find_nodes_by_type(NodeType::Function);
    if use_lsp == true {
        assert_eq!(functions.len(), 22, "Expected 22 functions/components");
    } else {
        assert_eq!(functions.len(), 16, "Expected 16 functions/components");
    }

    let mut sorted_functions = functions.clone();
    sorted_functions.sort_by(|a, b| a.name.cmp(&b.name));

    assert_eq!(
        sorted_functions[0].name, "App",
        "App component name is incorrect"
    );
    assert_eq!(
        normalize_path(&sorted_functions[0].file),
        "src/testing/react/src/App.tsx",
        "App component file path is incorrect"
    );

    let function_component = functions
        .iter()
        .find(|f| f.name == "FunctionComponent")
        .expect("FunctionComponent not found");
    assert_eq!(
        normalize_path(&function_component.file),
        "src/testing/react/src/ComponentPatterns.tsx",
        "FunctionComponent file path is incorrect"
    );

    let arrow_component = functions
        .iter()
        .find(|f| f.name == "ArrowComponent")
        .expect("ArrowComponent not found");
    assert_eq!(
        normalize_path(&arrow_component.file),
        "src/testing/react/src/ComponentPatterns.tsx",
        "ArrowComponent file path is incorrect"
    );

    let export_arrow_component = functions
        .iter()
        .find(|f| f.name == "ExportArrowComponent")
        .expect("ExportArrowComponent not found");
    assert_eq!(
        normalize_path(&export_arrow_component.file),
        "src/testing/react/src/ComponentPatterns.tsx",
        "ExportArrowComponent file path is incorrect"
    );

    let direct_assignment_component = functions
        .iter()
        .find(|f| f.name == "DirectAssignmentComponent")
        .expect("DirectAssignmentComponent not found");
    assert_eq!(
        normalize_path(&direct_assignment_component.file),
        "src/testing/react/src/ComponentPatterns.tsx",
        "DirectAssignmentComponent file path is incorrect"
    );

    let export_direct_assignment_component = functions
        .iter()
        .find(|f| f.name == "ExportDirectAssignmentComponent")
        .expect("ExportDirectAssignmentComponent not found");
    assert_eq!(
        normalize_path(&export_direct_assignment_component.file),
        "src/testing/react/src/ComponentPatterns.tsx",
        "ExportDirectAssignmentComponent file path is incorrect"
    );

    let submit_button = functions
        .iter()
        .find(|f| f.name == "SubmitButton")
        .expect("SubmitButton component not found");
    assert_eq!(
        submit_button.name, "SubmitButton",
        "SubmitButton component name is incorrect"
    );
    assert_eq!(
        normalize_path(&submit_button.file),
        "src/testing/react/src/components/NewPerson.tsx",
        "SubmitButton component file path is incorrect"
    );

    let requests = graph.find_nodes_by_type(NodeType::Request);
    assert_eq!(requests.len(), 2, "Expected 2 requests");

    let calls_edges_count = graph.count_edges_of_type(EdgeType::Calls);
    assert_eq!(calls_edges_count, 14, "Expected 14 calls edges");

    let pages = graph.find_nodes_by_type(NodeType::Page);
    assert_eq!(pages.len(), 2, "Expected 2 pages");

    let renders_edges_count = graph.count_edges_of_type(EdgeType::Renders);
    assert_eq!(renders_edges_count, 2, "Expected 2 renders edges");

    let people_page = pages
        .iter()
        .find(|p| p.name == "/people")
        .expect("Expected '/people' page not found");
    assert_eq!(people_page.name, "/people", "Page name should be '/people'");
    assert_eq!(
        normalize_path(&people_page.file),
        "src/testing/react/src/App.tsx",
        "Page file path is incorrect"
    );

    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_react_typescript() {
    use crate::lang::graphs::{ArrayGraph, BTreeMapGraph};
    test_react_typescript_generic::<ArrayGraph>().await.unwrap();
    test_react_typescript_generic::<BTreeMapGraph>()
        .await
        .unwrap();
}
