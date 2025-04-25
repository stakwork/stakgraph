use crate::lang::graphs::{EdgeType, NodeType};
use crate::lang::Graph;
use crate::utils::get_use_lsp;
use crate::{lang::Lang, repo::Repo};
use std::str::FromStr;
use tracing_test::traced_test;

async fn test_component_styles_generic<G: Graph>() -> Result<(), anyhow::Error> {
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
    graph.analysis();

    fn normalize_path(path: &str) -> String {
        path.replace("\\", "/")
    }

    let functions = graph.find_nodes_by_type(NodeType::Function);
    
    let regular_component = functions
        .iter()
        .find(|f| f.name == "RegularComponent")
        .expect("RegularComponent not found");
    assert_eq!(
        regular_component.name, "RegularComponent",
        "Regular function component name is incorrect"
    );
    assert_eq!(
        normalize_path(&regular_component.file),
        "src/testing/react/src/components/ComponentStyles.tsx",
        "Regular function component file path is incorrect"
    );

    let arrow_component = functions
        .iter()
        .find(|f| f.name == "ArrowComponent")
        .expect("ArrowComponent not found");
    assert_eq!(
        arrow_component.name, "ArrowComponent",
        "Arrow function component name is incorrect"
    );

    let direct_assignment = functions
        .iter()
        .find(|f| f.name == "DirectAssignmentComponent")
        .expect("DirectAssignmentComponent not found");
    assert_eq!(
        direct_assignment.name, "DirectAssignmentComponent",
        "Direct assignment component name is incorrect"
    );

    let exported_function = functions
        .iter()
        .find(|f| f.name == "ExportedFunctionComponent")
        .expect("ExportedFunctionComponent not found");
    assert_eq!(
        exported_function.name, "ExportedFunctionComponent",
        "Exported function component name is incorrect"
    );

    let exported_arrow = functions
        .iter()
        .find(|f| f.name == "ExportedArrowComponent")
        .expect("ExportedArrowComponent not found");
    assert_eq!(
        exported_arrow.name, "ExportedArrowComponent",
        "Exported arrow component name is incorrect"
    );

    Ok(())
}

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

    //graph.analysis();

    let (num_nodes, num_edges) = graph.get_graph_size();
    if use_lsp == true {
        assert_eq!(num_nodes, 56, "Expected 56 nodes");
        assert_eq!(num_edges, 77, "Expected 77 edges");
    } else {
        assert_eq!(num_nodes, 56, "Expected 56 nodes");
        assert_eq!(num_edges, 72, "Expected 72 edges");
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

    assert_eq!(
        sorted_functions[1].name, "ArrowComponent",
        "ArrowComponent component name is incorrect"
    );
    assert_eq!(
        normalize_path(&sorted_functions[1].file),
        "src/testing/react/src/components/ComponentStyles.tsx",
        "ArrowComponent component file path is incorrect"
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

    let calls_edges_count = graph.count_edges_of_type(EdgeType::Calls(Default::default()));
    assert_eq!(calls_edges_count, 18, "Expected 18 calls edges");

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

#[tokio::test(flavor = "multi_thread", worker_threads = 10)]
#[traced_test]
async fn test_component_styles() {
    use crate::lang::graphs::{ArrayGraph, BTreeMapGraph};
    test_component_styles_generic::<ArrayGraph>().await.unwrap();
    test_component_styles_generic::<BTreeMapGraph>()
        .await
        .unwrap();
}
