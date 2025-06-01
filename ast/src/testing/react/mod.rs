use crate::lang::graphs::{EdgeType, NodeType};
use crate::lang::Graph;
use crate::utils::get_use_lsp;
use crate::{lang::Lang, repo::Repo};
use std::str::FromStr;
pub async fn test_react_typescript_generic<G: Graph>() -> Result<(), anyhow::Error> {
    let use_lsp = get_use_lsp("react");
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

    let (num_nodes, num_edges) = graph.get_graph_size();
    if use_lsp == true {
        assert_eq!(num_nodes, 67, "Expected 67 nodes");
        assert_eq!(num_edges, 109, "Expected 109 edges");
    } else {
        assert_eq!(num_nodes, 61, "Expected 61 nodes");
        assert_eq!(num_edges, 74, "Expected 74 edges");
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
    for imp in &imports {
        let import_lines: Vec<&str> = imp
            .body
            .lines()
            .filter(|line| line.trim_start().starts_with("import "))
            .collect();

        assert!(
            import_lines.len() > 0,
            "Expected multiple import lines in {}",
            imp.file
        );
    }
    let import_test_file = imports
        .iter()
        .find(|imp| imp.file == "src/testing/react/src/App.tsx")
        .unwrap();

    let app_body = format!(
        r#"import React from "react";
import {{ BrowserRouter as Router, Route, Routes }} from "react-router-dom";
import "./App.css";
import People from "./components/People";
import NewPerson from "./components/NewPerson";"#
    );

    assert_eq!(import_test_file.body, app_body, "Body of App is incorrect");
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
    assert_eq!(calls_edges_count, 11, "Expected 11 calls edges");

    let pages = graph.find_nodes_by_type(NodeType::Page);
    assert_eq!(pages.len(), 2, "Expected 2 pages");

    let variables = graph.find_nodes_by_type(NodeType::Var);
    assert_eq!(variables.len(), 5, "Expected 5 variables");

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
    #[cfg(feature = "neo4j")]
    use crate::lang::graphs::Neo4jGraph;
    use crate::lang::graphs::{ArrayGraph, BTreeMapGraph};
    test_react_typescript_generic::<ArrayGraph>().await.unwrap();
    test_react_typescript_generic::<BTreeMapGraph>()
        .await
        .unwrap();

    #[cfg(feature = "neo4j")]
    {
        let mut graph = Neo4jGraph::default();
        graph.clear();
        test_react_typescript_generic::<Neo4jGraph>().await.unwrap();
    }
}
