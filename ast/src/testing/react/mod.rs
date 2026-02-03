use crate::lang::graphs::{EdgeType, NodeType};
use crate::lang::{Graph, Node};
// use crate::utils::get_use_lsp;
use crate::{lang::Lang, repo::Repo};
use shared::error::Result;
use std::str::FromStr;

pub async fn test_react_typescript_generic<G: Graph>() -> Result<()> {
    // TODO: LSP mode needs additional work for React TypeScript - disabled for now
    let use_lsp = false; // get_use_lsp();
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

    let mut nodes_count = 0;
    let mut edges_count = 0;

    fn normalize_path(path: &str) -> String {
        path.replace("\\", "/")
    }

    let language_nodes = graph.find_nodes_by_type(NodeType::Language);
    nodes_count += language_nodes.len();
    assert_eq!(language_nodes.len(), 1, "Expected 1 language node");
    assert_eq!(
        language_nodes[0].name, "typescript",
        "Language node name should be 'typescript'"
    );
    assert_eq!(
        normalize_path(&language_nodes[0].file),
        "src/testing/react",
        "Language node file path is incorrect"
    );

    let repository = graph.find_nodes_by_type(NodeType::Repository);
    nodes_count += repository.len();
    assert_eq!(repository.len(), 1, "Expected 1 repository node");

    let libraries = graph.find_nodes_by_type(NodeType::Library);
    nodes_count += libraries.len();
    assert_eq!(libraries.len(), 27, "Expected 27 library nodes");

    let pkg_files = graph.find_nodes_by_name(NodeType::File, "package.json");
    assert_eq!(pkg_files.len(), 1, "Expected 1 package.json file");
    assert_eq!(
        pkg_files[0].name, "package.json",
        "Package file name is incorrect"
    );
    assert!(
        pkg_files[0].body.contains("react"),
        "package.json should contain react dependency"
    );
    assert!(
        pkg_files[0].body.contains("react-dom"),
        "package.json should contain react-dom dependency"
    );
    assert!(
        pkg_files[0].body.contains("react-router-dom"),
        "package.json should contain react-router-dom dependency"
    );
    assert!(
        pkg_files[0].body.contains("typescript"),
        "package.json should contain typescript dependency"
    );

    let imports = graph.find_nodes_by_type(NodeType::Import);
    nodes_count += imports.len();
    for imp in &imports {
        let import_lines: Vec<&str> = imp
            .body
            .lines()
            .filter(|line| line.trim_start().starts_with("import "))
            .collect();

        assert!(
            !import_lines.is_empty(),
            "Expected multiple import lines in {}",
            imp.file
        );
    }
    let import_test_file = imports
        .iter()
        .find(|imp| imp.file == "src/testing/react/src/App.tsx")
        .unwrap();

    let app_body = format!(
        r#"import React, {{useEffect}} from "react";
import {{ BrowserRouter as Router, Route, Routes }} from "react-router-dom";
import "./App.css";
import People from "./components/People";
import NewPerson from "./components/NewPerson";"#
    );

    assert_eq!(import_test_file.body, app_body, "Body of App is incorrect");
    assert_eq!(imports.len(), 19, "Expected 19 imports");

    let people_import = imports
        .iter()
        .find(|imp| imp.file == "src/testing/react/src/components/People.tsx")
        .expect("People.tsx import not found");
    assert!(
        people_import.body.contains("import { useEffect }"),
        "People.tsx should import useEffect from react"
    );
    assert!(
        people_import.body.contains("Person, useStore"),
        "People.tsx should import Person and useStore from ./Person"
    );

    let new_person_import = imports
        .iter()
        .find(|imp| imp.file == "src/testing/react/src/components/NewPerson.tsx")
        .expect("NewPerson.tsx import not found");
    assert!(
        new_person_import.body.contains("import { useState }"),
        "NewPerson.tsx should import useState from react"
    );

    let person_import = imports
        .iter()
        .find(|imp| imp.file == "src/testing/react/src/components/Person.tsx")
        .expect("Person.tsx import not found");
    assert!(
        person_import.body.contains("import { useState"),
        "Person.tsx should import useState and useCallback from react"
    );

    let functions = graph.find_nodes_by_type(NodeType::Function);
    nodes_count += functions.len();
    if use_lsp == true {
        assert_eq!(functions.len(), 22, "Expected 22 functions/components");
    } else {
        assert_eq!(functions.len(), 61, "Expected 61 functions/components");
    }

    let classes = graph.find_nodes_by_type(NodeType::Class);
    nodes_count += classes.len();
    assert_eq!(classes.len(), 4, "Expected 4 classes");

    let class = classes
        .iter()
        .find(|c| c.name == "TestThing")
        .expect("TestThing class not found");
    assert_eq!(
        normalize_path(&class.file),
        "src/testing/react/src/App.tsx",
        "TestThing class file path is incorrect"
    );
    assert_eq!(
        class.body.contains("constructor"),
        true,
        "TestThing class should have a constructor"
    );

    let mut sorted_functions = functions.clone();
    sorted_functions.sort_by(|a, b| a.name.cmp(&b.name));

    let app_function = sorted_functions
        .iter()
        .find(|f| f.name == "App")
        .expect("App function not found");
    assert_eq!(app_function.name, "App", "App component name is incorrect");
    assert_eq!(
        normalize_path(&app_function.file),
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

    let app_function = functions
        .iter()
        .find(|f| f.name == "App")
        .expect("App component not found");
    assert!(
        app_function.body.contains("Router"),
        "App should use Router (BrowserRouter alias)"
    );
    assert!(
        app_function.body.contains("Routes"),
        "App should define Routes"
    );
    assert!(
        app_function.body.contains("Route"),
        "App should define Route components"
    );

    let people_function = functions
        .iter()
        .find(|f| {
            f.name == "People"
                && normalize_path(&f.file) == "src/testing/react/src/components/People.tsx"
        })
        .expect("People component not found");
    assert!(
        people_function.body.contains("useStore"),
        "People component should use useStore hook"
    );
    assert!(
        people_function.body.contains("useEffect"),
        "People component should use useEffect hook"
    );

    let new_person_function = functions
        .iter()
        .find(|f| f.name == "NewPerson")
        .expect("NewPerson component not found");
    assert!(
        new_person_function.body.contains("useState"),
        "NewPerson component should use useState"
    );
    assert!(
        new_person_function.body.contains("SubmitButton"),
        "NewPerson component should render SubmitButton"
    );

    let use_store_function = functions
        .iter()
        .find(|f| {
            f.name == "useStore"
                && normalize_path(&f.file) == "src/testing/react/src/components/Person.tsx"
        })
        .expect("useStore hook not found");
    let use_store_fn = Node::new(NodeType::Function, use_store_function.clone());
    assert!(
        use_store_function.body.contains("useState"),
        "useStore should use useState hook"
    );
    assert!(
        use_store_function.body.contains("initialState"),
        "useStore should reference initialState"
    );

    let function_component_fn = functions
        .iter()
        .find(|f| f.name == "FunctionComponent")
        .expect("FunctionComponent not found");
    assert!(
        function_component_fn.body.contains("return"),
        "FunctionComponent should have return statement"
    );

    let arrow_component_fn = functions
        .iter()
        .find(|f| f.name == "ArrowComponent")
        .expect("ArrowComponent not found");
    assert!(
        arrow_component_fn.body.contains("=>"),
        "ArrowComponent should be arrow function"
    );

    let requests = graph.find_nodes_by_type(NodeType::Request);
    nodes_count += requests.len();
    assert_eq!(requests.len(), 14, "Expected 14 requests");

    let pages = graph.find_nodes_by_type(NodeType::Page);
    nodes_count += pages.len();
    assert_eq!(pages.len(), 4, "Expected 4 pages");

    let new_person_page = pages
        .iter()
        .find(|p| p.name == "/new-person")
        .expect("'/new-person' page not found");
    assert_eq!(
        new_person_page.name, "/new-person",
        "Page name should be '/new-person'"
    );
    assert_eq!(
        normalize_path(&new_person_page.file),
        "src/testing/react/src/App.tsx",
        "Page file path is incorrect"
    );

    let variables = graph.find_nodes_by_type(NodeType::Var);
    nodes_count += variables.len();
    assert_eq!(variables.len(), 8, "Expected 8 variables");

    let initial_state_var = variables
        .iter()
        .find(|v| v.name == "initialState")
        .map(|n| Node::new(NodeType::Var, n.clone()))
        .expect("initialState variable not found");
    assert!(
        initial_state_var.node_data.body.contains("people: []"),
        "initialState should have empty people array"
    );
    let app_name_var = variables
        .iter()
        .find(|v| v.name == "AppName" && normalize_path(&v.file) == "src/testing/react/src/App.tsx")
        .expect("AppName variable not found");
    assert_eq!(
        app_name_var.docs,
        Some("The name of the application".to_string()),
        "AppName should have documentation"
    );

    let name_var = variables.iter().find(|v| {
        v.name == "name"
            && normalize_path(&v.file) == "src/testing/react/src/components/NewPerson.tsx"
    });
    if let Some(var) = name_var {
        assert!(
            var.body.contains("useState"),
            "name variable should use useState"
        );
    }

    let email_var = variables.iter().find(|v| {
        v.name == "email"
            && normalize_path(&v.file) == "src/testing/react/src/components/NewPerson.tsx"
    });
    if let Some(var) = email_var {
        assert!(
            var.body.contains("useState"),
            "email variable should use useState"
        );
    }

    let renders_edges_count = graph.count_edges_of_type(EdgeType::Renders);
    edges_count += renders_edges_count;
    assert_eq!(renders_edges_count, 4, "Expected 4 renders edges");

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

    let people_fn = functions
        .iter()
        .find(|f| {
            f.name == "People"
                && normalize_path(&f.file) == "src/testing/react/src/components/People.tsx"
        })
        .map(|n| Node::new(NodeType::Function, n.clone()))
        .expect("People component not found");
    let new_person_fn = functions
        .iter()
        .find(|f| {
            f.name == "NewPerson"
                && normalize_path(&f.file) == "src/testing/react/src/components/NewPerson.tsx"
        })
        .map(|n| Node::new(NodeType::Function, n.clone()))
        .expect("NewPerson component not found");
    let submit_button_fn = functions
        .iter()
        .find(|f| {
            f.name == "SubmitButton"
                && normalize_path(&f.file) == "src/testing/react/src/components/NewPerson.tsx"
        })
        .map(|n| Node::new(NodeType::Function, n.clone()))
        .expect("SubmitButton component not found");
    let people_page = pages
        .iter()
        .find(|p| p.name == "/people" && normalize_path(&p.file) == "src/testing/react/src/App.tsx")
        .map(|n| Node::new(NodeType::Page, n.clone()))
        .expect("'/people' page not found");

    assert!(
        graph.has_edge(&people_page, &people_fn, EdgeType::Renders),
        "Expected '/people' page to render People component"
    );
    assert!(
        graph.has_edge(&new_person_fn, &submit_button_fn, EdgeType::Calls),
        "Expected NewPerson component to call SubmitButton component"
    );

    let data_models = graph.find_nodes_by_type(NodeType::DataModel);
    nodes_count += data_models.len();
    assert_eq!(data_models.len(), 25, "Expected 25 data models");

    let person_data_model = data_models
        .iter()
        .find(|dm| dm.name == "Person")
        .expect("Person DataModel not found");
    assert!(
        person_data_model.body.contains("id"),
        "Person DataModel should have id field"
    );
    assert!(
        person_data_model.body.contains("name"),
        "Person DataModel should have name field"
    );
    assert!(
        person_data_model.body.contains("email"),
        "Person DataModel should have email field"
    );

    let store_state_data_model = data_models
        .iter()
        .find(|dm| dm.name == "StoreState")
        .expect("StoreState DataModel not found");
    assert!(
        store_state_data_model.body.contains("people"),
        "StoreState DataModel should have people field"
    );
    assert!(
        store_state_data_model.body.contains("Person[]"),
        "StoreState DataModel should have Person array type"
    );

    assert!(
        graph.has_edge(&use_store_fn, &initial_state_var, EdgeType::Contains),
        "Expected useStore to contain initialState variable"
    );

    let new_person_page_node = pages
        .iter()
        .find(|p| p.name == "/new-person")
        .map(|n| Node::new(NodeType::Page, n.clone()))
        .expect("'/new-person' page not found");

    assert!(
        graph.has_edge(&new_person_page_node, &new_person_fn, EdgeType::Renders),
        "Expected '/new-person' page to render NewPerson component"
    );

    let contains_edges_count = graph.count_edges_of_type(EdgeType::Contains);
    edges_count += contains_edges_count;
    assert_eq!(
        contains_edges_count, 223,
        "Expected 223 contains edges, got {}",
        contains_edges_count
    );

    let of_edges = graph.count_edges_of_type(EdgeType::Of);
    edges_count += of_edges;
    assert_eq!(of_edges, 1, "Expected 1 of edges");

    let calls = graph.count_edges_of_type(EdgeType::Calls);
    edges_count += calls;

    let imports = graph.count_edges_of_type(EdgeType::Imports);
    edges_count += imports;
    assert_eq!(imports, 5, "Expected 5 imports edges");

    let operand_edges = graph.count_edges_of_type(EdgeType::Operand);
    edges_count += operand_edges;
    assert_eq!(operand_edges, 1, "Expected 1 operand edges");

    let endpoints = graph.find_nodes_by_type(NodeType::Endpoint);

    nodes_count += endpoints.len();
    assert_eq!(endpoints.len(), 9, "Expected 9 endpoints");
    let users_get_endpoint = endpoints
        .iter()
        .find(|e| {
            e.name == "/users"
                && normalize_path(&e.file) == "src/testing/react/src/api/routes.ts"
                && e.meta.get("verb") == Some(&"GET".to_string())
        })
        .expect("/users endpoint not found");
    assert_eq!(
        users_get_endpoint.docs,
        Some("GET endpoint".to_string()),
        "/users endpoint should have documentation"
    );

    let unit_tests = graph.find_nodes_by_type(NodeType::UnitTest);
    nodes_count += unit_tests.len();
    assert_eq!(unit_tests.len(), 3, "Expected 3 unit tests");

    let integration_tests = graph.find_nodes_by_type(NodeType::IntegrationTest);
    nodes_count += integration_tests.len();
    assert_eq!(integration_tests.len(), 2, "Expected 2 integration tests");

    let e2e_tests = graph.find_nodes_by_type(NodeType::E2eTest);
    nodes_count += e2e_tests.len();
    assert_eq!(e2e_tests.len(), 2, "Expected 2 e2e tests");

    let file_nodes = graph.find_nodes_by_type(NodeType::File);
    nodes_count += file_nodes.len();
    let tsx_files = file_nodes
        .iter()
        .filter(|f| f.name.ends_with(".tsx"))
        .count();
    assert_eq!(tsx_files, 12, "Expected 12 TSX files, got {}", tsx_files);

    let component_pattern_functions = functions
        .iter()
        .filter(|f| normalize_path(&f.file) == "src/testing/react/src/ComponentPatterns.tsx")
        .count();
    assert_eq!(
        component_pattern_functions, 5,
        "Expected at least 5 component patterns, got {}",
        component_pattern_functions
    );

    let directories = graph.find_nodes_by_type(NodeType::Directory);
    nodes_count += directories.len();
    assert_eq!(directories.len(), 14, "Expected 14 directories");

    let uses = graph.count_edges_of_type(EdgeType::Uses);
    edges_count += uses;

    let nested_in = graph.count_edges_of_type(EdgeType::NestedIn);
    edges_count += nested_in;
    assert_eq!(nested_in, 2, "Expected 2 NestedIn edges");

    let handlers = graph.count_edges_of_type(EdgeType::Handler);

    edges_count += handlers;
    assert_eq!(handlers, 9, "Expected 9 handler edges");

    if use_lsp {}

    let (nodes, edges) = graph.get_graph_size();

    assert_eq!(
        nodes as usize, nodes_count,
        "Expected {nodes} found {nodes_count} nodes"
    );
    assert_eq!(
        edges as usize, edges_count,
        "Expected {edges} found {edges_count} edges"
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

    #[cfg(feature = "neo4j")]
    {
        use crate::lang::graphs::Neo4jGraph;
        let graph = Neo4jGraph::default();
        graph.clear().await.unwrap();
        test_react_typescript_generic::<Neo4jGraph>().await.unwrap();
    }
}
