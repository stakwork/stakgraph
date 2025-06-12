use crate::lang::graphs::{EdgeType, NodeType};
use crate::lang::Graph;
use crate::{lang::Lang, repo::Repo};
use anyhow::Ok;
use std::str::FromStr;

pub async fn test_angular_generic<G: Graph>() -> Result<(), anyhow::Error> {
    let repo = Repo::new(
        "src/testing/angular",
        Lang::from_str("angular").unwrap(),
        false,
        Vec::new(),
        Vec::new(),
    )
    .unwrap();

    let graph = repo.build_graph_inner::<G>().await?;

    let (num_nodes, num_edges) = graph.get_graph_size();
    assert_eq!(num_nodes, 94, "Expected 94 nodes");
    assert_eq!(num_edges, 104, "Expected 104 edges");

    let imports = graph.find_nodes_by_type(NodeType::Import);
    assert_eq!(imports.len(), 10, "Expected 10 imports");

    let main_import_body = format!(
        r#"import {{ bootstrapApplication }} from '@angular/platform-browser';
import {{ appConfig }} from './app/app.config';
import {{ AppComponent }} from './app/app.component';"#
    );
    let main = imports
        .iter()
        .find(|i| i.file == "src/testing/angular/src/main.ts")
        .unwrap();

    assert_eq!(
        main.body, main_import_body,
        "Model import body is incorrect"
    );

    let classes = graph.find_nodes_by_type(NodeType::Class);
    assert_eq!(classes.len(), 6, "Expected 6 classes");

    let data_models = graph.find_nodes_by_type(NodeType::DataModel);
    assert_eq!(data_models.len(), 1, "Expected 1 data model");
    assert_eq!(
        data_models[0].name, "Person",
        "Data model name should be 'Person'"
    );

    let functions = graph.find_nodes_by_type(NodeType::Function);
    assert_eq!(functions.len(), 9, "Expected 9 functions");

    let variables = graph.find_nodes_by_type(NodeType::Var);
    assert_eq!(variables.len(), 4, "Expected 4 variables");

    let constructor = functions.iter().find(|f| f.name == "constructor");
    assert!(
        constructor.is_some(),
        "Expected constructor function not found"
    );

    let requests = graph.find_nodes_by_type(NodeType::Request);
    assert_eq!(requests.len(), 7, "Expected 7 requests");

    let calls_edges_count = graph.count_edges_of_type(EdgeType::Calls);
    assert_eq!(calls_edges_count, 8, "Expected 8 calls edges");

    let imports_edges_count = graph.count_edges_of_type(EdgeType::Imports);
    assert_eq!(imports_edges_count, 8, "Expected 8 imports edges");
    
    let renders_edges_count = graph.count_edges_of_type(EdgeType::Renders);
    assert!(renders_edges_count > 0, "Expected at least one RENDERS edge");
    
    let pages = graph.find_nodes_by_type(NodeType::Page);
    assert!(pages.len() > 0, "Expected at least one Page node");
    
    let people_list_html = pages.iter()
        .find(|p| p.file.contains("people-list.component.html"));
    let person_item_html = pages.iter()
        .find(|p| p.file.contains("person-item.component.html"));
    
    assert!(people_list_html.is_some(), "Expected people-list.component.html Page node");
    assert!(person_item_html.is_some(), "Expected person-item.component.html Page node");
    
    let page_renders_page = graph.find_edges_between(
        NodeType::Page, 
        &people_list_html.unwrap().file, 
        NodeType::Page, 
        &person_item_html.unwrap().file
    );
    
    assert!(
        page_renders_page.len() > 0, 
        "Expected Page RENDERS Page relationship between people-list and person-item"
    );

    let app_html = pages.iter()
        .find(|p| p.file.contains("app.component.html"));
    assert!(app_html.is_some(), "Expected app.component.html Page node");

    let router_renders = graph.find_edges_between(
        NodeType::Page,
        &app_html.unwrap().file,
        NodeType::Page,
        &people_list_html.unwrap().file
    );
    assert!(
        router_renders.len() > 0,
        "Expected router-based Page RENDERS Page relationship from app to people-list"
    );

    let add_person_html = pages.iter()
        .find(|p| p.file.contains("add-person.component.html"));
    assert!(add_person_html.is_some(), "Expected add-person.component.html Page node");

    let add_person_renders = graph.find_edges_between(
        NodeType::Page,
        &app_html.unwrap().file,
        NodeType::Page,
        &add_person_html.unwrap().file
    );
    assert!(
        add_person_renders.len() > 0,
        "Expected router-based Page RENDERS Page relationship from app to add-person"
    );

    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_angular() {
    use crate::lang::graphs::{ArrayGraph, BTreeMapGraph};
    test_angular_generic::<ArrayGraph>().await.unwrap();
    test_angular_generic::<BTreeMapGraph>().await.unwrap();
}
