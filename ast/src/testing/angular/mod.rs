use crate::lang::graphs::{EdgeType, NodeType};
use crate::lang::{Graph, Node};
use crate::{lang::Lang, repo::Repo};
use anyhow::Ok;
use std::str::FromStr;
use test_log::test;

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

    graph.analysis();

    let (num_nodes, num_edges) = graph.get_graph_size();
    assert_eq!(num_nodes, 112, "Expected 112 nodes");
    assert_eq!(num_edges, 125, "Expected 125 edges");

    let imports = graph.find_nodes_by_type(NodeType::Import);
    assert_eq!(imports.len(), 14, "Expected 14 imports");

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
    assert_eq!(classes.len(), 5, "Expected 5 classes");

    let data_models = graph.find_nodes_by_type(NodeType::DataModel);
    assert_eq!(data_models.len(), 1, "Expected 1 data model");
    assert_eq!(
        data_models[0].name, "Person",
        "Data model name should be 'Person'"
    );

    let functions = graph.find_nodes_by_type(NodeType::Function);
    assert_eq!(functions.len(), 8, "Expected 8 functions");

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
    assert_eq!(imports_edges_count, 12, "Expected 12 imports edges");

    let renders_edges_count = graph.count_edges_of_type(EdgeType::Renders);
    assert_eq!(renders_edges_count, 7, "Expected 7 RENDERS edge");

    let pages = graph.find_nodes_by_type(NodeType::Page);
    assert_eq!(pages.len(), 11, "Expected at least one Page node");

    let index_page_nodes = graph.find_nodes_by_file_ends_with(NodeType::Page, "src/index.html");
    assert_eq!(
        index_page_nodes.len(),
        1,
        "Expected to find the index.html page"
    );
    let index_page = index_page_nodes.first().unwrap();

    let app_component_page_nodes =
        graph.find_nodes_by_file_ends_with(NodeType::Page, "src/app/app.component.html");
    assert_eq!(
        app_component_page_nodes.len(),
        1,
        "Expected to find the app.component.html page"
    );
    let app_component_page = app_component_page_nodes.first().unwrap();

    let app_node = Node::new(NodeType::Page, app_component_page.clone());
    let index_node = Node::new(NodeType::Page, index_page.clone());

    println!("Index page: {:?}", index_page);
    println!("App component page: {:?}", app_component_page);

    let has_render_edge = graph.has_edge(&index_node, &app_node, EdgeType::Renders);
    assert!(
        has_render_edge,
        "Expected index.html to render app.component.html"
    );
    let app_component_page_nodes =
        graph.find_nodes_by_file_ends_with(NodeType::Page, "src/app/app.component.html");
    assert_eq!(
        app_component_page_nodes.len(),
        1,
        "Expected to find the app.component.html page"
    );
    let app_component_page = app_component_page_nodes.first().unwrap();

    let people_list_page_nodes = graph.find_nodes_by_file_ends_with(
        NodeType::Page,
        "src/app/people-list/people-list.component.html",
    );
    assert_eq!(
        people_list_page_nodes.len(),
        1,
        "Expected to find the people-list.component.html page"
    );
    let people_list_page = people_list_page_nodes.first().unwrap();

    let app_component_node = Node::new(NodeType::Page, app_component_page.clone());
    let people_list_node = Node::new(NodeType::Page, people_list_page.clone());

    let has_render_edge = graph.has_edge(&app_component_node, &people_list_node, EdgeType::Renders);
    assert!(
        has_render_edge,
        "Expected app.component.html to render people-list.component.html"
    );

    Ok(())
}

#[test(tokio::test(flavor = "multi_thread", worker_threads = 2))]
async fn test_angular() {
    use crate::lang::graphs::{ArrayGraph, BTreeMapGraph};
    test_angular_generic::<ArrayGraph>().await.unwrap();
    test_angular_generic::<BTreeMapGraph>().await.unwrap();
}
