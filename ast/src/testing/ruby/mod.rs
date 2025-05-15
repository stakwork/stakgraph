use crate::lang::graphs::{EdgeType, NodeType};
use crate::lang::Graph;
use crate::{lang::Lang, repo::Repo};
use std::str::FromStr;
use test_log::test;

pub async fn test_ruby_generic<G: Graph>() -> Result<(), anyhow::Error> {
    let repo = Repo::new(
        "src/testing/ruby",
        Lang::from_str("ruby").unwrap(),
        false,
        Vec::new(),
        Vec::new(),
    )
    .unwrap();

    let graph = repo.build_graph_inner::<G>().await?;

    graph.analysis();

    let (num_nodes, num_edges) = graph.get_graph_size();
    assert!(num_nodes >= 58 && num_nodes <= 61, "Expected between 58-61 nodes, got {}", num_nodes);
    assert!(num_edges >= 88 && num_edges <= 99, "Expected between 88-99 edges, got {}", num_edges);

    let language_nodes = graph.find_nodes_by_type(NodeType::Language);
    assert_eq!(language_nodes.len(), 1, "Expected 1 language node");
    assert_eq!(
        language_nodes[0].name, "ruby",
        "Language node name should be 'ruby'"
    );
    assert_eq!(
        language_nodes[0].file, "src/testing/ruby/",
        "Language node file path is incorrect"
    );

    let pkg_files = graph.find_nodes_by_name(NodeType::File, "Gemfile");
    assert_eq!(pkg_files.len(), 1, "Expected 1 Gemfile");
    assert_eq!(
        pkg_files[0].name, "Gemfile",
        "Package file name is incorrect"
    );

    let endpoints = graph.find_nodes_by_type(NodeType::Endpoint);
    assert_eq!(endpoints.len(), 7, "Expected 7 endpoints");

    let mut sorted_endpoints = endpoints.clone();
    sorted_endpoints.sort_by(|a, b| a.name.cmp(&b.name));

    let get_person_endpoint = endpoints
        .iter()
        .find(|e| e.name == "person/:id" && e.meta.get("verb") == Some(&"GET".to_string()))
        .expect("GET person/:id endpoint not found");
    assert!(
        get_person_endpoint.file.ends_with("routes.rb"),
        "Endpoint file path is incorrect"
    );

    let post_person_endpoint = endpoints
        .iter()
        .find(|e| e.name == "person" && e.meta.get("verb") == Some(&"POST".to_string()))
        .expect("POST person endpoint not found");
    assert!(
        post_person_endpoint.file.ends_with("routes.rb"),
        "Endpoint file path is incorrect"
    );

    let delete_people_endpoint = endpoints
        .iter()
        .find(|e| e.name == "/people/:id" && e.meta.get("verb") == Some(&"DELETE".to_string()))
        .expect("DELETE /people/:id endpoint not found");
    assert!(
        delete_people_endpoint.file.ends_with("routes.rb"),
        "Endpoint file path is incorrect"
    );

    let get_articles_endpoint = endpoints
        .iter()
        .find(|e| e.name == "/people/articles" && e.meta.get("verb") == Some(&"GET".to_string()))
        .expect("GET /people/articles endpoint not found");
    assert!(
        get_articles_endpoint.file.ends_with("routes.rb"),
        "Endpoint file path is incorrect"
    );

    let post_articles_endpoint = endpoints
        .iter()
        .find(|e| {
            e.name == "/people/:id/articles" && e.meta.get("verb") == Some(&"POST".to_string())
        })
        .expect("POST /people/:id/articles endpoint not found");
    assert!(
        post_articles_endpoint.file.ends_with("routes.rb"),
        "Endpoint file path is incorrect"
    );

    let post_countries_endpoint = endpoints
        .iter()
        .find(|e| {
            e.name == "/countries/:country_id/process"
                && e.meta.get("verb") == Some(&"POST".to_string())
        })
        .expect("POST /countries/:country_id/process endpoint not found");
    assert!(
        post_countries_endpoint.file.ends_with("routes.rb"),
        "Endpoint file path is incorrect"
    );

    let handler_edges_count = graph.count_edges_of_type(EdgeType::Handler);
    assert_eq!(handler_edges_count, 7, "Expected 7 handler edges");

    let pages = graph.find_nodes_by_type(NodeType::Page);
    assert_eq!(pages.len(), 1, "Expected 1 page node");
    
    let show_person_profile = pages
        .iter()
        .find(|p| p.name.ends_with("show_person_profile.erb"))
        .expect("show_person_profile.erb page not found");
    
    let file_path_lower = show_person_profile.file.to_lowercase().replace("\\", "/");
    assert!(
        file_path_lower.contains("/views/people/show_person_profile.erb"),
        "Page file path is incorrect"
    );
    
    let renders_edges_count = graph.count_edges_of_type(EdgeType::Renders);
    assert!(
        renders_edges_count > 0,
        "Expected at least one RENDERS edge"
    );
    
    let functions = graph.find_nodes_by_name(NodeType::Function, "show_person_profile");
    assert!(!functions.is_empty(), "Function show_person_profile not found");
    
    let function_renders = renders_edges_count > 0 && !functions.is_empty();
    assert!(function_renders, "Expected a RENDERS edge from page to function");

    Ok(())
}

#[test(tokio::test)]
async fn test_ruby() {
    use crate::lang::graphs::{ArrayGraph, BTreeMapGraph};
    test_ruby_generic::<ArrayGraph>().await.unwrap();
    test_ruby_generic::<BTreeMapGraph>().await.unwrap();
}
