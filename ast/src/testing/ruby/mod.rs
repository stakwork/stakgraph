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

    let (num_nodes, num_edges) = graph.get_graph_size();
    assert_eq!(num_nodes, 55, "Expected 55 nodes");
    assert_eq!(num_edges, 79, "Expected 79 edges");

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
    assert_eq!(endpoints.len(), 6, "Expected 6 endpoints");

    let mut sorted_endpoints = endpoints.clone();
    sorted_endpoints.sort_by(|a, b| a.name.cmp(&b.name));

    let get_person_endpoint = endpoints
        .iter()
        .find(|e| e.name == "person/:id" && e.meta.get("verb") == Some(&"GET".to_string()))
        .expect("GET person/:id endpoint not found");
    assert_eq!(
        get_person_endpoint.file, "src/testing/ruby/config/routes.rb",
        "Endpoint file path is incorrect"
    );

    let post_person_endpoint = endpoints
        .iter()
        .find(|e| e.name == "person" && e.meta.get("verb") == Some(&"POST".to_string()))
        .expect("POST person endpoint not found");
    assert_eq!(
        post_person_endpoint.file, "src/testing/ruby/config/routes.rb",
        "Endpoint file path is incorrect"
    );

    let delete_people_endpoint = endpoints
        .iter()
        .find(|e| e.name == "/people/:id" && e.meta.get("verb") == Some(&"DELETE".to_string()))
        .expect("DELETE /people/:id endpoint not found");
    assert_eq!(
        delete_people_endpoint.file, "src/testing/ruby/config/routes.rb",
        "Endpoint file path is incorrect"
    );

    let get_articles_endpoint = endpoints
        .iter()
        .find(|e| e.name == "/people/articles" && e.meta.get("verb") == Some(&"GET".to_string()))
        .expect("GET /people/articles endpoint not found");
    assert_eq!(
        get_articles_endpoint.file, "src/testing/ruby/config/routes.rb",
        "Endpoint file path is incorrect"
    );

    let post_articles_endpoint = endpoints
        .iter()
        .find(|e| {
            e.name == "/people/:id/articles" && e.meta.get("verb") == Some(&"POST".to_string())
        })
        .expect("POST /people/:id/articles endpoint not found");
    assert_eq!(
        post_articles_endpoint.file, "src/testing/ruby/config/routes.rb",
        "Endpoint file path is incorrect"
    );

    let post_countries_endpoint = endpoints
        .iter()
        .find(|e| {
            e.name == "/countries/:country_id/process"
                && e.meta.get("verb") == Some(&"POST".to_string())
        })
        .expect("POST /countries/:country_id/process endpoint not found");
    assert_eq!(
        post_countries_endpoint.file, "src/testing/ruby/config/routes.rb",
        "Endpoint file path is incorrect"
    );

    let handler_edges_count = graph.count_edges_of_type(EdgeType::Handler);
    assert_eq!(handler_edges_count, 6, "Expected 6 handler edges");

    Ok(())
}

#[test(tokio::test)]
async fn test_ruby() {
    use crate::lang::graphs::{ArrayGraph, BTreeMapGraph};
    test_ruby_generic::<ArrayGraph>().await.unwrap();
    test_ruby_generic::<BTreeMapGraph>().await.unwrap();
}
