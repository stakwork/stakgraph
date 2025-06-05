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
    assert_eq!(num_nodes, 67, "Expected 67 nodes");
    assert_eq!(num_edges, 110, "Expected 110 edges");

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

    let imports = graph.find_nodes_by_type(NodeType::Import);
    assert_eq!(imports.len(), 6, "Expected 6 import node");

    for imp in &imports {
        println!("Import: {} in file: {}", imp.name, imp.file);
    }
    let import_body = imports
        .iter()
        .find(|i| i.file == "src/testing/ruby/config/environment.rb")
        .expect("Import body not found");
    let environment_body = format!(r#"require_relative "application""#,);

    assert_eq!(
        import_body.body, environment_body,
        "Import body is incorrect"
    );

    let endpoints = graph.find_nodes_by_type(NodeType::Endpoint);
    assert_eq!(endpoints.len(), 7, "Expected 7 endpoints");

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
    assert_eq!(handler_edges_count, 7, "Expected 7 handler edges");

    let class_counts = graph.count_edges_of_type(EdgeType::ParentOf);
    assert_eq!(class_counts, 6, "Expected 6 class edges");

    let class_calls =
        graph.find_nodes_with_edge_type(NodeType::Class, NodeType::Class, EdgeType::Calls);

    assert_eq!(class_calls.len(), 1, "Expected 1 class calls edges");

    let import_edges = graph.count_edges_of_type(EdgeType::Imports);
    assert_eq!(import_edges, 4, "Expected 4 import edges");

    let person_to_article_call = class_calls.iter().any(|(src, dst)| {
        (src.name == "Person" && dst.name == "Article")
            || (src.name == "Article" && dst.name == "Person")
    });
    assert!(
        person_to_article_call,
        "Expects a Person -> CALLS -> Article Class Call Edge"
    );

    Ok(())
}

#[test(tokio::test(flavor = "multi_thread", worker_threads = 2))]
async fn test_ruby() {
    #[cfg(feature = "neo4j")]
    use crate::lang::graphs::Neo4jGraph;
    use crate::lang::graphs::{ArrayGraph, BTreeMapGraph};
    test_ruby_generic::<ArrayGraph>().await.unwrap();
    test_ruby_generic::<BTreeMapGraph>().await.unwrap();

    #[cfg(feature = "neo4j")]
    {
        let mut graph = Neo4jGraph::default();
        graph.clear();
        test_ruby_generic::<Neo4jGraph>().await.unwrap();
    }
}
