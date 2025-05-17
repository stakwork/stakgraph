use crate::lang::graphs::{EdgeType, NodeType};
use crate::lang::Graph;
use crate::{lang::Lang, repo::Repo};
use std::str::FromStr;
use test_log::test;

fn normalize_path(path: &str) -> String {
    path.replace('\\', "/")
}

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
    assert_eq!(num_nodes, 61, "Expected 61 nodes");
    assert_eq!(num_edges, 99, "Expected 99 edges");

    let language_nodes = graph.find_nodes_by_type(NodeType::Language);
    assert_eq!(language_nodes.len(), 1, "Expected 1 language node");
    assert_eq!(
        language_nodes[0].name, "ruby",
        "Language node name should be 'ruby'"
    );
    assert_eq!(
        normalize_path(&language_nodes[0].file), "src/testing/ruby/",
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
    assert_eq!(
        normalize_path(&get_person_endpoint.file), "src/testing/ruby/config/routes.rb",
        "Endpoint file path is incorrect"
    );

    let post_person_endpoint = endpoints
        .iter()
        .find(|e| e.name == "person" && e.meta.get("verb") == Some(&"POST".to_string()))
        .expect("POST person endpoint not found");
    assert_eq!(
        normalize_path(&post_person_endpoint.file), "src/testing/ruby/config/routes.rb",
        "Endpoint file path is incorrect"
    );

    let delete_people_endpoint = endpoints
        .iter()
        .find(|e| e.name == "/people/:id" && e.meta.get("verb") == Some(&"DELETE".to_string()))
        .expect("DELETE /people/:id endpoint not found");
    assert_eq!(
        normalize_path(&delete_people_endpoint.file), "src/testing/ruby/config/routes.rb",
        "Endpoint file path is incorrect"
    );

    let get_articles_endpoint = endpoints
        .iter()
        .find(|e| e.name == "/people/articles" && e.meta.get("verb") == Some(&"GET".to_string()))
        .expect("GET /people/articles endpoint not found");
    assert_eq!(
        normalize_path(&get_articles_endpoint.file), "src/testing/ruby/config/routes.rb",
        "Endpoint file path is incorrect"
    );

    let post_articles_endpoint = endpoints
        .iter()
        .find(|e| {
            e.name == "/people/:id/articles" && e.meta.get("verb") == Some(&"POST".to_string())
        })
        .expect("POST /people/:id/articles endpoint not found");
    assert_eq!(
        normalize_path(&post_articles_endpoint.file), "src/testing/ruby/config/routes.rb",
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
        normalize_path(&post_countries_endpoint.file), "src/testing/ruby/config/routes.rb",
        "Endpoint file path is incorrect"
    );

    let handler_edges_count = graph.count_edges_of_type(EdgeType::Handler);
    assert_eq!(handler_edges_count, 7, "Expected 7 handler edges");

    Ok(())
}

pub async fn test_ruby_lsp_generic<G: Graph>() -> Result<(), anyhow::Error> {
    let repo = Repo::new(
        "src/testing/ruby",
        Lang::from_str("ruby").unwrap(),
        true,
        Vec::new(),
        Vec::new(),
    )
    .unwrap();

    let graph = repo.build_graph_inner::<G>().await?;

    graph.analysis();

    let (num_nodes, num_edges) = graph.get_graph_size();
    assert!(num_nodes > 61, "Expected more than 61 nodes with LSP");
    assert!(num_edges > 99, "Expected more than 99 edges with LSP");

    let language_nodes = graph.find_nodes_by_type(NodeType::Language);
    assert_eq!(language_nodes.len(), 1, "Expected 1 language node");
    assert_eq!(
        language_nodes[0].name, "ruby",
        "Language node name should be 'ruby'"
    );
    
    let pkg_files = graph.find_nodes_by_name(NodeType::File, "Gemfile");
    assert_eq!(pkg_files.len(), 1, "Expected 1 Gemfile");

    let endpoints = graph.find_nodes_by_type(NodeType::Endpoint);
    assert_eq!(endpoints.len(), 7, "Expected 7 endpoints");

    let classes = graph.find_nodes_by_type(NodeType::Class);
    assert!(classes.len() >= 3, "Expected at least 3 classes with LSP");

    let _people_controller = classes
        .iter()
        .find(|c| c.name.contains("PeopleController"))
        .expect("PeopleController class not found");
    
    let methods = graph.find_nodes_by_type(NodeType::Function);
    assert!(methods.len() >= 5, "Expected at least 5 methods with LSP");

    let _create_method = methods
        .iter()
        .find(|m| m.name == "create")
        .expect("Create method not found");
    
    let _show_method = methods
        .iter()
        .find(|m| m.name == "show")
        .expect("Show method not found");
    
    let calls_edges_count = graph.count_edges_of_type(EdgeType::Calls(Default::default()));
    assert!(calls_edges_count > 0, "Expected function call edges with LSP");

    let imports_edges_count = graph.count_edges_of_type(EdgeType::Imports);
    assert!(imports_edges_count > 0, "Expected import edges with LSP");

    let models = graph.find_nodes_by_type(NodeType::DataModel);
    assert!(models.len() > 0, "Expected data models with LSP");

    Ok(())
}

#[test(tokio::test)]
async fn test_ruby() {
    use crate::lang::graphs::{ArrayGraph, BTreeMapGraph};
    test_ruby_generic::<ArrayGraph>().await.unwrap();
    test_ruby_generic::<BTreeMapGraph>().await.unwrap();
}

#[test(tokio::test)]
async fn test_ruby_lsp() {
    use crate::lang::graphs::{ArrayGraph, BTreeMapGraph};
    
    if std::env::var("SKIP_LSP_TESTS").is_ok() {
        println!("Skipping Ruby LSP test due to SKIP_LSP_TESTS environment variable");
        return;
    }
    
    match test_ruby_lsp_generic::<ArrayGraph>().await {
        Ok(_) => println!("Ruby LSP test with ArrayGraph passed"),
        Err(e) => {
            if e.to_string().contains("sending on a closed channel") {
                println!("Ruby LSP test skipped: LSP server might not be available: {}", e);
                return;
            } else {
                panic!("Ruby LSP test with ArrayGraph failed: {}", e);
            }
        }
    }
    
    match test_ruby_lsp_generic::<BTreeMapGraph>().await {
        Ok(_) => println!("Ruby LSP test with BTreeMapGraph passed"),
        Err(e) => {
            if e.to_string().contains("sending on a closed channel") {
                println!("Ruby LSP test skipped: LSP server might not be available: {}", e);
            } else {
                panic!("Ruby LSP test with BTreeMapGraph failed: {}", e);
            }
        }
    }
}
