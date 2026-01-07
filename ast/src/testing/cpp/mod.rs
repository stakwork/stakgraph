use crate::lang::graphs::{EdgeType, NodeType};
use crate::lang::{Graph, Node};
use crate::{lang::Lang, repo::Repo};
use shared::error::Result;
use std::str::FromStr;

pub async fn test_cpp_generic<G: Graph>() -> Result<()> {
    let repo = Repo::new(
        "src/testing/cpp",
        Lang::from_str("cpp").unwrap(),
        false,
        Vec::new(),
        Vec::new(),
    )
    .unwrap();

    let graph = repo.build_graph_inner::<G>().await?;

    graph.analysis();

    let mut nodes = 0;
    let mut edges = 0;

    let language_nodes = graph.find_nodes_by_name(NodeType::Language, "cpp");
    nodes += language_nodes.len();
    assert_eq!(language_nodes.len(), 1, "Expected 1 language node");
    assert_eq!(
        language_nodes[0].name, "cpp",
        "Language node name should be 'cpp'"
    );
    assert!(
        "src/testing/cpp/".contains(language_nodes[0].file.as_str()),
        "Language node file path is incorrect"
    );
    let repositories = graph.find_nodes_by_type(NodeType::Repository);
    nodes += repositories.len();
    assert_eq!(repositories.len(), 1, "Expected 1 repository node");

    let files = graph.find_nodes_by_type(NodeType::File);
    nodes += files.len();
    assert_eq!(files.len(), 6, "Expected 6 files");

    let directories = graph.find_nodes_by_type(NodeType::Directory);
    nodes += directories.len();
    assert_eq!(directories.len(), 0, "Expected 0 directory");

    let imports = graph.find_nodes_by_type(NodeType::Import);
    nodes += imports.len();
    assert_eq!(imports.len(), 5, "Expected 5 imports");

    let main_import_body = format!(
        r#"#include "crow.h"

#include "routes.h"

#include "model.h"
"#
    );
    let main = imports
        .iter()
        .find(|i| i.file == "src/testing/cpp/main.cpp")
        .unwrap();

    assert_eq!(
        main.body, main_import_body,
        "Model import body is incorrect"
    );

    let classes = graph.find_nodes_by_type(NodeType::Class);
    nodes += classes.len();
    assert_eq!(classes.len(), 1, "Expected 1 class");
    assert_eq!(
        classes[0].name, "Database",
        "Class name should be 'Database'"
    );
    assert_eq!(
        classes[0].file, "src/testing/cpp/model.h",
        "Class file path is incorrect"
    );

    let data_models = graph.find_nodes_by_type(NodeType::DataModel);
    nodes += data_models.len();
    assert_eq!(data_models.len(), 1, "Expected 1 data models");
    assert!(
        data_models
            .iter()
            .any(|dm| dm.name == "Person" && dm.file == "src/testing/cpp/model.h"),
        "Expected Person data model not found"
    );

    let endpoints = graph.find_nodes_by_type(NodeType::Endpoint);
    nodes += endpoints.len();
    assert_eq!(endpoints.len(), 2, "Expected 2 endpoints");

    let get_endpoint = endpoints
        .iter()
        .find(|e| e.name == "/person/<int>" && e.meta.get("verb") == Some(&"ANY".to_string()))
        .expect("ANY endpoint not found");
    assert_eq!(get_endpoint.file, "src/testing/cpp/routes.cpp");

    let post_endpoint = endpoints
        .iter()
        .find(|e| e.name == "/person" && e.meta.get("verb") == Some(&"POST".to_string()))
        .expect("POST endpoint not found");
    assert_eq!(post_endpoint.file, "src/testing/cpp/routes.cpp");

    let handler_edges_count = graph.count_edges_of_type(EdgeType::Handler);
    edges += handler_edges_count;
    assert_eq!(handler_edges_count, 2, "Expected 2 handler edges");

    let function_calls = graph.count_edges_of_type(EdgeType::Calls);
    edges += function_calls;
    assert_eq!(function_calls, 3, "Expected 3 function calls");

    let contains = graph.count_edges_of_type(EdgeType::Contains);
    edges += contains;
    assert_eq!(contains, 21, "Expected 21 contains edges");

    let of_edges = graph.count_edges_of_type(EdgeType::Of);
    edges += of_edges;
    assert_eq!(of_edges, 1, "Expected 1 of edge");

    let variables = graph.find_nodes_by_type(NodeType::Var);
    nodes += variables.len();
    assert_eq!(variables.len(), 1, "Expected 1 variables");

    let instances = graph.find_nodes_by_type(NodeType::Instance);
    nodes += instances.len();
    assert_eq!(instances.len(), 1, "Expected 1 instances");

    let libraries = graph.find_nodes_by_type(NodeType::Library);
    nodes += libraries.len();
    assert_eq!(libraries.len(), 1, "Expected 1 libraries");

    let functions = graph.find_nodes_by_type(NodeType::Function);
    nodes += functions.len();
    assert_eq!(functions.len(), 4, "Expected 4 functions");

    let database = classes
        .into_iter()
        .find(|c| c.name == "Database" && c.file == "src/testing/cpp/model.h")
        .map(|c| Node::new(NodeType::Class, c))
        .expect("Database class not found");

    let db = instances
        .into_iter()
        .find(|i| i.name == "db" && i.file == "src/testing/cpp/main.cpp")
        .map(|i| Node::new(NodeType::Instance, i))
        .expect("db instance not found");

    assert!(
        graph.has_edge(&db, &database, EdgeType::Of),
        "Expected 'Database' class to be of 'db' instance"
    );

    let person_data_model = graph
        .find_nodes_by_name(NodeType::DataModel, "Person")
        .into_iter()
        .find(|n| n.file == "src/testing/cpp/model.h")
        .map(|n| Node::new(NodeType::DataModel, n))
        .expect("Person DataModel not found in model.h");

    let database_class = graph
        .find_nodes_by_name(NodeType::Class, "Database")
        .into_iter()
        .find(|n| n.file == "src/testing/cpp/model.h")
        .map(|n| Node::new(NodeType::Class, n))
        .expect("Database class not found in model.h");

    let main_fn = graph
        .find_nodes_by_name(NodeType::Function, "main")
        .into_iter()
        .find(|n| n.file == "src/testing/cpp/main.cpp")
        .map(|n| Node::new(NodeType::Function, n))
        .expect("main function not found in main.cpp");

    let setup_routes_fn = graph
        .find_nodes_by_name(NodeType::Function, "setup_routes")
        .into_iter()
        .find(|n| n.file == "src/testing/cpp/routes.cpp")
        .map(|n| Node::new(NodeType::Function, n))
        .expect("setup_routes function not found in routes.cpp");

    let post_endpoint = graph
        .find_nodes_by_name(NodeType::Endpoint, "/person")
        .into_iter()
        .find(|n| {
            n.file == "src/testing/cpp/routes.cpp"
                && n.meta.get("verb") == Some(&"POST".to_string())
        })
        .map(|n| Node::new(NodeType::Endpoint, n))
        .expect("POST /person endpoint not found in routes.cpp");
    let new_person_fn = graph
        .find_nodes_by_name(NodeType::Function, "new_person")
        .into_iter()
        .find(|n| n.file == "src/testing/cpp/routes.cpp")
        .map(|n| Node::new(NodeType::Function, n))
        .expect("new_person function not found in routes.cpp");

    let model_h_file = graph
        .find_nodes_by_name(NodeType::File, "model.h")
        .into_iter()
        .find(|n| n.file == "src/testing/cpp/model.h")
        .map(|n| Node::new(NodeType::File, n))
        .expect("model.h file node not found");

    assert!(
        graph.has_edge(&model_h_file, &person_data_model, EdgeType::Contains),
        "Expected 'Database' class to contain 'Person' DataModel"
    );
    assert!(
        graph.has_edge(&model_h_file, &database_class, EdgeType::Contains),
        "Expected 'Database' class to contain 'Person' DataModel"
    );
    assert!(
        graph.has_edge(&main_fn, &setup_routes_fn, EdgeType::Calls),
        "Expected 'main' function to call 'setup_routes' function"
    );
    assert!(
        graph.has_edge(&post_endpoint, &new_person_fn, EdgeType::Handler),
        "Expected '/person' endpoint to be handled by 'new_person'"
    );

    let (num_nodes, num_edges) = graph.get_graph_size();

    assert_eq!(
        num_nodes, nodes as u32,
        "Expected {} nodes found {}",
        nodes, num_nodes
    );
    assert_eq!(
        num_edges, edges as u32,
        "Expected {} edges found {}",
        edges, num_edges
    );

    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_cpp() {
    use crate::lang::graphs::{ArrayGraph, BTreeMapGraph};
    test_cpp_generic::<ArrayGraph>().await.unwrap();
    test_cpp_generic::<BTreeMapGraph>().await.unwrap();

    #[cfg(feature = "neo4j")]
    {
        use crate::lang::graphs::Neo4jGraph;
        std::env::set_var("NEO4J_TEST_NAMESPACE", "test_lang_cpp");
        let graph = Neo4jGraph::with_namespace("test_lang_cpp");
        graph.clear().await.unwrap();
        test_cpp_generic::<Neo4jGraph>().await.unwrap();
        std::env::remove_var("NEO4J_TEST_NAMESPACE");
    }
}
