use crate::lang::graphs::{EdgeType, NodeType};
use crate::lang::{Graph, Node};
use crate::{lang::Lang, repo::Repo};
use std::str::FromStr;

pub async fn test_rust_generic<G: Graph>() -> Result<(), anyhow::Error> {
    let repo = Repo::new(
        "src/testing/rust",
        Lang::from_str("rust").unwrap(),
        false,
        Vec::new(),
        Vec::new(),
    )
    .unwrap();

    let graph = repo.build_graph_inner::<G>().await?;

    let mut nodes_count = 0;
    let mut edges_count = 0;

    let language_nodes = graph.find_nodes_by_type(NodeType::Language);
    nodes_count += language_nodes.len();
    assert_eq!(language_nodes.len(), 1, "Expected 1 language node");
    assert_eq!(
        language_nodes[0].name, "rust",
        "Language node name should be 'rust'"
    );
    assert_eq!(
        language_nodes[0].file, "src/testing/rust",
        "Language node file path is incorrect"
    );

    let repositories = graph.find_nodes_by_type(NodeType::Repository);
    nodes_count += repositories.len();
    assert_eq!(repositories.len(), 1, "Expected 1 repository node");

    let directories = graph.find_nodes_by_type(NodeType::Directory);
    nodes_count += directories.len();
    assert_eq!(directories.len(), 2, "Expected 2 directory nodes");

    let files = graph.find_nodes_by_type(NodeType::File);
    nodes_count += files.len();
    assert_eq!(files.len(), 8, "Expected 8 files");

    let imports = graph.find_nodes_by_type(NodeType::Import);
    nodes_count += imports.len();
    assert_eq!(imports.len(), 5, "Expected 5 imports");

    let libraries = graph.find_nodes_by_type(NodeType::Library);
    nodes_count += libraries.len();
    //should be ~9
    assert_eq!(libraries.len(), 9, "Expected 9 library nodes");

    let main_import_body = format!(
        r#"use crate::db::init_db;
use crate::routes::{{
    actix_routes::config, axum_routes::create_router, rocket_routes::create_rocket,
}};

use anyhow::Result;
use std::net::SocketAddr;"#
    );
    let main = imports
        .iter()
        .find(|i| i.file == "src/testing/rust/src/main.rs")
        .unwrap();

    assert_eq!(
        main.body, main_import_body,
        "Model import body is incorrect"
    );

    let vars = graph.find_nodes_by_type(NodeType::Var);
    nodes_count += vars.len();
    assert_eq!(vars.len(), 5, "Expected 5 variables");

    let data_models = graph.find_nodes_by_type(NodeType::DataModel);
    nodes_count += data_models.len();
    assert_eq!(data_models.len(), 2, "Expected 2 data models");

    let endpoints = graph.find_nodes_by_type(NodeType::Endpoint);
    nodes_count += endpoints.len();
    assert_eq!(endpoints.len(), 6, "Expected 6 endpoints");

    let imported_edges = graph.count_edges_of_type(EdgeType::Imports);
    edges_count += imported_edges;
    assert_eq!(imported_edges, 4, "Expected 4 import edges");

    let calls_edges = graph.count_edges_of_type(EdgeType::Contains);
    edges_count += calls_edges;
    assert_eq!(calls_edges, 71, "Expected 71 contains edges");

    let functions = graph.find_nodes_by_type(NodeType::Function);
    nodes_count += functions.len();
    assert_eq!(functions.len(), 19, "Expected 19 functions");

    let handlers = graph.count_edges_of_type(EdgeType::Handler);
    edges_count += handlers;
    assert_eq!(handlers, 6, "Expected 6 handler edges");

    let get_person_fn = functions
        .iter()
        .find(|f| f.name == "get_person" && f.file.ends_with("src/routes/actix_routes.rs"))
        .map(|n| Node::new(NodeType::Function, n.clone()))
        .expect("get_person function not found in actix_routes.rs");

    let create_person_fn = functions
        .iter()
        .find(|f| f.name == "create_person" && f.file.ends_with("src/routes/actix_routes.rs"))
        .map(|n| Node::new(NodeType::Function, n.clone()))
        .expect("create_person function not found in actix_routes.rs");

    let endpoints = graph.find_nodes_by_type(NodeType::Endpoint);

    let get_person_endpoint = endpoints
        .iter()
        .find(|e| e.name == "/person/{id}" && e.file.ends_with("src/routes/actix_routes.rs"))
        .map(|n| Node::new(NodeType::Endpoint, n.clone()))
        .expect("GET /person/{id} endpoint not found in actix_routes.rs");

    let post_person_endpoint = endpoints
        .iter()
        .find(|e| e.name == "/person" && e.file.ends_with("src/routes/actix_routes.rs"))
        .map(|n| Node::new(NodeType::Endpoint, n.clone()))
        .expect("POST /person endpoint not found in actix_routes.rs");

    assert!(
        graph.has_edge(&get_person_endpoint, &get_person_fn, EdgeType::Handler),
        "Expected '/person/id' endpoint to be handled by get_person"
    );

    assert!(
        graph.has_edge(&post_person_endpoint, &create_person_fn, EdgeType::Handler),
        "Expected '/person' endpoint to be handled by create_person"
    );

    let (nodes, edges) = graph.get_graph_size();
    assert_eq!(
        nodes as usize, nodes_count,
        "Expected {} nodes, found {}",
        nodes_count, nodes
    );
    assert_eq!(
        edges as usize, edges_count,
        "Expected {} edges, found {}",
        edges_count, edges
    );
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_rust() {
    use crate::lang::graphs::{ArrayGraph, BTreeMapGraph};
    test_rust_generic::<ArrayGraph>().await.unwrap();
    test_rust_generic::<BTreeMapGraph>().await.unwrap();

    #[cfg(feature = "neo4j")]
    {
        use crate::lang::graphs::Neo4jGraph;
        let graph = Neo4jGraph::default();
        graph.clear().await.unwrap();
        test_rust_generic::<Neo4jGraph>().await.unwrap();
    }
}
