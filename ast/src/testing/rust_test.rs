use crate::lang::graphs::{EdgeType, NodeType};
use crate::lang::Graph;
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

    let (num_nodes, num_edges) = graph.get_graph_size();
    assert_eq!(num_nodes, 49, "Expected 49 nodes");
    assert_eq!(num_edges, 72, "Expected 72 edges");

    let language_nodes = graph.find_nodes_by_type(NodeType::Language);
    assert_eq!(language_nodes.len(), 1, "Expected 1 language node");
    assert_eq!(
        language_nodes[0].name, "rust",
        "Language node name should be 'rust'"
    );
    assert_eq!(
        language_nodes[0].file, "src/testing/rust/",
        "Language node file path is incorrect"
    );
    let files = graph.find_nodes_by_type(NodeType::File);
    assert_eq!(files.len(), 7, "Expected 7 files");

    let imports = graph.find_nodes_by_type(NodeType::Import);
    assert_eq!(imports.len(), 5, "Expected 5 imports");

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
    assert_eq!(vars.len(), 5, "Expected 5 variables");

    let data_models = graph.find_nodes_by_type(NodeType::DataModel);
    assert_eq!(data_models.len(), 2, "Expected 2 data models");

    let endpoints = graph.find_nodes_by_type(NodeType::Endpoint);
    assert_eq!(endpoints.len(), 6, "Expected 6 endpoints");

    let imported_edges = graph.count_edges_of_type(EdgeType::Imports);
    assert_eq!(imported_edges, 4, "Expected 4 import edges");

    let calls_edges = graph.count_edges_of_type(EdgeType::Contains);
    assert_eq!(calls_edges, 62, "Expected 62 contains edges");

    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_rust() {
    #[cfg(feature = "neo4j")]
    use crate::lang::graphs::Neo4jGraph;
    use crate::lang::graphs::{ArrayGraph, BTreeMapGraph};
    test_rust_generic::<ArrayGraph>().await.unwrap();
    test_rust_generic::<BTreeMapGraph>().await.unwrap();

    #[cfg(feature = "neo4j")]
    {
        let mut graph = Neo4jGraph::default();
        graph.clear();
        test_rust_generic::<Neo4jGraph>().await.unwrap();
    }
}
