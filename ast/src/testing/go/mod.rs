use crate::lang::graphs::{EdgeType, NodeType};
use crate::lang::Graph;
use crate::utils::get_use_lsp;
use crate::{lang::Lang, repo::Repo};
use std::str::FromStr;

pub async fn test_go_generic<G: Graph>() -> Result<(), anyhow::Error> {
    let use_lsp = get_use_lsp();
    let repo = Repo::new(
        "src/testing/go",
        Lang::from_str("go").unwrap(),
        use_lsp,
        Vec::new(),
        Vec::new(),
    )
    .unwrap();

    let graph = repo.build_graph_inner::<G>().await?;

    graph.analysis();

    let (num_nodes, num_edges) = graph.get_graph_size().await;
    if use_lsp == true {
        assert_eq!(num_nodes, 64, "Expected 64 nodes");
        assert_eq!(num_edges, 93, "Expected 93 edges");
    } else {
        assert_eq!(num_nodes, 30, "Expected 30 nodes");
        assert_eq!(num_edges, 47, "Expected 47 edges");
    }

    let language_nodes = graph.find_nodes_by_name(NodeType::Language, "go").await;
    assert_eq!(language_nodes.len(), 1, "Expected 1 language node");
    assert_eq!(
        language_nodes[0].name, "go",
        "Language node name should be 'go'"
    );
    assert!(
        "src/testing/go/".contains(language_nodes[0].file.as_str()),
        "Language node file path is incorrect"
    );

    let imports = graph.find_nodes_by_type(NodeType::Import).await;
    assert_eq!(imports.len(), 3, "Expected 3 imports");

    // Find classes
    let classes = graph.find_nodes_by_type(NodeType::Class).await;
    assert_eq!(classes.len(), 1, "Expected 1 class");
    assert_eq!(
        classes[0].name, "database",
        "Class name should be 'database'"
    );
    assert_eq!(
        classes[0].file, "src/testing/go/db.go",
        "Class file path is incorrect"
    );

    let class_function_edges = graph
        .find_nodes_with_edge_type(NodeType::Class, NodeType::Function, EdgeType::Operand)
        .await;
    assert_eq!(class_function_edges.len(), 4, "Expected 4 methods");

    let data_models = graph.find_nodes_by_type(NodeType::DataModel).await;
    assert_eq!(data_models.len(), 2, "Expected 2 data models");
    assert!(
        data_models
            .iter()
            .any(|dm| dm.name == "Person" && dm.file == "src/testing/go/db.go"),
        "Expected Person data model not found"
    );

    let endpoints = graph.find_nodes_by_type(NodeType::Endpoint).await;
    assert_eq!(endpoints.len(), 2, "Expected 2 endpoints");

    let get_endpoint = endpoints
        .iter()
        .find(|e| e.name == "/person/{id}" && e.meta.get("verb") == Some(&"GET".to_string()))
        .expect("GET endpoint not found");
    assert_eq!(get_endpoint.file, "src/testing/go/routes.go");

    let post_endpoint = endpoints
        .iter()
        .find(|e| e.name == "/person" && e.meta.get("verb") == Some(&"POST".to_string()))
        .expect("POST endpoint not found");
    assert_eq!(post_endpoint.file, "src/testing/go/routes.go");

    let handler_edges_count = graph.count_edges_of_type(EdgeType::Handler).await;
    assert_eq!(handler_edges_count, 2, "Expected 2 handler edges");

    let function_calls = graph.count_edges_of_type(EdgeType::Calls).await;
    assert_eq!(function_calls, 6, "Expected 6 function calls");

    let operands = graph.count_edges_of_type(EdgeType::Operand).await;
    assert_eq!(operands, 4, "Expected 4 operands");

    let of = graph.count_edges_of_type(EdgeType::Of).await;
    assert_eq!(of, 1, "Expected 1 of edges");

    if use_lsp {
        let contains = graph.count_edges_of_type(EdgeType::Contains).await;
        assert_eq!(contains, 36, "Expected 36 contains edges with lsp");
    } else {
        let contains = graph.count_edges_of_type(EdgeType::Contains).await;
        assert_eq!(contains, 34, "Expected 34 contains edges");
    }

    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_go() {
    #[cfg(feature = "neo4j")]
    use crate::lang::graphs::Neo4jGraph;
    use crate::lang::graphs::{ArrayGraph, BTreeMapGraph};
    test_go_generic::<ArrayGraph>().await.unwrap();
    test_go_generic::<BTreeMapGraph>().await.unwrap();
    #[cfg(feature = "neo4j")]
    {
        let mut graph = Neo4jGraph::default();
        graph.clear();
        test_go_generic::<Neo4jGraph>().await.unwrap();
    }
}
