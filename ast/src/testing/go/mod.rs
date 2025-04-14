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

    let (num_nodes, num_edges) = graph.get_graph_size();
    if use_lsp == true {
        assert_eq!(num_nodes, 64, "Expected 64 nodes");
        assert_eq!(num_edges, 108, "Expected 108 edges");
    } else {
        assert_eq!(num_nodes, 30, "Expected 30 nodes");
        assert_eq!(num_edges, 48, "Expected 48 edges");
    }

    let language_nodes = graph.find_nodes_by_name(NodeType::Language, "go");
    assert_eq!(language_nodes.len(), 1, "Expected 1 language node");
    assert_eq!(
        language_nodes[0].name, "go",
        "Language node name should be 'go'"
    );
    assert_eq!(
        language_nodes[0].file, "src/testing/go/",
        "Language node file path is incorrect"
    );

    let imports = graph.find_nodes_by_type(NodeType::Import);
    assert_eq!(imports.len(), 3, "Expected 3 imports");

    // Find classes
    let classes = graph.find_nodes_by_type(NodeType::Class);
    assert_eq!(classes.len(), 1, "Expected 1 class");
    assert_eq!(
        classes[0].name, "database",
        "Class name should be 'database'"
    );
    assert_eq!(
        classes[0].file, "src/testing/go/db.go",
        "Class file path is incorrect"
    );

    let class_function_edges =
        graph.find_nodes_with_edge_type(NodeType::Class, NodeType::Function, EdgeType::Operand);
    assert_eq!(class_function_edges.len(), 4, "Expected 4 methods");

    let data_models = graph.find_nodes_by_type(NodeType::DataModel);
    assert_eq!(data_models.len(), 2, "Expected 2 data models");
    assert!(
        data_models
            .iter()
            .any(|dm| dm.name == "Person" && dm.file == "src/testing/go/db.go"),
        "Expected Person data model not found"
    );

    let endpoints = graph.find_nodes_by_type(NodeType::Endpoint);
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

    let handler_edges_count = graph.count_edges_of_type(EdgeType::Handler);
    assert_eq!(handler_edges_count, 2, "Expected 2 handler edges");

    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_go() {
    use crate::lang::graphs::ArrayGraph;
    test_go_generic::<ArrayGraph>().await.unwrap();
}

// #[test(tokio::test)]
// async fn test_go() {
//     use crate::lang::graphs::ArrayGraph;
//     test_go_generic::<ArrayGraph>().await.unwrap();
// }

// #[test(tokio::test)]
// async fn test_go_btree() {
//     use crate::lang::graphs::BTreeMapGraph;
//     test_go_generic::<BTreeMapGraph>().await.unwrap();
// }
