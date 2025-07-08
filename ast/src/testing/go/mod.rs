use crate::lang::graphs::{EdgeType, NodeType};
use crate::lang::{Graph, Node};
use crate::utils::get_use_lsp;
use crate::{lang::Lang, repo::Repo};
use std::str::FromStr;
use test_log::test;

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

    let (num_nodes, num_edges) = graph.get_graph_size();
    if use_lsp == true {
        assert_eq!(num_nodes, 76, "Expected 76 nodes");
        assert_eq!(num_edges, 122, "Expected 122 edges");
    } else {
        assert_eq!(num_nodes, 42, "Expected 42 nodes");
        assert_eq!(num_edges, 68, "Expected 68 edges");
    }

    let language_nodes = graph.find_nodes_by_name(NodeType::Language, "go");
    assert_eq!(language_nodes.len(), 1, "Expected 1 language node");
    assert_eq!(
        language_nodes[0].name, "go",
        "Language node name should be 'go'"
    );
    assert!(
        "src/testing/go/".contains(language_nodes[0].file.as_str()),
        "Language node file path is incorrect"
    );

    let imports = graph.find_nodes_by_type(NodeType::Import);
    assert_eq!(imports.len(), 3, "Expected 3 imports");

    let main_import_body = format!(
        r#"import (
	"context"
	"fmt"
	"os"
	"os/signal"
	"syscall"
	"time"
)"#
    );
    let main = imports
        .iter()
        .find(|i| i.file == "src/testing/go/main.go")
        .unwrap();

    assert_eq!(
        main.body, main_import_body,
        "Model import body is incorrect"
    );

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

    let functions = graph.find_nodes_by_type(NodeType::Function);
    assert!(
        functions
            .iter()
            .any(|f| f.name == "NewRouter" && f.file == "src/testing/go/routes.go"),
        "Function 'NewRouter' not found"
    );

    let class_function_edges =
        graph.find_nodes_with_edge_type(NodeType::Class, NodeType::Function, EdgeType::Operand);
    assert_eq!(class_function_edges.len(), 4, "Expected 4 methods");

    let data_models = graph.find_nodes_by_type(NodeType::DataModel);
    assert_eq!(data_models.len(), 5, "Expected 5 data models");
    assert!(
        data_models
            .iter()
            .any(|dm| dm.name == "Person" && dm.file == "src/testing/go/db.go"),
        "Expected Person data model not found"
    );

    let endpoints = graph.find_nodes_by_type(NodeType::Endpoint);
    assert_eq!(endpoints.len(), 4, "Expected 4 endpoints");

    let get_endpoint = endpoints
        .iter()
        .find(|e| e.name == "/person/{id}" && e.meta.get("verb") == Some(&"GET".to_string()))
        .expect("GET endpoint not found");
    assert_eq!(get_endpoint.file, "src/testing/go/routes.go");

    let post_endpoint = endpoints
        .iter()
        .find(|e| e.name == "/person" && e.meta.get("verb") == Some(&"POST".to_string()))
        .map(|e| Node::new(NodeType::Endpoint, e.clone()))
        .expect("POST endpoint not found");
    assert_eq!(post_endpoint.node_data.file, "src/testing/go/routes.go");

    let create_person_fn = graph
        .find_nodes_by_name(NodeType::Function, "CreatePerson")
        .into_iter()
        .find(|n| n.file == "src/testing/go/routes.go")
        .map(|nd| Node::new(NodeType::Function, nd))
        .expect("CreatePerson function not found");

    assert!(
        graph.has_edge(&post_endpoint, &create_person_fn, EdgeType::Handler),
        "Expected '/person' endpoint to be handled by 'CreatePerson'"
    );

    let main_fn = graph
        .find_nodes_by_name(NodeType::Function, "main")
        .into_iter()
        .find(|n| n.file == "src/testing/go/main.go")
        .map(|nd| Node::new(NodeType::Function, nd))
        .expect("main function not found");

    let init_db_fn = graph
        .find_nodes_by_name(NodeType::Function, "InitDB")
        .into_iter()
        .find(|n| n.file == "src/testing/go/db.go")
        .map(|nd| Node::new(NodeType::Function, nd))
        .expect("InitDB function not found");

    assert!(
        graph.has_edge(&main_fn, &init_db_fn, EdgeType::Calls),
        "Expected 'main' to call 'InitDB'"
    );

    let new_router_fn = graph
        .find_nodes_by_name(NodeType::Function, "NewRouter")
        .into_iter()
        .find(|n| n.file == "src/testing/go/routes.go")
        .map(|nd| Node::new(NodeType::Function, nd))
        .expect("NewRouter function not found in routes.go");
    assert_eq!(new_router_fn.node_data.name, "NewRouter");
    assert!(
        graph.has_edge(&main_fn, &new_router_fn, EdgeType::Calls),
        "Expected 'main' to call 'NewRouter'"
    );

    let handler_edges_count = graph.count_edges_of_type(EdgeType::Handler);
    assert_eq!(handler_edges_count, 4, "Expected 4 handler edges");

    let function_calls = graph.count_edges_of_type(EdgeType::Calls);
    assert_eq!(function_calls, 8, "Expected 8 function calls");

    let operands = graph.count_edges_of_type(EdgeType::Operand);
    assert_eq!(operands, 5, "Expected 5 operands");

    let of = graph.count_edges_of_type(EdgeType::Of);
    assert_eq!(of, 1, "Expected 1 of edges");

    let contains = graph.count_edges_of_type(EdgeType::Contains);
    assert_eq!(contains, 50, "Expected 50 contains edges");

    let variables = graph.find_nodes_by_type(NodeType::Var);
    assert_eq!(variables.len(), 1, "Expected 1 variables");

    if use_lsp {
        let import_edges = graph.count_edges_of_type(EdgeType::Imports);
        assert_eq!(import_edges, 4, "Expected 4 import edges with lsp");
    }
    let handler_fn = graph
        .find_nodes_by_name(NodeType::Function, "GetBountiesLeaderboard")
        .into_iter()
        .find(|n| n.file.ends_with("db.go") && n.body.contains("http.ResponseWriter"))
        .map(|nd| Node::new(NodeType::Function, nd))
        .expect("Handler method GetBountiesLeaderboard not found");

    let db_fn = graph
        .find_nodes_by_name(NodeType::Function, "GetBountiesLeaderboard")
        .into_iter()
        .find(|n| {
            n.file.ends_with("db.go")
                && n.body.contains("[]LeaderboardEntry")
                && !n.body.contains("http.ResponseWriter")
        })
        .map(|nd| Node::new(NodeType::Function, nd))
        .expect("DB method GetBountiesLeaderboard not found");

    assert!(
        graph.has_edge(&handler_fn, &db_fn, EdgeType::Calls),
        "Expected handler to call DB method"
    );
    Ok(())
}

#[test(tokio::test(flavor = "multi_thread", worker_threads = 2))]
async fn test_go() {
    use crate::lang::graphs::{ArrayGraph, BTreeMapGraph};
    test_go_generic::<ArrayGraph>().await.unwrap();
    test_go_generic::<BTreeMapGraph>().await.unwrap();

    #[cfg(feature = "neo4j")]
    {
        use crate::lang::graphs::Neo4jGraph;
        let graph = Neo4jGraph::default();
        graph.clear().await.unwrap();
        test_go_generic::<Neo4jGraph>().await.unwrap();
    }
}
