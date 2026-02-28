use crate::lang::graphs::{EdgeType, NodeType};
use crate::lang::{Graph, Node};
use crate::utils::get_use_lsp;
use crate::{lang::Lang, repo::Repo};
use shared::error::Result;
use std::str::FromStr;
use test_log::test;

pub async fn test_go_generic<G: Graph + Sync>() -> Result<()> {
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

    // graph.analysis();

    let mut nodes_count = 0;
    let mut edges_count = 0;

    let language_nodes = graph.find_nodes_by_name(NodeType::Language, "go");
    nodes_count += language_nodes.len();

    let repo = graph.find_nodes_by_type(NodeType::Repository);
    nodes_count += repo.len();
    assert_eq!(repo.len(), 1, "Expected 1 repository node");

    assert_eq!(language_nodes.len(), 1, "Expected 1 language node");
    assert_eq!(
        language_nodes[0].name, "go",
        "Language node name should be 'go'"
    );
    assert!(
        "src/testing/go/".contains(language_nodes[0].file.as_str()),
        "Language node file path is incorrect"
    );

    let libraries = graph.find_nodes_by_type(NodeType::Library);
    nodes_count += libraries.len();
    assert_eq!(libraries.len(), 4, "Expected 4 library nodes");

    let unit_tests = graph.find_nodes_by_type(NodeType::UnitTest);
    nodes_count += unit_tests.len();
    assert_eq!(unit_tests.len(), 2, "Expected 2 unit tests");

    let integration_tests = graph.find_nodes_by_type(NodeType::IntegrationTest);
    nodes_count += integration_tests.len();
    assert_eq!(integration_tests.len(), 1, "Expected 1 integration test");

    let e2e_tests = graph.find_nodes_by_type(NodeType::E2eTest);
    nodes_count += e2e_tests.len();
    assert_eq!(e2e_tests.len(), 2, "Expected 2 e2e tests");

    let files = graph.find_nodes_by_type(NodeType::File);
    nodes_count += files.len();
    assert_eq!(files.len(), 13, "Expected 13 file nodes");

    let directories = graph.find_nodes_by_type(NodeType::Directory);
    nodes_count += directories.len();
    assert_eq!(directories.len(), 1, "Expected 1 directory node");

    let imports = graph.find_nodes_by_type(NodeType::Import);
    nodes_count += imports.len();
    assert_eq!(imports.len(), 9, "Expected 9 imports");

    let packages = graph.find_nodes_by_type(NodeType::Package);
    nodes_count += packages.len();
    assert_eq!(packages.len(), 0, "Expected 0 packages");

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
    nodes_count += classes.len();
    assert_eq!(classes.len(), 8, "Expected 8 classes");

    let database_class = classes
        .iter()
        .find(|c| c.name == "database")
        .expect("Class 'database' not found");

    assert_eq!(
        database_class.file, "src/testing/go/db.go",
        "Class file path is incorrect"
    );

    let instances = graph.find_nodes_by_type(NodeType::Instance);
    nodes_count += instances.len();
    assert_eq!(instances.len(), 1, "Expected 1 instance node");

    let traits = graph.find_nodes_by_type(NodeType::Trait);
    nodes_count += traits.len();
    assert_eq!(traits.len(), 1, "Expected 1 trait");
    assert_eq!(traits[0].name, "Shape", "Trait should be Shape");
    assert_eq!(
        traits[0].docs,
        Some("Shape is a geometric shape interface".to_string()),
        "Shape interface should have comment"
    );

    let functions = graph.find_nodes_by_type(NodeType::Function);
    nodes_count += functions.len();
    if use_lsp {
        assert_eq!(functions.len(), 61, "Expected 61 functions");
    } else {
        assert_eq!(functions.len(), 29, "Expected 29 functions");
    }
    assert!(
        functions
            .iter()
            .any(|f| f.name == "NewRouter" && f.file == "src/testing/go/routes.go"),
        "Function 'NewRouter' not found"
    );

    let class_function_edges =
        graph.find_nodes_with_edge_type(NodeType::Class, NodeType::Function, EdgeType::Operand);
    assert_eq!(class_function_edges.len(), 15, "Expected 15 methods");

    let data_models = graph.find_nodes_by_type(NodeType::DataModel);
    nodes_count += data_models.len();
    assert_eq!(data_models.len(), 12, "Expected 12 data models");

    let rect_dm = data_models
        .iter()
        .find(|dm| dm.name == "Rectangle")
        .expect("Rectangle struct not found");
    assert_eq!(
        rect_dm.docs,
        Some("Rectangle represents a rectangle".to_string()),
        "Rectangle should have comment"
    );

    let requests = graph.find_nodes_by_type(NodeType::Request);
    nodes_count += requests.len();

    let endpoints = graph.find_nodes_by_type(NodeType::Endpoint);
    nodes_count += endpoints.len();
    if use_lsp {
        assert_eq!(endpoints.len(), 6, "Expected 6 endpoints");
    } else {
        assert_eq!(endpoints.len(), 5, "Expected 5 endpoints");
    }

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

    let leaderboard_endpoint = endpoints
        .iter()
        .find(|e| e.name == "/leaderboard" && e.meta.get("verb") == Some(&"GET".to_string()))
        .expect("GET /leaderboard endpoint not found");
    assert_eq!(leaderboard_endpoint.file, "src/testing/go/routes.go");

    if use_lsp {
        let bounties_endpoint = endpoints
            .iter()
            .find(|e| {
                e.name == "/bounties/leaderboard" && e.meta.get("verb") == Some(&"GET".to_string())
            })
            .expect("GET /bounties/leaderboard endpoint not found");
        assert_eq!(bounties_endpoint.file, "src/testing/go/routes.go");
    }

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
    assert_eq!(
        new_router_fn.node_data.docs,
        Some("NewRouter creates a chi router".to_string()),
        "NewRouter should have documentation"
    );
    assert_eq!(new_router_fn.node_data.name, "NewRouter");
    assert!(
        graph.has_edge(&main_fn, &new_router_fn, EdgeType::Calls),
        "Expected 'main' to call 'NewRouter'"
    );
    let new_router = functions
        .iter()
        .find(|f| f.name == "NewRouter" && f.file == "src/testing/go/routes.go")
        .expect("NewRouter function not found");
    assert!(
        new_router.body.contains("initChi()"),
        "NewRouter should call initChi()"
    );
    let init_chi = functions
        .iter()
        .find(|f| f.name == "initChi" && f.file == "src/testing/go/routes.go")
        .expect("initChi function not found");
    assert!(
        init_chi.body.contains("chi.NewRouter()"),
        "initChi should create chi router"
    );

    assert!(
        new_router
            .body
            .contains("r.Get(\"/person/{id}\", GetPerson)"),
        "NewRouter should define GET /person/{{id}} route"
    );
    assert!(
        new_router
            .body
            .contains("r.Post(\"/person\", CreatePerson)"),
        "NewRouter should define POST /person route"
    );

    let handler_edges_count = graph.count_edges_of_type(EdgeType::Handler);
    edges_count += handler_edges_count;
    if use_lsp {
        assert_eq!(handler_edges_count, 4, "Expected 4 handler edges with lsp");
    } else {
        assert_eq!(
            handler_edges_count, 5,
            "Expected 3 handler edges without lsp"
        );
    }

    let function_calls = graph.count_edges_of_type(EdgeType::Calls);
    edges_count += function_calls;
    assert_eq!(function_calls, 12, "Expected 12 function calls");

    let operands = graph.count_edges_of_type(EdgeType::Operand);
    edges_count += operands;
    if use_lsp {
        assert_eq!(operands, 19, "Expected 19 operands with lsp");
    } else {
        assert_eq!(operands, 15, "Expected 15 operands without lsp");
    }

    let of = graph.count_edges_of_type(EdgeType::Of);
    edges_count += of;
    assert_eq!(of, 2, "Expected 2 of edges");

    let contains = graph.count_edges_of_type(EdgeType::Contains);
    edges_count += contains;
    assert_eq!(contains, 108, "Expected 108 contains edges");

    let variables = graph.find_nodes_by_type(NodeType::Var);
    nodes_count += variables.len();
    assert_eq!(variables.len(), 1, "Expected 1 variables");

    let import_edges = graph.count_edges_of_type(EdgeType::Imports);
    edges_count += import_edges;
    if use_lsp {
        assert_eq!(import_edges, 4, "Expected 4 import edges with lsp");
    }

    let uses = graph.count_edges_of_type(EdgeType::Uses);
    edges_count += uses;
    if use_lsp {
        assert_eq!(uses, 47, "Expected 47 uses edges with lsp");
    }

    let nested_in = graph.count_edges_of_type(EdgeType::NestedIn);
    edges_count += nested_in;
    assert_eq!(
        nested_in, 2,
        "Expected 2 NestedIn edges for anonymous functions"
    );

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

    let db_var = variables
        .iter()
        .find(|v| v.name == "DB")
        .expect("DB var not found");
    assert_eq!(db_var.name, "DB", "Variable name should be 'DB'");
    assert_eq!(
        db_var.file, "src/testing/go/db.go",
        "Variable file should be db.go"
    );
    assert!(
        db_var.body.contains("var DB database"),
        "DB variable should have correct declaration"
    );
    assert_eq!(
        db_var.docs,
        Some("DB is the object".to_string()),
        "DB variable should have documentation"
    );

    assert_eq!(
        post_endpoint.node_data.docs,
        Some("Create a new person".to_string()),
        "POST endpoint should have documentation"
    );

    let (nodes, edges) = graph.get_graph_size();

    assert_eq!(
        nodes as usize, nodes_count,
        "Expected {} nodes got {}",
        nodes, nodes_count
    );
    assert_eq!(
        edges as usize, edges_count,
        "Expected 144 edges got {}",
        edges
    );

    let anon_get = endpoints
        .iter()
        .find(|e| e.name == "/anon-get")
        .expect("GET /anon-get endpoint not found");

    let get_handler = anon_get.meta.get("handler").expect("Handler missing");
    assert!(
        get_handler.contains("GET_anon-get_func_L8"),
        "Incorrect GET handler: {}",
        get_handler
    );

    let anon_post = endpoints
        .iter()
        .find(|e| e.name == "/anon-post")
        .expect("POST /anon-post endpoint not found");

    let post_handler = anon_post.meta.get("handler").expect("Handler missing");
    assert!(
        post_handler.contains("POST_anon-post_func_L13"),
        "Incorrect POST handler: {}",
        post_handler
    );

    Ok(())
}

pub async fn test_go_non_web_generic<G: Graph + Sync>() -> Result<()> {
    let repo = Repo::new(
        "src/testing/go_non_web",
        Lang::from_str("go").unwrap(),
        false,
        Vec::new(),
        Vec::new(),
    )
    .unwrap();

    let graph = repo.build_graph_inner::<G>().await?;

    let language_nodes = graph.find_nodes_by_name(NodeType::Language, "go");
    assert_eq!(language_nodes.len(), 1, "Expected 1 language node");
    assert_eq!(
        language_nodes[0].file, "src/testing/go_non_web",
        "Language node file path is incorrect"
    );

    let files = graph.find_nodes_by_type(NodeType::File);
    assert_eq!(files.len(), 8, "Expected 8 file nodes");

    let endpoints = graph.find_nodes_by_type(NodeType::Endpoint);
    assert_eq!(endpoints.len(), 3, "Expected 3 stdlib endpoints");

    let health_endpoint = endpoints
        .iter()
        .find(|e| e.name == "/health")
        .map(|e| Node::new(NodeType::Endpoint, e.clone()))
        .expect("/health endpoint not found");

    let ready_endpoint = endpoints
        .iter()
        .find(|e| e.name == "/ready")
        .map(|e| Node::new(NodeType::Endpoint, e.clone()))
        .expect("/ready endpoint not found");

    let anon_endpoint = endpoints
        .iter()
        .find(|e| e.name == "/anon")
        .expect("/anon endpoint not found");
    let anon_handler = anon_endpoint
        .meta
        .get("handler")
        .expect("anonymous handler missing");
    assert!(
        anon_handler.contains("HANDLEFUNC_anon_func_"),
        "unexpected anonymous handler name: {}",
        anon_handler
    );

    let health_fn = graph
        .find_nodes_by_name(NodeType::Function, "HealthHandler")
        .into_iter()
        .find(|n| n.file == "src/testing/go_non_web/http_stdlib.go")
        .map(|nd| Node::new(NodeType::Function, nd))
        .expect("HealthHandler function not found");

    let ready_fn = graph
        .find_nodes_by_name(NodeType::Function, "ReadyHandler")
        .into_iter()
        .find(|n| n.file == "src/testing/go_non_web/http_stdlib.go")
        .map(|nd| Node::new(NodeType::Function, nd))
        .expect("ReadyHandler function not found");

    assert!(
        graph.has_edge(&health_endpoint, &health_fn, EdgeType::Handler),
        "Expected /health endpoint to be handled by HealthHandler"
    );
    assert!(
        graph.has_edge(&ready_endpoint, &ready_fn, EdgeType::Handler),
        "Expected /ready endpoint to be handled by ReadyHandler"
    );

    let unit_tests = graph.find_nodes_by_type(NodeType::UnitTest);
    assert_eq!(
        unit_tests.len(),
        4,
        "Expected 4 unit tests in non-web suite)"
    );

    let integration_tests = graph.find_nodes_by_type(NodeType::IntegrationTest);
    assert_eq!(
        integration_tests.len(),
        1,
        "Expected 1 integration test in non-web suite"
    );

    let e2e_tests = graph.find_nodes_by_type(NodeType::E2eTest);
    assert_eq!(e2e_tests.len(), 0, "Expected 0 e2e tests in non-web suite");

    let main_fn = graph
        .find_nodes_by_name(NodeType::Function, "main")
        .into_iter()
        .find(|n| n.file == "src/testing/go_non_web/main.go")
        .map(|nd| Node::new(NodeType::Function, nd))
        .expect("main function not found in non-web suite");

    let execute_fn = graph
        .find_nodes_by_name(NodeType::Function, "Execute")
        .into_iter()
        .find(|n| n.file == "src/testing/go_non_web/cli.go")
        .map(|nd| Node::new(NodeType::Function, nd))
        .expect("Execute function not found in non-web suite");

    assert!(
        graph.has_edge(&main_fn, &execute_fn, EdgeType::Calls),
        "Expected main to call Execute in non-web suite"
    );

    Ok(())
}

#[test(tokio::test(flavor = "multi_thread", worker_threads = 2))]
async fn test_go() {
    #[cfg(not(feature = "neo4j"))]
    {
        use crate::lang::graphs::{ArrayGraph, BTreeMapGraph};
        test_go_generic::<ArrayGraph>().await.unwrap();
        test_go_generic::<BTreeMapGraph>().await.unwrap();

        test_go_non_web_generic::<ArrayGraph>().await.unwrap();
        test_go_non_web_generic::<BTreeMapGraph>().await.unwrap();
    }

    #[cfg(feature = "neo4j")]
    {
        use crate::lang::graphs::Neo4jGraph;
        let graph = Neo4jGraph::default();
        graph.clear().await.unwrap();
        test_go_generic::<Neo4jGraph>().await.unwrap();
        graph.clear().await.unwrap();
        test_go_non_web_generic::<Neo4jGraph>().await.unwrap();
    }
}
