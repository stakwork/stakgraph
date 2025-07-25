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

    let files = graph.find_nodes_by_type(NodeType::File);
    nodes_count += files.len();
    assert_eq!(files.len(), 6, "Expected at least 6 file nodes");

    let directories = graph.find_nodes_by_type(NodeType::Directory);
    nodes_count += directories.len();
    assert_eq!(directories.len(), 0, "Expected 0 directory nodes");

    let imports = graph.find_nodes_by_type(NodeType::Import);
    nodes_count += imports.len();
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

    let mut classes = graph.find_nodes_by_type(NodeType::Class);
    classes.sort_by(|a, b| a.name.cmp(&b.name));
    nodes_count += classes.len();
    assert_eq!(classes.len(), 2, "Expected 2 class");
    assert_eq!(
        classes[1].name, "database",
        "Class name should be 'database'"
    );
    assert_eq!(
        classes[1].file, "src/testing/go/db.go",
        "Class file path is incorrect"
    );

    let instances = graph.find_nodes_by_type(NodeType::Instance);
    nodes_count += instances.len();
    assert_eq!(instances.len(), 1, "Expected 1 instance node");

    let functions = graph.find_nodes_by_type(NodeType::Function);
    nodes_count += functions.len();
    assert!(
        functions
            .iter()
            .any(|f| f.name == "NewRouter" && f.file == "src/testing/go/routes.go"),
        "Function 'NewRouter' not found"
    );

    let class_function_edges =
        graph.find_nodes_with_edge_type(NodeType::Class, NodeType::Function, EdgeType::Operand);
    assert_eq!(class_function_edges.len(), 7, "Expected 7 methods");

    let data_models = graph.find_nodes_by_type(NodeType::DataModel);
    nodes_count += data_models.len();
    assert_eq!(data_models.len(), 6, "Expected 6 data models");
    let person = data_models
        .iter()
        .find(|dm| dm.name == "Person" && dm.file == "src/testing/go/db.go")
        .expect("Person data model not found");
    assert!(
        person.body.contains("ID    int"),
        "Person should have ID field"
    );
    assert!(
        person.body.contains("Name  string"),
        "Person should have Name field"
    );
    assert!(
        person.body.contains("Email string"),
        "Person should have Email field"
    );
    let leaderboard = data_models
        .iter()
        .find(|dm| dm.name == "LeaderboardEntry")
        .expect("LeaderboardEntry data model not found");
    assert!(
        leaderboard.body.contains("Name  string"),
        "LeaderboardEntry should have Name field"
    );
    assert!(
        leaderboard.body.contains("Score int"),
        "LeaderboardEntry should have Score field"
    );

    let endpoints = graph.find_nodes_by_type(NodeType::Endpoint);
    nodes_count += endpoints.len();
    if use_lsp {
        assert_eq!(endpoints.len(), 4, "Expected 4 endpoints");
    } else {
        assert_eq!(endpoints.len(), 3, "Expected 3 endpoints");
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
            handler_edges_count, 3,
            "Expected 3 handler edges without lsp"
        );
    }

    let function_calls = graph.count_edges_of_type(EdgeType::Calls);
    edges_count += function_calls;
    assert_eq!(function_calls, 8, "Expected 8 function calls");

    let operands = graph.count_edges_of_type(EdgeType::Operand);
    edges_count += operands;
    assert_eq!(operands, 7, "Expected 7 operands");

    let of = graph.count_edges_of_type(EdgeType::Of);
    edges_count += of;
    assert_eq!(of, 1, "Expected 1 of edges");

    let contains = graph.count_edges_of_type(EdgeType::Contains);
    edges_count += contains;
    assert_eq!(contains, 57, "Expected 57 contains edges");

    let variables = graph.find_nodes_by_type(NodeType::Var);
    nodes_count += variables.len();
    assert_eq!(variables.len(), 1, "Expected 1 variables");

    let trts = graph.find_nodes_by_type(NodeType::Trait);
    nodes_count += trts.len();
    assert_eq!(trts.len(), 1, "Expected 1 traits");

    let import_edges = graph.count_edges_of_type(EdgeType::Imports);
    edges_count += import_edges;
    if use_lsp {
        assert_eq!(import_edges, 4, "Expected 4 import edges with lsp");
    }

    let uses = graph.count_edges_of_type(EdgeType::Uses);
    edges_count += uses;
    if use_lsp {
        assert_eq!(uses, 50, "Expected 50 uses edges with lsp");
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

    let db_var = &variables[0];
    assert_eq!(db_var.name, "DB", "Variable name should be 'DB'");
    assert_eq!(
        db_var.file, "src/testing/go/db.go",
        "Variable file should be db.go"
    );
    assert!(
        db_var.body.contains("var DB database"),
        "DB variable should have correct declaration"
    );

    let (nodes, edges) = graph.get_graph_size();
    assert_eq!(
        nodes as usize, nodes_count,
        "Expected {} nodes got {}",
        nodes, nodes_count
    );
    assert_eq!(
        edges as usize, edges_count,
        "Expected {} edges got {}",
        edges, edges_count
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
