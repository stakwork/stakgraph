use crate::lang::graphs::{EdgeType, NodeType};
use crate::lang::{Graph, Node};
//use crate::testing::utils::{assert_golden_standard, parse_golden_standard};
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
        assert_eq!(num_nodes, 67, "Expected 67 nodes");
        assert_eq!(num_edges, 102, "Expected 102 edges");
    } else {
        assert_eq!(num_nodes, 33, "Expected 33 nodes");
        assert_eq!(num_edges, 53, "Expected 53 edges");
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
    assert_eq!(handler_edges_count, 2, "Expected 2 handler edges");

    let function_calls = graph.count_edges_of_type(EdgeType::Calls);
    assert_eq!(function_calls, 6, "Expected 6 function calls");

    let operands = graph.count_edges_of_type(EdgeType::Operand);
    assert_eq!(operands, 4, "Expected 4 operands");

    let of = graph.count_edges_of_type(EdgeType::Of);
    assert_eq!(of, 1, "Expected 1 of edges");

    if use_lsp {
        let contains = graph.count_edges_of_type(EdgeType::Contains);
        assert_eq!(contains, 42, "Expected 42 contains edges with lsp");
    } else {
        let contains = graph.count_edges_of_type(EdgeType::Contains);
        assert_eq!(contains, 40, "Expected 40 contains edges");
    }

    let variables = graph.find_nodes_by_type(NodeType::Var);
    assert_eq!(variables.len(), 1, "Expected 1 variables");

    if use_lsp {
        let import_edges = graph.count_edges_of_type(EdgeType::Imports);
        assert_eq!(import_edges, 3, "Expected 3 import edges with lsp");
    }
    Ok(())
}

//INFO: RUST_LOG=debug cargo test --test go [--features neo4j -- --nocapture]
#[test(tokio::test(flavor = "multi_thread", worker_threads = 2))]
async fn test_go() {
    use crate::lang::graphs::{ArrayGraph, BTreeMapGraph};
    // if !get_use_lsp() {
    //     let golden_standard = parse_golden_standard(GO_GOLDEN_STANDARD);

    //     test_go_generic::<ArrayGraph>().await.unwrap();
    //     assert_golden_standard(&golden_standard);

    //     test_go_generic::<BTreeMapGraph>().await.unwrap();
    //     assert_golden_standard(&golden_standard);

    //     #[cfg(feature = "neo4j")]
    //     {
    //         use crate::lang::graphs::Neo4jGraph;
    //         let mut graph = Neo4jGraph::default();
    //         graph.clear().await.unwrap();
    //         test_go_generic::<Neo4jGraph>().await.unwrap();
    //         assert_golden_standard(&golden_standard);
    //     }
    // } else {}
    test_go_generic::<ArrayGraph>().await.unwrap();
    test_go_generic::<BTreeMapGraph>().await.unwrap();
    #[cfg(feature = "neo4j")]
    {
        let mut graph = crate::lang::graphs::Neo4jGraph::default();
        graph.clear().await.unwrap();
        test_go_generic::<crate::lang::graphs::Neo4jGraph>()
            .await
            .unwrap();

        //}
    }
}

const _GO_GOLDEN_STANDARD: &str = r#"
[Node] : repository-go-srctestinggomain-0
[Node] : language-go-srctestinggo-0
[Node] : file-gitignore-srctestinggogitignore-0
[Node] : file-dbgo-srctestinggodbgo-0
[Node] : file-gomod-srctestinggogomod-0
[Node] : file-gosum-srctestinggogosum-0
[Node] : file-maingo-srctestinggomaingo-0
[Node] : file-routesgo-srctestinggoroutesgo-0
[Node] : library-gormiodriverpostgres-srctestinggogomod-4
[Node] : library-gormiogorm-srctestinggogomod-4
[Node] : library-githubcomgochichi-srctestinggogomod-9
[Node] : library-githubcomrscors-srctestinggogomod-9
[Node] : import-importimportsdbgo2-srctestinggodbgo-2
[Node] : import-importimportsmaingo2-srctestinggomaingo-2
[Node] : import-importimportsroutesgo2-srctestinggoroutesgo-2
[Node] : var-db-srctestinggodbgo-14
[Node] : class-database-srctestinggodbgo-9
[Node] : instance-db-srctestinggodbgo-14
[Node] : datamodel-database-srctestinggodbgo-9
[Node] : datamodel-person-srctestinggodbgo-16
[Node] : function-tablename-srctestinggodbgo-22
[Node] : function-newperson-srctestinggodbgo-26
[Node] : function-createoreditperson-srctestinggodbgo-31
[Node] : function-updatepersonname-srctestinggodbgo-38
[Node] : function-getpersonbyid-srctestinggodbgo-47
[Node] : function-initdb-srctestinggodbgo-55
[Node] : function-main-srctestinggomaingo-11
[Node] : function-newrouter-srctestinggoroutesgo-17
[Node] : function-getperson-srctestinggoroutesgo-42
[Node] : function-createperson-srctestinggoroutesgo-55
[Node] : function-initchi-srctestinggoroutesgo-78
[Node] : endpoint-personid-srctestinggoroutesgo-21-get
[Node] : endpoint-person-srctestinggoroutesgo-22-post
[Edge] : repository-go-srctestinggomain-0 - CONTAINS -> language-go-srctestinggo-0
[Edge] : repository-go-srctestinggomain-0 - CONTAINS -> file-gitignore-srctestinggogitignore-0
[Edge] : repository-go-srctestinggomain-0 - CONTAINS -> file-dbgo-srctestinggodbgo-0
[Edge] : repository-go-srctestinggomain-0 - CONTAINS -> file-gomod-srctestinggogomod-0
[Edge] : repository-go-srctestinggomain-0 - CONTAINS -> file-gosum-srctestinggogosum-0
[Edge] : repository-go-srctestinggomain-0 - CONTAINS -> file-maingo-srctestinggomaingo-0
[Edge] : repository-go-srctestinggomain-0 - CONTAINS -> file-routesgo-srctestinggoroutesgo-0
[Edge] : file-gomod-srctestinggogomod-0 - CONTAINS -> library-gormiodriverpostgres-srctestinggogomod-4
[Edge] : file-gomod-srctestinggogomod-0 - CONTAINS -> library-gormiogorm-srctestinggogomod-4
[Edge] : file-gomod-srctestinggogomod-0 - CONTAINS -> library-githubcomgochichi-srctestinggogomod-9
[Edge] : file-gomod-srctestinggogomod-0 - CONTAINS -> library-githubcomrscors-srctestinggogomod-9
[Edge] : file-dbgo-srctestinggodbgo-0 - CONTAINS -> import-importimportsdbgo2-srctestinggodbgo-2
[Edge] : file-maingo-srctestinggomaingo-0 - CONTAINS -> import-importimportsmaingo2-srctestinggomaingo-2
[Edge] : file-routesgo-srctestinggoroutesgo-0 - CONTAINS -> import-importimportsroutesgo2-srctestinggoroutesgo-2
[Edge] : file-dbgo-srctestinggodbgo-0 - CONTAINS -> var-db-srctestinggodbgo-14
[Edge] : file-dbgo-srctestinggodbgo-0 - CONTAINS -> class-database-srctestinggodbgo-9
[Edge] : file-dbgo-srctestinggodbgo-0 - CONTAINS -> instance-db-srctestinggodbgo-14
[Edge] : instance-db-srctestinggodbgo-14 - OF -> class-database-srctestinggodbgo-9
[Edge] : file-dbgo-srctestinggodbgo-0 - CONTAINS -> datamodel-database-srctestinggodbgo-9
[Edge] : file-dbgo-srctestinggodbgo-0 - CONTAINS -> datamodel-person-srctestinggodbgo-16
[Edge] : file-dbgo-srctestinggodbgo-0 - CONTAINS -> function-tablename-srctestinggodbgo-22
[Edge] : function-tablename-srctestinggodbgo-22 - CONTAINS -> datamodel-person-srctestinggodbgo-16
[Edge] : file-dbgo-srctestinggodbgo-0 - CONTAINS -> function-newperson-srctestinggodbgo-26
[Edge] : class-database-srctestinggodbgo-9 - OPERAND -> function-newperson-srctestinggodbgo-26
[Edge] : function-newperson-srctestinggodbgo-26 - CONTAINS -> datamodel-person-srctestinggodbgo-16
[Edge] : file-dbgo-srctestinggodbgo-0 - CONTAINS -> function-createoreditperson-srctestinggodbgo-31
[Edge] : class-database-srctestinggodbgo-9 - OPERAND -> function-createoreditperson-srctestinggodbgo-31
[Edge] : function-createoreditperson-srctestinggodbgo-31 - CONTAINS -> datamodel-person-srctestinggodbgo-16
[Edge] : file-dbgo-srctestinggodbgo-0 - CONTAINS -> function-updatepersonname-srctestinggodbgo-38
[Edge] : class-database-srctestinggodbgo-9 - OPERAND -> function-updatepersonname-srctestinggodbgo-38
[Edge] : function-updatepersonname-srctestinggodbgo-38 - CONTAINS -> datamodel-person-srctestinggodbgo-16
[Edge] : file-dbgo-srctestinggodbgo-0 - CONTAINS -> function-getpersonbyid-srctestinggodbgo-47
[Edge] : class-database-srctestinggodbgo-9 - OPERAND -> function-getpersonbyid-srctestinggodbgo-47
[Edge] : function-getpersonbyid-srctestinggodbgo-47 - CONTAINS -> datamodel-person-srctestinggodbgo-16
[Edge] : file-dbgo-srctestinggodbgo-0 - CONTAINS -> function-initdb-srctestinggodbgo-55
[Edge] : function-initdb-srctestinggodbgo-55 - CONTAINS -> datamodel-person-srctestinggodbgo-16
[Edge] : function-initdb-srctestinggodbgo-55 - CONTAINS -> var-db-srctestinggodbgo-14
[Edge] : file-maingo-srctestinggomaingo-0 - CONTAINS -> function-main-srctestinggomaingo-11
[Edge] : file-routesgo-srctestinggoroutesgo-0 - CONTAINS -> function-newrouter-srctestinggoroutesgo-17
[Edge] : file-routesgo-srctestinggoroutesgo-0 - CONTAINS -> function-getperson-srctestinggoroutesgo-42
[Edge] : function-getperson-srctestinggoroutesgo-42 - CONTAINS -> var-db-srctestinggodbgo-14
[Edge] : file-routesgo-srctestinggoroutesgo-0 - CONTAINS -> function-createperson-srctestinggoroutesgo-55
[Edge] : function-createperson-srctestinggoroutesgo-55 - CONTAINS -> datamodel-person-srctestinggodbgo-16
[Edge] : function-createperson-srctestinggoroutesgo-55 - CONTAINS -> var-db-srctestinggodbgo-14
[Edge] : file-routesgo-srctestinggoroutesgo-0 - CONTAINS -> function-initchi-srctestinggoroutesgo-78
[Edge] : endpoint-personid-srctestinggoroutesgo-21-get - HANDLER -> function-getperson-srctestinggoroutesgo-42
[Edge] : endpoint-person-srctestinggoroutesgo-22-post - HANDLER -> function-createperson-srctestinggoroutesgo-55
[Edge] : function-main-srctestinggomaingo-13 - CALLS -> function-initdb-srctestinggodbgo-0
[Edge] : function-main-srctestinggomaingo-15 - CALLS -> function-newrouter-srctestinggoroutesgo-0
[Edge] : function-newrouter-srctestinggoroutesgo-18 - CALLS -> function-initchi-srctestinggoroutesgo-78
[Edge] : function-getperson-srctestinggoroutesgo-45 - CALLS -> function-getpersonbyid-srctestinggodbgo-0
[Edge] : function-createperson-srctestinggoroutesgo-68 - CALLS -> function-newperson-srctestinggodbgo-0
[Edge] : function-initchi-srctestinggoroutesgo-79 - CALLS -> function-newrouter-srctestinggoroutesgo-17
"#;
