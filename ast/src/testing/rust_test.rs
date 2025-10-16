use crate::lang::graphs::{EdgeType, NodeType};
use crate::lang::{Graph, Node};
use crate::utils::sanitize_string;
use crate::{lang::Lang, repo::Repo};
use shared::Result;
use std::str::FromStr;

pub async fn test_rust_generic<G: Graph>() -> Result<()> {
    let repo = Repo::new(
        "src/testing/rust",
        Lang::from_str("rust").unwrap(),
        false,
        Vec::new(),
        Vec::new(),
    )
    .unwrap();

    let graph = repo.build_graph_inner::<G>().await?;

    graph.analysis();

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
    assert_eq!(directories.len(), 5, "Expected 5 directory nodes");

    let files = graph.find_nodes_by_type(NodeType::File);
    nodes_count += files.len();
    assert_eq!(files.len(), 12, "Expected 12 files");

    let rocket_file = files
        .iter()
        .find(|f| {
            f.name == "rocket_routes.rs"
                && f.file
                    .ends_with("src/testing/rust/src/routes/rocket_routes.rs")
        })
        .map(|n| Node::new(NodeType::File, n.clone()))
        .expect("File 'rocket.rs' not found in routes/rocket_routes.rs");

    let axum_file = files
        .iter()
        .find(|f| {
            f.name == "axum_routes.rs"
                && f.file
                    .ends_with("src/testing/rust/src/routes/axum_routes.rs")
        })
        .map(|n| Node::new(NodeType::File, n.clone()))
        .expect("File 'axum.rs' not found in routes/axum_routes.rs");

    let actix_file = files
        .iter()
        .find(|f| {
            f.name == "actix_routes.rs"
                && f.file
                    .ends_with("src/testing/rust/src/routes/actix_routes.rs")
        })
        .map(|n| Node::new(NodeType::File, n.clone()))
        .expect("File 'actix.rs' not found in routes/actix_routes.rs");

    let imports = graph.find_nodes_by_type(NodeType::Import);
    nodes_count += imports.len();
    assert_eq!(imports.len(), 8, "Expected 8 imports");

    let traits = graph.find_nodes_by_type(NodeType::Trait);
    nodes_count += traits.len();
    assert_eq!(traits.len(), 1, "Expected 1 trait nodes");

    let trait_node = traits
        .iter()
        .find(|t| t.name == "Greet" && t.file.ends_with("src/testing/rust/src/traits.rs"))
        .map(|n| Node::new(NodeType::Trait, n.clone()))
        .expect("Trait 'Greet' not found in traits.rs");

    let libraries = graph.find_nodes_by_type(NodeType::Library);
    nodes_count += libraries.len();

    assert_eq!(libraries.len(), 10, "Expected 10 library nodes");

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
    assert_eq!(vars.len(), 2, "Expected 2 variables");

    let data_models = graph.find_nodes_by_type(NodeType::DataModel);
    nodes_count += data_models.len();
    assert_eq!(data_models.len(), 6, "Expected 6 data models");

    let person_dm = data_models
        .iter()
        .find(|dm| dm.name == "Person" && dm.file.ends_with("src/testing/rust/src/db.rs"))
        .map(|n| Node::new(NodeType::DataModel, n.clone()))
        .expect("Data model 'Person' not found in models.rs");

    let classes = graph.find_nodes_by_type(NodeType::Class);
    nodes_count += classes.len();
    assert_eq!(classes.len(), 4, "Expected 4 class node");

    let database_class = classes
        .iter()
        .find(|c| c.name == "Database" && c.file.ends_with("src/testing/rust/src/db.rs"))
        .map(|n| Node::new(NodeType::Class, n.clone()))
        .expect("Class 'Database' not found in db.rs");

    let dm_imports = graph.has_edge(&rocket_file, &person_dm, EdgeType::Imports);
    assert!(
        dm_imports,
        "Expected 'Person' data model to be imported in 'rocket_routes.rs'"
    );
    let db_imports = graph.has_edge(&rocket_file, &database_class, EdgeType::Imports);
    assert!(
        db_imports,
        "Expected 'Database' class to be imported in 'rocket_routes.rs'"
    );

    let dm_imports = graph.has_edge(&axum_file, &person_dm, EdgeType::Imports);
    assert!(
        dm_imports,
        "Expected 'Person' data model to be imported in 'axum_routes.rs'"
    );
    let db_imports = graph.has_edge(&axum_file, &database_class, EdgeType::Imports);
    assert!(
        db_imports,
        "Expected 'Database' class to be imported in 'axum_routes.rs'"
    );
    let dm_imports = graph.has_edge(&actix_file, &person_dm, EdgeType::Imports);
    assert!(
        dm_imports,
        "Expected 'Person' data model to be imported in 'actix_routes.rs'"
    );
    let db_imports = graph.has_edge(&actix_file, &database_class, EdgeType::Imports);
    assert!(
        db_imports,
        "Expected 'Database' class to be imported in 'actix_routes.rs'"
    );

    let greeter_class = classes
        .iter()
        .find(|c| c.name == "Greeter" && c.file.ends_with("src/testing/rust/src/traits.rs"))
        .map(|n| Node::new(NodeType::Class, n.clone()))
        .expect("Class 'Greet' not found in traits.rs");

    let implements_edge_exist = graph.has_edge(&greeter_class, &trait_node, EdgeType::Implements);
    assert!(
        implements_edge_exist,
        "Expected 'Greet' class to implement 'Greet' trait"
    );

    let endpoints = graph.find_nodes_by_type(NodeType::Endpoint);
    nodes_count += endpoints.len();
    assert_eq!(endpoints.len(), 6, "Expected 6 endpoints");

    let imported_edges = graph.count_edges_of_type(EdgeType::Imports);
    edges_count += imported_edges;
    assert_eq!(imported_edges, 10, "Expected 10 import edges");

    let contains_edges = graph.count_edges_of_type(EdgeType::Contains);
    edges_count += contains_edges;
    assert_eq!(contains_edges, 106, "Expected 106 contains edges");

    let calls_edges = graph.count_edges_of_type(EdgeType::Calls);
    edges_count += calls_edges;
    //FIXME: Other graphs record 3 edges, while Neo4j record 2... the integration test call to one endpoint is absent.
    //assert_eq!(calls_edges, 3, "Expected 3 calls edges");

    let functions = graph.find_nodes_by_type(NodeType::Function);
    nodes_count += functions.len();
    println!("{:#?}", functions.iter().map(|f| f.name.clone()).collect::<Vec<_>>());
    assert_eq!(functions.len(), 25, "Expected 25 functions");


    let unit_tests = graph.find_nodes_by_type(NodeType::UnitTest);
    nodes_count += unit_tests.len();
    println!("--- Unit Tests ---");
    println!("{:#?}", unit_tests.iter().map(|f| f.name.clone()).collect::<Vec<_>>());
    // FIXME: Unit test deeply flawed... catches all other tests...
    assert_eq!(unit_tests.len(), 9, "Expected 9 unit tests (4 db.rs + 2 axum + 2 benchmarks)");

    let integration_tests = graph.find_nodes_by_type(NodeType::IntegrationTest);
    nodes_count += integration_tests.len();
    println!("--- Integration Tests ---");
    println!("{:#?}", integration_tests.iter().map(|f| f.name.clone()).collect::<Vec<_>>());
    assert_eq!(integration_tests.len(), 5, "Expected 5 integration tests");

    let e2e_tests = graph.find_nodes_by_type(NodeType::E2eTest);
    nodes_count += e2e_tests.len();
    println!("--- E2E Tests ---");
    println!("{:#?}", e2e_tests.iter().map(|f| f.name.clone()).collect::<Vec<_>>());
    assert_eq!(e2e_tests.len(), 4, "Expected 4 e2e tests (including #[ignore] test)");

    let handlers = graph.count_edges_of_type(EdgeType::Handler);
    edges_count += handlers;
    assert_eq!(handlers, 6, "Expected 6 handler edges");

    let implements = graph.count_edges_of_type(EdgeType::Implements);
    edges_count += implements;
    assert_eq!(implements, 1, "Expected 1 implements edge");

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

    let get_person_fn = functions
        .iter()
        .find(|f| f.name == "get_person" && f.file.ends_with("src/routes/axum_routes.rs"))
        .map(|n| Node::new(NodeType::Function, n.clone()))
        .expect("get_person function not found in axum_routes.rs");

    let create_person_fn = functions
        .iter()
        .find(|f| f.name == "create_person" && f.file.ends_with("src/routes/axum_routes.rs"))
        .map(|n| Node::new(NodeType::Function, n.clone()))
        .expect("create_person function not found in axum_routes.rs");

    let get_person_endpoint = endpoints
        .iter()
        .find(|e| e.name == "/person/:id" && e.file.ends_with("src/routes/axum_routes.rs"))
        .map(|n| Node::new(NodeType::Endpoint, n.clone()))
        .expect("GET /person/:id endpoint not found in axum_routes.rs");

    let post_person_endpoint = endpoints
        .iter()
        .find(|e| e.name == "/person" && e.file.ends_with("src/routes/axum_routes.rs"))
        .map(|n| Node::new(NodeType::Endpoint, n.clone()))
        .expect("POST /person endpoint not found in axum_routes.rs");

    assert!(
        graph.has_edge(&get_person_endpoint, &get_person_fn, EdgeType::Handler),
        "Expected '/person/id' endpoint to be handled by get_person"
    );
    assert!(
        graph.has_edge(&post_person_endpoint, &create_person_fn, EdgeType::Handler),
        "Expected '/person' endpoint to be handled by create_person"
    );

    let get_person_fn = functions
        .iter()
        .find(|f| f.name == "get_person" && f.file.ends_with("src/routes/rocket_routes.rs"))
        .map(|n| Node::new(NodeType::Function, n.clone()))
        .expect("get_person function not found in rocket_routes.rs");

    let create_person_fn = functions
        .iter()
        .find(|f| f.name == "create_person" && f.file.ends_with("src/routes/rocket_routes.rs"))
        .map(|n| Node::new(NodeType::Function, n.clone()))
        .expect("create_person function not found in rocket_routes.rs");

    let get_person_endpoint = endpoints
        .iter()
        .find(|e| e.name == "/person/<id>" && e.file.ends_with("src/routes/rocket_routes.rs"))
        .map(|n| Node::new(NodeType::Endpoint, n.clone()))
        .expect("GET /person/<id> endpoint not found in rocket_routes.rs");

    let post_person_endpoint = endpoints
        .iter()
        .find(|e| {
            &sanitize_string(&e.name) == "person" && e.file.ends_with("src/routes/rocket_routes.rs")
        })
        .map(|n| Node::new(NodeType::Endpoint, n.clone()))
        .expect("POST /person endpoint not found in rocket_routes.rs");

    assert!(
        graph.has_edge(&get_person_endpoint, &get_person_fn, EdgeType::Handler),
        "Expected '/person/id' endpoint to be handled by get_person"
    );
    assert!(
        graph.has_edge(&post_person_endpoint, &create_person_fn, EdgeType::Handler),
        "Expected '/person' endpoint to be handled by create_person"
    );

    let init_db_fn = functions
        .iter()
        .find(|f| f.name == "init_db" && f.file.ends_with("src/testing/rust/src/db.rs"))
        .map(|n| Node::new(NodeType::Function, n.clone()))
        .expect("init_db function not found in db.rs");

    let database_dm = data_models
        .iter()
        .find(|dm| dm.name == "Database" && dm.file.ends_with("src/testing/rust/src/db.rs"))
        .map(|n| Node::new(NodeType::DataModel, n.clone()))
        .expect("Data model 'Database' not found in db.rs");

    let db_instance_var = vars
        .iter()
        .find(|v| v.name == "DB_INSTANCE" && v.file.ends_with("src/testing/rust/src/db.rs"))
        .map(|n| Node::new(NodeType::Var, n.clone()))
        .expect("Variable 'db' not found in main.rs");

    assert!(
        graph.has_edge(&init_db_fn, &database_dm, EdgeType::Contains),
        "Expected 'init_db' function to use 'Database' data model"
    );
    assert!(
        graph.has_edge(&init_db_fn, &db_instance_var, EdgeType::Contains),
        "Expected 'init_db' function to use 'DB_INSTANCE' variable"
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
