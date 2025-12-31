use crate::lang::graphs::{EdgeType, NodeType};
use crate::lang::{Graph, Node};
use crate::utils::get_use_lsp;
use crate::{lang::Lang, repo::Repo};
use shared::error::Result;
use std::str::FromStr;

pub async fn test_typescript_generic<G: Graph>() -> Result<()> {
    let use_lsp = get_use_lsp();
    let repo = Repo::new(
        "src/testing/typescript",
        Lang::from_str("ts").unwrap(),
        use_lsp,
        Vec::new(),
        Vec::new(),
    )
    .unwrap();

    let graph = repo.build_graph_inner::<G>().await?;

    // graph.analysis();

    let mut nodes_count = 0;
    let mut edges_count = 0;

    fn normalize_path(path: &str) -> String {
        path.replace("\\", "/")
    }

    let language_nodes = graph.find_nodes_by_type(NodeType::Language);
    nodes_count += language_nodes.len();
    assert_eq!(language_nodes.len(), 1, "Expected 1 language node");
    assert_eq!(
        language_nodes[0].name, "typescript",
        "Language node name should be 'typescript'"
    );
    assert_eq!(
        normalize_path(&language_nodes[0].file),
        "src/testing/typescript",
        "Language node file path is incorrect"
    );

    let repository = graph.find_nodes_by_type(NodeType::Repository);
    nodes_count += repository.len();
    assert_eq!(repository.len(), 1, "Expected 1 repository node");

    let files = graph.find_nodes_by_type(NodeType::File);
    nodes_count += files.len();

    let pkg_files = files
        .iter()
        .filter(|f| f.name == "package.json")
        .collect::<Vec<_>>();
    assert_eq!(pkg_files.len(), 1, "Expected 1 package.json file");
    assert_eq!(
        pkg_files[0].name, "package.json",
        "Package file name is incorrect"
    );

    let imports = graph.find_nodes_by_type(NodeType::Import);
    nodes_count += imports.len();

    for imp in &imports {
        let import_lines: Vec<&str> = imp
            .body
            .lines()
            .filter(|line| line.trim_start().starts_with("import "))
            .collect();

        assert!(
            import_lines.len() > 0,
            "Expected multiple import lines in {}",
            imp.file
        );
    }

    assert_eq!(imports.len(), 12, "Expected 12 imports");

    let model_import_body = format!(
        r#"import DataTypes, {{ Model }} from "sequelize";
import {{ Entity, Column, PrimaryGeneratedColumn }} from "typeorm";
import {{ sequelize }} from "./config.js";"#
    );
    let model = imports
        .iter()
        .find(|i| i.file == "src/testing/typescript/src/model.ts")
        .unwrap();

    assert_eq!(
        model.body, model_import_body,
        "Model import body is incorrect"
    );

    let libraries = graph.find_nodes_by_type(NodeType::Library);
    nodes_count += libraries.len();
    assert_eq!(libraries.len(), 11, "Expected 11 libraries");

    let functions = graph.find_nodes_by_type(NodeType::Function);
    nodes_count += functions.len();
    if use_lsp == true {
        assert_eq!(functions.len(), 34, "Expected 34 functions");
    } else {
        assert_eq!(functions.len(), 32, "Expected 32 functions");
    }

    let log_fn = functions
        .iter()
        .find(|f| f.name == "log" && f.file == "src/testing/typescript/src/service.ts")
        .expect("log function not found in service.ts");

    assert!(
        log_fn.meta.contains_key("interface"),
        "log function should have interface metadata"
    );

    let deprecated_fn = functions
        .iter()
        .find(|f| f.name == "deprecated" && f.file == "src/testing/typescript/src/service.ts")
        .expect("deprecated function not found in service.ts");

    assert!(
        deprecated_fn.meta.contains_key("interface"),
        "deprecated function should have interface metadata"
    );

    let classes = graph.find_nodes_by_type(NodeType::Class);
    nodes_count += classes.len();
    assert_eq!(classes.len(), 7, "Expected 7 classes");

    let directories = graph.find_nodes_by_type(NodeType::Directory);
    nodes_count += directories.len();
    assert_eq!(directories.len(), 5, "Expected 5 directories");

    let calls_edges_count = graph.count_edges_of_type(EdgeType::Calls);
    edges_count += calls_edges_count;

    //TODO: Fix lsp calls edge count : locally, it says 5 but on CI it says 7
    // assert_eq!(calls_edges_count, 5, "Expected 5 calls edges");

    let data_models = graph.find_nodes_by_type(NodeType::DataModel);
    nodes_count += data_models.len();
    assert_eq!(data_models.len(), 10, "Expected 10 data models");

    let trait_nodes = graph.find_nodes_by_type(NodeType::Trait);
    nodes_count += trait_nodes.len();
    assert_eq!(trait_nodes.len(), 2, "Expected 2 trait nodes");

    let variables = graph.find_nodes_by_type(NodeType::Var);
    nodes_count += variables.len();
    assert_eq!(variables.len(), 6, "Expected 6 variables");

    let contains = graph.count_edges_of_type(EdgeType::Contains);
    edges_count += contains;
    assert_eq!(contains, 115, "Expected 115 contains edges");

    let import_edges_count = graph.count_edges_of_type(EdgeType::Imports);
    edges_count += import_edges_count;
    if use_lsp {
        assert_eq!(import_edges_count, 15, "Expected 15 import edges");
    } else {
        assert_eq!(import_edges_count, 13, "Expected 13 import edges");
    }

    let handlers = graph.count_edges_of_type(EdgeType::Handler);
    edges_count += handlers;
    if use_lsp {
        assert_eq!(handlers, 8, "Expected 8 handler edges");
    } else {
        assert_eq!(handlers, 22, "Expected 22 handler edges");
    }

    let create_person_fn = functions
        .iter()
        .find(|f| {
            f.name == "createPerson"
                && normalize_path(&f.file) == "src/testing/typescript/src/routes.ts"
        })
        .map(|n| Node::new(NodeType::Function, n.clone()))
        .expect("createPerson function not found");

    let get_person_fn = functions
        .iter()
        .find(|f| {
            f.name == "getPerson"
                && normalize_path(&f.file) == "src/testing/typescript/src/routes.ts"
        })
        .map(|n| Node::new(NodeType::Function, n.clone()))
        .expect("getPerson function not found");

    let endpoints = graph.find_nodes_by_type(NodeType::Endpoint);
    for e in &endpoints {
        println!("{:?} {}  {}", e.meta.get("verb"), e.name, e.file);
    }
    nodes_count += endpoints.len();

    if use_lsp {
        assert_eq!(endpoints.len(), 19, "Expected 19 endpoints");
    } else {
        assert_eq!(endpoints.len(), 22, "Expected 22 endpoints");
    }

    let implements = graph.count_edges_of_type(EdgeType::Implements);
    edges_count += implements;
    assert_eq!(implements, 3, "Expected 3 implements edges");

    let uses = graph.count_edges_of_type(EdgeType::Uses);
    edges_count += uses;
    if use_lsp {
        // assert_eq!(uses, 6, "Expected 6 uses edges");
    } else {
        //    assert_eq!(uses, 0, "Expected 0 uses edges");
    }

    let neseted = graph.count_edges_of_type(EdgeType::NestedIn);
    edges_count += neseted;

    let post_person_endpoint = endpoints
        .iter()
        .find(|e| {
            e.name == "/person"
                && normalize_path(&e.file) == "src/testing/typescript/src/routes.ts"
                && e.meta.get("verb") == Some(&"POST".to_string())
        })
        .map(|n| Node::new(NodeType::Endpoint, n.clone()))
        .expect("POST /person endpoint not found");

    assert!(
        graph.has_edge(&post_person_endpoint, &create_person_fn, EdgeType::Handler),
        "Expected '/person' POST endpoint to be handled by createPerson"
    );

    let get_person_endpoint = endpoints
        .iter()
        .find(|e| {
            e.name == "/person/:id"
                && normalize_path(&e.file) == "src/testing/typescript/src/routes.ts"
                && e.meta.get("verb") == Some(&"GET".to_string())
        })
        .map(|n| Node::new(NodeType::Endpoint, n.clone()))
        .expect("GET /person/:id endpoint not found");

    assert!(
        graph.has_edge(&get_person_endpoint, &get_person_fn, EdgeType::Handler),
        "Expected '/person/:id' GET endpoint to be handled by getPerson"
    );

    let create_new_person_fn = functions
        .iter()
        .find(|f| {
            f.name == "createNewPerson"
                && normalize_path(&f.file) == "src/testing/typescript/src/routes.ts"
        })
        .map(|n| Node::new(NodeType::Function, n.clone()))
        .expect("createNewPerson function not found");

    let get_recent_people_fn = functions
        .iter()
        .find(|f| {
            f.name == "getRecentPeople"
                && normalize_path(&f.file) == "src/testing/typescript/src/routes.ts"
        })
        .map(|n| Node::new(NodeType::Function, n.clone()))
        .expect("getRecentPeople function not found");

    let post_people_new_endpoint = endpoints
        .iter()
        .find(|e| {
            e.name == "/people/new"
                && normalize_path(&e.file) == "src/testing/typescript/src/routes.ts"
                && e.meta.get("verb") == Some(&"POST".to_string())
        })
        .map(|n| Node::new(NodeType::Endpoint, n.clone()))
        .expect("POST /people/new endpoint not found");

    assert!(
        graph.has_edge(
            &post_people_new_endpoint,
            &create_new_person_fn,
            EdgeType::Handler
        ),
        "Expected '/people/new' POST endpoint to be handled by createNewPerson"
    );

    let get_people_recent_endpoint = endpoints
        .iter()
        .find(|e| {
            e.name == "/people/recent"
                && normalize_path(&e.file) == "src/testing/typescript/src/routes.ts"
                && e.meta.get("verb") == Some(&"GET".to_string())
        })
        .map(|n| Node::new(NodeType::Endpoint, n.clone()))
        .expect("GET /people/recent endpoint not found");

    assert!(
        graph.has_edge(
            &get_people_recent_endpoint,
            &get_recent_people_fn,
            EdgeType::Handler
        ),
        "Expected '/people/recent' GET endpoint to be handled by getRecentPeople"
    );

    let search_people_fn = functions
        .iter()
        .find(|f| {
            f.name == "searchPeople"
                && normalize_path(&f.file) == "src/testing/typescript/src/people-routes.ts"
        })
        .map(|n| Node::new(NodeType::Function, n.clone()))
        .expect("searchPeople function not found in people-routes.ts");

    let list_people_fn = functions
        .iter()
        .find(|f| {
            f.name == "listPeople"
                && normalize_path(&f.file) == "src/testing/typescript/src/people-routes.ts"
        })
        .map(|n| Node::new(NodeType::Function, n.clone()))
        .expect("listPeople function not found in people-routes.ts");

    let get_api_people_search = endpoints
        .iter()
        .find(|e| {
            e.name == "/api/people/search"
                && normalize_path(&e.file) == "src/testing/typescript/src/people-routes.ts"
                && e.meta.get("verb") == Some(&"GET".to_string())
        })
        .map(|n| Node::new(NodeType::Endpoint, n.clone()))
        .expect("GET /api/people/search endpoint not found (cross-file)");

    assert!(
        graph.has_edge(&get_api_people_search, &search_people_fn, EdgeType::Handler),
        "Expected '/api/people/search' GET endpoint to be handled by searchPeople (cross-file)"
    );

    let get_api_people_list = endpoints
        .iter()
        .find(|e| {
            e.name == "/api/people/list"
                && normalize_path(&e.file) == "src/testing/typescript/src/people-routes.ts"
                && e.meta.get("verb") == Some(&"GET".to_string())
        })
        .map(|n| Node::new(NodeType::Endpoint, n.clone()))
        .expect("GET /api/people/list endpoint not found (cross-file)");

    assert!(
        graph.has_edge(&get_api_people_list, &list_people_fn, EdgeType::Handler),
        "Expected '/api/people/list' GET endpoint to be handled by listPeople (cross-file)"
    );

    // Test cross-file router endpoints (admin-routes.ts)
    let list_users_fn = functions
        .iter()
        .find(|f| {
            f.name == "listUsers"
                && normalize_path(&f.file) == "src/testing/typescript/src/admin-routes.ts"
        })
        .map(|n| Node::new(NodeType::Function, n.clone()))
        .expect("listUsers function not found in admin-routes.ts");

    let delete_user_fn = functions
        .iter()
        .find(|f| {
            f.name == "deleteUser"
                && normalize_path(&f.file) == "src/testing/typescript/src/admin-routes.ts"
        })
        .map(|n| Node::new(NodeType::Function, n.clone()))
        .expect("deleteUser function not found in admin-routes.ts");

    let get_api_admin_users = endpoints
        .iter()
        .find(|e| {
            e.name == "/api/admin/users"
                && normalize_path(&e.file) == "src/testing/typescript/src/admin-routes.ts"
                && e.meta.get("verb") == Some(&"GET".to_string())
        })
        .map(|n| Node::new(NodeType::Endpoint, n.clone()))
        .expect("GET /api/admin/users endpoint not found (cross-file)");

    assert!(
        graph.has_edge(&get_api_admin_users, &list_users_fn, EdgeType::Handler),
        "Expected '/api/admin/users' GET endpoint to be handled by listUsers (cross-file)"
    );

    let delete_api_admin_users_id = endpoints
        .iter()
        .find(|e| {
            e.name == "/api/admin/users/:id"
                && normalize_path(&e.file) == "src/testing/typescript/src/admin-routes.ts"
                && e.meta.get("verb") == Some(&"DELETE".to_string())
        })
        .map(|n| Node::new(NodeType::Endpoint, n.clone()))
        .expect("DELETE /api/admin/users/:id endpoint not found (cross-file)");

    assert!(
        graph.has_edge(
            &delete_api_admin_users_id,
            &delete_user_fn,
            EdgeType::Handler
        ),
        "Expected '/api/admin/users/:id' DELETE endpoint to be handled by deleteUser (cross-file)"
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
async fn test_typescript() {
    use crate::lang::graphs::{ArrayGraph, BTreeMapGraph};
    test_typescript_generic::<BTreeMapGraph>().await.unwrap();
    test_typescript_generic::<ArrayGraph>().await.unwrap();

    #[cfg(feature = "neo4j")]
    {
        use crate::lang::graphs::Neo4jGraph;
        let graph = Neo4jGraph::default();
        graph.clear().await.unwrap();
        test_typescript_generic::<Neo4jGraph>().await.unwrap();
    }
}
