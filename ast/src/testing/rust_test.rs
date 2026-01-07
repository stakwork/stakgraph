use crate::lang::graphs::{EdgeType, NodeType};
use crate::lang::{Graph, Node};
use crate::utils::sanitize_string;
use crate::{lang::Lang, repo::Repo};
use shared::Result;
use std::str::FromStr;

pub async fn verify_rust<G: Graph>(graph: &G) -> Result<()> {
    // graph.analysis();

    // graph.analysis();

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
    assert_eq!(files.len(), 18, "Expected 18 files (added lib.rs)");

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
    assert_eq!(imports.len(), 13, "Expected 13 imports");

    let traits = graph.find_nodes_by_type(NodeType::Trait);
    nodes_count += traits.len();
    assert_eq!(traits.len(), 4, "Expected 4 trait nodes");

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
    assert_eq!(data_models.len(), 18, "Expected 18 data models");

    let person_dm = data_models
        .iter()
        .find(|dm| dm.name == "Person" && dm.file.ends_with("src/testing/rust/src/db.rs"))
        .map(|n| Node::new(NodeType::DataModel, n.clone()))
        .expect("Data model 'Person' not found in models.rs");

    let person_attrs = person_dm.node_data.meta.get("attributes");
    assert!(person_attrs.is_some(), "Person should have attributes");
    let attrs = person_attrs.unwrap();
    assert!(
        attrs.contains("derive"),
        "Person attributes should contain 'derive'"
    );
    assert!(
        attrs.contains("Debug"),
        "Person attributes should contain 'Debug'"
    );
    assert!(
        attrs.contains("Serialize"),
        "Person attributes should contain 'Serialize'"
    );
    assert!(
        attrs.contains("FromRow"),
        "Person attributes should contain 'FromRow'"
    );

    let classes = graph.find_nodes_by_type(NodeType::Class);
    nodes_count += classes.len();
    assert_eq!(classes.len(), 7, "Expected 7 class nodes");

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

    let greeter_dm = data_models
        .iter()
        .find(|dm| dm.name == "Greeter" && dm.file.ends_with("src/testing/rust/src/traits.rs"))
        .expect("Data model 'Greeter' not found in traits.rs");
    let greeter_attrs = greeter_dm.meta.get("attributes");
    assert!(greeter_attrs.is_some(), "Greeter should have attributes");
    let attrs = greeter_attrs.unwrap();
    assert!(
        attrs.contains("derive"),
        "Greeter attributes should contain 'derive'"
    );
    assert!(
        attrs.contains("Debug"),
        "Greeter attributes should contain 'Debug'"
    );
    assert!(
        attrs.contains("Clone"),
        "Greeter attributes should contain 'Clone'"
    );

    let multi_attr_dm = data_models
        .iter()
        .find(|dm| {
            dm.name == "MultiAttrStruct" && dm.file.ends_with("src/testing/rust/src/traits.rs")
        })
        .expect("Data model 'MultiAttrStruct' not found in traits.rs");
    let multi_attrs = multi_attr_dm.meta.get("attributes");
    assert!(
        multi_attrs.is_some(),
        "MultiAttrStruct should have attributes"
    );
    let attrs = multi_attrs.unwrap();
    assert!(
        attrs.contains("derive"),
        "MultiAttrStruct attributes should contain 'derive'"
    );

    assert!(
        attrs.contains("Debug") || attrs.contains("Clone") || attrs.contains("PartialEq"),
        "MultiAttrStruct attributes should contain derive traits"
    );

    let endpoints = graph.find_nodes_by_type(NodeType::Endpoint);
    nodes_count += endpoints.len();
    assert_eq!(endpoints.len(), 18, "Expected 18 endpoints");

    let imported_edges = graph.count_edges_of_type(EdgeType::Imports);
    edges_count += imported_edges;
    assert_eq!(imported_edges, 12, "Expected 12 import edges");

    let contains_edges = graph.count_edges_of_type(EdgeType::Contains);
    edges_count += contains_edges;
    assert_eq!(contains_edges, 254, "Expected 254 contains edges (was 197)");

    let calls_edges = graph.count_edges_of_type(EdgeType::Calls);
    edges_count += calls_edges;
    assert_eq!(calls_edges, 104, "Expected 104 calls edges");

    let functions = graph.find_nodes_by_type(NodeType::Function);
    nodes_count += functions.len();
    assert_eq!(functions.len(), 74, "Expected 74 functions");

    let macros: Vec<_> = functions
        .iter()
        .filter(|f| f.meta.get("macro") == Some(&"true".to_string()))
        .collect();
    assert_eq!(
        macros.len(),
        5,
        "Expected 5 macros (say_hello, create_function, log_expr, make_struct, impl_display)"
    );

    let say_hello_macro = macros
        .iter()
        .find(|m| m.name == "say_hello" && m.file.ends_with("src/testing/rust/src/macros.rs"));
    assert!(
        say_hello_macro.is_some(),
        "Expected say_hello! macro to be captured"
    );

    let create_function_macro = macros.iter().find(|m| {
        m.name == "create_function" && m.file.ends_with("src/testing/rust/src/macros.rs")
    });
    assert!(
        create_function_macro.is_some(),
        "Expected create_function! macro to be captured"
    );

    let internal_helper_fn = functions
        .iter()
        .find(|f| f.name == "internal_helper" && f.file.ends_with("src/testing/rust/src/db.rs"))
        .expect("internal_helper function not found in db.rs");
    let helper_attrs = internal_helper_fn.meta.get("attributes");
    assert!(
        helper_attrs.is_some(),
        "internal_helper should have attributes"
    );
    let attrs = helper_attrs.unwrap();
    assert!(
        attrs.contains("inline"),
        "internal_helper attributes should contain 'inline'"
    );

    let advanced_fn = functions
        .iter()
        .find(|f| f.name == "advanced_feature" && f.file.ends_with("src/testing/rust/src/db.rs"))
        .expect("advanced_feature function not found in db.rs");
    let cfg_attrs = advanced_fn.meta.get("attributes");
    assert!(
        cfg_attrs.is_some(),
        "advanced_feature should have attributes"
    );
    let attrs = cfg_attrs.unwrap();
    assert!(
        attrs.contains("cfg"),
        "advanced_feature attributes should contain 'cfg'"
    );

    let multi_attr_fn = functions
        .iter()
        .find(|f| {
            f.name == "multi_attribute_function" && f.file.ends_with("src/testing/rust/src/db.rs")
        })
        .expect("multi_attribute_function not found in db.rs");
    let multi_fn_attrs = multi_attr_fn.meta.get("attributes");
    assert!(
        multi_fn_attrs.is_some(),
        "multi_attribute_function should have attributes"
    );
    let attrs = multi_fn_attrs.unwrap();
    assert!(
        attrs.contains("inline") || attrs.contains("must_use") || attrs.contains("deprecated"),
        "multi_attribute_function should have at least one attribute captured"
    );

    let multi_fn_interface = multi_attr_fn.meta.get("interface");
    assert!(
        multi_fn_interface.is_some(),
        "multi_attribute_function should have interface"
    );
    let interface = multi_fn_interface.unwrap();
    assert!(
        interface.contains("#["),
        "interface should contain attribute markers"
    );
    assert!(
        interface.contains("pub fn multi_attribute_function"),
        "interface should contain function signature"
    );

    assert!(
        multi_attr_fn.body.contains("#["),
        "body should contain attribute markers"
    );
    assert!(
        multi_attr_fn
            .body
            .contains("pub fn multi_attribute_function"),
        "body should contain full function with attributes"
    );

    let unit_tests = graph.find_nodes_by_type(NodeType::UnitTest);
    nodes_count += unit_tests.len();
    assert_eq!(unit_tests.len(), 43, "Expected 43 unit tests");

    let integration_tests = graph.find_nodes_by_type(NodeType::IntegrationTest);
    nodes_count += integration_tests.len();
    assert_eq!(integration_tests.len(), 17, "Expected 17 integration tests");

    let e2e_tests = graph.find_nodes_by_type(NodeType::E2eTest);
    nodes_count += e2e_tests.len();
    assert_eq!(e2e_tests.len(), 8, "Expected 8 e2e tests");

    let handlers = graph.count_edges_of_type(EdgeType::Handler);
    edges_count += handlers;
    assert_eq!(handlers, 18, "Expected 18 handler edges");

    let implements = graph.count_edges_of_type(EdgeType::Implements);
    edges_count += implements;
    assert_eq!(implements, 2, "Expected 2 implements edges");

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
        .find(|e| e.name == "/api/person/{id}" && e.file.ends_with("src/routes/actix_routes.rs"))
        .map(|n| Node::new(NodeType::Endpoint, n.clone()))
        .expect("GET /api/person/{id} endpoint not found in actix_routes.rs");

    let post_person_endpoint = endpoints
        .iter()
        .find(|e| e.name == "/api/person" && e.file.ends_with("src/routes/actix_routes.rs"))
        .map(|n| Node::new(NodeType::Endpoint, n.clone()))
        .expect("POST /api/person endpoint not found in actix_routes.rs");

    assert!(
        graph.has_edge(&get_person_endpoint, &get_person_fn, EdgeType::Handler),
        "Expected '/api/person/id' endpoint to be handled by get_person"
    );

    assert!(
        graph.has_edge(&post_person_endpoint, &create_person_fn, EdgeType::Handler),
        "Expected '/api/person' endpoint to be handled by create_person"
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

    let get_profile_fn = functions
        .iter()
        .find(|f| f.name == "get_profile" && f.file.ends_with("src/routes/axum_routes.rs"))
        .map(|n| Node::new(NodeType::Function, n.clone()))
        .expect("get_profile function not found in axum_routes.rs");

    let update_profile_fn = functions
        .iter()
        .find(|f| f.name == "update_profile" && f.file.ends_with("src/routes/axum_routes.rs"))
        .map(|n| Node::new(NodeType::Function, n.clone()))
        .expect("update_profile function not found in axum_routes.rs");

    let get_profile_endpoint = endpoints
        .iter()
        .find(|e| e.name == "/user/profile" && e.file.ends_with("src/routes/axum_routes.rs"))
        .map(|n| Node::new(NodeType::Endpoint, n.clone()))
        .expect("GET /user/profile endpoint not found (same-file nested)");

    let update_profile_endpoint = endpoints
        .iter()
        .find(|e| e.name == "/user/profile/update" && e.file.ends_with("src/routes/axum_routes.rs"))
        .map(|n| Node::new(NodeType::Endpoint, n.clone()))
        .expect("POST /user/profile/update endpoint not found (same-file nested)");

    assert!(
        graph.has_edge(&get_profile_endpoint, &get_profile_fn, EdgeType::Handler),
        "Expected '/user/profile' endpoint to be handled by get_profile (same-file nested)"
    );

    assert!(
        graph.has_edge(&update_profile_endpoint, &update_profile_fn, EdgeType::Handler),
        "Expected '/user/profile/update' endpoint to be handled by update_profile (same-file nested)"
    );

    let list_users_fn = functions
        .iter()
        .find(|f| f.name == "list_users" && f.file.ends_with("src/routes/admin_axum.rs"))
        .map(|n| Node::new(NodeType::Function, n.clone()))
        .expect("list_users function not found in admin_axum.rs");

    let delete_user_fn = functions
        .iter()
        .find(|f| f.name == "delete_user" && f.file.ends_with("src/routes/admin_axum.rs"))
        .map(|n| Node::new(NodeType::Function, n.clone()))
        .expect("delete_user function not found in admin_axum.rs");

    let list_users_endpoint = endpoints
        .iter()
        .find(|e| e.name == "/admin/users" && e.file.ends_with("src/routes/admin_axum.rs"))
        .map(|n| Node::new(NodeType::Endpoint, n.clone()))
        .expect("GET /admin/users endpoint not found (cross-file)");

    let delete_user_endpoint = endpoints
        .iter()
        .find(|e| e.name == "/admin/users/:id" && e.file.ends_with("src/routes/admin_axum.rs"))
        .map(|n| Node::new(NodeType::Endpoint, n.clone()))
        .expect("DELETE /admin/users/:id endpoint not found (cross-file)");

    assert!(
        graph.has_edge(&list_users_endpoint, &list_users_fn, EdgeType::Handler),
        "Expected '/admin/users' endpoint to be handled by list_users (cross-file)"
    );

    assert!(
        graph.has_edge(&delete_user_endpoint, &delete_user_fn, EdgeType::Handler),
        "Expected '/admin/users/:id' endpoint to be handled by delete_user (cross-file)"
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

    let get_profile_rocket_fn = functions
        .iter()
        .find(|f| f.name == "get_profile" && f.file.ends_with("src/routes/rocket_routes.rs"))
        .map(|n| Node::new(NodeType::Function, n.clone()))
        .expect("get_profile function not found in rocket_routes.rs");

    let update_profile_rocket_fn = functions
        .iter()
        .find(|f| f.name == "update_profile" && f.file.ends_with("src/routes/rocket_routes.rs"))
        .map(|n| Node::new(NodeType::Function, n.clone()))
        .expect("update_profile function not found in rocket_routes.rs");

    let get_profile_rocket_endpoint = endpoints
        .iter()
        .find(|e| e.name == "/user/profile" && e.file.ends_with("src/routes/rocket_routes.rs"))
        .map(|n| Node::new(NodeType::Endpoint, n.clone()))
        .expect("GET /user/profile endpoint not found (Rocket same-file)");

    let update_profile_rocket_endpoint = endpoints
        .iter()
        .find(|e| {
            e.name == "/user/profile/update" && e.file.ends_with("src/routes/rocket_routes.rs")
        })
        .map(|n| Node::new(NodeType::Endpoint, n.clone()))
        .expect("POST /user/profile/update endpoint not found (Rocket same-file)");

    assert!(
        graph.has_edge(
            &get_profile_rocket_endpoint,
            &get_profile_rocket_fn,
            EdgeType::Handler
        ),
        "Expected '/user/profile' GET endpoint to be handled by get_profile (Rocket same-file)"
    );

    assert!(
        graph.has_edge(&update_profile_rocket_endpoint, &update_profile_rocket_fn, EdgeType::Handler),
        "Expected '/user/profile/update' POST endpoint to be handled by update_profile (Rocket same-file)"
    );

    let list_users_rocket_fn = functions
        .iter()
        .find(|f| f.name == "list_users" && f.file.ends_with("src/routes/admin_rocket.rs"))
        .map(|n| Node::new(NodeType::Function, n.clone()))
        .expect("list_users function not found in admin_rocket.rs");

    let delete_user_rocket_fn = functions
        .iter()
        .find(|f| f.name == "delete_user" && f.file.ends_with("src/routes/admin_rocket.rs"))
        .map(|n| Node::new(NodeType::Function, n.clone()))
        .expect("delete_user function not found in admin_rocket.rs");

    let list_users_rocket_endpoint = endpoints
        .iter()
        .find(|e| e.name == "/admin/users" && e.file.ends_with("src/routes/admin_rocket.rs"))
        .map(|n| Node::new(NodeType::Endpoint, n.clone()))
        .expect("GET /admin/users endpoint not found (Rocket cross-file)");

    let delete_user_rocket_endpoint = endpoints
        .iter()
        .find(|e| e.name == "/admin/users/<id>" && e.file.ends_with("src/routes/admin_rocket.rs"))
        .map(|n| Node::new(NodeType::Endpoint, n.clone()))
        .expect("DELETE /admin/users/<id> endpoint not found (Rocket cross-file)");

    assert!(
        graph.has_edge(
            &list_users_rocket_endpoint,
            &list_users_rocket_fn,
            EdgeType::Handler
        ),
        "Expected '/admin/users' GET endpoint to be handled by list_users (Rocket cross-file)"
    );

    assert!(
        graph.has_edge(&delete_user_rocket_endpoint, &delete_user_rocket_fn, EdgeType::Handler),
        "Expected '/admin/users/<id>' DELETE endpoint to be handled by delete_user (Rocket cross-file)"
    );

    let get_profile_fn = functions
        .iter()
        .find(|f| f.name == "get_profile" && f.file.ends_with("src/routes/actix_routes.rs"))
        .map(|n| Node::new(NodeType::Function, n.clone()))
        .expect("get_profile function not found in actix_routes.rs");

    let update_profile_fn = functions
        .iter()
        .find(|f| f.name == "update_profile" && f.file.ends_with("src/routes/actix_routes.rs"))
        .map(|n| Node::new(NodeType::Function, n.clone()))
        .expect("update_profile function not found in actix_routes.rs");

    let get_profile_endpoint = endpoints
        .iter()
        .find(|e| e.name == "/user/profile" && e.file.ends_with("src/routes/actix_routes.rs"))
        .map(|n| Node::new(NodeType::Endpoint, n.clone()))
        .expect("GET /user/profile endpoint not found");

    let update_profile_endpoint = endpoints
        .iter()
        .find(|e| {
            e.name == "/user/profile/update" && e.file.ends_with("src/routes/actix_routes.rs")
        })
        .map(|n| Node::new(NodeType::Endpoint, n.clone()))
        .expect("PUT /user/profile/update endpoint not found");

    assert!(
        graph.has_edge(&get_profile_endpoint, &get_profile_fn, EdgeType::Handler),
        "Expected '/user/profile' GET endpoint to be handled by get_profile"
    );

    assert!(
        graph.has_edge(
            &update_profile_endpoint,
            &update_profile_fn,
            EdgeType::Handler
        ),
        "Expected '/user/profile/update' POST endpoint to be handled by update_profile"
    );

    let list_users_fn = functions
        .iter()
        .find(|f| f.name == "list_users" && f.file.ends_with("src/routes/admin_actix.rs"))
        .map(|n| Node::new(NodeType::Function, n.clone()))
        .expect("list_users function not found in admin_actix.rs");

    let delete_user_fn = functions
        .iter()
        .find(|f| f.name == "delete_user" && f.file.ends_with("src/routes/admin_actix.rs"))
        .map(|n| Node::new(NodeType::Function, n.clone()))
        .expect("delete_user function not found in admin_actix.rs");

    let list_users_endpoint = endpoints
        .iter()
        .find(|e| e.name == "/admin/users" && e.file.ends_with("src/routes/admin_actix.rs"))
        .map(|n| Node::new(NodeType::Endpoint, n.clone()))
        .expect("GET /admin/users endpoint not found (cross-file)");

    let delete_user_endpoint = endpoints
        .iter()
        .find(|e| e.name == "/admin/users/{id}" && e.file.ends_with("src/routes/admin_actix.rs"))
        .map(|n| Node::new(NodeType::Endpoint, n.clone()))
        .expect("DELETE /admin/users/{id} endpoint not found (cross-file)");

    assert!(
        graph.has_edge(&list_users_endpoint, &list_users_fn, EdgeType::Handler),
        "Expected '/admin/users' GET endpoint to be handled by list_users (cross-file)"
    );

    assert!(
        graph.has_edge(&delete_user_endpoint, &delete_user_fn, EdgeType::Handler),
        "Expected '/admin/users/{{id}}' DELETE endpoint to be handled by delete_user (cross-file)"
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
    verify_rust(&graph).await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_rust() {
    use crate::lang::graphs::{ArrayGraph, BTreeMapGraph};
    test_rust_generic::<ArrayGraph>().await.unwrap();
    test_rust_generic::<BTreeMapGraph>().await.unwrap();

    #[cfg(feature = "neo4j")]
    {
        use crate::lang::graphs::Neo4jGraph;
        let graph = Neo4jGraph::with_namespace("test_lang_rust");
        graph.clear().await.unwrap();

        let repo = Repo::new(
            "src/testing/rust",
            Lang::from_str("rust").unwrap(),
            false,
            Vec::new(),
            Vec::new(),
        )
        .unwrap();

        // Build with custom instance
        #[cfg(feature = "neo4j")]
        std::env::set_var("NEO4J_TEST_NAMESPACE", "test_lang_rust");
        let graph = repo.build_graph_with_instance(graph, false).await.unwrap();
        #[cfg(feature = "neo4j")]
        std::env::remove_var("NEO4J_TEST_NAMESPACE");
        verify_rust(&graph).await.unwrap();
    }
}
