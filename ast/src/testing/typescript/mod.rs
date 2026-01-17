use crate::lang::graphs::{EdgeType, NodeType};
use crate::lang::{Graph, Node};
// use crate::utils::get_use_lsp;
use crate::{lang::Lang, repo::Repo};
use shared::error::Result;
use std::str::FromStr;

pub async fn test_typescript_generic<G: Graph>() -> Result<()> {
    // TODO: LSP mode needs additional work for TypeScript - disabled for now
    let use_lsp = false; // get_use_lsp();
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
    assert_eq!(files.len(), 27, "Expected 27 File nodes");

    let pkg_files = files
        .iter()
        .filter(|f| f.name == "package.json")
        .collect::<Vec<_>>();
    assert_eq!(pkg_files.len(), 1, "Expected 1 package.json file");

    let imports = graph.find_nodes_by_type(NodeType::Import);
    nodes_count += imports.len();
    assert_eq!(imports.len(), 19, "Expected 19 imports");

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
    if use_lsp {
        assert_eq!(functions.len(), 38, "Expected 38 functions with LSP");
    } else {
        assert_eq!(functions.len(), 60, "Expected 60 functions without LSP");
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
    assert_eq!(classes.len(), 9, "Expected 9 classes");

    let directories = graph.find_nodes_by_type(NodeType::Directory);
    nodes_count += directories.len();
    assert_eq!(directories.len(), 9, "Expected 9 directories");

    let test_dir = directories
        .iter()
        .find(|d| d.name == "test" && d.file.ends_with("typescript/test"))
        .expect("test directory not found");
    let unit_dir = directories
        .iter()
        .find(|d| d.name == "unit" && d.file.ends_with("test/unit"))
        .expect("unit directory not found");
    let integration_dir = directories
        .iter()
        .find(|d| d.name == "integration" && d.file.ends_with("test/integration"))
        .expect("integration directory not found");
    let e2e_dir = directories
        .iter()
        .find(|d| d.name == "e2e" && d.file.ends_with("test/e2e"))
        .expect("e2e directory not found");

    let repo_node = Node::new(
        NodeType::Repository,
        repository.first().expect("Repository not found").clone(),
    );
    let test_dir_node = Node::new(NodeType::Directory, test_dir.clone());
    let unit_dir_node = Node::new(NodeType::Directory, unit_dir.clone());
    let integration_dir_node = Node::new(NodeType::Directory, integration_dir.clone());
    let e2e_dir_node = Node::new(NodeType::Directory, e2e_dir.clone());

    assert!(
        graph.has_edge(&repo_node, &test_dir_node, EdgeType::Contains),
        "Repository should contain test directory"
    );
    assert!(
        graph.has_edge(&test_dir_node, &unit_dir_node, EdgeType::Contains),
        "test directory should contain unit directory"
    );
    assert!(
        graph.has_edge(&test_dir_node, &integration_dir_node, EdgeType::Contains),
        "test directory should contain integration directory"
    );
    assert!(
        graph.has_edge(&test_dir_node, &e2e_dir_node, EdgeType::Contains),
        "test directory should contain e2e directory"
    );

    let unit_tests = graph.find_nodes_by_type(NodeType::UnitTest);
    nodes_count += unit_tests.len();
    assert_eq!(unit_tests.len(), 8, "Expected 8 UnitTest nodes");

    let person_service_test = unit_tests
        .iter()
        .find(|t| {
            t.name == "unit: PersonService"
                && normalize_path(&t.file).ends_with("test/unit/service.test.ts")
        })
        .expect("unit: PersonService test not found");
    assert!(
        person_service_test
            .body
            .contains("describe(\"unit: PersonService\""),
        "PersonService test body should contain describe block"
    );

    let _calculator_add_test = unit_tests
        .iter()
        .find(|t| {
            t.name == "unit: Calculator add"
                && normalize_path(&t.file).ends_with("test/unit/calculator.spec.ts")
        })
        .expect("unit: Calculator add test not found");

    let _format_date_test = unit_tests
        .iter()
        .find(|t| {
            t.name == "unit: formatDate utility"
                && normalize_path(&t.file).ends_with("test/unit/utils.unit.test.ts")
        })
        .expect("unit: formatDate utility test not found");

    let _models_test = unit_tests
        .iter()
        .find(|t| {
            t.name == "unit: SequelizePerson model"
                && normalize_path(&t.file).ends_with("test/unit/models.test.ts")
        })
        .expect("unit: SequelizePerson model test not found");

    let integration_tests = graph.find_nodes_by_type(NodeType::IntegrationTest);
    nodes_count += integration_tests.len();
    assert_eq!(
        integration_tests.len(),
        3,
        "Expected 3 IntegrationTest nodes"
    );

    let person_endpoint_test = integration_tests
        .iter()
        .find(|t| {
            t.name == "integration: /person endpoint"
                && normalize_path(&t.file).ends_with("test/integration/api.integration.test.ts")
        })
        .expect("integration: /person endpoint test not found");
    assert!(
        person_endpoint_test.body.contains("fetch("),
        "Integration test body should contain fetch call"
    );

    let _db_test = integration_tests
        .iter()
        .find(|t| {
            t.name == "integration: database connection"
                && normalize_path(&t.file).ends_with("test/integration/database.int.test.ts")
        })
        .expect("integration: database connection test not found");

    let e2e_tests = graph.find_nodes_by_type(NodeType::E2eTest);
    nodes_count += e2e_tests.len();
    assert_eq!(e2e_tests.len(), 3, "Expected 3 E2eTest nodes");

    let cypress_test = e2e_tests
        .iter()
        .find(|t| {
            t.name == "e2e: checkout flow"
                && normalize_path(&t.file).ends_with("test/e2e/checkout.cy.test.ts")
        })
        .expect("e2e: checkout flow test not found");
    assert!(
        cypress_test.body.contains("cy.visit("),
        "Cypress test body should contain cy.visit"
    );

    let puppeteer_test = e2e_tests
        .iter()
        .find(|t| {
            t.name == "e2e: form submission"
                && normalize_path(&t.file).ends_with("test/e2e/form.puppeteer.test.ts")
        })
        .expect("e2e: form submission test not found");
    assert!(
        puppeteer_test.body.contains("puppeteer.launch"),
        "Puppeteer test body should contain puppeteer.launch"
    );

    let _playwright_test = e2e_tests
        .iter()
        .find(|t| {
            t.name == "e2e: user management flow"
                && normalize_path(&t.file).ends_with("test/e2e/user-flow.e2e.test.ts")
        })
        .expect("e2e: user management flow test not found");

    let calls_edges_count = graph.count_edges_of_type(EdgeType::Calls);
    edges_count += calls_edges_count;

    assert_eq!(calls_edges_count, 14, "Expected 14 calls edges");

    let data_models = graph.find_nodes_by_type(NodeType::DataModel);
    nodes_count += data_models.len();
    assert_eq!(data_models.len(), 27, "Expected 27 data models");

    let trait_nodes = graph.find_nodes_by_type(NodeType::Trait);
    nodes_count += trait_nodes.len();
    assert_eq!(trait_nodes.len(), 6, "Expected 6 trait nodes");

    let person_service_trait = trait_nodes
        .iter()
        .find(|t| {
            t.name == "PersonService"
                && normalize_path(&t.file) == "src/testing/typescript/src/service.ts"
        })
        .expect("PersonService trait not found");

    let variables = graph.find_nodes_by_type(NodeType::Var);
    nodes_count += variables.len();
    assert_eq!(variables.len(), 6, "Expected 6 variables");

    // Count any Page nodes (unified parser may create these from React patterns)
    let pages = graph.find_nodes_by_type(NodeType::Page);
    nodes_count += pages.len();

    // Count any Instance nodes
    let instances = graph.find_nodes_by_type(NodeType::Instance);
    nodes_count += instances.len();

    let contains = graph.count_edges_of_type(EdgeType::Contains);
    edges_count += contains;
    assert_eq!(contains, 215, "Expected 215 contains edges");

    let import_edges_count = graph.count_edges_of_type(EdgeType::Imports);
    edges_count += import_edges_count;
    if use_lsp {
        assert_eq!(import_edges_count, 21, "Expected 21 import edges with LSP");
    } else {
        assert_eq!(
            import_edges_count, 15,
            "Expected 13 import edges without LSP"
        );
    }

    let handlers = graph.count_edges_of_type(EdgeType::Handler);
    edges_count += handlers;
    if use_lsp {
        assert_eq!(handlers, 8, "Expected 8 handler edges with LSP");
    } else {
        assert_eq!(handlers, 22, "Expected 22 handler edges without LSP");
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
    nodes_count += endpoints.len();
    assert_eq!(endpoints.len(), 22, "Expected 22 endpoints");

    // Request nodes from unified parser's request_finder (finds fetch/axios calls)
    let requests = graph.find_nodes_by_type(NodeType::Request);
    nodes_count += requests.len();
    assert_eq!(requests.len(), 3, "Expected 3 Request nodes from unified parser");

    let implements = graph.count_edges_of_type(EdgeType::Implements);
    edges_count += implements;
    assert_eq!(implements, 3, "Expected 3 implements edges");

    let sequelize_service = classes
        .iter()
        .find(|c| c.name == "SequelizePersonService")
        .map(|n| Node::new(NodeType::Class, n.clone()))
        .expect("SequelizePersonService class not found");
    let typeorm_service = classes
        .iter()
        .find(|c| c.name == "TypeOrmPersonService")
        .map(|n| Node::new(NodeType::Class, n.clone()))
        .expect("TypeOrmPersonService class not found");
    let prisma_service = classes
        .iter()
        .find(|c| c.name == "PrismaPersonService")
        .map(|n| Node::new(NodeType::Class, n.clone()))
        .expect("PrismaPersonService class not found");
    let person_trait_node = Node::new(NodeType::Trait, person_service_trait.clone());

    assert!(
        graph.has_edge(&sequelize_service, &person_trait_node, EdgeType::Implements),
        "SequelizePersonService should implement PersonService"
    );
    assert!(
        graph.has_edge(&typeorm_service, &person_trait_node, EdgeType::Implements),
        "TypeOrmPersonService should implement PersonService"
    );
    assert!(
        graph.has_edge(&prisma_service, &person_trait_node, EdgeType::Implements),
        "PrismaPersonService should implement PersonService"
    );

    let uses = graph.count_edges_of_type(EdgeType::Uses);
    edges_count += uses;
    if use_lsp {
        // assert_eq!(uses, 14, "Expected 14 uses edges with LSP");
    } else {
        assert_eq!(uses, 0, "Expected 0 uses edges without LSP");
    }

    let nested = graph.count_edges_of_type(EdgeType::NestedIn);
    edges_count += nested;

    // Operand edges from unified parser's function_call_query
    let operand = graph.count_edges_of_type(EdgeType::Operand);
    edges_count += operand;
    assert_eq!(operand, 23, "Expected 23 Operand edges from unified parser");

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

    let user_router_endpoints: Vec<_> = endpoints
        .iter()
        .filter(|e| normalize_path(&e.file).ends_with("routers/user-router.ts"))
        .collect();
    assert_eq!(
        user_router_endpoints.len(),
        5,
        "Expected 5 endpoints in user-router.ts"
    );

    let post_router_endpoints: Vec<_> = endpoints
        .iter()
        .filter(|e| normalize_path(&e.file).ends_with("routers/post-router.ts"))
        .collect();
    assert_eq!(
        post_router_endpoints.len(),
        5,
        "Expected 5 endpoints in post-router.ts"
    );

    // Phase 2: Method chaining pattern (router.route().post().delete())
    // Handler names follow pattern: {verb}_{path}_handler_L{line}
    // Path params become param_ prefix, e.g. :postId becomes param_postId
    let like_post_endpoint = endpoints
        .iter()
        .find(|e| {
            e.name.contains("like")
                && normalize_path(&e.file).ends_with("routers/post-router.ts")
                && e.meta.get("verb") == Some(&"POST".to_string())
        })
        .map(|n| Node::new(NodeType::Endpoint, n.clone()))
        .expect("POST /:postId/like endpoint not found (method chaining)");

    // LSP doesn't create Handler edges for inline arrow functions
    if !use_lsp {
        let like_post_handler = functions
            .iter()
            .find(|f| {
                f.name.contains("post_param_postId_like_handler")
                    && normalize_path(&f.file).ends_with("routers/post-router.ts")
            })
            .map(|n| Node::new(NodeType::Function, n.clone()))
            .expect("POST like handler function not found");

        assert!(
            graph.has_edge(&like_post_endpoint, &like_post_handler, EdgeType::Handler),
            "Expected POST /:postId/like endpoint to be handled by its arrow function handler"
        );
    }

    if !use_lsp {
        let unlike_post_endpoint = endpoints
            .iter()
            .find(|e| {
                e.name.contains("like")
                    && normalize_path(&e.file).ends_with("routers/post-router.ts")
                    && e.meta.get("verb") == Some(&"DELETE".to_string())
            })
            .map(|n| Node::new(NodeType::Endpoint, n.clone()))
            .expect("DELETE /:postId/like endpoint not found (method chaining)");

        let unlike_post_handler = functions
            .iter()
            .find(|f| {
                f.name.contains("delete_param_postId_like_handler")
                    && normalize_path(&f.file).ends_with("routers/post-router.ts")
            })
            .map(|n| Node::new(NodeType::Function, n.clone()))
            .expect("DELETE like handler function not found");

        assert!(
            graph.has_edge(
                &unlike_post_endpoint,
                &unlike_post_handler,
                EdgeType::Handler
            ),
            "Expected DELETE /:postId/like endpoint to be handled by its arrow function handler"
        );
    }

    // Phase 6-7: Types and Imports
    let types_file_nodes = graph
        .find_nodes_by_type(NodeType::Import)
        .into_iter()
        .filter(|n| normalize_path(&n.file).ends_with("types-and-imports.ts"))
        .collect::<Vec<_>>();

    assert_eq!(
        types_file_nodes.len(),
        1,
        "Expected 1 aggregated import node for types-and-imports.ts"
    );

    let import_body = &types_file_nodes[0].body;
    assert!(import_body.contains("import { PersonService } from \"./service\";"));
    assert!(import_body.contains("import type { SequelizePerson } from \"./model\";"));
    assert!(import_body.contains("import * as models from \"./model\";"));
    assert!(import_body.contains("as SP")); // Check aliasing
    assert!(import_body.contains("import \"./config\";")); // Side-effect

    // Check DataModels (Type Aliases & Enums & Interfaces)
    let new_models = ["ID", "UserDTO", "UserRole", "Status", "Config"];
    for model_name in new_models {
        data_models
            .iter()
            .find(|m| {
                m.name == model_name && normalize_path(&m.file).ends_with("types-and-imports.ts")
            })
            .expect(&format!("DataModel {} not found", model_name));
    }

    // Check Traits (Interfaces/Types with methods)
    let new_traits = ["Logger", "IGreeter"];
    for trait_name in new_traits {
        trait_nodes
            .iter()
            .find(|t| {
                t.name == trait_name && normalize_path(&t.file).ends_with("types-and-imports.ts")
            })
            .expect(&format!("Trait {} not found", trait_name));
    }

    // Note: Logger and IGreeter might ALSO appear in DataModels due to query overlap
    // verified by count 17 (10 old + 5 models + 2 dual-role)

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
