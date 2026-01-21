use crate::lang::{BTreeMapGraph, EdgeType, Graph, NodeType};
use crate::repo::Repo;

use shared::error::Result;
use std::path::Path;

const MONOREPO_TEST_DIR: &str = "src/testing/monorepo";

#[test]
fn test_monorepo_rust_structure() {
    let base = Path::new(MONOREPO_TEST_DIR).join("monorepo_rust");
    assert!(base.join("Cargo.toml").exists(), "Missing root Cargo.toml");
    assert!(
        base.join("api/Cargo.toml").exists(),
        "Missing api/Cargo.toml"
    );
    assert!(
        base.join("shared/Cargo.toml").exists(),
        "Missing shared/Cargo.toml"
    );
}

#[test]
fn test_monorepo_npm_go_structure() {
    let base = Path::new(MONOREPO_TEST_DIR).join("monorepo_npm_go");
    assert!(
        base.join("package.json").exists(),
        "Missing root package.json"
    );
    assert!(base.join("go.mod").exists(), "Missing go.mod");
    assert!(
        base.join("apps/web/package.json").exists(),
        "Missing apps/web/package.json"
    );
}

#[test]
fn test_monorepo_turbo_python_structure() {
    let base = Path::new(MONOREPO_TEST_DIR).join("monorepo_turbo_python");
    assert!(base.join("turbo.json").exists(), "Missing turbo.json");
    assert!(
        base.join("package.json").exists(),
        "Missing root package.json"
    );
    assert!(
        base.join("services/api/requirements.txt").exists(),
        "Missing Python requirements"
    );
}

#[test]
fn test_monorepo_go_structure() {
    let base = Path::new(MONOREPO_TEST_DIR).join("monorepo_go");
    assert!(base.join("go.work").exists(), "Missing go.work");
    assert!(base.join("api/go.mod").exists(), "Missing api/go.mod");
    assert!(base.join("worker/go.mod").exists(), "Missing worker/go.mod");
    assert!(base.join("pkg/go.mod").exists(), "Missing pkg/go.mod");
}

#[test]
fn test_monorepo_python_rust_structure() {
    let base = Path::new(MONOREPO_TEST_DIR).join("monorepo_python_rust");
    assert!(
        base.join("pyproject.toml").exists(),
        "Missing pyproject.toml"
    );
    assert!(
        base.join("services/web/requirements.txt").exists(),
        "Missing Python requirements"
    );
    assert!(
        base.join("services/processor/Cargo.toml").exists(),
        "Missing Rust Cargo.toml"
    );
}

#[test]
fn test_monorepo_nx_mixed_structure() {
    let base = Path::new(MONOREPO_TEST_DIR).join("monorepo_nx_mixed");
    assert!(base.join("nx.json").exists(), "Missing nx.json");
    assert!(
        base.join("apps/web/package.json").exists(),
        "Missing Angular app"
    );
    assert!(base.join("apps/api/go.mod").exists(), "Missing Go backend");
}

#[test]
fn test_monorepo_turbo_ts_structure() {
    let base = Path::new(MONOREPO_TEST_DIR).join("monorepo_turbo_ts");
    assert!(base.join("turbo.json").exists(), "Missing turbo.json");
    assert!(
        base.join("apps/web/package.json").exists(),
        "Missing Next.js app"
    );
    assert!(
        base.join("apps/api/package.json").exists(),
        "Missing Express app"
    );
    assert!(
        base.join("packages/ui/package.json").exists(),
        "Missing UI package"
    );
}

#[test]
fn test_monorepo_simple_ts_structure() {
    let base = Path::new(MONOREPO_TEST_DIR).join("monorepo_simple_ts");
    assert!(
        base.join("package.json").exists(),
        "Missing root package.json"
    );
    assert!(
        base.join("frontend/package.json").exists(),
        "Missing frontend"
    );
    assert!(
        base.join("backend/package.json").exists(),
        "Missing backend"
    );
}

#[test]
fn test_monorepo_rust_root_files() {
    let base = Path::new(MONOREPO_TEST_DIR).join("monorepo_rust");
    assert!(base.join("CLAUDE.md").exists(), "Missing CLAUDE.md");
    assert!(base.join(".cursorrules").exists(), "Missing .cursorrules");
    assert!(base.join("README.md").exists(), "Missing README.md");
    assert!(base.join("Dockerfile").exists(), "Missing Dockerfile");
    assert!(base.join(".gitignore").exists(), "Missing .gitignore");
}

#[test]
fn test_monorepo_npm_go_root_files() {
    let base = Path::new(MONOREPO_TEST_DIR).join("monorepo_npm_go");
    assert!(base.join("README.md").exists(), "Missing README.md");
    assert!(
        base.join(".windsurfrules").exists(),
        "Missing .windsurfrules"
    );
    assert!(base.join("Dockerfile").exists(), "Missing Dockerfile");
    assert!(
        base.join("docker-compose.yml").exists(),
        "Missing docker-compose.yml"
    );
}

#[test]
fn test_monorepo_turbo_ts_root_files() {
    let base = Path::new(MONOREPO_TEST_DIR).join("monorepo_turbo_ts");
    assert!(base.join("AGENTS.md").exists(), "Missing AGENTS.md");
    assert!(base.join("README.md").exists(), "Missing README.md");
    assert!(base.join("Dockerfile").exists(), "Missing Dockerfile");
}

async fn check_monorepo_graph<G: Graph + 'static>(
    fixture: &str,
    expected_langs: &[&str],
    expected_files: &[&str],
) -> Result<()> {
    let root = Path::new(MONOREPO_TEST_DIR).join(fixture);
    let abs_root = std::env::current_dir()?.join(&root).canonicalize()?;
    let abs_root_str = abs_root.to_str().unwrap();

    #[cfg(feature = "neo4j")]
    if std::any::TypeId::of::<G>() == std::any::TypeId::of::<crate::lang::graphs::Neo4jGraph>() {
        let neo_graph = crate::lang::graphs::Neo4jGraph::default();
        neo_graph.clear_existing_graph(abs_root_str).await?;
    }
    let repos = Repo::new_multi_detect(
        root.to_str().unwrap(),
        None,
        Vec::new(),
        Vec::new(),
        Some(false),
    )
    .await?;

    let graph = repos.build_graphs_inner::<G>().await?;

    let all_languages = graph.find_nodes_by_type(NodeType::Language);
    let languages: Vec<_> = all_languages
        .into_iter()
        .filter(|n| n.file.starts_with(abs_root_str))
        .collect();

    assert_eq!(
        languages.len(),
        expected_langs.len(),
        "Language count mismatch for {}: \n  Expected Languages: {:?}\n  Actual Languages: {:?}\n  Details: {}",
        fixture,
        expected_langs,
        languages.iter().map(|l| &l.name).collect::<Vec<_>>(),
        languages
            .iter()
            .enumerate()
            .map(|(i, l)| format!("\n    {}: {} (path: {})", i+1, l.name, l.file))
            .collect::<String>()
    );

    for lang in expected_langs {
        assert!(
            languages.iter().any(|l| l.name.eq_ignore_ascii_case(lang)),
            "Missing language '{}' in {}",
            lang,
            fixture
        );
    }

    // Check Files
    let all_files = graph.find_nodes_by_type(NodeType::File);
    let files: Vec<_> = all_files
        .into_iter()
        .filter(|n| n.file.starts_with(abs_root_str))
        .collect();

    for file in expected_files {
        let matching_files: Vec<_> = files.iter().filter(|f| f.file.ends_with(file)).collect();
        assert!(
            !matching_files.is_empty(),
            "Missing file '{}' in {}. \nAll detected files: {}\n",
            file,
            fixture,
            files
                .iter()
                .map(|f| format!("\n  - {}", f.file))
                .collect::<String>()
        );
    }

    Ok(())
}

#[tokio::test]
async fn test_monorepo_graph_construction() -> Result<()> {
    check_monorepo_graph::<crate::lang::BTreeMapGraph>("monorepo_nx_mixed", &[], &[]).await?;

    check_monorepo_graph::<crate::lang::BTreeMapGraph>("monorepo_npm_go", &[], &[]).await?;

    check_monorepo_graph::<BTreeMapGraph>("monorepo_rust", &[], &[]).await?;

    check_monorepo_graph::<BTreeMapGraph>("monorepo_npm_go", &[], &[]).await?;

    check_monorepo_graph::<BTreeMapGraph>("monorepo_turbo_ts", &[], &[]).await?;

    check_monorepo_graph::<BTreeMapGraph>("monorepo_python_rust", &[], &[]).await?;

    check_monorepo_graph::<BTreeMapGraph>("monorepo_simple_ts", &[], &[]).await?;

    Ok(())
}

#[cfg(feature = "neo4j")]
#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn test_monorepo_graph_construction_neo4j() -> Result<()> {
    check_monorepo_graph::<crate::lang::graphs::Neo4jGraph>("monorepo_nx_mixed", &[], &[]).await?;

    check_monorepo_graph::<crate::lang::graphs::Neo4jGraph>("monorepo_rust", &[], &[]).await?;

    check_monorepo_graph::<crate::lang::graphs::Neo4jGraph>("monorepo_npm_go", &[], &[]).await?;

    check_monorepo_graph::<crate::lang::graphs::Neo4jGraph>("monorepo_turbo_ts", &[], &[]).await?;
    check_monorepo_graph::<crate::lang::graphs::Neo4jGraph>("monorepo_python_rust", &[], &[])
        .await?;

    check_monorepo_graph::<crate::lang::graphs::Neo4jGraph>("monorepo_simple_ts", &[], &[]).await?;

    Ok(())
}

#[tokio::test]
async fn test_monorepo_root_files_in_graph() -> Result<()> {
    let root = Path::new(MONOREPO_TEST_DIR).join("monorepo_rust");
    let abs_root = std::env::current_dir()?.join(&root).canonicalize()?;
    let abs_root_str = abs_root.to_str().unwrap();

    let repos = Repo::new_multi_detect(
        root.to_str().unwrap(),
        None,
        Vec::new(),
        Vec::new(),
        Some(false),
    )
    .await?;

    let graph = repos
        .build_graphs_inner::<crate::lang::BTreeMapGraph>()
        .await?;

    let all_files = graph.find_nodes_by_type(NodeType::File);
    let root_str = root.to_str().unwrap();
    let files: Vec<_> = all_files
        .into_iter()
        .filter(|n| n.file.starts_with(abs_root_str) || n.file.starts_with(root_str))
        .collect();
    assert!(
        files.iter().any(|f| f.file.ends_with("CLAUDE.md")),
        "Missing CLAUDE.md in graph"
    );
    assert!(
        files.iter().any(|f| f.file.ends_with(".cursorrules")),
        "Missing .cursorrules in graph"
    );
    assert!(
        files.iter().any(|f| f.file.ends_with("Dockerfile")),
        "Missing Dockerfile in graph"
    );
    assert!(
        files.iter().any(|f| f.file.ends_with("README.md")),
        "Missing README.md in graph"
    );

    Ok(())
}

#[tokio::test]
async fn test_remote_monorepo_root_detection() -> Result<()> {
    use crate::repo::Repo;

    let repo_url = "https://github.com/fayekelmith/test-monorepo";

    // Clone and detect workspace
    let repos = Repo::new_clone_multi_detect(
        repo_url,
        None,
        None,
        Vec::new(),
        Vec::new(),
        None,
        None,
        Some(false),
    )
    .await?;

    assert_eq!(
        repos.0.len(),
        1,
        "Should detect 1 repo, got {}",
        repos.0.len()
    );

    let graph = repos
        .build_graphs_inner::<crate::lang::BTreeMapGraph>()
        .await?;

    let all_repos = graph.find_nodes_by_type(NodeType::Repository);

    assert_eq!(
        all_repos.len(),
        1,
        "Should have 1 Repository node (workspace root only), got {}",
        all_repos.len()
    );

    let root_repo = all_repos.iter().find(|r| r.name.contains("test-monorepo"));
    assert!(
        root_repo.is_some(),
        "Should have root Repository node containing 'test-monorepo', got: {:?}",
        all_repos.iter().map(|r| &r.name).collect::<Vec<_>>()
    );

    let all_files = graph.find_nodes_by_type(NodeType::File);
    let root_files: Vec<_> = all_files
        .iter()
        .filter(|f| {
            f.file.contains("test-monorepo")
                && !f.file.contains("packages/")
                && !f.file.contains("docs/")
        })
        .collect();

    assert!(
        root_files.iter().any(|f| f.name == "CLAUDE.md"),
        "Should capture CLAUDE.md at root level"
    );

    assert!(
        root_files.iter().any(|f| f.name == "README.md"),
        "Should capture README.md at root level"
    );

    assert!(
        root_files.iter().any(|f| f.name == "Dockerfile"),
        "Should capture Dockerfile at root level"
    );

    assert!(
        root_files.iter().any(|f| f.name == ".cursorrules"),
        "Should capture .cursorrules at root level"
    );

    let claude_file = root_files.iter().find(|f| f.name == "CLAUDE.md").unwrap();
    assert!(
        !claude_file.body.is_empty(),
        "CLAUDE.md should have body content"
    );

    let readme_file = root_files.iter().find(|f| f.name == "README.md").unwrap();
    assert!(
        !readme_file.body.is_empty(),
        "README.md should have body content"
    );
    assert!(claude_file.hash.is_some(), "CLAUDE.md should have hash set");

    Ok(())
}

#[tokio::test]
async fn test_remote_monorepo_comprehensive_graph() -> Result<()> {
    use crate::repo::Repo;

    let repo_url = "https://github.com/fayekelmith/test-monorepo";

    let repos = Repo::new_clone_multi_detect(
        repo_url,
        None,
        None,
        Vec::new(),
        Vec::new(),
        None,
        None,
        Some(false),
    )
    .await?;

    let graph = repos
        .build_graphs_inner::<crate::lang::BTreeMapGraph>()
        .await?;

    let all_endpoints = graph.find_nodes_by_type(NodeType::Endpoint);

    assert!(
        all_endpoints.len() >= 8,
        "Should have at least 8 endpoints (users CRUD + products + health), got {}",
        all_endpoints.len()
    );

    // Check for specific endpoints
    let has_get_users = all_endpoints
        .iter()
        .any(|e| e.name.contains("/users") && e.meta.get("verb") == Some(&"GET".to_string()));
    assert!(has_get_users, "Should have GET /users endpoint");

    let has_post_users = all_endpoints
        .iter()
        .any(|e| e.name.contains("/users") && e.meta.get("verb") == Some(&"POST".to_string()));
    assert!(has_post_users, "Should have POST /users endpoint");

    let has_put_users = all_endpoints
        .iter()
        .any(|e| e.name.contains("/users") && e.meta.get("verb") == Some(&"PUT".to_string()));
    assert!(has_put_users, "Should have PUT /users/:id endpoint");

    let has_delete_users = all_endpoints
        .iter()
        .any(|e| e.name.contains("/users") && e.meta.get("verb") == Some(&"DELETE".to_string()));
    assert!(has_delete_users, "Should have DELETE /users/:id endpoint");

    let has_health = all_endpoints
        .iter()
        .any(|e| e.name.contains("/health") && e.meta.get("verb") == Some(&"GET".to_string()));
    assert!(has_health, "Should have GET /health endpoint");

    let all_requests = graph.find_nodes_by_type(NodeType::Request);

    assert!(
        all_requests.len() >= 5,
        "Should have at least 5 requests (getUsers, getUserById, createUser, deleteUser, updateUser), got {}",
        all_requests.len()
    );

    let all_functions = graph.find_nodes_by_type(NodeType::Function);

    let has_format_response = all_functions.iter().any(|f| f.name == "formatResponse");
    assert!(
        has_format_response,
        "Should have formatResponse function from shared/utils.ts"
    );

    let has_validate_email = all_functions.iter().any(|f| f.name == "validateEmail");
    assert!(
        has_validate_email,
        "Should have validateEmail function from shared/utils.ts"
    );

    let has_capitalize = all_functions.iter().any(|f| f.name == "capitalize");
    assert!(
        has_capitalize,
        "Should have capitalize function from shared/utils.ts"
    );

    let has_get_users_fn = all_functions.iter().any(|f| f.name == "getUsers");
    assert!(
        has_get_users_fn,
        "Should have getUsers function from api/client.ts"
    );

    let has_create_user_fn = all_functions.iter().any(|f| f.name == "createUser");
    assert!(
        has_create_user_fn,
        "Should have createUser function from api/client.ts"
    );

    let all_unit_tests = graph.find_nodes_by_type(NodeType::UnitTest);

    assert!(
        all_unit_tests.len() >= 3,
        "Should have at least 3 unit tests (validateEmail, capitalize, formatResponse tests), got {}",
        all_unit_tests.len()
    );

    let all_e2e_tests = graph.find_nodes_by_type(NodeType::E2eTest);

    assert!(
        all_e2e_tests.len() >= 1,
        "Should have at least 1 E2E test/suite from Playwright, got {}",
        all_e2e_tests.len()
    );

    let all_datamodels = graph.find_nodes_by_type(NodeType::DataModel);

    let has_user_type = all_datamodels.iter().any(|d| d.name == "User");
    assert!(
        has_user_type,
        "Should have User interface from shared/types.ts"
    );

    let has_product_type = all_datamodels.iter().any(|d| d.name == "Product");
    assert!(
        has_product_type,
        "Should have Product interface from shared/types.ts"
    );

    let has_api_response = all_datamodels.iter().any(|d| d.name == "ApiResponse");
    assert!(
        has_api_response,
        "Should have ApiResponse interface from shared"
    );

    let all_pages = graph.find_nodes_by_type(NodeType::Page);

    let has_users_page_as_page = all_pages.iter().any(|p| p.name == "UsersPage");
    let has_users_page_as_func = all_functions.iter().any(|f| f.name == "UsersPage");

    assert!(
        has_users_page_as_page || has_users_page_as_func,
        "Should have UsersPage component (as Page or Function)"
    );

    Ok(())
}
#[tokio::test]
async fn test_polyglot_monorepo() -> Result<()> {
    use crate::repo::Repo;

    let repo_url = "https://github.com/fayekelmith/polyglot-monorepo";

    let repos = Repo::new_clone_multi_detect(
        repo_url,
        None,
        None,
        Vec::new(),
        Vec::new(),
        None,
        None,
        Some(false),
    )
    .await?;

    assert_eq!(repos.0.len(), 2, "Should detect 2 repos");

    let go_packages: Vec<_> = repos
        .0
        .iter()
        .filter(|r| format!("{:?}", r.lang.kind).contains("Go"))
        .collect();
    let ts_packages: Vec<_> = repos
        .0
        .iter()
        .filter(|r| format!("{:?}", r.lang.kind).contains("Typescript"))
        .collect();
    assert_eq!(go_packages.len(), 1, "Should have 1 Go repo");
    assert_eq!(ts_packages.len(), 1, "Should have 1 TypeScript repo");

    let graph = repos
        .build_graphs_inner::<crate::lang::BTreeMapGraph>()
        .await?;

    let all_repos = graph.find_nodes_by_type(NodeType::Repository);
    assert_eq!(
        all_repos.len(),
        1,
        "Polyglot monorepo should have exactly 1 Repository node, got {} (one per language detected: {:?})",
        all_repos.len(),
        all_repos.iter().map(|r| &r.name).collect::<Vec<_>>()
    );

    let all_languages = graph.find_nodes_by_type(NodeType::Language);
    assert_eq!(
        all_languages.len(),
        2,
        "Polyglot monorepo should have 2 Language nodes (Go + TypeScript), got {:?}",
        all_languages.iter().map(|l| &l.name).collect::<Vec<_>>()
    );

    let repo_to_lang_edges =
        graph.find_nodes_with_edge_type(NodeType::Repository, NodeType::Language, EdgeType::Of);
    assert_eq!(
        repo_to_lang_edges.len(),
        2,
        "Repository should have Of edges to BOTH Language nodes (Go + TypeScript), got {} edges: {:?}",
        repo_to_lang_edges.len(),
        repo_to_lang_edges.iter().map(|(r, l)| format!("{} -> {}", r.name, l.name)).collect::<Vec<_>>()
    );

    let all_packages = graph.find_nodes_by_type(NodeType::Package);
    assert!(
        all_packages.len() >= 2,
        "Should have at least 2 Package nodes for polyglot monorepo, got {}",
        all_packages.len()
    );

    let repo_to_pkg_edges = graph.find_nodes_with_edge_type(
        NodeType::Repository,
        NodeType::Package,
        EdgeType::Contains,
    );
    assert_eq!(
        repo_to_pkg_edges.len(),
        all_packages.len(),
        "Each Package should have a CONTAINS edge from Repository"
    );

    let pkg_to_lang_edges =
        graph.find_nodes_with_edge_type(NodeType::Package, NodeType::Language, EdgeType::Of);
    assert_eq!(
        pkg_to_lang_edges.len(),
        all_packages.len(),
        "Each Package should have an OF edge to its Language"
    );

    let web_pkg = all_packages.iter().find(|p| p.name == "web");
    if let Some(pkg) = web_pkg {
        assert!(
            pkg.meta.contains_key("framework"),
            "web package should have framework metadata"
        );
    }

    let all_endpoints = graph.find_nodes_by_type(NodeType::Endpoint);
    assert_eq!(all_endpoints.len(), 0, "Go endpoints require LSP");

    let all_requests = graph.find_nodes_by_type(NodeType::Request);
    assert_eq!(all_requests.len(), 4, "Should have 4 fetch requests");

    let all_functions = graph.find_nodes_by_type(NodeType::Function);
    assert_eq!(all_functions.len(), 19, "Should have 19 functions");

    let go_handlers = all_functions
        .iter()
        .filter(|f| {
            f.name == "GetUsers"
                || f.name == "CreateUser"
                || f.name == "DeleteUser"
                || f.name == "UpdateUser"
                || f.name == "GetUserByID"
                || f.name == "HealthCheck"
        })
        .count();
    assert_eq!(go_handlers, 6, "Should have 6 Go handlers");

    let all_e2e_tests = graph.find_nodes_by_type(NodeType::E2eTest);
    assert_eq!(all_e2e_tests.len(), 1, "Should have 1 E2E test suite");

    let all_datamodels = graph.find_nodes_by_type(NodeType::DataModel);
    assert_eq!(all_datamodels.len(), 7, "Should have 7 data models");

    assert!(
        all_datamodels.iter().any(|d| d.name == "User"),
        "Should have User type"
    );
    assert!(
        all_datamodels.iter().any(|d| d.name == "Product"),
        "Should have Product type"
    );

    // Root files: 8
    let all_files = graph.find_nodes_by_type(NodeType::File);
    let root_files: Vec<_> = all_files
        .iter()
        .filter(|f| {
            f.file.contains("polyglot-monorepo")
                && !f.file.contains("/apps/")
                && !f.file.contains("/packages/")
                && !f.file.contains("/services/")
                && !f.file.contains("/tests/")
        })
        .collect();
    assert_eq!(root_files.len(), 8, "Should have 8 root files");
    assert!(
        root_files.iter().any(|f| f.name == "CLAUDE.md"),
        "Should capture CLAUDE.md"
    );
    assert!(
        root_files.iter().any(|f| f.name == "README.md"),
        "Should capture README.md"
    );

    Ok(())
}

#[tokio::test]
async fn test_simple_api_non_monorepo() -> Result<()> {
    use crate::repo::Repo;

    let repo_url = "https://github.com/fayekelmith/test-simple-api";

    let repos = Repo::new_clone_multi_detect(
        repo_url,
        None,
        None,
        Vec::new(),
        Vec::new(),
        None,
        None,
        Some(false),
    )
    .await?;

    assert_eq!(repos.0.len(), 1, "Non-monorepo should have 1 repo entry");

    let graph = repos
        .build_graphs_inner::<crate::lang::BTreeMapGraph>()
        .await?;

    let all_repos = graph.find_nodes_by_type(NodeType::Repository);
    assert_eq!(all_repos.len(), 1, "Should have 1 Repository node");

    let all_endpoints = graph.find_nodes_by_type(NodeType::Endpoint);
    assert_eq!(all_endpoints.len(), 9, "Should have 9 Express endpoints");

    let all_functions = graph.find_nodes_by_type(NodeType::Function);
    assert_eq!(all_functions.len(), 18, "Should have 18 functions");

    // Controller functions: 8 specific ones
    let controller_fns = [
        "getUsers",
        "getUserById",
        "createUser",
        "updateUser",
        "deleteUser",
        "getProducts",
        "getProductById",
        "createProduct",
    ];
    let found_controller_count = controller_fns
        .iter()
        .filter(|fn_name| all_functions.iter().any(|f| f.name == **fn_name))
        .count();
    assert_eq!(
        found_controller_count, 8,
        "Should find all 8 controller functions"
    );

    // Validator function
    assert!(
        all_functions.iter().any(|f| f.name == "validateEmail"),
        "Should have validateEmail"
    );

    // Unit tests: 6 (vitest tests detected as UnitTest nodes)
    let all_unit_tests = graph.find_nodes_by_type(NodeType::UnitTest);
    assert_eq!(all_unit_tests.len(), 6, "Should have 6 UnitTest nodes");

    // Data models: 4 (User, Product, ApiResponse, PaginationParams)
    let all_datamodels = graph.find_nodes_by_type(NodeType::DataModel);
    assert_eq!(all_datamodels.len(), 4, "Should have 4 data models");

    // Specific model checks
    assert!(
        all_datamodels.iter().any(|d| d.name == "User"),
        "Should have User type"
    );
    assert!(
        all_datamodels.iter().any(|d| d.name == "Product"),
        "Should have Product type"
    );

    // Root files: 7
    let all_files = graph.find_nodes_by_type(NodeType::File);
    let root_files: Vec<_> = all_files
        .iter()
        .filter(|f| {
            f.file.contains("test-simple-api")
                && !f.file.contains("/src/")
                && !f.file.contains("/tests/")
        })
        .collect();
    assert_eq!(root_files.len(), 7, "Should have 7 root files");
    assert!(
        root_files.iter().any(|f| f.name == "CLAUDE.md"),
        "Should capture CLAUDE.md"
    );
    assert!(
        root_files.iter().any(|f| f.name == "README.md"),
        "Should capture README.md"
    );

    Ok(())
}
