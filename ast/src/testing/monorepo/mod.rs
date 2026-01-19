use crate::lang::graphs::EdgeType;
use crate::lang::{Graph, NodeType};
use crate::repo::Repo;
use crate::workspace::detect_workspaces;
use lsp::language::Language;
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


#[test]
fn test_detect_monorepo_rust() {
    let root = Path::new(MONOREPO_TEST_DIR).join("monorepo_rust");
    let packages = detect_workspaces(&root).unwrap().unwrap();
    assert_eq!(packages.len(), 2);
    assert!(packages.iter().all(|p| p.language == Language::Rust));
}

#[test]
fn test_detect_monorepo_go() {
    let root = Path::new(MONOREPO_TEST_DIR).join("monorepo_go");
    let packages = detect_workspaces(&root).unwrap().unwrap();
    assert_eq!(packages.len(), 3);
    assert!(packages.iter().all(|p| p.language == Language::Go));
}

#[test]
fn test_detect_monorepo_npm_go() {
    let root = Path::new(MONOREPO_TEST_DIR).join("monorepo_npm_go");
    let packages = detect_workspaces(&root).unwrap().unwrap();
    assert_eq!(packages.len(), 2);
    assert!(packages.iter().any(|p| p.language == Language::Go));
    assert!(packages.iter().any(|p| p.language == Language::Typescript));
}

#[test]
fn test_detect_monorepo_turbo_ts() {
    let root = Path::new(MONOREPO_TEST_DIR).join("monorepo_turbo_ts");
    let packages = detect_workspaces(&root).unwrap().unwrap();
    assert_eq!(packages.len(), 3);
}

#[test]
fn test_detect_monorepo_simple_ts() {
    let root = Path::new(MONOREPO_TEST_DIR).join("monorepo_simple_ts");
    let packages = detect_workspaces(&root).unwrap().unwrap();
    assert_eq!(packages.len(), 2);
}

#[test]
fn test_detect_monorepo_turbo_python() {
    let root = Path::new(MONOREPO_TEST_DIR).join("monorepo_turbo_python");
    let packages = detect_workspaces(&root).unwrap().unwrap();
    assert_eq!(packages.len(), 2);
    assert!(packages.iter().any(|p| p.language == Language::Typescript));
}

#[test]
fn test_detect_monorepo_python_rust() {
    let root = Path::new(MONOREPO_TEST_DIR).join("monorepo_python_rust");
    let packages = detect_workspaces(&root).unwrap().unwrap();
    assert_eq!(packages.len(), 3);
    assert!(packages.iter().any(|p| p.language == Language::Rust));
    assert!(packages.iter().any(|p| p.language == Language::Python));
}

#[test]
fn test_detect_monorepo_nx_mixed() {
    let root = Path::new(MONOREPO_TEST_DIR).join("monorepo_nx_mixed");
    let packages = detect_workspaces(&root).unwrap().unwrap();
    assert_eq!(packages.len(), 3);
}

// Graph Construction Tests

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

    let all_packages = graph.find_nodes_by_type(NodeType::Package);
    let packages: Vec<_> = all_packages
        .into_iter()
        .filter(|n| n.file.starts_with(abs_root_str))
        .collect();

    assert_eq!(
        packages.len(),
        expected_files.len(),
        "Package count mismatch for {}: \n  Expected: {} packages\n  Actual: {} packages\n  Detected Packages: {}\n  Details: {}",
        fixture,
        expected_files.len(),
        packages.len(),
        packages.len(),
        packages
            .iter()
            .enumerate()
            .map(|(i, p)| format!("\n    {}: {} (path: {})", i+1, p.name, p.file))
            .collect::<String>()
    );

    // Check Languages
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
    // 1. Mixed Go/Angular (NX) - 3 packages, 3 unique languages: Go, Typescript, Angular
    check_monorepo_graph::<crate::lang::BTreeMapGraph>(
        "monorepo_nx_mixed",
        &["Go", "Typescript", "Angular"],
        &["apps/api/main.go", "libs/shared/package.json", "apps/web/package.json"],
    )
    .await?;

    // 2. Mixed Go/Typescript (NPM) - 2 packages, 2 unique languages: Go, Typescript
    check_monorepo_graph::<crate::lang::BTreeMapGraph>(
        "monorepo_npm_go",
        &["Go", "Typescript"],
        &["go.mod", "apps/web/package.json"],
    )
    .await?;

    // 3. Rust Workspace - 2 packages, 1 unique language: Rust
    check_monorepo_graph::<crate::lang::BTreeMapGraph>(
        "monorepo_rust",
        &["Rust"],
        &["api/Cargo.toml", "shared/Cargo.toml"],
    )
    .await?;

    // 4. NPM + Go (duplicate of #2, keeping for coverage)
    check_monorepo_graph::<crate::lang::BTreeMapGraph>(
        "monorepo_npm_go",
        &["Go", "Typescript"],
        &["main.go", "apps/web/package.json"],
    )
    .await?;

    // 5. Turbo (TS) - 3 packages, 1 unique language: Typescript
    check_monorepo_graph::<crate::lang::BTreeMapGraph>(
        "monorepo_turbo_ts",
        &["Typescript"],
        &[
            "apps/api/package.json",
            "apps/web/package.json",
            "packages/ui/package.json",
        ],
    )
    .await?;

    // 6. Python + Rust - 3 packages, 2 unique languages: Python, Rust
    check_monorepo_graph::<crate::lang::BTreeMapGraph>(
        "monorepo_python_rust",
        &["Python", "Rust"],
        &[
            "services/web/app.py",
            "services/processor/Cargo.toml",
            "libs/common/setup.py",
        ],
    )
    .await?;

    // 7. Simple TS - 2 packages, 1 unique language: Typescript
    check_monorepo_graph::<crate::lang::BTreeMapGraph>(
        "monorepo_simple_ts",
        &["Typescript"],
        &["frontend/package.json", "backend/package.json"],
    )
    .await?;

    Ok(())
}

#[cfg(feature = "neo4j")]
#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn test_monorepo_graph_construction_neo4j() -> Result<()> {
    // 1. Mixed Go/Angular (NX) - 3 packages, 3 unique languages
    check_monorepo_graph::<crate::lang::graphs::Neo4jGraph>(
        "monorepo_nx_mixed",
        &["Go", "Typescript", "Angular"],
        &["apps/api/main.go", "libs/shared/package.json", "apps/web/package.json"],
    )
    .await?;

    // 3. Rust Workspace - 2 packages, 1 unique language
    check_monorepo_graph::<crate::lang::graphs::Neo4jGraph>(
        "monorepo_rust",
        &["Rust"],
        &["api/Cargo.toml", "shared/Cargo.toml"],
    )
    .await?;

    // 4. NPM + Go - 2 packages, 2 unique languages
    check_monorepo_graph::<crate::lang::graphs::Neo4jGraph>(
        "monorepo_npm_go",
        &["Go", "Typescript"],
        &["main.go", "apps/web/package.json"],
    )
    .await?;

    // 5. Turbo (TS) - 3 packages, 1 unique language
    check_monorepo_graph::<crate::lang::graphs::Neo4jGraph>(
        "monorepo_turbo_ts",
        &["Typescript"],
        &[
            "apps/api/package.json",
            "apps/web/package.json",
            "packages/ui/package.json",
        ],
    )
    .await?;
    // 6. Python + Rust - 3 packages, 2 unique languages
    check_monorepo_graph::<crate::lang::graphs::Neo4jGraph>(
        "monorepo_python_rust",
        &["Python", "Rust"],
        &[
            "services/web/app.py",
            "services/processor/Cargo.toml",
            "libs/common/setup.py",
        ],
    )
    .await?;

    // 7. Simple TS - 2 packages, 1 unique language
    check_monorepo_graph::<crate::lang::graphs::Neo4jGraph>(
        "monorepo_simple_ts",
        &["Typescript"],
        &["frontend/package.json", "backend/package.json"],
    )
    .await?;

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
async fn test_monorepo_repository_hierarchy() -> Result<()> {
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

    let all_repos = graph.find_nodes_by_type(NodeType::Repository);
    let root_str = root.to_str().unwrap();
    let repo_nodes: Vec<_> = all_repos
        .into_iter()
        .filter(|n| n.file.starts_with(abs_root_str) || n.file.starts_with(root_str))
        .collect();

    // With is_package=true, we should have only 1 Repository (workspace root)
    assert_eq!(
        repo_nodes.len(),
        1,
        "Should have 1 Repository (workspace root only), got {}",
        repo_nodes.len()
    );

    let all_packages = graph.find_nodes_by_type(NodeType::Package);
    let package_nodes: Vec<_> = all_packages
        .into_iter()
        .filter(|n| n.file.starts_with(abs_root_str) || n.file.starts_with(root_str))
        .collect();

    assert_eq!(
        package_nodes.len(),
        2,
        "Should have 2 Package nodes (api, shared), got {}",
        package_nodes.len()
    );

    let edges = graph.get_edges_vec();
    
    let repo_to_pkg_edges: Vec<_> = edges
        .iter()
        .filter(|e| {
            e.edge == EdgeType::Contains
                && e.source.node_type == NodeType::Repository
                && e.target.node_type == NodeType::Package
        })
        .collect();

    assert_eq!(
        repo_to_pkg_edges.len(),
        2,
        "Should have 2 Repository->Package CONTAINS edges (root->api, root->shared), got {}",
        repo_to_pkg_edges.len()
    );

    // No Repository -> Repository edges anymore
    let repo_to_repo_edges: Vec<_> = edges
        .iter()
        .filter(|e| {
            e.edge == EdgeType::Contains
                && e.source.node_type == NodeType::Repository
                && e.target.node_type == NodeType::Repository
        })
        .collect();

    assert_eq!(
        repo_to_repo_edges.len(),
        0,
        "Should have 0 Repository->Repository edges, got {}",
        repo_to_repo_edges.len()
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

    assert!(
        repos.workspace_root.is_some(),
        "Workspace root should be detected for monorepo"
    );

    assert_eq!(
        repos.repos.len(),
        4,
        "Should detect 4 packages (api, web, shared, e2e), got {}",
        repos.repos.len()
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

    let all_packages = graph.find_nodes_by_type(NodeType::Package);

    assert_eq!(
        all_packages.len(),
        4,
        "Should have 4 Package nodes (api, web, shared, e2e), got {}",
        all_packages.len()
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
    assert!(
        claude_file.hash.is_some(),
        "CLAUDE.md should have hash set"
    );

    // === EDGE ANALYSIS ===
    let edges = graph.get_edges_vec();
    
    // Repository -> Package CONTAINS edges (should have 4: root -> each package)
    let repo_to_pkg_edges: Vec<_> = edges
        .iter()
        .filter(|e| {
            e.edge == EdgeType::Contains
                && e.source.node_type == NodeType::Repository
                && e.target.node_type == NodeType::Package
        })
        .collect();

    assert_eq!(
        repo_to_pkg_edges.len(),
        4,
        "Should have 4 CONTAINS edges from Repository to Packages, got {}",
        repo_to_pkg_edges.len()
    );

    // Package -> Language OF edges (should have 4: each package -> its language)
    let pkg_to_lang_edges: Vec<_> = edges
        .iter()
        .filter(|e| {
            e.edge == EdgeType::Of
                && e.source.node_type == NodeType::Package
                && e.target.node_type == NodeType::Language
        })
        .collect();

    assert_eq!(
        pkg_to_lang_edges.len(),
        4,
        "Should have 4 OF edges from Packages to Languages, got {}",
        pkg_to_lang_edges.len()
    );


    let repo_to_repo_edges: Vec<_> = edges
        .iter()
        .filter(|e| {
            e.edge == EdgeType::Contains
                && e.source.node_type == NodeType::Repository
                && e.target.node_type == NodeType::Repository
        })
        .collect();

    assert_eq!(
        repo_to_repo_edges.len(),
        0,
        "Should have 0 Repository->Repository edges (packages are Package nodes), got {}",
        repo_to_repo_edges.len()
    );

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
    let has_get_users = all_endpoints.iter().any(|e| {
        e.name.contains("/users") && e.meta.get("verb") == Some(&"GET".to_string())
    });
    assert!(has_get_users, "Should have GET /users endpoint");

    let has_post_users = all_endpoints.iter().any(|e| {
        e.name.contains("/users") && e.meta.get("verb") == Some(&"POST".to_string())
    });
    assert!(has_post_users, "Should have POST /users endpoint");

    let has_put_users = all_endpoints.iter().any(|e| {
        e.name.contains("/users") && e.meta.get("verb") == Some(&"PUT".to_string())
    });
    assert!(has_put_users, "Should have PUT /users/:id endpoint");

    let has_delete_users = all_endpoints.iter().any(|e| {
        e.name.contains("/users") && e.meta.get("verb") == Some(&"DELETE".to_string())
    });
    assert!(has_delete_users, "Should have DELETE /users/:id endpoint");

    let has_health = all_endpoints.iter().any(|e| {
        e.name.contains("/health") && e.meta.get("verb") == Some(&"GET".to_string())
    });
    assert!(has_health, "Should have GET /health endpoint");

  
    let all_requests = graph.find_nodes_by_type(NodeType::Request);

    assert!(
        all_requests.len() >= 5,
        "Should have at least 5 requests (getUsers, getUserById, createUser, deleteUser, updateUser), got {}",
        all_requests.len()
    );

    let all_functions = graph.find_nodes_by_type(NodeType::Function);

    let has_format_response = all_functions.iter().any(|f| f.name == "formatResponse");
    assert!(has_format_response, "Should have formatResponse function from shared/utils.ts");

    let has_validate_email = all_functions.iter().any(|f| f.name == "validateEmail");
    assert!(has_validate_email, "Should have validateEmail function from shared/utils.ts");

    let has_capitalize = all_functions.iter().any(|f| f.name == "capitalize");
    assert!(has_capitalize, "Should have capitalize function from shared/utils.ts");

    let has_get_users_fn = all_functions.iter().any(|f| f.name == "getUsers");
    assert!(has_get_users_fn, "Should have getUsers function from api/client.ts");

    let has_create_user_fn = all_functions.iter().any(|f| f.name == "createUser");
    assert!(has_create_user_fn, "Should have createUser function from api/client.ts");

 
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
    assert!(has_user_type, "Should have User interface from shared/types.ts");

    let has_product_type = all_datamodels.iter().any(|d| d.name == "Product");
    assert!(has_product_type, "Should have Product interface from shared/types.ts");

    let has_api_response = all_datamodels.iter().any(|d| d.name == "ApiResponse");
    assert!(has_api_response, "Should have ApiResponse interface from shared");

    
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

    // Workspace detection
    assert!(repos.workspace_root.is_some(), "Workspace root should be detected");

    // Package count: 6 total (2 Go + 4 TS)
    assert_eq!(repos.repos.len(), 6, "Should detect 6 packages");

    // Language split
    let go_packages: Vec<_> = repos.repos.iter().filter(|r| format!("{:?}", r.lang.kind).contains("Go")).collect();
    let ts_packages: Vec<_> = repos.repos.iter().filter(|r| format!("{:?}", r.lang.kind).contains("Typescript")).collect();
    assert_eq!(go_packages.len(), 2, "Should have 2 Go packages");
    assert_eq!(ts_packages.len(), 4, "Should have 4 TypeScript packages");

    let graph = repos
        .build_graphs_inner::<crate::lang::BTreeMapGraph>()
        .await?;

    // Package nodes
    let all_packages = graph.find_nodes_by_type(NodeType::Package);
    assert_eq!(all_packages.len(), 6, "Should have 6 Package nodes");
    
    // Endpoints: 0 (Go endpoints require LSP)
    let all_endpoints = graph.find_nodes_by_type(NodeType::Endpoint);
    assert_eq!(all_endpoints.len(), 0, "Go endpoints require LSP");

    // Requests: 4 (from apps/web/src/api/client.ts)
    let all_requests = graph.find_nodes_by_type(NodeType::Request);
    assert_eq!(all_requests.len(), 4, "Should have 4 fetch requests");

    // Functions: 19 total
    let all_functions = graph.find_nodes_by_type(NodeType::Function);
    assert_eq!(all_functions.len(), 19, "Should have 19 functions");

    // Go handlers: 6
    let go_handlers = all_functions.iter().filter(|f| 
        f.name == "GetUsers" || f.name == "CreateUser" || f.name == "DeleteUser" || 
        f.name == "UpdateUser" || f.name == "GetUserByID" || f.name == "HealthCheck"
    ).count();
    assert_eq!(go_handlers, 6, "Should have 6 Go handlers");

    
    let all_e2e_tests = graph.find_nodes_by_type(NodeType::E2eTest);
    assert_eq!(all_e2e_tests.len(), 1, "Should have 1 E2E test suite");
 
   
    let all_datamodels = graph.find_nodes_by_type(NodeType::DataModel);
    assert_eq!(all_datamodels.len(), 7, "Should have 7 data models");

  
    assert!(all_datamodels.iter().any(|d| d.name == "User"), "Should have User type");
    assert!(all_datamodels.iter().any(|d| d.name == "Product"), "Should have Product type");
    

    let all_unit_tests = graph.find_nodes_by_type(NodeType::UnitTest);
    assert_eq!(all_unit_tests.len(), 0, "No unit tests in this repo");
    
    // Request -> Endpoint edges: 0 (no endpoints without LSP)
    let edges = graph.get_edges_vec();
    let request_to_endpoint_edges: Vec<_> = edges
        .iter()
        .filter(|e| e.source.node_type == NodeType::Request && e.target.node_type == NodeType::Endpoint)
        .collect();
    assert_eq!(request_to_endpoint_edges.len(), 0, "No request-endpoint edges without LSP");

    // Root files: 8
    let all_files = graph.find_nodes_by_type(NodeType::File);
    let root_files: Vec<_> = all_files
        .iter()
        .filter(|f| f.file.contains("polyglot-monorepo") && !f.file.contains("/apps/") && !f.file.contains("/packages/") && !f.file.contains("/services/") && !f.file.contains("/tests/"))
        .collect();
    assert_eq!(root_files.len(), 8, "Should have 8 root files");
    assert!(root_files.iter().any(|f| f.name == "CLAUDE.md"), "Should capture CLAUDE.md");
    assert!(root_files.iter().any(|f| f.name == "README.md"), "Should capture README.md");

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

    // Non-monorepo should NOT have workspace_root
    assert!(repos.workspace_root.is_none(), "Non-monorepo should NOT have workspace_root");

    // Should have exactly 1 repo entry
    assert_eq!(repos.repos.len(), 1, "Non-monorepo should have 1 repo entry");

    // No packages for non-monorepo
    assert!(repos.packages.is_empty(), "Non-monorepo should have 0 packages");

    let graph = repos
        .build_graphs_inner::<crate::lang::BTreeMapGraph>()
        .await?;

    // Repository nodes: 1
    let all_repos = graph.find_nodes_by_type(NodeType::Repository);
    assert_eq!(all_repos.len(), 1, "Should have 1 Repository node");

    // Package nodes: 0 (NOT a monorepo)
    let all_packages = graph.find_nodes_by_type(NodeType::Package);
    assert_eq!(all_packages.len(), 0, "Non-monorepo should have 0 Package nodes");

    // Endpoints: 9 Express endpoints detected
    let all_endpoints = graph.find_nodes_by_type(NodeType::Endpoint);
    assert_eq!(all_endpoints.len(), 9, "Should have 9 Express endpoints");

    // Functions: 18
    let all_functions = graph.find_nodes_by_type(NodeType::Function);
    assert_eq!(all_functions.len(), 18, "Should have 18 functions");

    // Controller functions: 8 specific ones
    let controller_fns = ["getUsers", "getUserById", "createUser", "updateUser", "deleteUser", 
                          "getProducts", "getProductById", "createProduct"];
    let found_controller_count = controller_fns.iter()
        .filter(|fn_name| all_functions.iter().any(|f| f.name == **fn_name))
        .count();
    assert_eq!(found_controller_count, 8, "Should find all 8 controller functions");

    // Validator function
    assert!(all_functions.iter().any(|f| f.name == "validateEmail"), "Should have validateEmail");

    // Unit tests: 6 (vitest tests detected as UnitTest nodes)
    let all_unit_tests = graph.find_nodes_by_type(NodeType::UnitTest);
    assert_eq!(all_unit_tests.len(), 6, "Should have 6 UnitTest nodes");

    // Data models: 4 (User, Product, ApiResponse, PaginationParams)
    let all_datamodels = graph.find_nodes_by_type(NodeType::DataModel);
    assert_eq!(all_datamodels.len(), 4, "Should have 4 data models");

    // Specific model checks
    assert!(all_datamodels.iter().any(|d| d.name == "User"), "Should have User type");
    assert!(all_datamodels.iter().any(|d| d.name == "Product"), "Should have Product type");

    // Root files: 7
    let all_files = graph.find_nodes_by_type(NodeType::File);
    let root_files: Vec<_> = all_files
        .iter()
        .filter(|f| f.file.contains("test-simple-api") && !f.file.contains("/src/") && !f.file.contains("/tests/"))
        .collect();
    assert_eq!(root_files.len(), 7, "Should have 7 root files");
    assert!(root_files.iter().any(|f| f.name == "CLAUDE.md"), "Should capture CLAUDE.md");
    assert!(root_files.iter().any(|f| f.name == "README.md"), "Should capture README.md");

    Ok(())
}
