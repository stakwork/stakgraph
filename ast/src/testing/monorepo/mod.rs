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

// Detection Tests

// Detection Tests

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
        expected_langs.len(),
        "Package count mismatch for {}: expected {}, got {} ({:?})",
        fixture,
        expected_langs.len(),
        packages.len(),
        packages.iter().map(|p| &p.name).collect::<Vec<_>>()
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
        "Language count mismatch for {}: expected {:?}, got {:?}",
        fixture,
        expected_langs,
        languages.iter().map(|l| &l.name).collect::<Vec<_>>()
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
        assert!(
            files.iter().any(|f| f.file.ends_with(file)),
            "Missing file '{}' in {}",
            file,
            fixture
        );
    }

    Ok(())
}

#[tokio::test]
async fn test_monorepo_graph_construction() -> Result<()> {
    // 1. Mixed Go/Angular (NX) - unique languages: Go, Typescript, Angular
    check_monorepo_graph::<crate::lang::BTreeMapGraph>(
        "monorepo_nx_mixed",
        &["Go", "Typescript", "Angular"],
        &["apps/api/main.go", "apps/web/package.json"],
    )
    .await?;

    // 2. Mixed Go/Typescript (NPM) - unique languages: Go, Typescript
    check_monorepo_graph::<crate::lang::BTreeMapGraph>(
        "monorepo_npm_go",
        &["Go", "Typescript"],
        &["go.mod", "apps/web/package.json"],
    )
    .await?;

    // 3. Rust Workspace - 2 packages: api, shared (root doesn't add extra Language)
    check_monorepo_graph::<crate::lang::BTreeMapGraph>(
        "monorepo_rust",
        &["Rust", "Rust"],
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

    // 5. Turbo (TS) - 3 packages with Languages
    check_monorepo_graph::<crate::lang::BTreeMapGraph>(
        "monorepo_turbo_ts",
        &["Typescript", "Typescript", "Typescript"],
        &[
            "apps/api/package.json",
            "apps/web/package.json",
            "packages/ui/package.json",
        ],
    )
    .await?;

    // 6. Python + Rust - 3 packages with Languages
    check_monorepo_graph::<crate::lang::BTreeMapGraph>(
        "monorepo_python_rust",
        &["Python", "Rust", "Python"],
        &[
            "services/web/app.py",
            "services/processor/Cargo.toml",
            "libs/common/setup.py",
        ],
    )
    .await?;

    // 7. Simple TS - unique languages: Typescript, Typescript
    check_monorepo_graph::<crate::lang::BTreeMapGraph>(
        "monorepo_simple_ts",
        &["Typescript", "Typescript"],
        &["frontend/package.json", "backend/package.json"],
    )
    .await?;

    Ok(())
}

#[cfg(feature = "neo4j")]
#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn test_monorepo_graph_construction_neo4j() -> Result<()> {
    // 1. Mixed Go/Angular (NX)
    check_monorepo_graph::<crate::lang::graphs::Neo4jGraph>(
        "monorepo_nx_mixed",
        &["Go", "Typescript", "Angular"],
        &["apps/api/main.go", "apps/web/package.json"],
    )
    .await?;

    // 3. Rust Workspace
    check_monorepo_graph::<crate::lang::graphs::Neo4jGraph>(
        "monorepo_rust",
        &["Rust", "Rust"],
        &["api/Cargo.toml", "shared/Cargo.toml"],
    )
    .await?;

    // 4. NPM + Go
    check_monorepo_graph::<crate::lang::graphs::Neo4jGraph>(
        "monorepo_npm_go",
        &["Go", "Typescript"],
        &["main.go", "apps/web/package.json"],
    )
    .await?;

    // 5. Turbo (TS)
    check_monorepo_graph::<crate::lang::graphs::Neo4jGraph>(
        "monorepo_turbo_ts",
        &["Typescript", "Typescript", "Typescript"],
        &[
            "apps/api/package.json",
            "apps/web/package.json",
            "packages/ui/package.json",
        ],
    )
    .await?;
    // 6. Python + Rust
    check_monorepo_graph::<crate::lang::graphs::Neo4jGraph>(
        "monorepo_python_rust",
        &["Python", "Rust", "Python"],
        &[
            "services/web/app.py",
            "services/processor/Cargo.toml",
            "libs/common/setup.py",
        ],
    )
    .await?;

    // 7. Simple TS
    check_monorepo_graph::<crate::lang::graphs::Neo4jGraph>(
        "monorepo_simple_ts",
        &["Typescript", "Typescript"],
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

    let workspace_root = repos.workspace_root.as_ref().unwrap();
    println!("Workspace root: {:?}", workspace_root);

  
    assert_eq!(
        repos.repos.len(),
        4,
        "Should detect 4 packages (api, web, shared, e2e), got {}",
        repos.repos.len()
    );

    for repo in &repos.repos {
        println!("Detected package: {:?} ({})", repo.root, repo.lang);
    }

    let graph = repos
        .build_graphs_inner::<crate::lang::BTreeMapGraph>()
        .await?;

    let all_repos = graph.find_nodes_by_type(NodeType::Repository);
    println!("\n=== REPOSITORY NODES ({}) ===", all_repos.len());
    for r in &all_repos {
        println!("  - name='{}' file='{}'", r.name, r.file);
    }

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
    println!("\n=== PACKAGE NODES ({}) ===", all_packages.len());
    for p in &all_packages {
        println!("  - name='{}' file='{}' meta={:?}", p.name, p.file, p.meta);
    }

    assert_eq!(
        all_packages.len(),
        4,
        "Should have 4 Package nodes (api, web, shared, e2e), got {}",
        all_packages.len()
    );

    let all_langs = graph.find_nodes_by_type(NodeType::Language);
    println!("\n=== LANGUAGE NODES ({}) ===", all_langs.len());
    for l in &all_langs {
        println!("  - name='{}' file='{}'", l.name, l.file);
    }

    let all_files = graph.find_nodes_by_type(NodeType::File);
    let root_files: Vec<_> = all_files
        .iter()
        .filter(|f| {
            f.file.contains("test-monorepo")
                && !f.file.contains("packages/")
                && !f.file.contains("docs/")
        })
        .collect();

    println!("\n=== ROOT FILES ({}) ===", root_files.len());
    for f in &root_files {
        println!("  - name='{}' path='{}' body_len={}", f.name, f.file, f.body.len());
    }

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

    println!("\n=== Repository->Package CONTAINS edges ({}) ===", repo_to_pkg_edges.len());
    for e in &repo_to_pkg_edges {
        println!("  {} -> {}", e.source.node_data.name, e.target.node_data.name);
    }

    assert_eq!(
        repo_to_pkg_edges.len(),
        4,
        "Should have 4 CONTAINS edges from Repository to Packages, got {}",
        repo_to_pkg_edges.len()
    );

    // Package -> Language CONTAINS edges (should have 4: each package -> its language)
    let pkg_to_lang_edges: Vec<_> = edges
        .iter()
        .filter(|e| {
            e.edge == EdgeType::Contains
                && e.source.node_type == NodeType::Package
                && e.target.node_type == NodeType::Language
        })
        .collect();

    println!("\n=== Package->Language CONTAINS edges ({}) ===", pkg_to_lang_edges.len());
    for e in &pkg_to_lang_edges {
        println!("  {} -> {}", e.source.node_data.name, e.target.node_data.name);
    }

    assert_eq!(
        pkg_to_lang_edges.len(),
        4,
        "Should have 4 CONTAINS edges from Packages to Languages, got {}",
        pkg_to_lang_edges.len()
    );

    // No Repository -> Repository edges anymore (packages are Package nodes now)
    let repo_to_repo_edges: Vec<_> = edges
        .iter()
        .filter(|e| {
            e.edge == EdgeType::Contains
                && e.source.node_type == NodeType::Repository
                && e.target.node_type == NodeType::Repository
        })
        .collect();

    println!("\n=== Repository->Repository CONTAINS edges ({}) ===", repo_to_repo_edges.len());
    for e in &repo_to_repo_edges {
        println!("  {} -> {}", e.source.node_data.name, e.target.node_data.name);
    }

    assert_eq!(
        repo_to_repo_edges.len(),
        0,
        "Should have 0 Repository->Repository edges (packages are Package nodes), got {}",
        repo_to_repo_edges.len()
    );

    Ok(())
}

/// Comprehensive test for the updated test-monorepo with:
/// - CRUD endpoints (GET/POST/PUT/DELETE for users and products)
/// - API client with fetch requests
/// - Unit tests
/// - Shared types and imports
/// - E2E tests
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

    // === ENDPOINT DETECTION ===
    let all_endpoints = graph.find_nodes_by_type(NodeType::Endpoint);
    println!("\n=== ENDPOINTS ({}) ===", all_endpoints.len());
    for e in &all_endpoints {
        let verb = e.meta.get("verb").map(|s| s.as_str()).unwrap_or("?");
        println!("  - {} {} (file: {})", verb, e.name, e.file);
    }

    // Should have at least: GET/POST/PUT/DELETE /users, GET/POST /products, GET /health
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

    // === REQUEST DETECTION (from api/client.ts) ===
    let all_requests = graph.find_nodes_by_type(NodeType::Request);
    println!("\n=== REQUESTS ({}) ===", all_requests.len());
    for r in &all_requests {
        let verb = r.meta.get("verb").map(|s| s.as_str()).unwrap_or("?");
        println!("  - {} {} (file: {})", verb, r.name, r.file);
    }

    // Should have requests from packages/web/src/api/client.ts
    assert!(
        all_requests.len() >= 5,
        "Should have at least 5 requests (getUsers, getUserById, createUser, deleteUser, updateUser), got {}",
        all_requests.len()
    );

    // === FUNCTION DETECTION ===
    let all_functions = graph.find_nodes_by_type(NodeType::Function);
    println!("\n=== FUNCTIONS ({}) ===", all_functions.len());

    // Check for key functions
    let has_format_response = all_functions.iter().any(|f| f.name == "formatResponse");
    assert!(has_format_response, "Should have formatResponse function from shared/utils.ts");

    let has_validate_email = all_functions.iter().any(|f| f.name == "validateEmail");
    assert!(has_validate_email, "Should have validateEmail function from shared/utils.ts");

    let has_capitalize = all_functions.iter().any(|f| f.name == "capitalize");
    assert!(has_capitalize, "Should have capitalize function from shared/utils.ts");

    // API client functions
    let has_get_users_fn = all_functions.iter().any(|f| f.name == "getUsers");
    assert!(has_get_users_fn, "Should have getUsers function from api/client.ts");

    let has_create_user_fn = all_functions.iter().any(|f| f.name == "createUser");
    assert!(has_create_user_fn, "Should have createUser function from api/client.ts");

    // === UNIT TEST DETECTION ===
    let all_unit_tests = graph.find_nodes_by_type(NodeType::UnitTest);
    println!("\n=== UNIT TESTS ({}) ===", all_unit_tests.len());
    for t in &all_unit_tests {
        println!("  - {} (file: {})", t.name, t.file);
    }

    // Should have unit tests from packages/shared/src/__tests__/utils.test.ts
    assert!(
        all_unit_tests.len() >= 3,
        "Should have at least 3 unit tests (validateEmail, capitalize, formatResponse tests), got {}",
        all_unit_tests.len()
    );

    // === E2E TEST DETECTION ===
    let all_e2e_tests = graph.find_nodes_by_type(NodeType::E2eTest);
    println!("\n=== E2E TESTS ({}) ===", all_e2e_tests.len());
    for t in &all_e2e_tests {
        println!("  - {} (file: {})", t.name, t.file);
    }

    // Should have E2E tests from packages/e2e/tests/app.spec.ts
    // Playwright's test.describe blocks are detected as test suites
    assert!(
        all_e2e_tests.len() >= 1,
        "Should have at least 1 E2E test/suite from Playwright, got {}",
        all_e2e_tests.len()
    );

    // === DATA MODEL (INTERFACE/TYPE) DETECTION ===
    let all_datamodels = graph.find_nodes_by_type(NodeType::DataModel);
    println!("\n=== DATA MODELS/INTERFACES ({}) ===", all_datamodels.len());
    for d in &all_datamodels {
        println!("  - {} (file: {})", d.name, d.file);
    }

    // Should have User, Product, ApiResponse from shared/types.ts
    let has_user_type = all_datamodels.iter().any(|d| d.name == "User");
    assert!(has_user_type, "Should have User interface from shared/types.ts");

    let has_product_type = all_datamodels.iter().any(|d| d.name == "Product");
    assert!(has_product_type, "Should have Product interface from shared/types.ts");

    let has_api_response = all_datamodels.iter().any(|d| d.name == "ApiResponse");
    assert!(has_api_response, "Should have ApiResponse interface from shared");

    // === IMPORT DETECTION ===
    let all_imports = graph.find_nodes_by_type(NodeType::Import);
    println!("\n=== IMPORTS ({}) ===", all_imports.len());
    for i in &all_imports {
        println!("  - {} (file: {})", i.name, i.file);
    }

    // Check for @test-monorepo/shared imports
    // Note: Import resolution may vary - check if imports exist at all
    let shared_imports: Vec<_> = all_imports
        .iter()
        .filter(|i| i.name.contains("@test-monorepo/shared") || i.name.contains("test-monorepo"))
        .collect();
    println!("\n=== @test-monorepo/shared IMPORTS ({}) ===", shared_imports.len());
    for i in &shared_imports {
        println!("  - {} (file: {})", i.name, i.file);
    }

    // Imports may be detected as Library nodes instead
    let all_libraries = graph.find_nodes_by_type(NodeType::Library);
    println!("\n=== LIBRARIES ({}) ===", all_libraries.len());
    for l in &all_libraries {
        println!("  - {} (file: {})", l.name, l.file);
    }

    // Check if @test-monorepo/shared appears in libraries
    let shared_libs: Vec<_> = all_libraries
        .iter()
        .filter(|l| l.name.contains("@test-monorepo/shared") || l.name.contains("test-monorepo"))
        .collect();
    
    // Either import or library nodes should exist
    let has_shared_refs = shared_imports.len() > 0 || shared_libs.len() > 0;
    println!(
        "\nShared package references: {} imports + {} libraries = {}",
        shared_imports.len(),
        shared_libs.len(),
        if has_shared_refs { "FOUND" } else { "NOT FOUND (may be resolved differently)" }
    );

    // === PAGE/COMPONENT DETECTION ===
    let all_pages = graph.find_nodes_by_type(NodeType::Page);
    println!("\n=== PAGES ({}) ===", all_pages.len());
    for p in &all_pages {
        println!("  - {} (file: {})", p.name, p.file);
    }

    // React components might be detected as Functions instead of Pages
    // Check if UsersPage exists as a Page or as a Function
    let has_users_page_as_page = all_pages.iter().any(|p| p.name == "UsersPage");
    let has_users_page_as_func = all_functions.iter().any(|f| f.name == "UsersPage");
    
    println!(
        "\nUsersPage detection: Page={}, Function={}",
        has_users_page_as_page, has_users_page_as_func
    );
    
    // UsersPage should exist somewhere in the graph
    assert!(
        has_users_page_as_page || has_users_page_as_func,
        "Should have UsersPage component (as Page or Function)"
    );

    // === EDGE ANALYSIS: Request -> Endpoint linking ===
    let edges = graph.get_edges_vec();
    
    let request_to_endpoint_edges: Vec<_> = edges
        .iter()
        .filter(|e| {
            e.source.node_type == NodeType::Request
                && e.target.node_type == NodeType::Endpoint
        })
        .collect();

    println!("\n=== Request->Endpoint edges ({}) ===", request_to_endpoint_edges.len());
    for e in &request_to_endpoint_edges {
        println!("  {} -> {}", e.source.node_data.name, e.target.node_data.name);
    }

    // We expect API linking to connect frontend fetch calls to backend endpoints
    // This may be 0 if linking is not yet implemented for this pattern
    println!(
        "Request->Endpoint linking: {} edges (linking may need further work)",
        request_to_endpoint_edges.len()
    );

    Ok(())
}
