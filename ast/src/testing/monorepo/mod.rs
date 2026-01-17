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

    assert_eq!(
        repo_nodes.len(),
        3,
        "Should have root + 2 packages = 3 repositories, got {}",
        repo_nodes.len()
    );

    let edges = graph.get_edges_vec();
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
        2,
        "Should have exactly 2 Repository->Repository CONTAINS edges (root->api, root->shared), got {}",
        repo_to_repo_edges.len()
    );

    Ok(())
}

#[tokio::test]
#[ignore]
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
        repos.1.is_some(),
        "Workspace root should be detected for monorepo"
    );

    let workspace_root = repos.1.as_ref().unwrap();
    println!("Workspace root: {:?}", workspace_root);

  
    assert_eq!(
        repos.0.len(),
        4,
        "Should detect 4 packages (api, web, shared, e2e), got {}",
        repos.0.len()
    );

    for repo in &repos.0 {
        println!("Detected package: {:?} ({})", repo.root, repo.lang);
    }

    let graph = repos
        .build_graphs_inner::<crate::lang::BTreeMapGraph>()
        .await?;

    let all_repos = graph.find_nodes_by_type(NodeType::Repository);
    println!("\nAll Repository nodes:");
    for r in &all_repos {
        println!("  - {} (file: {})", r.name, r.file);
    }

    assert_eq!(
        all_repos.len(),
        5,
        "Should have 5 Repository nodes (1 root + 4 packages), got {}",
        all_repos.len()
    );

    let root_repo = all_repos.iter().find(|r| r.name == "test-monorepo");
    assert!(
        root_repo.is_some(),
        "Should have root Repository node named 'test-monorepo'"
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

    println!("\nRoot files found:");
    for f in &root_files {
        println!("  - {} (path: {}, body_len: {})", f.name, f.file, f.body.len());
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

    let edges = graph.get_edges_vec();
    let repo_contains_edges: Vec<_> = edges
        .iter()
        .filter(|e| {
            e.edge == EdgeType::Contains
                && e.source.node_type == NodeType::Repository
                && e.target.node_type == NodeType::Repository
        })
        .collect();

    println!("\nRepository CONTAINS edges:");
    for e in &repo_contains_edges {
        println!("  {} -> {}", e.source.node_data.name, e.target.node_data.name);
    }

    assert_eq!(
        repo_contains_edges.len(),
        4,
        "Should have 4 CONTAINS edges from root to packages, got {}",
        repo_contains_edges.len()
    );

    Ok(())
}
