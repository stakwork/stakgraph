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
    assert!(packages.iter().any(|p| p.language == Language::React));
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
    assert!(packages.iter().any(|p| p.language == Language::React));
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
    // 1. Mixed Go/Angular (NX)
    check_monorepo_graph::<crate::lang::BTreeMapGraph>(
        "monorepo_nx_mixed",
        &["Go", "Typescript", "Angular"],
        &["apps/api/main.go", "apps/web/package.json"],
    )
    .await?;

    // 2. Mixed Go/React (NPM)
    check_monorepo_graph::<crate::lang::BTreeMapGraph>(
        "monorepo_npm_go",
        &["Go", "React"],
        &["go.mod", "apps/web/package.json"],
    )
    .await?;

    // 3. Rust Workspace
    check_monorepo_graph::<crate::lang::BTreeMapGraph>(
        "monorepo_rust",
        &["Rust", "Rust"],
        &["api/Cargo.toml", "shared/Cargo.toml"],
    )
    .await?;

    // 4. NPM + Go
    check_monorepo_graph::<crate::lang::BTreeMapGraph>(
        "monorepo_npm_go",
        &["Go", "React"],
        &["main.go", "apps/web/package.json"],
    )
    .await?;

    // 5. Turbo (TS + React)
    check_monorepo_graph::<crate::lang::BTreeMapGraph>(
        "monorepo_turbo_ts",
        &["Typescript", "React", "React"],
        &[
            "apps/api/package.json",
            "apps/web/package.json",
            "packages/ui/package.json",
        ],
    )
    .await?;

    // 6. Python + Rust
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

    // 7. Simple TS
    check_monorepo_graph::<crate::lang::BTreeMapGraph>(
        "monorepo_simple_ts",
        &["React", "Typescript"],
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
        &["Go", "React"],
        &["main.go", "apps/web/package.json"],
    )
    .await?;

    // 5. Turbo (TS + React)
    check_monorepo_graph::<crate::lang::graphs::Neo4jGraph>(
        "monorepo_turbo_ts",
        &["Typescript", "React", "React"],
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
        &["React", "Typescript"],
        &["frontend/package.json", "backend/package.json"],
    )
    .await?;

    Ok(())
}
