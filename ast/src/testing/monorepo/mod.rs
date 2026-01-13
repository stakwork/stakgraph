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
