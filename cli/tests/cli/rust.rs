mod common;

use common::{count_prefix, fixture_path, run_stakgraph};

#[test]
fn smoke_traits_rs_exact_counts() {
    let traits = fixture_path("src/testing/rust/src/traits.rs");
    let out = run_stakgraph(&[&traits]);

    assert_eq!(out.exit_code, 0);
    assert_eq!(count_prefix(&out.stdout, "Function:"), 12);
    assert_eq!(count_prefix(&out.stdout, "Datamodel:"), 4);
    assert_eq!(count_prefix(&out.stdout, "Trait:"), 4);

    let test_nodes = count_prefix(&out.stdout, "UnitTest:") + count_prefix(&out.stdout, "IntegrationTest:");
    assert_eq!(test_nodes, 10);
}

#[test]
fn default_prunes_item_datamodel_exactly() {
    let traits = fixture_path("src/testing/rust/src/traits.rs");
    let out = run_stakgraph(&[&traits]);

    assert_eq!(out.exit_code, 0);
    assert_eq!(count_prefix(&out.stdout, "Function:"), 12);
    assert_eq!(count_prefix(&out.stdout, "Datamodel:"), 4);
    assert_eq!(count_prefix(&out.stdout, "Trait:"), 4);
    assert_eq!(count_prefix(&out.stdout, "Datamodel: Item"), 0);
}

#[test]
fn traits_rs_contains_exact_named_nodes() {
    let traits = fixture_path("src/testing/rust/src/traits.rs");
    let out = run_stakgraph(&[&traits]);

    assert_eq!(out.exit_code, 0);
    assert_eq!(out.stdout.contains("Trait: Greet (4-6)"), true);
    assert_eq!(out.stdout.contains("Datamodel: Greeter (9-11)"), true);
    assert_eq!(out.stdout.contains("Function: Greeter::greet (20-22)"), true);
}

#[test]
fn invalid_path_exits_nonzero_with_exact_error() {
    let out = run_stakgraph(&["nonexistent_file.rs"]);

    assert_ne!(out.exit_code, 0);
    assert!(out.stderr.contains("file does not exist"));
}

#[test]
fn quiet_mode_has_empty_stderr() {
    let traits = fixture_path("src/testing/rust/src/traits.rs");
    let out = run_stakgraph(&["-q", &traits]);

    assert_eq!(out.exit_code, 0);
    assert_eq!(out.stderr, "");
}
