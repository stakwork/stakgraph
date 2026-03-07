mod common;

use common::{count_prefix, fixture_path, run_stakgraph};

#[test]
fn smoke_main_py_exact_counts() {
    let main_py = fixture_path("src/testing/python/web/main.py");
    let out = run_stakgraph(&[&main_py]);

    assert_eq!(out.exit_code, 0);
    assert_eq!(count_prefix(&out.stdout, "Function:"), 3);
    assert_eq!(count_prefix(&out.stdout, "Var:"), 5);
}

#[test]
fn main_py_contains_exact_named_nodes() {
    let main_py = fixture_path("src/testing/python/web/main.py");
    let out = run_stakgraph(&[&main_py]);

    assert_eq!(out.exit_code, 0);
    assert_eq!(out.stdout.contains("Function: cleanup (37-41)"), true);
    assert_eq!(out.stdout.contains("Function: run_servers (52-91)"), true);
    assert_eq!(out.stdout.contains("Var: fastapi_app (26-29)"), true);
}
