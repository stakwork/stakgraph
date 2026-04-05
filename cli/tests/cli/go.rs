mod common;

use common::{count_prefix, fixture_path, run_stakgraph};

#[test]
fn smoke_routes_go_exact_counts() {
    let routes = fixture_path("src/testing/go/routes.go");
    let out = run_stakgraph(&[&routes]);

    assert_eq!(out.exit_code, 0);
    assert_eq!(count_prefix(&out.stdout, "Function:"), 4);
    assert_eq!(count_prefix(&out.stdout, "Endpoint:"), 4);
    assert_eq!(count_prefix(&out.stdout, "  → "), 2);
}

#[test]
fn routes_go_contains_exact_named_nodes() {
    let routes = fixture_path("src/testing/go/routes.go");
    let out = run_stakgraph(&[&routes]);

    assert_eq!(out.exit_code, 0);
    assert_eq!(out.stdout.contains("Endpoint: GET /person/{id} (22)"), true);
    assert_eq!(out.stdout.contains("Function: NewRouter (18-45)"), true);
    assert_eq!(out.stdout.contains("Function: initChi (83-98)"), true);
}

#[test]
fn default_prunes_nested_functions_exactly() {
    let file = fixture_path("src/testing/go/anonymous_functions.go");
    let out = run_stakgraph(&[&file]);

    assert_eq!(out.exit_code, 0);
    assert_eq!(count_prefix(&out.stdout, "Function:"), 1);
    assert_eq!(out.stdout.contains("Function: GET_anon-get_func_L8 (9-11)"), false);
    assert_eq!(out.stdout.contains("Function: POST_anon-post_func_L13 (14-16)"), false);
}

#[test]
fn skip_calls_removes_call_arrows_exactly() {
    let routes = fixture_path("src/testing/go/routes.go");
    let normal = run_stakgraph(&[&routes]);
    let skipped = run_stakgraph(&["--skip-calls", &routes]);

    assert_eq!(normal.exit_code, 0);
    assert_eq!(skipped.exit_code, 0);
    assert_eq!(count_prefix(&normal.stdout, "  → "), 2);
    assert_eq!(count_prefix(&skipped.stdout, "  → "), 0);
}
