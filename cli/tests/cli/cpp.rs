mod common;

use common::{count_prefix, fixture_path, run_stakgraph};

#[test]
fn smoke_routes_cpp_exact_counts() {
    let file = fixture_path("src/testing/cpp/web_api/routes.cpp");
    let out = run_stakgraph(&[&file]);

    assert_eq!(out.exit_code, 0);
    assert_eq!(count_prefix(&out.stdout, "Import:"), 1);
    assert_eq!(count_prefix(&out.stdout, "Function:"), 3);
    assert_eq!(count_prefix(&out.stdout, "Endpoint:"), 2);
    assert_eq!(count_prefix(&out.stdout, "  → "), 2);
}

#[test]
fn routes_cpp_contains_exact_named_nodes() {
    let file = fixture_path("src/testing/cpp/web_api/routes.cpp");
    let out = run_stakgraph(&[&file]);

    assert_eq!(out.exit_code, 0);
    assert_eq!(out.stdout.contains("Endpoint: ANY /person/<int> (26-29)"), true);
    assert_eq!(out.stdout.contains("Endpoint: POST /person (31-34)"), true);
    assert_eq!(out.stdout.contains("Function: setup_routes (25-35)"), true);
}
