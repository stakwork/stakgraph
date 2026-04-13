mod common;

use common::{count_prefix, fixture_path, run_stakgraph};

#[test]
fn smoke_routes_react_exact_counts() {
    let file = fixture_path("src/testing/react/src/api/routes.ts");
    let out = run_stakgraph(&[&file]);

    assert_eq!(out.exit_code, 0);
    assert_eq!(count_prefix(&out.stdout, "Import:"), 1);
    assert_eq!(count_prefix(&out.stdout, "Function:"), 5);
    assert_eq!(count_prefix(&out.stdout, "Endpoint:"), 4);
}

#[test]
fn routes_react_contains_exact_named_nodes() {
    let file = fixture_path("src/testing/react/src/api/routes.ts");
    let out = run_stakgraph(&[&file]);

    assert_eq!(out.exit_code, 0);
    assert_eq!(out.stdout.contains("Endpoint: GET /users (6-9)"), true);
    assert_eq!(out.stdout.contains("Endpoint: POST /users (12-16)"), true);
    assert_eq!(
        out.stdout.contains("Endpoint: PUT /users/:id (19-23)"),
        true
    );
    assert_eq!(
        out.stdout.contains("Endpoint: DELETE /users/:id (26-29)"),
        true
    );
}
