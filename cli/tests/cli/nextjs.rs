mod common;

use common::{count_prefix, fixture_path, run_stakgraph};

#[test]
fn smoke_users_route_nextjs_exact_counts() {
    let file = fixture_path("src/testing/nextjs/app/api/users/route.ts");
    let out = run_stakgraph(&[&file]);

    assert_eq!(out.exit_code, 0);
    assert_eq!(count_prefix(&out.stdout, "Function:"), 1);
    assert_eq!(count_prefix(&out.stdout, "Endpoint:"), 1);
}

#[test]
fn users_route_nextjs_contains_exact_named_nodes() {
    let file = fixture_path("src/testing/nextjs/app/api/users/route.ts");
    let out = run_stakgraph(&[&file]);

    assert_eq!(out.exit_code, 0);
    assert_eq!(out.stdout.contains("Function: POST (1-4)"), true);
    assert_eq!(out.stdout.contains("Endpoint: POST /api/users (1-4)"), true);
}
