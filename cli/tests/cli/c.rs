mod common;

use common::{count_prefix, fixture_path, run_stakgraph};

#[test]
fn smoke_routes_c_exact_counts() {
    let file = fixture_path("src/testing/c/web-http/routes.c");
    let out = run_stakgraph(&[&file]);

    assert_eq!(out.exit_code, 0);
    assert_eq!(count_prefix(&out.stdout, "Import:"), 1);
    assert_eq!(count_prefix(&out.stdout, "Function:"), 3);
    assert_eq!(count_prefix(&out.stdout, "Endpoint:"), 0);
}

#[test]
fn routes_c_contains_exact_named_functions() {
    let file = fixture_path("src/testing/c/web-http/routes.c");
    let out = run_stakgraph(&[&file]);

    assert_eq!(out.exit_code, 0);
    assert_eq!(
        out.stdout.contains("Function: handler_get_user (5-18)"),
        true
    );
    assert_eq!(
        out.stdout.contains("Function: handler_post_user (20-33)"),
        true
    );
    assert_eq!(
        out.stdout
            .contains("Function: handler_list_products (35-39)"),
        true
    );
}
