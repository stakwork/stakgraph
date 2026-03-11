mod common;

use common::{count_prefix, fixture_path, run_stakgraph};

#[test]
fn smoke_app_routes_angular_exact_counts() {
    let file = fixture_path("src/testing/angular/src/app/app.routes.ts");
    let out = run_stakgraph(&[&file]);

    assert_eq!(out.exit_code, 0);
    assert_eq!(count_prefix(&out.stdout, "Import:"), 1);
    assert_eq!(count_prefix(&out.stdout, "Var:"), 1);
    assert_eq!(count_prefix(&out.stdout, "Class:"), 1);
    assert_eq!(count_prefix(&out.stdout, "Function:"), 0);
}

#[test]
fn app_routes_angular_contains_exact_named_nodes() {
    let file = fixture_path("src/testing/angular/src/app/app.routes.ts");
    let out = run_stakgraph(&[&file]);

    assert_eq!(out.exit_code, 0);
    assert_eq!(out.stdout.contains("Var: routes (6-10)"), true);
    assert_eq!(out.stdout.contains("Class: AppRoutingModule (16)"), true);
}
