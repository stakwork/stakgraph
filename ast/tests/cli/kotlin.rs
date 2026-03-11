mod common;

use common::{count_prefix, fixture_path, run_stakgraph};

#[test]
fn smoke_api_service_kotlin_exact_counts() {
    let file = fixture_path("src/testing/kotlin/app/src/main/java/com/kotlintestapp/data/api/ApiService.kt");
    let out = run_stakgraph(&[&file]);

    assert_eq!(out.exit_code, 0);
    assert_eq!(count_prefix(&out.stdout, "Import:"), 1);
    assert_eq!(count_prefix(&out.stdout, "Class:"), 1);
    assert_eq!(count_prefix(&out.stdout, "Function:"), 3);
    assert_eq!(count_prefix(&out.stdout, "Request:"), 3);
    assert_eq!(count_prefix(&out.stdout, "  → "), 3);
}

#[test]
fn api_service_kotlin_contains_exact_named_nodes() {
    let file = fixture_path("src/testing/kotlin/app/src/main/java/com/kotlintestapp/data/api/ApiService.kt");
    let out = run_stakgraph(&[&file]);

    assert_eq!(out.exit_code, 0);
    assert_eq!(out.stdout.contains("Class: ApiService (9-18)"), true);
    assert_eq!(out.stdout.contains("Function: ApiService.getUsers (10-11)"), true);
    assert_eq!(out.stdout.contains("Request: GET /users (10-11)"), true);
    assert_eq!(out.stdout.contains("Request: POST /users (16-17)"), true);
}
