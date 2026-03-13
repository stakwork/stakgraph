mod common;

use common::{count_prefix, fixture_path, run_stakgraph};

#[test]
fn smoke_api_swift_exact_counts() {
    let file = fixture_path("src/testing/swift/LegacyApp/SphinxTestApp/API.swift");
    let out = run_stakgraph(&[&file]);

    assert_eq!(out.exit_code, 0);
    assert_eq!(count_prefix(&out.stdout, "Import:"), 1);
    assert_eq!(count_prefix(&out.stdout, "Class:"), 2);
    assert_eq!(count_prefix(&out.stdout, "Function:"), 3);
    assert_eq!(count_prefix(&out.stdout, "Request:"), 2);
    assert_eq!(count_prefix(&out.stdout, "  → "), 2);
}

#[test]
fn api_swift_contains_exact_named_nodes() {
    let file = fixture_path("src/testing/swift/LegacyApp/SphinxTestApp/API.swift");
    let out = run_stakgraph(&[&file]);

    assert_eq!(out.exit_code, 0);
    assert_eq!(out.stdout.contains("Class: API (11-126)"), true);
    assert_eq!(out.stdout.contains("Function: API.getPeopleList (62-85)"), true);
    assert_eq!(out.stdout.contains("Request: GET /people (68-72)"), true);
    assert_eq!(out.stdout.contains("Request: POST /person (103-107)"), true);
}
