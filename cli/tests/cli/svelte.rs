mod common;

use common::{count_prefix, fixture_path, run_stakgraph};

#[test]
fn smoke_test_person_svelte_exact_counts() {
    let file = fixture_path("src/testing/svelte/tests/test_person.svelte");
    let out = run_stakgraph(&[&file]);

    assert_eq!(out.exit_code, 0);
    assert_eq!(count_prefix(&out.stdout, "Import:"), 1);
    assert_eq!(count_prefix(&out.stdout, "Class:"), 1);
    assert_eq!(count_prefix(&out.stdout, "Datamodel:"), 3);
    assert_eq!(count_prefix(&out.stdout, "Function:"), 4);
}

#[test]
fn test_person_svelte_contains_exact_named_nodes() {
    let file = fixture_path("src/testing/svelte/tests/test_person.svelte");
    let out = run_stakgraph(&[&file]);

    assert_eq!(out.exit_code, 0);
    assert_eq!(out.stdout.contains("Datamodel: <script> (1-31)"), true);
    assert_eq!(out.stdout.contains("Datamodel: <svelte:head> (1-31)"), true);
    assert_eq!(out.stdout.contains("Datamodel: <div> (1-31)"), true);
}
