mod common;

use common::{count_prefix, fixture_path, run_stakgraph};

#[test]
fn smoke_billing_service_java_exact_counts() {
    let file = fixture_path(
        "src/testing/java/src/main/java/graph/stakgraph/java/nonweb/BillingService.java",
    );
    let out = run_stakgraph(&[&file]);

    assert_eq!(out.exit_code, 0);
    assert_eq!(count_prefix(&out.stdout, "Import:"), 1);
    assert_eq!(count_prefix(&out.stdout, "Class:"), 1);
    assert_eq!(count_prefix(&out.stdout, "Function:"), 0);
}

#[test]
fn billing_service_java_contains_exact_named_nodes() {
    let file = fixture_path(
        "src/testing/java/src/main/java/graph/stakgraph/java/nonweb/BillingService.java",
    );
    let out = run_stakgraph(&[&file]);

    assert_eq!(out.exit_code, 0);
    assert_eq!(out.stdout.contains("Class: BillingService (9-15)"), true);
    assert_eq!(
        out.stdout
            .contains("import graph.stakgraph.java.model.Person;"),
        true
    );
}
