mod common;

use common::{count_prefix, fixture_path, run_stakgraph};

#[test]
fn smoke_person_controller_cs_exact_counts() {
    let file = fixture_path("src/testing/csharp/Controllers/PersonController.cs");
    let out = run_stakgraph(&[&file]);

    assert_eq!(out.exit_code, 0);
    assert_eq!(count_prefix(&out.stdout, "Import:"), 1);
    assert_eq!(count_prefix(&out.stdout, "Class:"), 1);
    assert_eq!(count_prefix(&out.stdout, "Var:"), 3);
    assert_eq!(count_prefix(&out.stdout, "Function:"), 10);
    assert_eq!(count_prefix(&out.stdout, "Endpoint:"), 9);
}

#[test]
fn person_controller_cs_contains_exact_named_nodes() {
    let file = fixture_path("src/testing/csharp/Controllers/PersonController.cs");
    let out = run_stakgraph(&[&file]);

    assert_eq!(out.exit_code, 0);
    assert_eq!(out.stdout.contains("Class: PersonController (7-107)"), true);
    assert_eq!(out.stdout.contains("Function: ApiController.GetById (33-42)"), true);
    assert_eq!(out.stdout.contains("Endpoint: HTTPGET {id:int} (33-42)"), true);
    assert_eq!(
        out.stdout
            .contains("Endpoint: HTTPPOST {id:int}/avatar (101-106)"),
        true
    );
}
