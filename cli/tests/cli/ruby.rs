mod common;

use common::{count_prefix, fixture_path, run_stakgraph};

#[test]
fn smoke_person_service_rb_exact_counts() {
    let file = fixture_path("src/testing/ruby/app/services/person_service.rb");
    let out = run_stakgraph(&[&file]);

    assert_eq!(out.exit_code, 0);
    assert_eq!(count_prefix(&out.stdout, "Class:"), 1);
    assert_eq!(count_prefix(&out.stdout, "Function:"), 3);
}

#[test]
fn person_service_rb_contains_exact_named_nodes() {
    let file = fixture_path("src/testing/ruby/app/services/person_service.rb");
    let out = run_stakgraph(&[&file]);

    assert_eq!(out.exit_code, 0);
    assert_eq!(out.stdout.contains("Class: PersonService (1-13)"), true);
    assert_eq!(
        out.stdout
            .contains("Function: PersonService.get_person_by_id (2-4)"),
        true
    );
    assert_eq!(
        out.stdout
            .contains("Function: PersonService.new_person (6-8)"),
        true
    );
    assert_eq!(
        out.stdout
            .contains("Function: PersonService.delete (10-12)"),
        true
    );
}
