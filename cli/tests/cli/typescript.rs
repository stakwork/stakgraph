mod common;

use common::{count_prefix, fixture_path, run_stakgraph};

#[test]
fn smoke_routes_ts_exact_counts() {
    let routes = fixture_path("src/testing/typescript/src/routes.ts");
    let out = run_stakgraph(&[&routes]);

    assert_eq!(out.exit_code, 0);
    assert_eq!(count_prefix(&out.stdout, "Function:"), 5);
    assert_eq!(count_prefix(&out.stdout, "Endpoint:"), 4);
    assert_eq!(count_prefix(&out.stdout, "Datamodel:"), 3);
}

#[test]
fn routes_ts_contains_exact_named_nodes() {
    let routes = fixture_path("src/testing/typescript/src/routes.ts");
    let out = run_stakgraph(&[&routes]);

    assert_eq!(out.exit_code, 0);
    assert_eq!(out.stdout.contains("Endpoint: POST /people/new (18)"), true);
    assert_eq!(out.stdout.contains("Function: getPerson (32-49)"), true);
    assert_eq!(out.stdout.contains("Datamodel: PersonRequest (6)"), true);
}

#[test]
fn comma_separated_single_arg_runs_both_files() {
    let traits = fixture_path("src/testing/rust/src/traits.rs");
    let routes = fixture_path("src/testing/typescript/src/routes.ts");
    let files = format!("{traits},{routes}");
    let out = run_stakgraph(&[&files]);

    assert_eq!(out.exit_code, 0);
    assert_eq!(count_prefix(&out.stdout, "File:"), 2);
    assert_eq!(out.stdout.contains("Datamodel: Greeter (9-11)"), true);
    assert_eq!(out.stdout.contains("Endpoint: POST /people/new (18)"), true);
}

#[test]
fn multiple_separate_args_runs_both_files() {
    let traits = fixture_path("src/testing/rust/src/traits.rs");
    let routes = fixture_path("src/testing/typescript/src/routes.ts");
    let out = run_stakgraph(&[&traits, &routes]);

    assert_eq!(out.exit_code, 0);
    assert_eq!(count_prefix(&out.stdout, "File:"), 2);
    assert_eq!(out.stdout.contains("Datamodel: Greeter (9-11)"), true);
    assert_eq!(out.stdout.contains("Function: getPerson (32-49)"), true);
}
