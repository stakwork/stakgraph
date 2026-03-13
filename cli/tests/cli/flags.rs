mod common;

use common::{fixture_path, run_stakgraph, workspace_path};

#[test]
fn filter_is_case_insensitive() {
    let routes = fixture_path("src/testing/typescript/src/routes.ts");
    let out = run_stakgraph(&["--filter", "endpoint,FUNCTION", &routes]);

    assert_eq!(out.exit_code, 0);
    assert!(out.stdout.contains("Endpoint:"));
    assert!(out.stdout.contains("Function:"));
    assert!(!out.stdout.contains("Datamodel:"));
}

#[test]
fn stats_prints_summary_table() {
    let traits = fixture_path("src/testing/rust/src/traits.rs");
    let out = run_stakgraph(&["--stats", &traits]);

    assert_eq!(out.exit_code, 0);
    assert!(out.stdout.contains("--- Node type counts ---"));
    assert!(out.stdout.contains("Function"));
}

#[test]
fn endpoint_output_includes_handler_metadata() {
    let index_ts = workspace_path("mcp/src/index.ts");
    let out = run_stakgraph(&["--filter", "endpoint", &index_ts]);

    assert_eq!(out.exit_code, 0);
    assert!(out.stdout.contains("Endpoint:"));
    assert!(out.stdout.contains("Handler:"));
}

#[test]
fn invalid_filter_type_fails() {
    let traits = fixture_path("src/testing/rust/src/traits.rs");
    let out = run_stakgraph(&["--filter", "NotAType", &traits]);

    assert_eq!(out.exit_code, 1);
    assert!(out.stderr.contains("Unknown node type"));
}

#[test]
fn allow_verbose_perf_flags_smoke() {
    let routes = fixture_path("src/testing/go/routes.go");

    let allow = run_stakgraph(&["--allow", &routes]);
    assert_eq!(allow.exit_code, 0);

    let verbose = run_stakgraph(&["-v", &routes]);
    assert_eq!(verbose.exit_code, 0);

    let perf = run_stakgraph(&["--perf", &routes]);
    assert_eq!(perf.exit_code, 0);
}

#[test]
fn quiet_conflicts_with_verbose_and_perf() {
    let routes = fixture_path("src/testing/go/routes.go");

    let qv = run_stakgraph(&["-q", "-v", &routes]);
    assert_ne!(qv.exit_code, 0);

    let qp = run_stakgraph(&["-q", "--perf", &routes]);
    assert_ne!(qp.exit_code, 0);
}
