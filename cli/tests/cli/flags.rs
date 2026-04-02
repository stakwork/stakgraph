mod common;

use common::{fixture_path, run_stakgraph, workspace_path};
use serde_json::Value;

#[test]
fn filter_is_case_insensitive() {
    let routes = fixture_path("src/testing/typescript/src/routes.ts");
    let out = run_stakgraph(&["--type", "endpoint,FUNCTION", &routes]);

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
    let out = run_stakgraph(&["--type", "endpoint", &index_ts]);

    assert_eq!(out.exit_code, 0);
    assert!(out.stdout.contains("Endpoint:"));
    assert!(out.stdout.contains("Handler:"));
}

#[test]
fn invalid_filter_type_fails() {
    let traits = fixture_path("src/testing/rust/src/traits.rs");
    let out = run_stakgraph(&["--type", "NotAType", &traits]);

    assert_ne!(out.exit_code, 0);
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

#[test]
fn name_lookup_returns_named_function_with_body() {
    let routes = fixture_path("src/testing/typescript/src/routes.ts");
    let out = run_stakgraph(&["--name", "getPerson", &routes]);

    assert_eq!(out.exit_code, 0);
    assert!(out.stdout.contains("Function: getPerson"));
    assert!(out.stdout.contains("const { id } = req.params;"));
}

#[test]
fn name_lookup_with_type_filters_named_node() {
    let routes = fixture_path("src/testing/typescript/src/routes.ts");
    let out = run_stakgraph(&["--name", "getPerson", "--type", "Function", &routes]);

    assert_eq!(out.exit_code, 0);
    assert!(out.stdout.contains("Function: getPerson"));
    assert!(!out.stdout.contains("Endpoint:"));
}

#[test]
fn name_lookup_not_found_reports_clear_message() {
    let traits = fixture_path("src/testing/rust/src/traits.rs");
    let out = run_stakgraph(&["--name", "definitely_not_real_node", &traits]);

    assert_eq!(out.exit_code, 0);
    assert!(out.stdout.contains("No node named 'definitely_not_real_node'"));
}

#[test]
fn name_lookup_runs_per_file_when_multiple_files_are_passed() {
    let traits = fixture_path("src/testing/rust/src/traits.rs");
    let routes = fixture_path("src/testing/typescript/src/routes.ts");
    let out = run_stakgraph(&["--name", "getPerson", &traits, &routes]);

    assert_eq!(out.exit_code, 0);
    assert!(out.stdout.contains("No node named 'getPerson'"));
    assert!(out.stdout.contains("Function: getPerson"));
}

#[test]
fn invalid_type_with_name_fails_validation() {
    let routes = fixture_path("src/testing/typescript/src/routes.ts");
    let out = run_stakgraph(&["--name", "getPerson", "--type", "NotAType", &routes]);

    assert_ne!(out.exit_code, 0);
    assert!(out.stderr.contains("Unknown node type"));
}

#[test]
fn type_and_stats_can_be_used_together() {
    let traits = fixture_path("src/testing/rust/src/traits.rs");
    let out = run_stakgraph(&["--type", "Function", "--stats", &traits]);

    assert_eq!(out.exit_code, 0);
    assert!(out.stdout.contains("Function:"));
    assert!(out.stdout.contains("--- Node type counts ---"));
}

#[test]
fn json_parse_mode_returns_machine_readable_payload() {
    let traits = fixture_path("src/testing/rust/src/traits.rs");
    let out = run_stakgraph(&["--json", &traits]);

    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    let payload: Value = serde_json::from_str(&out.stdout).expect("valid json stdout");
    assert_eq!(payload["ok"], true);
    assert_eq!(payload["command"], "parse");
    assert_eq!(payload["data"]["mode"], "parse");
    assert!(payload["data"]["files"].is_array());
}
