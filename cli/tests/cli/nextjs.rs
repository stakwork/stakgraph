mod common;

use common::{count_prefix, fixture_path, run_stakgraph};
use serde_json::Value;

#[test]
fn smoke_users_route_nextjs_exact_counts() {
    let file = fixture_path("src/testing/nextjs/app/api/users/route.ts");
    let out = run_stakgraph(&[&file]);

    assert_eq!(out.exit_code, 0);
    assert_eq!(count_prefix(&out.stdout, "Function:"), 1);
    assert_eq!(count_prefix(&out.stdout, "Endpoint:"), 1);
}

#[test]
fn users_route_nextjs_contains_exact_named_nodes() {
    let file = fixture_path("src/testing/nextjs/app/api/users/route.ts");
    let out = run_stakgraph(&[&file]);

    assert_eq!(out.exit_code, 0);
    assert_eq!(out.stdout.contains("Function: POST (1-4)"), true);
    assert_eq!(out.stdout.contains("Endpoint: POST /api/users (1-4)"), true);
}

// ── parse: --type filter ──────────────────────────────────────────────────────

#[test]
fn parse_type_filter_endpoints_only() {
    let api = fixture_path("src/testing/nextjs/app/api");
    let out = run_stakgraph(&["--type", "Endpoint", &api]);

    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    assert!(out.stdout.contains("Endpoint:"), "stdout: {}", out.stdout);
    assert!(!out.stdout.contains("Function:"), "stdout: {}", out.stdout);
}

#[test]
fn parse_type_filter_functions_calculator() {
    let calc = fixture_path("src/testing/nextjs/lib/calculator.ts");
    let out = run_stakgraph(&["--type", "Function", &calc]);

    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    assert_eq!(count_prefix(&out.stdout, "Function:"), 4);
    assert!(!out.stdout.contains("Class:"), "stdout: {}", out.stdout);
}

#[test]
fn parse_name_filter_single_file() {
    let utils = fixture_path("src/testing/nextjs/lib/utils.ts");
    let out = run_stakgraph(&["--name", "cn", "--type", "Function", &utils]);

    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    assert!(out.stdout.contains("cn"), "stdout: {}", out.stdout);
    assert!(out.stdout.contains("Docs:"), "stdout: {}", out.stdout);
}

#[test]
fn parse_stats_api_dir() {
    let api = fixture_path("src/testing/nextjs/app/api");
    let out = run_stakgraph(&["--stats", &api]);

    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    assert!(out.stdout.contains("Endpoint"), "stdout: {}", out.stdout);
    assert!(out.stdout.contains("21"), "stdout: {}", out.stdout);
}

#[test]
fn parse_json_envelope_single_file() {
    let route = fixture_path("src/testing/nextjs/app/api/users/route.ts");
    let out = run_stakgraph(&["--json", &route]);

    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    let v: Value = serde_json::from_str(&out.stdout).expect("invalid JSON");
    assert_eq!(v["ok"], true);
    assert_eq!(v["command"], "parse");
    assert!(v["data"]["files"].is_array());
}

// ── deps ──────────────────────────────────────────────────────────────────────

#[test]
fn deps_leaf_node_has_no_children() {
    let calc = fixture_path("src/testing/nextjs/lib/calculator.ts");
    let out = run_stakgraph(&["deps", "add", &calc]);

    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    assert!(out.stdout.contains("add"), "stdout: {}", out.stdout);
    assert!(!out.stdout.contains("└──"), "stdout: {}", out.stdout);
}

#[test]
fn deps_cross_file_chain_lib() {
    let lib = fixture_path("src/testing/nextjs/lib");
    let out = run_stakgraph(&["deps", "format", &lib]);

    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    assert!(out.stdout.contains("convertSatsToUSD"), "stdout: {}", out.stdout);
    assert!(out.stdout.contains("toFixed"), "stdout: {}", out.stdout);
}

#[test]
fn deps_json_edges_cross_file() {
    let lib = fixture_path("src/testing/nextjs/lib");
    let out = run_stakgraph(&["--json", "deps", "format", &lib]);

    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    let v: Value = serde_json::from_str(&out.stdout).expect("invalid JSON");
    assert_eq!(v["ok"], true);
    assert_eq!(v["command"], "deps");
    let edges = v["data"]["edges"].as_array().expect("edges array");
    assert_eq!(edges.len(), 2);
    assert_eq!(edges[0]["target_name"], "convertSatsToUSD");
    assert_eq!(edges[1]["target_name"], "toFixed");
}

// ── impact ────────────────────────────────────────────────────────────────────

#[test]
fn impact_cn_counts_full_repo() {
    let dir = fixture_path("src/testing/nextjs");
    let out = run_stakgraph(&["impact", "--name", "cn", &dir]);

    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    assert!(out.stdout.contains("16 Functions"), "stdout: {}", out.stdout);
    assert!(out.stdout.contains("8 Pages"), "stdout: {}", out.stdout);
    assert!(out.stdout.contains("2 UnitTests"), "stdout: {}", out.stdout);
}

#[test]
fn impact_scoped_to_lib_dir() {
    let lib = fixture_path("src/testing/nextjs/lib");
    let out = run_stakgraph(&["impact", "--name", "convertSatsToUSD", &lib]);

    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    assert!(out.stdout.contains("format"), "stdout: {}", out.stdout);
    assert!(out.stdout.contains("api-handlers.ts"), "stdout: {}", out.stdout);
}

#[test]
fn impact_file_flag_matches_name_flag() {
    let utils = fixture_path("src/testing/nextjs/lib/utils.ts");
    let dir = fixture_path("src/testing/nextjs");
    let out = run_stakgraph(&["impact", "--file", &utils, &dir]);

    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    assert!(out.stdout.contains("16 Functions"), "stdout: {}", out.stdout);
    assert!(out.stdout.contains("8 Pages"), "stdout: {}", out.stdout);
}

#[test]
fn impact_json_summary_counts_cn() {
    let dir = fixture_path("src/testing/nextjs");
    let out = run_stakgraph(&["--json", "impact", "--name", "cn", &dir]);

    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    let v: Value = serde_json::from_str(&out.stdout).expect("invalid JSON");
    assert_eq!(v["ok"], true);
    assert_eq!(v["command"], "impact");
    assert_eq!(v["data"]["summary"]["total"], 27);
    assert_eq!(v["data"]["summary"]["by_type"]["Function"], 16);
    assert_eq!(v["data"]["summary"]["by_type"]["Page"], 8);
}

// ── search ────────────────────────────────────────────────────────────────────

#[test]
fn search_get_endpoints_api_dir() {
    let api = fixture_path("src/testing/nextjs/app/api");
    let out = run_stakgraph(&["search", "GET", "--type", "Endpoint", &api]);

    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    assert!(out.stdout.contains("7 results"), "stdout: {}", out.stdout);
    assert!(out.stdout.contains("Endpoint: GET"), "stdout: {}", out.stdout);
    assert!(!out.stdout.contains("Function:"), "stdout: {}", out.stdout);
}

#[test]
fn search_card_components() {
    let components = fixture_path("src/testing/nextjs/components");
    let out = run_stakgraph(&["search", "Card", "--type", "Function", &components]);

    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    assert!(out.stdout.contains("7 results"), "stdout: {}", out.stdout);
    assert!(out.stdout.contains("Card"), "stdout: {}", out.stdout);
}

#[test]
fn search_file_filter_scopes_to_button() {
    let components = fixture_path("src/testing/nextjs/components");
    let out = run_stakgraph(&[
        "search", "Button", "--type", "Function", "--file", "button", &components,
    ]);

    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    assert!(out.stdout.contains("Button"), "stdout: {}", out.stdout);
    assert_eq!(count_prefix(&out.stdout, "Function:"), 1);
}

#[test]
fn search_json_total_get_endpoints() {
    let api = fixture_path("src/testing/nextjs/app/api");
    let out = run_stakgraph(&["--json", "search", "GET", "--type", "Endpoint", &api]);

    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    let v: Value = serde_json::from_str(&out.stdout).expect("invalid JSON");
    assert_eq!(v["ok"], true);
    assert_eq!(v["command"], "search");
    assert_eq!(v["data"]["total"], 7);
}

// ── overview ──────────────────────────────────────────────────────────────────

#[test]
fn overview_nextjs_tree_shape() {
    let dir = fixture_path("src/testing/nextjs");
    let out = run_stakgraph(&["overview", &dir]);

    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    assert!(out.stdout.contains("app/"), "stdout: {}", out.stdout);
    assert!(out.stdout.contains("lib/"), "stdout: {}", out.stdout);
    assert!(out.stdout.contains("components/"), "stdout: {}", out.stdout);
    assert!(out.stdout.contains("package.json"), "stdout: {}", out.stdout);
}

#[test]
fn overview_json_nextjs() {
    let dir = fixture_path("src/testing/nextjs");
    let out = run_stakgraph(&["--json", "overview", &dir]);

    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    let v: Value = serde_json::from_str(&out.stdout).expect("invalid JSON");
    assert_eq!(v["ok"], true);
    assert_eq!(v["command"], "overview");
    assert_eq!(v["data"]["mode"], "overview");
    assert!(v["data"]["stats"]["total_files"].is_number());
}

// ── summarize (--max-tokens) ──────────────────────────────────────────────────

#[test]
fn summarize_single_file_nextjs() {
    let utils = fixture_path("src/testing/nextjs/lib/utils.ts");
    let out = run_stakgraph(&[&utils, "--max-tokens", "500"]);

    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    assert!(out.stdout.contains("Summary:"), "stdout: {}", out.stdout);
    assert!(out.stdout.contains("cn"), "stdout: {}", out.stdout);
}

#[test]
fn summarize_directory_nextjs_lib() {
    let lib = fixture_path("src/testing/nextjs/lib");
    let out = run_stakgraph(&[&lib, "--max-tokens", "1000", "--depth", "1"]);

    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    assert!(out.stdout.contains("Directory Structure"), "stdout: {}", out.stdout);
    assert!(out.stdout.contains("File Summaries"), "stdout: {}", out.stdout);
    assert!(out.stdout.contains("calculator.ts"), "stdout: {}", out.stdout);
}
