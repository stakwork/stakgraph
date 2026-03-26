mod common;

use common::run_stakgraph;

#[test]
fn completions_bash_contains_command_and_flags() {
    let out = run_stakgraph(&["completions", "bash"]);

    assert_eq!(out.exit_code, 0);
    assert!(out.stdout.contains("stakgraph"));
    assert!(out.stdout.contains("--type"));
    assert!(out.stdout.contains("--max-tokens"));
    assert!(out.stdout.contains("changes"));
}

#[test]
fn completions_zsh_non_empty() {
    let out = run_stakgraph(&["completions", "zsh"]);

    assert_eq!(out.exit_code, 0);
    assert!(!out.stdout.trim().is_empty());
}
