mod common;

use common::{fixture_path, run_stakgraph};

#[test]
fn unsupported_text_file_prints_preview() {
    let file = fixture_path("src/testing/monorepo/monorepo_rust/README.md");
    let out = run_stakgraph(&[&file]);

    assert_eq!(out.exit_code, 0);
    assert!(out.stdout.contains("File:"));
    assert!(out.stdout.contains("Rust Monorepo"));
    assert!(!out
        .stdout
        .contains("[binary or unprintable content skipped]"));
}

#[test]
fn unsupported_binary_file_prints_skip_message() {
    let file = fixture_path("src/testing/svelte/static/favicon.png");
    let out = run_stakgraph(&[&file]);

    assert_eq!(out.exit_code, 0);
    assert!(out.stdout.contains("File:"));
    assert!(out
        .stdout
        .contains("[binary or unprintable content skipped]"));
}
