mod common;

use common::{count_prefix, fixture_path, run_stakgraph};

#[test]
fn smoke_api_php_exact_counts() {
    let api = fixture_path("src/testing/php/routes/api.php");
    let out = run_stakgraph(&[&api]);

    assert_eq!(out.exit_code, 0);
    assert_eq!(count_prefix(&out.stdout, "Import:"), 1);
    assert_eq!(count_prefix(&out.stdout, "Function:"), 1);
    assert_eq!(count_prefix(&out.stdout, "Endpoint:"), 15);
}

#[test]
fn api_php_contains_exact_named_endpoints() {
    let api = fixture_path("src/testing/php/routes/api.php");
    let out = run_stakgraph(&[&api]);

    assert_eq!(out.exit_code, 0);
    assert_eq!(out.stdout.contains("Endpoint: GET /user (20-22)"), true);
    assert_eq!(out.stdout.contains("Endpoint: POST /login (24)"), true);
    assert_eq!(
        out.stdout.contains("Endpoint: DELETE /posts/{post} (29)"),
        true
    );
    assert_eq!(
        out.stdout
            .contains("Endpoint: POST /posts/{post}/like (32)"),
        true
    );
}
