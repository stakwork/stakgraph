use standalone::types::ProcessBody;
use standalone::utils::resolve_repo;

#[test]
fn test_resolve_single_repo_url() {
    let body = ProcessBody {
        repo_url: Some("https://github.com/fayekelmith/test-simple-api".to_string()),
        repo_path: None,
        username: None,
        pat: None,
        use_lsp: None,
        commit: None,
        branch: None,
        callback_url: None,
        realtime: None,
        docs: None,
        mocks: None,
        embeddings: None,
        embeddings_limit: None,
    };

    let result = resolve_repo(&body);
    assert!(result.is_ok());

    let (paths, urls, _, _, _, _) = result.unwrap();
    assert_eq!(paths.len(), 1);
    assert_eq!(urls.len(), 1);
    assert_eq!(urls[0], "https://github.com/fayekelmith/test-simple-api");
    assert_eq!(paths[0], "/tmp/fayekelmith/test-simple-api");
}

#[test]
fn test_resolve_multi_repo_urls() {
    let body = ProcessBody {
        repo_url: Some(
            "https://github.com/fayekelmith/test-simple-api,https://github.com/fayekelmith/demorepo"
                .to_string(),
        ),
        repo_path: None,
        username: None,
        pat: None,
        use_lsp: None,
        commit: None,
        branch: None,
        callback_url: None,
        realtime: None,
        docs: None,
        mocks: None,
        embeddings: None,
        embeddings_limit: None,
    };

    let result = resolve_repo(&body);
    assert!(result.is_ok());

    let (paths, urls, _, _, _, _) = result.unwrap();
    assert_eq!(paths.len(), 2);
    assert_eq!(urls.len(), 2);
    assert_eq!(urls[0], "https://github.com/fayekelmith/test-simple-api");
    assert_eq!(urls[1], "https://github.com/fayekelmith/demorepo");
    assert_eq!(paths[0], "/tmp/fayekelmith/test-simple-api");
    assert_eq!(paths[1], "/tmp/fayekelmith/demorepo");
}

#[test]
fn test_resolve_multi_repo_urls_with_spaces() {
    let body = ProcessBody {
        repo_url: Some(
            "https://github.com/fayekelmith/test-simple-api , https://github.com/fayekelmith/demorepo"
                .to_string(),
        ),
        repo_path: None,
        username: None,
        pat: None,
        use_lsp: None,
        commit: None,
        branch: None,
        callback_url: None,
        realtime: None,
        docs: None,
        mocks: None,
        embeddings: None,
        embeddings_limit: None,
    };

    let result = resolve_repo(&body);
    assert!(result.is_ok());

    let (paths, urls, _, _, _, _) = result.unwrap();
    assert_eq!(paths.len(), 2);
    assert_eq!(urls.len(), 2);
    // Verify trimming works
    assert_eq!(urls[0], "https://github.com/fayekelmith/test-simple-api");
    assert_eq!(urls[1], "https://github.com/fayekelmith/demorepo");
}

#[test]
fn test_resolve_repo_path_returns_single_element_vectors() {
    let body = ProcessBody {
        repo_url: None,
        repo_path: Some("/local/path/to/repo".to_string()),
        username: None,
        pat: None,
        use_lsp: None,
        commit: None,
        branch: None,
        callback_url: None,
        realtime: None,
        docs: None,
        mocks: None,
        embeddings: None,
        embeddings_limit: None,
    };

    let result = resolve_repo(&body);
    assert!(result.is_ok());

    let (paths, urls, _, _, _, _) = result.unwrap();
    assert_eq!(paths.len(), 1);
    assert_eq!(urls.len(), 1);
    assert_eq!(paths[0], "/local/path/to/repo");
    assert_eq!(urls[0], ""); // Empty URL when using local path
}

#[tokio::test]
async fn test_validate_multi_repo_credentials() {
    use lsp::git::validate_git_credentials_multi;

    let repos = vec![
        "https://github.com/fayekelmith/test-simple-api".to_string(),
        "https://github.com/fayekelmith/demorepo".to_string(),
    ];

    // Public repos should validate without credentials
    let result = validate_git_credentials_multi(&repos, None, None).await;
    assert!(
        result.is_ok(),
        "Failed to validate public repos: {:?}",
        result.err()
    );
}

#[tokio::test]
async fn test_validate_multi_repo_one_invalid() {
    use lsp::git::validate_git_credentials_multi;

    let repos = vec![
        "https://github.com/fayekelmith/test-simple-api".to_string(),
        "https://github.com/fayekelmith/this-repo-definitely-does-not-exist-12345".to_string(),
    ];

    let result = validate_git_credentials_multi(&repos, None, None).await;
    assert!(result.is_err(), "Should fail with invalid repo");

    let error_msg = result.unwrap_err().to_string();
    assert!(
        error_msg.contains("this-repo-definitely-does-not-exist-12345"),
        "Error should mention the invalid repo"
    );
}

#[test]
fn test_resolve_empty_repo_url() {
    let body = ProcessBody {
        repo_url: Some("".to_string()),
        repo_path: None,
        username: None,
        pat: None,
        use_lsp: None,
        commit: None,
        branch: None,
        callback_url: None,
        realtime: None,
        docs: None,
        mocks: None,
        embeddings: None,
        embeddings_limit: None,
    };

    let result = resolve_repo(&body);
    assert!(result.is_err());
    assert!(result
        .unwrap_err()
        .to_string()
        .contains("empty after parsing"));
}
