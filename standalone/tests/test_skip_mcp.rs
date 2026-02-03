use standalone::utils::{extract_repo_name, should_call_mcp_for_repo};

#[test]
fn test_extract_repo_name() {
    assert_eq!(
        extract_repo_name("https://github.com/stakwork/hive").unwrap(),
        "hive"
    );
    assert_eq!(
        extract_repo_name("https://github.com/stakwork/sphinx-tribes").unwrap(),
        "sphinx-tribes"
    );
}

#[test]
fn test_should_call_mcp_for_repo_true() {
    let param = Some("true".to_string());
    assert!(should_call_mcp_for_repo(
        &param,
        "https://github.com/stakwork/hive"
    ));
    assert!(should_call_mcp_for_repo(
        &param,
        "https://github.com/stakwork/sphinx-tribes"
    ));
}

#[test]
fn test_should_call_mcp_for_repo_single_match() {
    let param = Some("hive".to_string());
    assert!(should_call_mcp_for_repo(
        &param,
        "https://github.com/stakwork/hive"
    ));
    assert!(!should_call_mcp_for_repo(
        &param,
        "https://github.com/stakwork/sphinx-tribes"
    ));
}

#[test]
fn test_should_call_mcp_for_repo_multiple_match() {
    let param = Some("hive,sphinx-tribes".to_string());
    assert!(should_call_mcp_for_repo(
        &param,
        "https://github.com/stakwork/hive"
    ));
    assert!(should_call_mcp_for_repo(
        &param,
        "https://github.com/stakwork/sphinx-tribes"
    ));
}

#[test]
fn test_should_call_mcp_for_repo_ends_with() {
    let param = Some("hive".to_string());
    assert!(should_call_mcp_for_repo(
        &param,
        "https://github.com/user/my-hive"
    ));
    assert!(should_call_mcp_for_repo(
        &param,
        "https://github.com/stakwork/hive"
    ));
    assert!(!should_call_mcp_for_repo(
        &param,
        "https://github.com/stakwork/hive-fork"
    ));
}

#[test]
fn test_should_call_mcp_for_repo_none() {
    assert!(!should_call_mcp_for_repo(
        &None,
        "https://github.com/stakwork/hive"
    ));
}

#[test]
fn test_should_call_mcp_for_repo_case_insensitive() {
    let param = Some("TRUE".to_string());
    assert!(should_call_mcp_for_repo(
        &param,
        "https://github.com/stakwork/hive"
    ));

    let param2 = Some("True".to_string());
    assert!(should_call_mcp_for_repo(
        &param2,
        "https://github.com/stakwork/sphinx-tribes"
    ));
}

#[test]
fn test_should_call_mcp_for_repo_with_spaces() {
    let param = Some("hive , sphinx-tribes ".to_string());
    assert!(should_call_mcp_for_repo(
        &param,
        "https://github.com/stakwork/hive"
    ));
    assert!(should_call_mcp_for_repo(
        &param,
        "https://github.com/stakwork/sphinx-tribes"
    ));
}

#[test]
fn test_should_call_mcp_for_repo_empty_string() {
    let param = Some("".to_string());
    assert!(!should_call_mcp_for_repo(
        &param,
        "https://github.com/stakwork/hive"
    ));
}

#[test]
fn test_should_call_mcp_for_repo_no_match() {
    let param = Some("nonexistent".to_string());
    assert!(!should_call_mcp_for_repo(
        &param,
        "https://github.com/stakwork/hive"
    ));
    assert!(!should_call_mcp_for_repo(
        &param,
        "https://github.com/stakwork/sphinx-tribes"
    ));
}

#[test]
fn test_should_call_mcp_for_repo_partial_name() {
    let param = Some("tribes".to_string());
    assert!(should_call_mcp_for_repo(
        &param,
        "https://github.com/stakwork/sphinx-tribes"
    ));
    assert!(!should_call_mcp_for_repo(
        &param,
        "https://github.com/stakwork/hive"
    ));
}
