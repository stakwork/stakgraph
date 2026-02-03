use standalone::utils::should_skip_mcp_call;

#[test]
fn test_should_skip_mcp_call_docs() {
    let skip = Some("docs".to_string());
    assert!(should_skip_mcp_call(&skip, "docs"));
    assert!(!should_skip_mcp_call(&skip, "mocks"));
}

#[test]
fn test_should_skip_mcp_call_mocks() {
    let skip = Some("mocks".to_string());
    assert!(!should_skip_mcp_call(&skip, "docs"));
    assert!(should_skip_mcp_call(&skip, "mocks"));
}

#[test]
fn test_should_skip_mcp_call_both() {
    let skip = Some("docs,mocks".to_string());
    assert!(should_skip_mcp_call(&skip, "docs"));
    assert!(should_skip_mcp_call(&skip, "mocks"));
}

#[test]
fn test_should_skip_mcp_call_case_insensitive() {
    let skip = Some("DOCS,Mocks".to_string());
    assert!(should_skip_mcp_call(&skip, "docs"));
    assert!(should_skip_mcp_call(&skip, "mocks"));
}

#[test]
fn test_should_skip_mcp_call_none() {
    assert!(!should_skip_mcp_call(&None, "docs"));
    assert!(!should_skip_mcp_call(&None, "mocks"));
}

#[test]
fn test_should_skip_mcp_call_with_spaces() {
    let skip = Some("docs , mocks ".to_string());
    assert!(should_skip_mcp_call(&skip, "docs"));
    assert!(should_skip_mcp_call(&skip, "mocks"));
}

#[test]
fn test_should_skip_mcp_call_partial() {
    let skip = Some("docs".to_string());
    assert!(should_skip_mcp_call(&skip, "docs"));
    assert!(!should_skip_mcp_call(&skip, "mocks"));

    let skip = Some("mocks".to_string());
    assert!(!should_skip_mcp_call(&skip, "docs"));
    assert!(should_skip_mcp_call(&skip, "mocks"));
}

#[test]
fn test_should_skip_mcp_call_empty_string() {
    let skip = Some("".to_string());
    assert!(!should_skip_mcp_call(&skip, "docs"));
    assert!(!should_skip_mcp_call(&skip, "mocks"));
}

#[test]
fn test_should_skip_mcp_call_invalid_value() {
    let skip = Some("invalid".to_string());
    assert!(!should_skip_mcp_call(&skip, "docs"));
    assert!(!should_skip_mcp_call(&skip, "mocks"));
}

#[test]
fn test_should_skip_mcp_call_mixed_case() {
    let skip = Some("DoCs,MoCkS".to_string());
    assert!(should_skip_mcp_call(&skip, "docs"));
    assert!(should_skip_mcp_call(&skip, "mocks"));
    assert!(should_skip_mcp_call(&skip, "DOCS"));
    assert!(should_skip_mcp_call(&skip, "MOCKS"));
}
