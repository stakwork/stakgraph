use super::NodeType;

pub fn classify_test_type(test_name: &str, test_file: &str) -> NodeType {
    let lname = test_name.to_lowercase();
    let lfile = test_file.replace('\\', "/").to_lowercase();

    if lname.contains("e2e")
        || lname.contains("system")
        || lname.contains("feature ")
        || lfile.contains("/e2e/")
        || lfile.contains("/system/")
        || lfile.contains("/features/")
        || lfile.contains("/feature/")
        || lfile.contains("/acceptance/")
        || lfile.contains("/spec/system/")
        || lfile.contains("/spec/features/")
        || lfile.contains("/test/system/")
    {
        return NodeType::E2eTest;
    }

    if lname.contains("integration")
        || lname.contains("request ")
        || lname.contains(" api")
        || lfile.contains("/integration/")
        || lfile.contains("/requests/")
        || lfile.contains("/request/")
        || lfile.contains("/controllers/")
        || lfile.contains("/spec/api/")
        || lfile.contains("/spec/integration/")
        || lfile.contains("/spec/requests/")
        || lfile.contains("/test/integration/")
    {
        return NodeType::IntegrationTest;
    }

    NodeType::UnitTest
}

pub fn tests_sources(tests_filter: Option<&str>) -> Vec<NodeType> {
    let raw = tests_filter.unwrap_or("all").trim();
    let lower = raw.to_lowercase();
    if lower == "all" || lower == "both" || lower.is_empty() {
        return vec![
            NodeType::UnitTest,
            NodeType::IntegrationTest,
            NodeType::E2eTest,
        ];
    }
    let mut ordered: Vec<NodeType> = Vec::new();
    for part in lower.split(',') {
        let nt = match part.trim() {
            "unit" => Some(NodeType::UnitTest),
            "integration" => Some(NodeType::IntegrationTest),
            "e2e" => Some(NodeType::E2eTest),
            _ => None,
        };
        if let Some(t) = nt {
            if !ordered.contains(&t) {
                ordered.push(t);
            }
        }
    }
    if ordered.is_empty() {
        return vec![
            NodeType::UnitTest,
            NodeType::IntegrationTest,
            NodeType::E2eTest,
        ];
    }
    let mut sources = Vec::new();
    for t in [
        NodeType::UnitTest,
        NodeType::IntegrationTest,
        NodeType::E2eTest,
    ] {
        if ordered.contains(&t) {
            sources.push(t);
        }
    }
    sources
}
