use super::NodeType;


pub fn tests_sources(tests_filter: Option<&str>) -> Vec<NodeType> {
    let raw = tests_filter.unwrap_or("all").trim();
    let lower = raw.to_lowercase();
    if lower == "all" || lower == "both" || lower.is_empty() {
        return vec![NodeType::UnitTest, NodeType::IntegrationTest, NodeType::E2eTest];
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
            if !ordered.contains(&t) { ordered.push(t); }
        }
    }
    if ordered.is_empty() {
        return vec![NodeType::UnitTest, NodeType::IntegrationTest, NodeType::E2eTest];
    }
    let mut sources = Vec::new();
    for t in [NodeType::UnitTest, NodeType::IntegrationTest, NodeType::E2eTest] {
        if ordered.contains(&t) { sources.push(t); }
    }
    sources
}
