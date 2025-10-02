use ast::lang::NodeType;
use shared::Result;
use std::str::FromStr;

pub fn parse_node_type(node_type: &str) -> Result<NodeType> {
    let mut chars: Vec<char> = node_type.chars().collect();
    if !chars.is_empty() {
        chars[0] = chars[0].to_uppercase().next().unwrap_or(chars[0]);
    }
    let titled_case = chars.into_iter().collect::<String>();
    NodeType::from_str(&titled_case)
}
