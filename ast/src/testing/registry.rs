use crate::lang::asg::NodeData;
use crate::lang::graphs::{BTreeMapGraph, Graph, NodeType};
use crate::lang::registry::{typescript::TypeScriptRegistry, Registry};
use lsp::Language;

fn ts_var(file: &str, name: &str, data_type: &str) -> NodeData {
    let mut nd = NodeData::in_file(file);
    nd.name = name.to_string();
    nd.data_type = Some(data_type.to_string());
    nd
}

fn ts_class(file: &str, name: &str) -> NodeData {
    let mut nd = NodeData::in_file(file);
    nd.name = name.to_string();
    nd
}

#[test]
fn test_resolve_type_from_var() {
    let mut graph = BTreeMapGraph::new("".to_string(), Language::Typescript);
    graph.add_node(&NodeType::Var, &ts_var("/repo/src/a.ts", "userService", "UserService"));
    let reg = TypeScriptRegistry::new(&graph);
    assert_eq!(
        reg.resolve_type("/repo/src/a.ts", "userService"),
        Some("UserService")
    );
}

#[test]
fn test_resolve_method_via_type_def() {
    let mut graph = BTreeMapGraph::new("".to_string(), Language::Typescript);
    graph.add_node(&NodeType::Class, &ts_class("/repo/src/user.ts", "UserService"));
    let reg = TypeScriptRegistry::new(&graph);
    assert_eq!(
        reg.resolve_method("UserService", "getUser"),
        Some("/repo/src/user.ts")
    );
}

#[test]
fn test_resolve_unknown_returns_none() {
    let graph = BTreeMapGraph::new("".to_string(), Language::Typescript);
    let reg = TypeScriptRegistry::new(&graph);
    assert_eq!(reg.resolve_type("/repo/src/a.ts", "mystery"), None);
    assert_eq!(reg.resolve_method("Mystery", "doThing"), None);
}
