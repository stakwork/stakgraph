use crate::lang::asg::NodeData;
use crate::lang::graphs::{BTreeMapGraph, Graph, NodeType};
use crate::lang::registry::{py_resolver, ts_resolver, typescript::TypeScriptRegistry, Registry};
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

fn ts_method(file: &str, name: &str, operand: &str) -> NodeData {
    let mut nd = NodeData::in_file(file);
    nd.name = name.to_string();
    nd.meta.insert("operand".to_string(), operand.to_string());
    nd
}

#[test]
fn test_resolve_type_from_var() {
    let mut graph = BTreeMapGraph::new("".to_string(), Language::Typescript);
    graph.add_node(&NodeType::Var, &ts_var("/repo/src/a.ts", "userService", "UserService"));
    let reg = TypeScriptRegistry::new(&graph, &[]);
    assert_eq!(
        reg.resolve_type("/repo/src/a.ts", "userService"),
        Some("UserService")
    );
}

#[test]
fn test_resolve_method_via_type_def() {
    let mut graph = BTreeMapGraph::new("".to_string(), Language::Typescript);
    graph.add_node(&NodeType::Class, &ts_class("/repo/src/user.ts", "UserService"));
    graph.add_node(&NodeType::Function, &ts_method("/repo/src/user.ts", "getUser", "UserService"));
    let reg = TypeScriptRegistry::new(&graph, &[]);
    assert_eq!(
        reg.resolve_method("UserService", "getUser"),
        Some("/repo/src/user.ts")
    );
    // method not on this type returns None
    assert_eq!(reg.resolve_method("UserService", "unrelated"), None);
}

#[test]
fn test_resolve_unknown_returns_none() {
    let graph = BTreeMapGraph::new("".to_string(), Language::Typescript);
    let reg = TypeScriptRegistry::new(&graph, &[]);
    assert_eq!(reg.resolve_type("/repo/src/a.ts", "mystery"), None);
    assert_eq!(reg.resolve_method("Mystery", "doThing"), None);
}

#[test]
fn test_resolve_import_relative() {
    assert_eq!(
        ts_resolver::resolve_import_relative(
            "src/testing/nextjs/app/api-demo/page.tsx",
            "../../lib/api/apiClient"
        ),
        "src/testing/nextjs/lib/api/apiClient"
    );
    assert_eq!(
        ts_resolver::resolve_import_relative("src/app/page.tsx", "../utils"),
        "src/utils"
    );
    assert_eq!(
        ts_resolver::resolve_import_relative("src/app/page.tsx", "react"),
        "react"
    );
}

#[test]
fn test_extract_top_level_vars_finds_api() {
    let source = r#"
class APIClient {
  users = new UsersAPI();
  posts = new PostsAPI();
}
export const api = new APIClient();
"#;
    let vars = ts_resolver::extract_top_level_vars(source);
    let fields = ts_resolver::extract_class_fields(source);
    assert_eq!(vars.get("api").map(|s| s.as_str()), Some("APIClient"));
    assert!(fields.contains_key("APIClient"));
    assert_eq!(
        fields.get("APIClient").and_then(|m| m.get("users")).map(|s| s.as_str()),
        Some("UsersAPI")
    );
}

#[test]
fn test_extract_fn_returns() {
    let source = r#"
export function useUserQuery(userId: string): QueryResult {
    return {} as QueryResult;
}
export const useActions = (): ActionsResult => ({} as ActionsResult);
export function useGeneric<T>(): Promise<T> { return Promise.resolve({} as T); }
"#;
    let returns = ts_resolver::extract_fn_returns(source);
    assert_eq!(returns.get("useUserQuery").map(|s| s.as_str()), Some("QueryResult"));
    assert_eq!(returns.get("useActions").map(|s| s.as_str()), Some("ActionsResult"));
    // generic return types are excluded
    assert_eq!(returns.get("useGeneric"), None);
}

// ── Python resolver unit tests ────────────────────────────────────────────────

#[test]
fn test_py_extract_class_fields_constructor() {
    let source = r#"
class UserService:
    def __init__(self):
        self.repo = UserRepository()
        self.cache = CacheService()
"#;
    let fields = py_resolver::extract_class_fields(source);
    assert!(fields.contains_key("UserService"));
    let svc = fields.get("UserService").unwrap();
    assert_eq!(svc.get("repo").map(|s| s.as_str()), Some("UserRepository"));
    assert_eq!(svc.get("cache").map(|s| s.as_str()), Some("CacheService"));
}

#[test]
fn test_py_extract_class_fields_param_annotation() {
    let source = r#"
class UserController:
    def __init__(self, service: UserService):
        self.service = service
"#;
    let fields = py_resolver::extract_class_fields(source);
    let ctrl = fields.get("UserController").unwrap();
    assert_eq!(ctrl.get("service").map(|s| s.as_str()), Some("UserService"));
}

#[test]
fn test_py_extract_class_fields_explicit_annotation() {
    let source = r#"
class App:
    def __init__(self):
        self.db: Database = Database()
"#;
    let fields = py_resolver::extract_class_fields(source);
    let app = fields.get("App").unwrap();
    // explicit annotation wins over constructor inference
    assert_eq!(app.get("db").map(|s| s.as_str()), Some("Database"));
}

#[test]
fn test_py_extract_top_level_vars() {
    let source = r#"
service = UserService()
typed_svc: UserService = UserService()
"#;
    let vars = py_resolver::extract_top_level_vars(source);
    assert_eq!(vars.get("service").map(|s| s.as_str()), Some("UserService"));
    assert_eq!(vars.get("typed_svc").map(|s| s.as_str()), Some("UserService"));
}

#[test]
fn test_py_extract_fn_returns() {
    let source = r#"
def get_service() -> UserService:
    return UserService()

def generic_fn():
    pass
"#;
    let returns = py_resolver::extract_fn_returns(source);
    assert_eq!(returns.get("get_service").map(|s| s.as_str()), Some("UserService"));
    assert_eq!(returns.get("generic_fn"), None);
}
