use crate::lang::asg::NodeData;
use crate::lang::graphs::{BTreeMapGraph, Graph, NodeType};
use crate::lang::registry::{go_resolver, py_resolver, ts_resolver, typescript::TypeScriptRegistry, Registry};
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

#[test]
fn test_extract_fn_returns_promise_unwrapping() {
    let source = r#"
async function fetchUser(id: string): Promise<User> { return {} as User; }
async function listUsers(): Promise<User[]> { return []; }
async function complexFn(): Promise<Map<string, User>> { return new Map(); }
function syncFn(): UserService { return new UserService(); }
"#;
    let returns = ts_resolver::extract_fn_returns(source);
    assert_eq!(returns.get("fetchUser").map(|s| s.as_str()), Some("User"));
    assert_eq!(returns.get("listUsers").map(|s| s.as_str()), Some("User[]"));
    // Promise<Map<…>> inner still contains < → filtered
    assert_eq!(returns.get("complexFn"), None);
    assert_eq!(returns.get("syncFn").map(|s| s.as_str()), Some("UserService"));
}

#[test]
fn test_extract_class_fields_includes_method_returns() {
    let source = r#"
class UserService {
  users = new UsersAPI();
  async getUser(id: string): Promise<User> { return {} as User; }
  async listUsers(): Promise<User[]> { return []; }
  async complex(): Promise<Map<string, User>> { return new Map(); }
}
"#;
    let fields = ts_resolver::extract_class_fields(source);
    let svc = fields.get("UserService").unwrap();
    // Regular new-expression field still captured
    assert_eq!(svc.get("users").map(|s| s.as_str()), Some("UsersAPI"));
    // Method return types stored with () suffix, Promise<T> unwrapped
    assert_eq!(svc.get("getUser()").map(|s| s.as_str()), Some("User"));
    assert_eq!(svc.get("listUsers()").map(|s| s.as_str()), Some("User[]"));
    // Promise<Map<…>> inner still has < → filtered
    assert_eq!(svc.get("complex()"), None);
}

// ── Go resolver unit tests ─────────────────────────────────────────────────────

#[test]
fn test_go_extract_fn_returns() {
    let source = r#"
package main

func newItemStore() *ItemStore { return &ItemStore{} }
func getCount() (int, error) { return 0, nil }
func justError() error { return nil }
"#;
    let returns = go_resolver::extract_fn_returns(source);
    assert_eq!(returns.get("newItemStore").map(|s| s.as_str()), Some("ItemStore"));
    // Multi-return (int, error): first non-error type is int (primitive, kept)
    assert_eq!(returns.get("getCount").map(|s| s.as_str()), Some("int"));
    // error-only return: skipped
    assert_eq!(returns.get("justError"), None);
}

// ── Python resolver unit tests ────────────────────────────────────────────────

#[test]
fn test_py_extract_fn_returns_optional_unwrapping() {
    let source = r#"
def get_user(user_id: str) -> Optional[User]:
    return None

def list_users() -> List[User]:
    return []

def sync_fn() -> UserService:
    return UserService()
"#;
    let returns = py_resolver::extract_fn_returns(source);
    // Optional[User] → unwrapped to User
    assert_eq!(returns.get("get_user").map(|s| s.as_str()), Some("User"));
    // List[User] — outer is not Optional → skipped
    assert_eq!(returns.get("list_users"), None);
    // Plain return type unchanged
    assert_eq!(returns.get("sync_fn").map(|s| s.as_str()), Some("UserService"));
}

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
