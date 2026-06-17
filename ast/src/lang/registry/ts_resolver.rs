use crate::lang::asg::NodeKeys;
use crate::lang::graphs::{Graph, NodeType};
use std::collections::HashMap;
use tree_sitter::{Node, Parser};

type Scope = Vec<HashMap<String, String>>;

fn scope_push(s: &mut Scope) {
    s.push(HashMap::new());
}

fn scope_pop(s: &mut Scope) {
    s.pop();
}

fn scope_bind(s: &mut Scope, name: &str, type_name: &str) {
    if let Some(frame) = s.last_mut() {
        frame.insert(name.to_string(), type_name.to_string());
    }
}

fn scope_lookup<'a>(s: &'a Scope, name: &str) -> Option<&'a str> {
    s.iter().rev().find_map(|f| f.get(name).map(|s| s.as_str()))
}

fn eval_expr_type(
    scope: &Scope,
    class_fields: &HashMap<String, HashMap<String, String>>,
    node: Node,
    source: &[u8],
) -> Option<String> {
    match node.kind() {
        "identifier" => scope_lookup(scope, node.utf8_text(source).ok()?).map(|s| s.to_string()),
        "member_expression" => {
            let obj_type =
                eval_expr_type(scope, class_fields, node.child_by_field_name("object")?, source)?;
            let prop = node.child_by_field_name("property")?.utf8_text(source).ok()?;
            class_fields.get(obj_type.as_str())?.get(prop).cloned()
        }
        "new_expression" => node
            .child_by_field_name("constructor")?
            .utf8_text(source)
            .ok()
            .map(|s| s.to_string()),
        "await_expression" => {
            for i in 0..node.named_child_count() {
                if let Some(child) = node.named_child(i) {
                    if let Some(t) = eval_expr_type(scope, class_fields, child, source) {
                        return Some(t);
                    }
                }
            }
            None
        }
        _ => None,
    }
}

fn resolve_callee<G: Graph>(
    scope: &Scope,
    class_fields: &HashMap<String, HashMap<String, String>>,
    graph: &G,
    node: Node,
    source: &[u8],
) -> Option<NodeKeys> {
    if node.kind() != "member_expression" {
        return None;
    }
    let receiver_type =
        eval_expr_type(scope, class_fields, node.child_by_field_name("object")?, source)?;
    let method_name = node
        .child_by_field_name("property")?
        .utf8_text(source)
        .ok()?;
    graph
        .find_nodes_by_name(NodeType::Function, method_name)
        .into_iter()
        .find(|n| n.meta.get("operand").map(|s| s.as_str()) == Some(receiver_type.as_str()))
        .map(|n| NodeKeys::from(&n))
}

fn try_bind_declaration(
    node: Node,
    source: &[u8],
    scope: &mut Scope,
    class_fields: &HashMap<String, HashMap<String, String>>,
) {
    for i in 0..node.named_child_count() {
        let Some(child) = node.named_child(i) else {
            continue;
        };
        if child.kind() != "variable_declarator" {
            continue;
        }
        let Some(name_node) = child.child_by_field_name("name") else {
            continue;
        };
        let Ok(name) = name_node.utf8_text(source) else {
            continue;
        };
        if let Some(type_node) = child.child_by_field_name("type") {
            if let Some(inner) = type_node.named_child(0) {
                if let Ok(type_str) = inner.utf8_text(source) {
                    scope_bind(scope, name, type_str);
                    continue;
                }
            }
        }
        if let Some(value_node) = child.child_by_field_name("value") {
            if let Some(type_name) = eval_expr_type(scope, class_fields, value_node, source) {
                scope_bind(scope, name, &type_name);
            }
        }
    }
}

fn bind_params(params_node: Node, source: &[u8], scope: &mut Scope) {
    for i in 0..params_node.named_child_count() {
        let Some(param) = params_node.named_child(i) else {
            continue;
        };
        if !matches!(param.kind(), "required_parameter" | "optional_parameter") {
            continue;
        }
        let Some(name_node) = param.child_by_field_name("pattern") else {
            continue;
        };
        let Ok(name) = name_node.utf8_text(source) else {
            continue;
        };
        if let Some(type_node) = param.child_by_field_name("type") {
            if let Some(inner) = type_node.named_child(0) {
                if let Ok(type_str) = inner.utf8_text(source) {
                    scope_bind(scope, name, type_str);
                }
            }
        }
    }
}

fn walk_node<G: Graph>(
    node: Node,
    source: &[u8],
    scope: &mut Scope,
    class_fields: &HashMap<String, HashMap<String, String>>,
    graph: &G,
    out: &mut HashMap<(usize, usize), NodeKeys>,
) {
    match node.kind() {
        "call_expression" => {
            if let Some(func_node) = node.child_by_field_name("function") {
                if let Some(target) = resolve_callee(scope, class_fields, graph, func_node, source)
                {
                    let pos = func_node
                        .child_by_field_name("property")
                        .map(|p| p.start_position())
                        .unwrap_or_else(|| func_node.start_position());
                    out.insert((pos.row, pos.column), target);
                }
                if let Some(args) = node.child_by_field_name("arguments") {
                    walk_node(args, source, scope, class_fields, graph, out);
                }
            }
        }
        "lexical_declaration" | "variable_declaration" => {
            try_bind_declaration(node, source, scope, class_fields);
            for i in 0..node.named_child_count() {
                if let Some(child) = node.named_child(i) {
                    walk_node(child, source, scope, class_fields, graph, out);
                }
            }
        }
        "function_declaration"
        | "function"
        | "arrow_function"
        | "method_definition"
        | "generator_function"
        | "generator_function_declaration" => {
            scope_push(scope);
            if let Some(params) = node.child_by_field_name("parameters") {
                bind_params(params, source, scope);
            }
            if let Some(body) = node.child_by_field_name("body") {
                walk_node(body, source, scope, class_fields, graph, out);
            }
            scope_pop(scope);
        }
        "class_declaration" | "class" => {
            if let Some(body) = node.child_by_field_name("body") {
                for i in 0..body.named_child_count() {
                    if let Some(child) = body.named_child(i) {
                        walk_node(child, source, scope, class_fields, graph, out);
                    }
                }
            }
        }
        _ => {
            for i in 0..node.named_child_count() {
                if let Some(child) = node.named_child(i) {
                    walk_node(child, source, scope, class_fields, graph, out);
                }
            }
        }
    }
}

fn make_parser() -> Option<Parser> {
    let mut parser = Parser::new();
    let lang: tree_sitter::Language = tree_sitter_typescript::LANGUAGE_TSX.into();
    parser.set_language(&lang).ok()?;
    Some(parser)
}

/// Walk a TypeScript/TSX source file and return a map from
/// (row, col) of each method-call's property identifier to the target NodeKeys.
pub fn resolve_file_calls<G: Graph>(
    source: &str,
    file: &str,
    class_fields: &HashMap<String, HashMap<String, String>>,
    import_sources: &HashMap<(String, String), String>,
    var_types: &HashMap<(String, String), String>,
    graph: &G,
) -> HashMap<(usize, usize), NodeKeys> {
    let mut out = HashMap::new();
    let Some(mut parser) = make_parser() else {
        return out;
    };
    let Some(tree) = parser.parse(source, None) else {
        return out;
    };
    let src = source.as_bytes();

    let mut scope: Scope = vec![HashMap::new()];
    for ((caller_file, name), source_file) in import_sources {
        if caller_file != file {
            continue;
        }
        // Try direct lookup; fall back to resolving relative import path
        let type_name = lookup_with_extensions(source_file, name, var_types).or_else(|| {
            let resolved = resolve_import_relative(file, source_file);
            lookup_with_extensions(&resolved, name, var_types)
        });
        if let Some(t) = type_name {
            scope_bind(&mut scope, name, t.as_str());
        }
    }

    walk_node(tree.root_node(), src, &mut scope, class_fields, graph, &mut out);
    out
}

fn lookup_with_extensions(
    base: &str,
    name: &str,
    var_types: &HashMap<(String, String), String>,
) -> Option<String> {
    for ext in &["", ".ts", ".tsx"] {
        let key = (format!("{}{}", base, ext), name.to_string());
        if let Some(v) = var_types.get(&key) {
            return Some(v.clone());
        }
    }
    None
}

/// Normalize a relative import path like "../../lib/api/client" relative to
/// the directory of `current_file`, returning the canonical relative path.
fn resolve_import_relative(current_file: &str, import_path: &str) -> String {
    use std::path::{Component, Path, PathBuf};
    if !import_path.starts_with('.') {
        return import_path.to_string();
    }
    let base_dir = Path::new(current_file)
        .parent()
        .unwrap_or(Path::new(""));
    let joined = base_dir.join(import_path);
    let mut normalized = PathBuf::new();
    for component in joined.components() {
        match component {
            Component::ParentDir => {
                normalized.pop();
            }
            Component::CurDir => {}
            c => normalized.push(c),
        }
    }
    normalized.display().to_string()
}

/// Extract top-level `const/let x = new Class()` var types from a TS/TSX source.
/// Returns { var_name → constructor_type } for unannotated new-expression assignments.
pub fn extract_top_level_vars(source: &str) -> HashMap<String, String> {
    let mut out = HashMap::new();
    let Some(mut parser) = make_parser() else {
        return out;
    };
    let Some(tree) = parser.parse(source, None) else {
        return out;
    };
    let src = source.as_bytes();
    let root = tree.root_node();
    for i in 0..root.named_child_count() {
        let Some(stmt) = root.named_child(i) else {
            continue;
        };
        // unwrap export_statement to get the inner declaration
        let decl = if stmt.kind() == "export_statement" {
            stmt.named_child(0).unwrap_or(stmt)
        } else {
            stmt
        };
        if !matches!(decl.kind(), "lexical_declaration" | "variable_declaration") {
            continue;
        }
        for j in 0..decl.named_child_count() {
            let Some(declarator) = decl.named_child(j) else {
                continue;
            };
            if declarator.kind() != "variable_declarator" {
                continue;
            }
            let Some(name_node) = declarator.child_by_field_name("name") else {
                continue;
            };
            let Some(value_node) = declarator.child_by_field_name("value") else {
                continue;
            };
            if value_node.kind() != "new_expression" {
                continue;
            }
            let Some(ctor_node) = value_node.child_by_field_name("constructor") else {
                continue;
            };
            if let (Ok(var_name), Ok(type_name)) =
                (name_node.utf8_text(src), ctor_node.utf8_text(src))
            {
                out.insert(var_name.to_string(), type_name.to_string());
            }
        }
    }
    out
}

/// Extract class field types from a TypeScript/TSX source file.
/// Returns class_name → { field_name → constructor_type }.
pub fn extract_class_fields(source: &str) -> HashMap<String, HashMap<String, String>> {
    let mut out: HashMap<String, HashMap<String, String>> = HashMap::new();
    let Some(mut parser) = make_parser() else {
        return out;
    };
    let Some(tree) = parser.parse(source, None) else {
        return out;
    };
    walk_classes(tree.root_node(), source.as_bytes(), &mut out);
    out
}

fn walk_classes(
    node: Node,
    source: &[u8],
    out: &mut HashMap<String, HashMap<String, String>>,
) {
    if matches!(node.kind(), "class_declaration" | "class") {
        if let (Some(name_node), Some(body_node)) = (
            node.child_by_field_name("name"),
            node.child_by_field_name("body"),
        ) {
            if let Ok(class_name) = name_node.utf8_text(source) {
                let mut fields: HashMap<String, String> = HashMap::new();
                for i in 0..body_node.named_child_count() {
                    let Some(member) = body_node.named_child(i) else {
                        continue;
                    };
                    if member.kind() != "public_field_definition" {
                        continue;
                    }
                    let Some(fname_node) = member.child_by_field_name("name") else {
                        continue;
                    };
                    let Some(value_node) = member.child_by_field_name("value") else {
                        continue;
                    };
                    if value_node.kind() != "new_expression" {
                        continue;
                    }
                    let Some(ctor_node) = value_node.child_by_field_name("constructor") else {
                        continue;
                    };
                    if let (Ok(field_name), Ok(type_name)) = (
                        fname_node.utf8_text(source),
                        ctor_node.utf8_text(source),
                    ) {
                        fields.insert(field_name.to_string(), type_name.to_string());
                    }
                }
                if !fields.is_empty() {
                    out.entry(class_name.to_string()).or_default().extend(fields);
                }
            }
        }
    }
    for i in 0..node.named_child_count() {
        if let Some(child) = node.named_child(i) {
            walk_classes(child, source, out);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resolve_import_relative() {
        assert_eq!(
            resolve_import_relative(
                "src/testing/nextjs/app/api-demo/page.tsx",
                "../../lib/api/apiClient"
            ),
            "src/testing/nextjs/lib/api/apiClient"
        );
        assert_eq!(
            resolve_import_relative("src/app/page.tsx", "../utils"),
            "src/utils"
        );
        // non-relative paths pass through unchanged
        assert_eq!(
            resolve_import_relative("src/app/page.tsx", "react"),
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
        let vars = extract_top_level_vars(source);
        println!("top_level_vars: {:?}", vars);
        let fields = extract_class_fields(source);
        println!("class_fields: {:?}", fields);
        assert_eq!(vars.get("api").map(|s| s.as_str()), Some("APIClient"), "api not found in top_level_vars");
        assert!(fields.contains_key("APIClient"), "APIClient not found in class_fields");
        assert_eq!(fields.get("APIClient").and_then(|m| m.get("users")).map(|s| s.as_str()), Some("UsersAPI"));
    }
}
