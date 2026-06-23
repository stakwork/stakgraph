use crate::lang::asg::NodeKeys;
use crate::lang::graphs::{Graph, NodeType};
use std::collections::HashMap;
use std::path::Path;
use tree_sitter::{Node, Parser};

use super::scope::{scope_bind, scope_lookup, scope_pop, scope_push, Scope};

fn make_parser() -> Option<Parser> {
    let mut parser = Parser::new();
    let lang: tree_sitter::Language = tree_sitter_go::LANGUAGE.into();
    parser.set_language(&lang).ok()?;
    Some(parser)
}

/// Returns the parent directory of `file` as a string (empty if none).
fn parent_dir(file: &str) -> String {
    Path::new(file)
        .parent()
        .map(|p| p.to_string_lossy().into_owned())
        .unwrap_or_default()
}

/// Strip pointer/generic wrappers and return the base type identifier.
/// Returns None for qualified types like `chi.Router` (external packages).
fn strip_go_type(node: Node, source: &[u8]) -> Option<String> {
    match node.kind() {
        "type_identifier" => node.utf8_text(source).ok().map(str::to_string),
        "pointer_type" => node.named_child(0).and_then(|n| strip_go_type(n, source)),
        "generic_type" => node
            .child_by_field_name("type")
            .and_then(|n| n.utf8_text(source).ok())
            .map(str::to_string),
        _ => None, // qualified_type (pkg.Type) and everything else → external, skip
    }
}

// ── Extraction helpers ─────────────────────────────────────────────────────────

fn walk_struct_fields(node: Node, source: &[u8], fields: &mut HashMap<String, String>) {
    for i in 0..node.named_child_count() {
        let Some(child) = node.named_child(i) else {
            continue;
        };
        if child.kind() == "field_declaration" {
            let Some(type_node) = child.child_by_field_name("type") else {
                walk_struct_fields(child, source, fields);
                continue;
            };
            let Some(field_type) = strip_go_type(type_node, source) else {
                continue;
            };
            // Collect all field_identifier name children (handles `X, Y Type`)
            let mut found = false;
            for j in 0..child.named_child_count() {
                let Some(nc) = child.named_child(j) else {
                    continue;
                };
                if nc.kind() == "field_identifier" {
                    if let Ok(name) = nc.utf8_text(source) {
                        fields.insert(name.to_string(), field_type.clone());
                        found = true;
                    }
                }
            }
            if !found {
                if let Some(name_node) = child.child_by_field_name("name") {
                    if let Ok(name) = name_node.utf8_text(source) {
                        fields.insert(name.to_string(), field_type);
                    }
                }
            }
        } else {
            walk_struct_fields(child, source, fields);
        }
    }
}

/// Extract struct field types from Go source.
/// Returns struct_name → { field_name → base_type }.
/// External-package types (qualified_type) are omitted.
pub fn extract_struct_fields(source: &str) -> HashMap<String, HashMap<String, String>> {
    let mut out: HashMap<String, HashMap<String, String>> = HashMap::new();
    let Some(mut parser) = make_parser() else {
        return out;
    };
    let Some(tree) = parser.parse(source, None) else {
        return out;
    };
    let src = source.as_bytes();
    let root = tree.root_node();

    for i in 0..root.named_child_count() {
        let Some(type_decl) = root.named_child(i) else {
            continue;
        };
        if type_decl.kind() != "type_declaration" {
            continue;
        }
        for j in 0..type_decl.named_child_count() {
            let Some(type_spec) = type_decl.named_child(j) else {
                continue;
            };
            if type_spec.kind() != "type_spec" {
                continue;
            }
            let Some(name_node) = type_spec.child_by_field_name("name") else {
                continue;
            };
            let Ok(struct_name) = name_node.utf8_text(src) else {
                continue;
            };
            let Some(type_node) = type_spec.child_by_field_name("type") else {
                continue;
            };
            if type_node.kind() != "struct_type" {
                continue;
            }
            let mut fields: HashMap<String, String> = HashMap::new();
            walk_struct_fields(type_node, src, &mut fields);
            if !fields.is_empty() {
                out.entry(struct_name.to_string()).or_default().extend(fields);
            }
        }
    }
    out
}

/// Extract top-level function return types from Go source.
/// Returns { func_name → base_return_type }.
/// For multi-return `(T, error)`, records T. Skips external-package and error-only returns.
pub fn extract_fn_returns(source: &str) -> HashMap<String, String> {
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
        let Some(node) = root.named_child(i) else {
            continue;
        };
        extract_fn_return_from_node(node, src, &mut out);
    }
    out
}

fn extract_fn_return_from_node(node: Node, source: &[u8], out: &mut HashMap<String, String>) {
    if !matches!(node.kind(), "function_declaration" | "method_declaration") {
        return;
    }
    let Some(name_node) = node.child_by_field_name("name") else {
        return;
    };
    let Ok(func_name) = name_node.utf8_text(source) else {
        return;
    };
    let Some(result_node) = node.child_by_field_name("result") else {
        return;
    };

    let base_type: Option<String> = match result_node.kind() {
        "type_identifier" | "pointer_type" | "generic_type" => {
            strip_go_type(result_node, source).filter(|t| t != "error")
        }
        "parameter_list" => {
            // Multi-return: (Store, error) — take first non-error type.
            // Unnamed returns have type nodes directly; named returns have parameter_declaration.
            (0..result_node.named_child_count())
                .filter_map(|j| result_node.named_child(j))
                .find_map(|child| {
                    let type_node = if child.kind() == "parameter_declaration" {
                        child.child_by_field_name("type")?
                    } else {
                        child
                    };
                    let t = strip_go_type(type_node, source)?;
                    if t == "error" {
                        None
                    } else {
                        Some(t)
                    }
                })
        }
        _ => None,
    };

    if let Some(t) = base_type {
        out.entry(func_name.to_string()).or_insert(t);
    }
}

// ── Type evaluator ─────────────────────────────────────────────────────────────

fn eval_expr_type(
    scope: &Scope,
    struct_fields: &HashMap<String, HashMap<String, String>>,
    fn_returns: &HashMap<String, String>,
    node: Node,
    source: &[u8],
) -> Option<String> {
    match node.kind() {
        "identifier" => scope_lookup(scope, node.utf8_text(source).ok()?).map(str::to_string),
        "selector_expression" => {
            let obj_type = eval_expr_type(
                scope,
                struct_fields,
                fn_returns,
                node.child_by_field_name("operand")?,
                source,
            )?;
            let field = node
                .child_by_field_name("field")?
                .utf8_text(source)
                .ok()?;
            struct_fields.get(&obj_type)?.get(field).cloned()
        }
        "unary_expression" => {
            // &T{} or *ptr → eval the operand
            node.named_child(0)
                .and_then(|inner| eval_expr_type(scope, struct_fields, fn_returns, inner, source))
        }
        "composite_literal" => {
            // SomeType{...} or &SomeType{...} → extract type child
            node.child_by_field_name("type")
                .and_then(|t| strip_go_type(t, source))
        }
        "call_expression" => {
            let func = node.child_by_field_name("function")?;
            if func.kind() == "identifier" {
                let func_name = func.utf8_text(source).ok()?;
                fn_returns.get(func_name).cloned()
            } else {
                None
            }
        }
        _ => None,
    }
}

// ── Callee resolution ──────────────────────────────────────────────────────────

fn resolve_callee<G: Graph>(
    scope: &Scope,
    struct_fields: &HashMap<String, HashMap<String, String>>,
    fn_returns: &HashMap<String, String>,
    pkg_fns: &HashMap<String, HashMap<String, NodeKeys>>,
    graph: &G,
    node: Node,
    source: &[u8],
    file: &str,
) -> Option<NodeKeys> {
    match node.kind() {
        "selector_expression" => {
            let obj_type = eval_expr_type(
                scope,
                struct_fields,
                fn_returns,
                node.child_by_field_name("operand")?,
                source,
            )?;
            let method_name = node
                .child_by_field_name("field")?
                .utf8_text(source)
                .ok()?;
            // Find a Function node with matching operand (receiver type)
            graph
                .find_nodes_by_name(NodeType::Function, method_name)
                .into_iter()
                .find(|n| n.meta.get("operand").map(|s| s.as_str()) == Some(obj_type.as_str()))
                .map(|n| NodeKeys::from(&n))
        }
        "identifier" => {
            let fname = node.utf8_text(source).ok()?;
            // Same-file first
            if let Some(n) = graph.find_node_by_name_in_file(NodeType::Function, fname, file) {
                return Some(NodeKeys::from(&n));
            }
            // Same-package (all .go files in the same directory)
            let dir = parent_dir(file);
            pkg_fns.get(&dir)?.get(fname).cloned()
        }
        _ => None,
    }
}

// ── Parameter binding ──────────────────────────────────────────────────────────

fn bind_params(params: Node, source: &[u8], scope: &mut Scope) {
    for i in 0..params.named_child_count() {
        let Some(decl) = params.named_child(i) else {
            continue;
        };
        if decl.kind() != "parameter_declaration" {
            continue;
        }
        let Some(type_node) = decl.child_by_field_name("type") else {
            continue;
        };
        let Some(type_name) = strip_go_type(type_node, source) else {
            continue;
        };
        // Bind all identifier names in this declaration (handles `x, y Type`)
        let mut found = false;
        for j in 0..decl.named_child_count() {
            let Some(nc) = decl.named_child(j) else {
                continue;
            };
            if nc.kind() == "identifier" {
                let Ok(name) = nc.utf8_text(source) else {
                    continue;
                };
                if name != "_" {
                    scope_bind(scope, name, &type_name);
                    found = true;
                }
            }
        }
        if !found {
            if let Some(name_node) = decl.child_by_field_name("name") {
                if let Ok(name) = name_node.utf8_text(source) {
                    if name != "_" {
                        scope_bind(scope, name, &type_name);
                    }
                }
            }
        }
    }
}

// ── Short variable declaration binding ────────────────────────────────────────

fn try_bind_short_var(
    node: Node,
    source: &[u8],
    scope: &mut Scope,
    struct_fields: &HashMap<String, HashMap<String, String>>,
    fn_returns: &HashMap<String, String>,
) {
    let Some(left) = node.child_by_field_name("left") else {
        return;
    };
    let Some(right) = node.child_by_field_name("right") else {
        return;
    };

    let names: Vec<String> = (0..left.named_child_count())
        .filter_map(|i| left.named_child(i))
        .filter(|n| n.kind() == "identifier")
        .filter_map(|n| n.utf8_text(source).ok().map(str::to_string))
        .collect();

    let types: Vec<Option<String>> = (0..right.named_child_count())
        .filter_map(|i| right.named_child(i))
        .map(|n| eval_expr_type(scope, struct_fields, fn_returns, n, source))
        .collect();

    for (i, name) in names.iter().enumerate() {
        if name == "_" {
            continue;
        }
        if let Some(Some(t)) = types.get(i) {
            scope_bind(scope, name, t);
        }
    }
}

// ── AST walker ────────────────────────────────────────────────────────────────

fn walk_node<G: Graph>(
    node: Node,
    source: &[u8],
    scope: &mut Scope,
    struct_fields: &HashMap<String, HashMap<String, String>>,
    fn_returns: &HashMap<String, String>,
    pkg_fns: &HashMap<String, HashMap<String, NodeKeys>>,
    graph: &G,
    out: &mut HashMap<(usize, usize), NodeKeys>,
    file: &str,
) {
    match node.kind() {
        "function_declaration" => {
            scope_push(scope);
            if let Some(params) = node.child_by_field_name("parameters") {
                bind_params(params, source, scope);
            }
            if let Some(body) = node.child_by_field_name("body") {
                walk_node(body, source, scope, struct_fields, fn_returns, pkg_fns, graph, out, file);
            }
            scope_pop(scope);
        }
        "method_declaration" => {
            scope_push(scope);
            // Bind receiver variable: receiver is a parameter_list with one parameter_declaration
            if let Some(recv_list) = node.child_by_field_name("receiver") {
                if let Some(pd) = recv_list.named_child(0) {
                    if pd.kind() == "parameter_declaration" {
                        if let Some(type_node) = pd.child_by_field_name("type") {
                            if let Some(recv_type) = strip_go_type(type_node, source) {
                                // Try field name "name" first, then iterate identifiers
                                let recv_name = pd
                                    .child_by_field_name("name")
                                    .and_then(|n| n.utf8_text(source).ok().map(str::to_string))
                                    .or_else(|| {
                                        (0..pd.named_child_count())
                                            .filter_map(|j| pd.named_child(j))
                                            .find(|n| n.kind() == "identifier")
                                            .and_then(|n| {
                                                n.utf8_text(source).ok().map(str::to_string)
                                            })
                                    });
                                if let Some(name) = recv_name {
                                    if name != "_" {
                                        scope_bind(scope, &name, &recv_type);
                                    }
                                }
                            }
                        }
                    }
                }
            }
            if let Some(params) = node.child_by_field_name("parameters") {
                bind_params(params, source, scope);
            }
            if let Some(body) = node.child_by_field_name("body") {
                walk_node(body, source, scope, struct_fields, fn_returns, pkg_fns, graph, out, file);
            }
            scope_pop(scope);
        }
        "func_literal" => {
            scope_push(scope);
            if let Some(params) = node.child_by_field_name("parameters") {
                bind_params(params, source, scope);
            }
            if let Some(body) = node.child_by_field_name("body") {
                walk_node(body, source, scope, struct_fields, fn_returns, pkg_fns, graph, out, file);
            }
            scope_pop(scope);
        }
        "short_var_declaration" => {
            // Bind types before recursing so later calls can see them in scope
            try_bind_short_var(node, source, scope, struct_fields, fn_returns);
            if let Some(right) = node.child_by_field_name("right") {
                walk_node(right, source, scope, struct_fields, fn_returns, pkg_fns, graph, out, file);
            }
        }
        "call_expression" => {
            if let Some(func_node) = node.child_by_field_name("function") {
                if let Some(target) =
                    resolve_callee(scope, struct_fields, fn_returns, pkg_fns, graph, func_node, source, file)
                {
                    let pos = if func_node.kind() == "selector_expression" {
                        func_node
                            .child_by_field_name("field")
                            .map(|f| f.start_position())
                            .unwrap_or_else(|| func_node.start_position())
                    } else {
                        func_node.start_position()
                    };
                    out.insert((pos.row, pos.column), target);
                }
                // Recurse into func_node to find nested calls (e.g. chained method calls)
                walk_node(func_node, source, scope, struct_fields, fn_returns, pkg_fns, graph, out, file);
                // Recurse into arguments for nested calls within args
                if let Some(args) = node.child_by_field_name("arguments") {
                    walk_node(args, source, scope, struct_fields, fn_returns, pkg_fns, graph, out, file);
                }
            }
        }
        _ => {
            for i in 0..node.named_child_count() {
                if let Some(child) = node.named_child(i) {
                    walk_node(child, source, scope, struct_fields, fn_returns, pkg_fns, graph, out, file);
                }
            }
        }
    }
}

// ── Public entry point ────────────────────────────────────────────────────────

/// Walk a Go source file and return a map from (row, col) of each resolved
/// call's function-name position to the target NodeKeys.
pub fn resolve_file_calls<G: Graph>(
    source: &str,
    file: &str,
    struct_fields: &HashMap<String, HashMap<String, String>>,
    fn_returns: &HashMap<String, String>,
    pkg_fns: &HashMap<String, HashMap<String, NodeKeys>>,
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

    // Seed root scope with package-level vars from ALL files in the same directory.
    // Go's package scope spans all files in the directory, so `var DB database` in
    // db.go is visible in routes.go without an import.
    let file_dir = parent_dir(file);
    let mut scope: Scope = vec![HashMap::new()];
    for ((f, name), type_name) in var_types {
        if parent_dir(f) == file_dir {
            scope_bind(&mut scope, name, type_name);
        }
    }

    walk_node(
        tree.root_node(),
        src,
        &mut scope,
        struct_fields,
        fn_returns,
        pkg_fns,
        graph,
        &mut out,
        file,
    );
    out
}
