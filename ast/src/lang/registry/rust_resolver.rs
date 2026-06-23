use crate::lang::asg::NodeKeys;
use crate::lang::graphs::{Graph, NodeType};
use std::collections::HashMap;
use std::path::Path;
use tree_sitter::{Node, Parser};

use super::scope::{scope_bind, scope_lookup, scope_pop, scope_push, Scope};

fn make_parser() -> Option<Parser> {
    let mut parser = Parser::new();
    let lang: tree_sitter::Language = tree_sitter_rust::LANGUAGE.into();
    parser.set_language(&lang).ok()?;
    Some(parser)
}

fn parent_dir(file: &str) -> String {
    Path::new(file)
        .parent()
        .map(|p| p.to_string_lossy().into_owned())
        .unwrap_or_default()
}

// Wrappers that are transparent for method-dispatch purposes:
// Arc<T> → T, Box<T> → T, etc.
const TRANSPARENT_WRAPPERS: &[&str] = &[
    "Arc", "Box", "Mutex", "RwLock", "Rc", "Option", "Cell", "RefCell",
];

/// Strip reference/pointer/transparent-generic wrappers and return the base type name.
/// Returns None for dyn/impl Trait, primitive types, tuple types, and array types.
fn strip_rust_type(node: Node, source: &[u8]) -> Option<String> {
    match node.kind() {
        "type_identifier" => node.utf8_text(source).ok().map(str::to_string),

        // &T or &mut T — skip the optional lifetime and mutable modifier
        "reference_type" => node
            .child_by_field_name("type")
            .and_then(|n| strip_rust_type(n, source)),

        // *const T or *mut T
        "pointer_type" => node
            .child_by_field_name("type")
            .and_then(|n| strip_rust_type(n, source)),

        // Arc<T>, Vec<T>, Option<T>, etc.
        "generic_type" => {
            let outer = node.child_by_field_name("type")?;
            let outer_name = match outer.kind() {
                "type_identifier" => outer.utf8_text(source).ok()?.to_string(),
                // std::sync::Arc → take the rightmost name segment
                "scoped_type_identifier" => outer
                    .child_by_field_name("name")
                    .and_then(|n| n.utf8_text(source).ok())
                    .map(str::to_string)?,
                _ => return None,
            };
            if TRANSPARENT_WRAPPERS.contains(&outer_name.as_str()) {
                // Recurse into first non-lifetime type argument
                if let Some(args) = node.child_by_field_name("type_arguments") {
                    for i in 0..args.named_child_count() {
                        if let Some(arg) = args.named_child(i) {
                            if arg.kind() == "lifetime" {
                                continue;
                            }
                            if let Some(t) = strip_rust_type(arg, source) {
                                return Some(t);
                            }
                        }
                    }
                }
                None
            } else {
                Some(outer_name)
            }
        }

        // path::Type — take the rightmost segment
        "scoped_type_identifier" => node
            .child_by_field_name("name")
            .and_then(|n| n.utf8_text(source).ok())
            .map(str::to_string),

        // dyn Trait / impl Trait — no concrete type to dispatch on
        "dynamic_type" | "abstract_type" => None,

        _ => None,
    }
}

// ── Struct field extraction ────────────────────────────────────────────────────

fn walk_field_declaration_list(
    node: Node,
    source: &[u8],
    fields: &mut HashMap<String, String>,
) {
    for i in 0..node.named_child_count() {
        let Some(child) = node.named_child(i) else {
            continue;
        };
        if child.kind() != "field_declaration" {
            continue;
        }
        let Some(type_node) = child.child_by_field_name("type") else {
            continue;
        };
        let Some(field_type) = strip_rust_type(type_node, source) else {
            continue;
        };
        if let Some(name_node) = child.child_by_field_name("name") {
            if let Ok(name) = name_node.utf8_text(source) {
                fields.insert(name.to_string(), field_type);
            }
        }
    }
}

/// Extract `struct_name → { field_name → base_type }` from a Rust source file.
/// Enum variants and tuple struct fields are skipped.
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
        let Some(item) = root.named_child(i) else {
            continue;
        };
        if item.kind() != "struct_item" {
            continue;
        }
        let Some(name_node) = item.child_by_field_name("name") else {
            continue;
        };
        let Ok(struct_name) = name_node.utf8_text(src) else {
            continue;
        };
        let Some(body) = item.child_by_field_name("body") else {
            continue;
        };
        // field_declaration_list for named structs; ordered_field_declaration_list for tuples
        if body.kind() != "field_declaration_list" {
            continue;
        }
        let mut fields = HashMap::new();
        walk_field_declaration_list(body, src, &mut fields);
        if !fields.is_empty() {
            out.entry(struct_name.to_string()).or_default().extend(fields);
        }
    }
    out
}

// ── Type evaluator ─────────────────────────────────────────────────────────────

fn eval_expr_type(
    scope: &Scope,
    struct_fields: &HashMap<String, HashMap<String, String>>,
    node: Node,
    source: &[u8],
) -> Option<String> {
    match node.kind() {
        // In tree-sitter-rust, `self` is its own node kind, not an "identifier".
        "self" => scope_lookup(scope, "self").map(str::to_string),

        "identifier" => scope_lookup(scope, node.utf8_text(source).ok()?).map(str::to_string),

        // obj.field — look up field type in struct_fields
        "field_expression" => {
            let obj_type = eval_expr_type(
                scope,
                struct_fields,
                node.child_by_field_name("value")?,
                source,
            )?;
            let field = node
                .child_by_field_name("field")?
                .utf8_text(source)
                .ok()?;
            struct_fields.get(&obj_type)?.get(field).cloned()
        }

        // Type::new() → "Type" (constructor heuristic)
        "call_expression" => {
            let func = node.child_by_field_name("function")?;
            if func.kind() == "scoped_identifier" {
                let path = func.child_by_field_name("path")?;
                let name = func.child_by_field_name("name")?;
                let name_text = name.utf8_text(source).ok()?;
                if name_text == "new" || name_text.starts_with("new_") {
                    // path could be identifier or scoped_identifier; take rightmost name
                    let path_name = match path.kind() {
                        "identifier" => path.utf8_text(source).ok()?.to_string(),
                        "scoped_identifier" => path
                            .child_by_field_name("name")
                            .and_then(|n| n.utf8_text(source).ok())
                            .map(str::to_string)?,
                        _ => return None,
                    };
                    return Some(path_name);
                }
            }
            None
        }

        // &expr or *expr — strip the operator
        "reference_expression" | "unary_expression" => node
            .named_child(0)
            .and_then(|inner| eval_expr_type(scope, struct_fields, inner, source)),

        // Foo { ... } struct literal → type name
        "struct_expression" => {
            let name_node = node.child_by_field_name("name")?;
            name_node.utf8_text(source).ok().map(str::to_string)
        }

        // (expr) — transparent
        "parenthesized_expression" => node
            .named_child(0)
            .and_then(|inner| eval_expr_type(scope, struct_fields, inner, source)),

        // expr as Type — use the cast type
        "type_cast_expression" => node
            .child_by_field_name("type")
            .and_then(|t| strip_rust_type(t, source)),

        // expr.await — recurse; Future<Output=T> → we don't track T, but try the inner expr
        "await_expression" => node
            .child_by_field_name("value")
            .and_then(|inner| eval_expr_type(scope, struct_fields, inner, source)),

        _ => None,
    }
}

// ── Callee resolution ──────────────────────────────────────────────────────────

fn position_of_callee(func_node: Node) -> (usize, usize) {
    match func_node.kind() {
        "field_expression" => func_node
            .child_by_field_name("field")
            .map(|n| {
                let p = n.start_position();
                (p.row, p.column)
            })
            .unwrap_or_else(|| {
                let p = func_node.start_position();
                (p.row, p.column)
            }),
        "scoped_identifier" => func_node
            .child_by_field_name("name")
            .map(|n| {
                let p = n.start_position();
                (p.row, p.column)
            })
            .unwrap_or_else(|| {
                let p = func_node.start_position();
                (p.row, p.column)
            }),
        _ => {
            let p = func_node.start_position();
            (p.row, p.column)
        }
    }
}

fn resolve_callee<G: Graph>(
    scope: &Scope,
    struct_fields: &HashMap<String, HashMap<String, String>>,
    pkg_fns: &HashMap<String, HashMap<String, NodeKeys>>,
    graph: &G,
    node: Node,
    source: &[u8],
    file: &str,
) -> Option<NodeKeys> {
    match node.kind() {
        // obj.method() — evaluate receiver type, then scan for function with matching operand.
        // We don't use methods_idx + find_node_by_name_in_file here because when multiple
        // methods share the same name in one file, find_node_by_name_in_file returns the first
        // match regardless of operand. Filtering by meta["operand"] == obj_type is the only
        // unambiguous way to identify the right function.
        "field_expression" => {
            let obj_type = eval_expr_type(
                scope,
                struct_fields,
                node.child_by_field_name("value")?,
                source,
            )?;
            let method_name = node
                .child_by_field_name("field")?
                .utf8_text(source)
                .ok()?;
            graph
                .find_nodes_by_name(NodeType::Function, method_name)
                .into_iter()
                .find(|n| n.meta.get("operand").map(|s| s.as_str()) == Some(obj_type.as_str()))
                .map(|n| NodeKeys::from(&n))
        }

        // Type::fn() — associated function via scoped identifier
        "scoped_identifier" => {
            let path_node = node.child_by_field_name("path")?;
            let name_node = node.child_by_field_name("name")?;
            let fn_name = name_node.utf8_text(source).ok()?;
            let path_text = match path_node.kind() {
                "identifier" => path_node.utf8_text(source).ok()?.to_string(),
                "scoped_identifier" => path_node
                    .child_by_field_name("name")
                    .and_then(|n| n.utf8_text(source).ok())
                    .map(str::to_string)?,
                _ => return None,
            };
            // Find Function with name==fn_name and meta["operand"]==path_text
            if let Some(nd) = graph
                .find_nodes_by_name(NodeType::Function, fn_name)
                .into_iter()
                .find(|n| n.meta.get("operand").map(|s| s.as_str()) == Some(path_text.as_str()))
            {
                return Some(NodeKeys::from(&nd));
            }
            // Fallback: might be a free function in a module
            let dir = parent_dir(file);
            pkg_fns.get(&dir)?.get(fn_name).cloned()
        }

        // bare fn() — same file first, then same package directory
        "identifier" => {
            let fname = node.utf8_text(source).ok()?;
            if let Some(nd) =
                graph.find_node_by_name_in_file(NodeType::Function, fname, file)
            {
                return Some(NodeKeys::from(&nd));
            }
            let dir = parent_dir(file);
            pkg_fns.get(&dir)?.get(fname).cloned()
        }

        _ => None,
    }
}

// ── Parameter binding ──────────────────────────────────────────────────────────

fn bind_params(params: Node, source: &[u8], scope: &mut Scope, impl_type: Option<&str>) {
    for i in 0..params.named_child_count() {
        let Some(child) = params.named_child(i) else {
            continue;
        };
        match child.kind() {
            // &self, &mut self, self
            "self_parameter" => {
                if let Some(t) = impl_type {
                    scope_bind(scope, "self", t);
                }
            }
            "parameter" => {
                // pattern could be identifier or destructuring
                let Some(pat) = child.child_by_field_name("pattern") else {
                    continue;
                };
                let Some(type_node) = child.child_by_field_name("type") else {
                    continue;
                };
                let Some(type_name) = strip_rust_type(type_node, source) else {
                    continue;
                };
                if pat.kind() == "identifier" {
                    let Ok(name) = pat.utf8_text(source) else {
                        continue;
                    };
                    if name != "_" && name != "self" {
                        scope_bind(scope, name, &type_name);
                    }
                }
            }
            _ => {}
        }
    }
}

// ── AST walker ────────────────────────────────────────────────────────────────

fn walk_node<G: Graph>(
    node: Node,
    source: &[u8],
    scope: &mut Scope,
    struct_fields: &HashMap<String, HashMap<String, String>>,
    pkg_fns: &HashMap<String, HashMap<String, NodeKeys>>,
    graph: &G,
    out: &mut HashMap<(usize, usize), NodeKeys>,
    file: &str,
    impl_type: Option<&str>,
) {
    match node.kind() {
        // An impl block: extract the concrete type (ignoring the trait, if any)
        // and propagate it as impl_type to all direct children (function_items).
        "impl_item" => {
            let this_type = node
                .child_by_field_name("type")
                .and_then(|n| strip_rust_type(n, source));
            if let Some(body) = node.child_by_field_name("body") {
                for i in 0..body.named_child_count() {
                    if let Some(child) = body.named_child(i) {
                        walk_node(
                            child,
                            source,
                            scope,
                            struct_fields,
                            pkg_fns,
                            graph,
                            out,
                            file,
                            this_type.as_deref(),
                        );
                    }
                }
            }
        }

        // A function (free or inside an impl): push scope, bind params, walk body.
        "function_item" => {
            scope_push(scope);
            if let Some(params) = node.child_by_field_name("parameters") {
                bind_params(params, source, scope, impl_type);
            }
            if let Some(body) = node.child_by_field_name("body") {
                walk_node(body, source, scope, struct_fields, pkg_fns, graph, out, file, None);
            }
            scope_pop(scope);
        }

        // Closure: push scope, bind typed parameters, walk body.
        "closure_expression" => {
            scope_push(scope);
            if let Some(params) = node.child_by_field_name("parameters") {
                // Closure params use "closure_parameter" kind, which may or may not have a type.
                for i in 0..params.named_child_count() {
                    if let Some(p) = params.named_child(i) {
                        if p.kind() == "closure_parameter" {
                            if let (Some(pat), Some(ty)) = (
                                p.child_by_field_name("pattern"),
                                p.child_by_field_name("type"),
                            ) {
                                if let Some(type_name) = strip_rust_type(ty, source) {
                                    if pat.kind() == "identifier" {
                                        if let Ok(name) = pat.utf8_text(source) {
                                            scope_bind(scope, name, &type_name);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            if let Some(body) = node.child_by_field_name("body") {
                walk_node(body, source, scope, struct_fields, pkg_fns, graph, out, file, None);
            }
            scope_pop(scope);
        }

        // let x: Type = value  OR  let x = Type::new()
        "let_declaration" => {
            // Determine the type: explicit annotation wins, then constructor inference.
            let type_name = if let Some(type_node) = node.child_by_field_name("type") {
                strip_rust_type(type_node, source)
            } else {
                node.child_by_field_name("value")
                    .and_then(|v| eval_expr_type(scope, struct_fields, v, source))
            };

            // Bind the pattern identifier (simple ident only; skip tuple/struct patterns).
            if let Some(pat) = node.child_by_field_name("pattern") {
                if pat.kind() == "identifier" {
                    if let (Ok(name), Some(t)) = (pat.utf8_text(source), &type_name) {
                        if name != "_" {
                            scope_bind(scope, name, t);
                        }
                    }
                }
            }

            // Recurse into the value expression to capture nested call edges.
            if let Some(val) = node.child_by_field_name("value") {
                walk_node(val, source, scope, struct_fields, pkg_fns, graph, out, file, impl_type);
            }
        }

        // A call site: resolve the callee and record the position → NodeKeys mapping.
        "call_expression" => {
            if let Some(func_node) = node.child_by_field_name("function") {
                if let Some(target) =
                    resolve_callee(scope, struct_fields, pkg_fns, graph, func_node, source, file)
                {
                    let pos = position_of_callee(func_node);
                    out.entry(pos).or_insert(target);
                }
                // Recurse into the function node (handles chained receivers).
                walk_node(
                    func_node, source, scope, struct_fields, pkg_fns, graph, out, file, impl_type,
                );
            }
            // Recurse into arguments for nested calls.
            if let Some(args) = node.child_by_field_name("arguments") {
                walk_node(args, source, scope, struct_fields, pkg_fns, graph, out, file, impl_type);
            }
        }

        _ => {
            for i in 0..node.named_child_count() {
                if let Some(child) = node.named_child(i) {
                    walk_node(
                        child, source, scope, struct_fields, pkg_fns, graph, out, file, impl_type,
                    );
                }
            }
        }
    }
}

// ── Public entry point ────────────────────────────────────────────────────────

/// Walk a Rust source file and return a map from (row, col) of each resolved
/// call-site's name position to the target NodeKeys.
pub fn resolve_file_calls<G: Graph>(
    source: &str,
    file: &str,
    struct_fields: &HashMap<String, HashMap<String, String>>,
    pkg_fns: &HashMap<String, HashMap<String, NodeKeys>>,
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

    // Root scope starts empty — Rust doesn't have cross-file package-level var scope.
    let mut scope: Scope = vec![HashMap::new()];

    walk_node(
        tree.root_node(),
        src,
        &mut scope,
        struct_fields,
        pkg_fns,
        graph,
        &mut out,
        file,
        None,
    );
    out
}
