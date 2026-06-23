use crate::lang::asg::NodeKeys;
use crate::lang::graphs::{EdgeType, Graph, NodeType};
use crate::lang::NodeData;
use std::collections::HashMap;
use std::path::Path;
use tree_sitter::{Node, Parser};

use super::scope::{scope_bind, scope_lookup, scope_pop, scope_push, Scope};

fn make_parser() -> Option<Parser> {
    let mut parser = Parser::new();
    let lang: tree_sitter::Language = tree_sitter_swift::LANGUAGE.into();
    parser.set_language(&lang).ok()?;
    Some(parser)
}

fn parent_dir(file: &str) -> String {
    Path::new(file)
        .parent()
        .map(|p| p.to_string_lossy().into_owned())
        .unwrap_or_default()
}

const SKIP_TYPES: &[&str] = &[
    "Void", "Never", "Bool", "Int", "Int8", "Int16", "Int32", "Int64", "UInt", "UInt8",
    "UInt16", "UInt32", "UInt64", "Float", "Double", "String", "Character", "Any", "AnyObject",
];

// ── Type stripping ─────────────────────────────────────────────────────────────

fn strip_swift_type(node: Node, source: &[u8]) -> Option<String> {
    match node.kind() {
        "type_identifier" => node.utf8_text(source).ok().map(str::to_string),

        // user_type: TypeName<Args>  — keep outer name, discard args
        "user_type" => {
            // type_identifier is always the first named child
            let type_id = (0..node.named_child_count())
                .filter_map(|i| node.named_child(i))
                .find(|n| n.kind() == "type_identifier")?;
            Some(type_id.utf8_text(source).ok()?.to_string())
        }

        // Optional<T> / T?  →  inner type
        "optional_type" => node
            .child_by_field_name("wrapped")
            .or_else(|| node.named_child(0))
            .and_then(|n| strip_swift_type(n, source)),

        // some Protocol  /  any Protocol  →  inner
        "existential_type" | "opaque_type" => {
            node.named_child(0).and_then(|n| strip_swift_type(n, source))
        }

        // [T], (A) -> B, (A, B), etc. — not dispatchable class types
        "array_type" | "dictionary_type" | "function_type" | "tuple_type" | "metatype" => None,

        _ => None,
    }
}

// ── Class / property extraction ────────────────────────────────────────────────

/// Extract the type name from a `type_annotation` node.
fn type_from_annotation(ta: Node, source: &[u8]) -> Option<String> {
    // type_annotation named children: the type node (user_type, optional_type, …)
    // Skip any ':' anonymous token — just take the first named child.
    let type_node = ta.named_child(0)?;
    strip_swift_type(type_node, source)
}

/// Extract stored-property name from a `property_declaration` node.
fn property_name(node: Node, source: &[u8]) -> Option<String> {
    // property_declaration.name field → pattern.bound_identifier → simple_identifier
    let pat = node.child_by_field_name("name")?;
    if pat.kind() == "pattern" {
        if let Some(id) = pat
            .child_by_field_name("bound_identifier")
            .or_else(|| {
                (0..pat.named_child_count())
                    .filter_map(|i| pat.named_child(i))
                    .find(|n| n.kind() == "simple_identifier")
            })
        {
            return id.utf8_text(source).ok().map(str::to_string);
        }
    }
    // Fallback: walk value_binding_pattern → pattern → simple_identifier
    let vbp = (0..node.named_child_count())
        .filter_map(|i| node.named_child(i))
        .find(|n| n.kind() == "value_binding_pattern")?;
    let inner_pat = (0..vbp.named_child_count())
        .filter_map(|i| vbp.named_child(i))
        .find(|n| n.kind() == "pattern")?;
    (0..inner_pat.named_child_count())
        .filter_map(|i| inner_pat.named_child(i))
        .find(|n| n.kind() == "simple_identifier")
        .and_then(|n| n.utf8_text(source).ok().map(str::to_string))
}

pub fn extract_class_fields(source: &str) -> HashMap<String, HashMap<String, String>> {
    let mut out: HashMap<String, HashMap<String, String>> = HashMap::new();
    let Some(mut parser) = make_parser() else {
        return out;
    };
    let Some(tree) = parser.parse(source, None) else {
        return out;
    };
    let src = source.as_bytes();
    walk_for_class_fields(tree.root_node(), src, &mut out);
    out
}

fn walk_for_class_fields(
    node: Node,
    source: &[u8],
    out: &mut HashMap<String, HashMap<String, String>>,
) {
    // class_declaration covers class / struct / actor / extension / enum in Swift
    if node.kind() == "class_declaration" {
        let Some(name_node) = node.child_by_field_name("name") else {
            recurse_class_fields(node, source, out);
            return;
        };
        let Some(class_name) = name_node.utf8_text(source).ok().map(str::to_string) else {
            recurse_class_fields(node, source, out);
            return;
        };

        let mut fields = HashMap::new();
        if let Some(body) = node.child_by_field_name("body") {
            for i in 0..body.named_child_count() {
                let Some(child) = body.named_child(i) else {
                    continue;
                };
                match child.kind() {
                    "property_declaration" => {
                        let Some(name) = property_name(child, source) else {
                            continue;
                        };
                        let type_ann = (0..child.named_child_count())
                            .filter_map(|j| child.named_child(j))
                            .find(|n| n.kind() == "type_annotation");
                        if let Some(ta) = type_ann {
                            if let Some(t) = type_from_annotation(ta, source) {
                                if !SKIP_TYPES.contains(&t.as_str()) {
                                    fields.insert(name, t);
                                }
                            }
                        } else {
                            // No explicit annotation — try to infer from constructor call initializer
                            let init_val = child.child_by_field_name("value").or_else(|| {
                                (0..child.named_child_count())
                                    .filter_map(|j| child.named_child(j))
                                    .find(|n| n.kind() == "call_expression")
                            });
                            if let Some(init) = init_val {
                                let ctor = if init.kind() == "call_expression" {
                                    init
                                } else {
                                    (0..init.named_child_count())
                                        .filter_map(|j| init.named_child(j))
                                        .find(|n| n.kind() == "call_expression")
                                        .unwrap_or(init)
                                };
                                if ctor.kind() == "call_expression" {
                                    if let Some(first) = ctor.named_child(0) {
                                        if first.kind() == "simple_identifier" {
                                            if let Ok(cname) = first.utf8_text(source) {
                                                if cname
                                                    .chars()
                                                    .next()
                                                    .map(|c| c.is_uppercase())
                                                    .unwrap_or(false)
                                                    && !SKIP_TYPES.contains(&cname)
                                                {
                                                    fields.insert(name, cname.to_string());
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    // Recurse into nested class/struct declarations
                    "class_declaration" => {
                        walk_for_class_fields(child, source, out);
                    }
                    _ => {}
                }
            }
        }

        if !fields.is_empty() {
            out.entry(class_name).or_default().extend(fields);
        }
        return;
    }
    recurse_class_fields(node, source, out);
}

fn recurse_class_fields(
    node: Node,
    source: &[u8],
    out: &mut HashMap<String, HashMap<String, String>>,
) {
    for i in 0..node.named_child_count() {
        if let Some(child) = node.named_child(i) {
            walk_for_class_fields(child, source, out);
        }
    }
}

// ── Method return type extraction ──────────────────────────────────────────────

pub fn extract_method_return_types(source: &str) -> HashMap<(String, String), String> {
    let mut out = HashMap::new();
    let Some(mut parser) = make_parser() else {
        return out;
    };
    let Some(tree) = parser.parse(source, None) else {
        return out;
    };
    let src = source.as_bytes();
    walk_for_method_returns(tree.root_node(), src, None, &mut out);
    out
}

fn walk_for_method_returns(
    node: Node,
    source: &[u8],
    current_class: Option<&str>,
    out: &mut HashMap<(String, String), String>,
) {
    if node.kind() == "class_declaration" {
        let class_name = node
            .child_by_field_name("name")
            .and_then(|n| n.utf8_text(source).ok().map(str::to_string));
        if let Some(body) = node.child_by_field_name("body") {
            for i in 0..body.named_child_count() {
                if let Some(child) = body.named_child(i) {
                    walk_for_method_returns(child, source, class_name.as_deref(), out);
                }
            }
        }
        return;
    }

    if node.kind() == "function_declaration" {
        if let Some(class_name) = current_class {
            let func_name = node
                .child_by_field_name("name")
                .and_then(|n| n.utf8_text(source).ok().map(str::to_string));
            let ret_type = node
                .child_by_field_name("return_type")
                .and_then(|n| strip_swift_type(n, source))
                .filter(|t| !SKIP_TYPES.contains(&t.as_str()));
            if let (Some(fname), Some(rt)) = (func_name, ret_type) {
                out.entry((class_name.to_string(), fname)).or_insert(rt);
            }
        }
        return;
    }

    for i in 0..node.named_child_count() {
        if let Some(child) = node.named_child(i) {
            walk_for_method_returns(child, source, current_class, out);
        }
    }
}

// ── Method lookup ──────────────────────────────────────────────────────────────

fn find_method_in_class<G: Graph>(graph: &G, class_name: &str, method_name: &str) -> Option<NodeKeys> {
    let find_in = |class_nd: &NodeData| -> Option<NodeKeys> {
        let mut candidates: Vec<_> = graph
            .find_nodes_by_name(NodeType::Function, method_name)
            .into_iter()
            .filter(|f| f.file == class_nd.file && f.start >= class_nd.start)
            .collect();
        candidates.sort_by_key(|f| f.start);
        candidates.into_iter().next().map(|nd| NodeKeys::from(&nd))
    };

    // Direct class lookup — includes extensions (same name, same file)
    for class_nd in &graph.find_nodes_by_name(NodeType::Class, class_name) {
        if let Some(k) = find_in(class_nd) {
            return Some(k);
        }
    }

    // Protocol conformance dispatch
    if graph.find_nodes_by_name(NodeType::Trait, class_name).is_empty() {
        return None;
    }
    let implementing: Vec<_> = graph
        .find_nodes_with_edge_type(NodeType::Class, NodeType::Trait, EdgeType::Implements)
        .into_iter()
        .filter(|(_, t)| t.name == class_name)
        .map(|(c, _)| c)
        .collect();
    for class_nd in &implementing {
        if let Some(k) = find_in(class_nd) {
            return Some(k);
        }
    }

    None
}

// ── Type evaluator ─────────────────────────────────────────────────────────────

fn eval_expr_type(
    scope: &Scope,
    class_fields: &HashMap<String, HashMap<String, String>>,
    method_returns: &HashMap<(String, String), String>,
    node: Node,
    source: &[u8],
) -> Option<String> {
    match node.kind() {
        "simple_identifier" => {
            scope_lookup(scope, node.utf8_text(source).ok()?).map(str::to_string)
        }

        // `self` → current class type from scope
        "self_expression" => scope_lookup(scope, "self").map(str::to_string),

        // expr.property → resolve expr type, look up property
        "navigation_expression" => {
            let recv = node.child_by_field_name("target")?;
            let nav_suffix = node.child_by_field_name("suffix")?;
            let prop = nav_suffix
                .child_by_field_name("suffix")
                .filter(|n| n.kind() == "simple_identifier")
                .and_then(|n| n.utf8_text(source).ok())?;
            let recv_type = eval_expr_type(scope, class_fields, method_returns, recv, source)?;
            class_fields.get(&recv_type)?.get(prop).cloned()
        }

        "call_expression" => {
            let first = node.named_child(0)?;
            match first.kind() {
                // ClassName(...)  →  constructor: type = class name
                "simple_identifier" => {
                    let name = first.utf8_text(source).ok()?;
                    if name.chars().next().map(|c| c.is_uppercase()).unwrap_or(false) {
                        Some(name.to_string())
                    } else {
                        None
                    }
                }
                // obj.method(...)  →  look up return type
                "navigation_expression" => {
                    let recv = first.child_by_field_name("target")?;
                    let nav_suffix = first.child_by_field_name("suffix")?;
                    let method_name = nav_suffix
                        .child_by_field_name("suffix")
                        .filter(|n| n.kind() == "simple_identifier")
                        .and_then(|n| n.utf8_text(source).ok())?;
                    let recv_type =
                        eval_expr_type(scope, class_fields, method_returns, recv, source)?;
                    method_returns
                        .get(&(recv_type, method_name.to_string()))
                        .cloned()
                }
                _ => None,
            }
        }

        // try/await pass through to inner expression
        "try_expression" | "await_expression" => node
            .child_by_field_name("expr")
            .or_else(|| (0..node.named_child_count()).filter_map(|i| node.named_child(i)).find(|n| !matches!(n.kind(), "try_operator" | "await")))
            .and_then(|n| eval_expr_type(scope, class_fields, method_returns, n, source)),

        _ => None,
    }
}

// ── Parameter binding ──────────────────────────────────────────────────────────

/// Bind function parameters into scope. Swift uses `parameter` nodes with `name`/`type` fields.
fn bind_parameters(node: Node, source: &[u8], scope: &mut Scope) {
    for i in 0..node.named_child_count() {
        let Some(child) = node.named_child(i) else {
            continue;
        };
        if child.kind() != "parameter" {
            continue;
        }
        // Internal name: field "name" (the second label, used inside the body)
        let Some(name_node) = child.child_by_field_name("name") else {
            continue;
        };
        let name = match name_node.utf8_text(source) {
            Ok(n) if n != "_" => n,
            _ => continue,
        };
        // Type: field "type"
        let type_node = child.child_by_field_name("type");
        if let Some(t) = type_node.and_then(|n| strip_swift_type(n, source)) {
            if !SKIP_TYPES.contains(&t.as_str()) {
                scope_bind(scope, name, &t);
            }
        }
    }
}

// ── AST walker ────────────────────────────────────────────────────────────────

fn walk_node<G: Graph>(
    node: Node,
    source: &[u8],
    scope: &mut Scope,
    class_fields: &HashMap<String, HashMap<String, String>>,
    method_returns: &HashMap<(String, String), String>,
    dir_fns: &HashMap<String, HashMap<String, NodeKeys>>,
    graph: &G,
    out: &mut HashMap<(usize, usize), NodeKeys>,
    file: &str,
) {
    match node.kind() {
        // class / struct / actor / extension / enum all share class_declaration
        "class_declaration" => {
            let class_name = node
                .child_by_field_name("name")
                .and_then(|n| n.utf8_text(source).ok().map(str::to_string));
            scope_push(scope);
            if let Some(ref name) = class_name {
                scope_bind(scope, "self", name);
                if let Some(fields) = class_fields.get(name.as_str()) {
                    for (field_name, field_type) in fields {
                        scope_bind(scope, field_name, field_type);
                    }
                }
            }
            if let Some(body) = node.child_by_field_name("body") {
                for i in 0..body.named_child_count() {
                    if let Some(child) = body.named_child(i) {
                        walk_node(
                            child, source, scope, class_fields, method_returns,
                            dir_fns, graph, out, file,
                        );
                    }
                }
            }
            scope_pop(scope);
        }

        "function_declaration" | "init_declaration" => {
            scope_push(scope);
            // Bind parameters: function_declaration has `parameter` children
            bind_parameters(node, source, scope);
            if let Some(body) = node.child_by_field_name("body") {
                walk_node(
                    body, source, scope, class_fields, method_returns,
                    dir_fns, graph, out, file,
                );
            }
            scope_pop(scope);
        }

        // Closures / anonymous functions
        "lambda_literal" => {
            scope_push(scope);
            recurse(
                node, source, scope, class_fields, method_returns,
                dir_fns, graph, out, file,
            );
            scope_pop(scope);
        }

        // Local let/var binding: bind name → type, recurse into initializer
        "property_declaration" => {
            let name = property_name(node, source);
            let type_ann = (0..node.named_child_count())
                .filter_map(|i| node.named_child(i))
                .find(|n| n.kind() == "type_annotation")
                .and_then(|ta| type_from_annotation(ta, source));

            let init_node = node.child_by_field_name("value").or_else(|| {
                (0..node.named_child_count())
                    .filter_map(|i| node.named_child(i))
                    .find(|n| {
                        !matches!(
                            n.kind(),
                            "value_binding_pattern"
                                | "modifiers"
                                | "type_annotation"
                                | "computed_property"
                                | "willset_didset_block"
                        )
                    })
            });

            let inferred = init_node.and_then(|n| {
                eval_expr_type(scope, class_fields, method_returns, n, source)
            });

            let bound = type_ann.or(inferred);
            if let (Some(n), Some(t)) = (&name, &bound) {
                if !SKIP_TYPES.contains(&t.as_str()) && n != "_" {
                    scope_bind(scope, n, t);
                }
            }

            if let Some(init) = init_node {
                walk_node(
                    init, source, scope, class_fields, method_returns,
                    dir_fns, graph, out, file,
                );
            }
        }

        // Call site: resolve then recurse into all children
        "call_expression" => {
            if let Some(first) = node.named_child(0) {
                match first.kind() {
                    // obj.method(...) or self.method(...)
                    "navigation_expression" => {
                        let recv_node = first.child_by_field_name("target");
                        let nav_suffix = first.child_by_field_name("suffix");

                        if let (Some(recv_node), Some(nav_suffix)) = (recv_node, nav_suffix) {
                            let method_name_node = nav_suffix
                                .child_by_field_name("suffix")
                                .filter(|n| n.kind() == "simple_identifier");
                            if let Some(method_name_node) = method_name_node {
                                if let Ok(method_name) = method_name_node.utf8_text(source) {
                                    let recv_type = eval_expr_type(
                                        scope, class_fields, method_returns, recv_node, source,
                                    );
                                    if let Some(recv_type) = recv_type {
                                        if let Some(target) =
                                            find_method_in_class(graph, &recv_type, method_name)
                                        {
                                            let p = method_name_node.start_position();
                                            out.entry((p.row, p.column)).or_insert(target);
                                        }
                                    }
                                }
                            }
                        }
                    }

                    // Bare function call: same file first, then same dir
                    "simple_identifier" => {
                        if let Ok(fname) = first.utf8_text(source) {
                            let target = graph
                                .find_node_by_name_in_file(NodeType::Function, fname, file)
                                .map(|n| NodeKeys::from(&n))
                                .or_else(|| {
                                    let dir = parent_dir(file);
                                    dir_fns.get(&dir)?.get(fname).cloned()
                                });
                            if let Some(target) = target {
                                let p = first.start_position();
                                out.entry((p.row, p.column)).or_insert(target);
                            }
                        }
                    }

                    _ => {}
                }
            }
            // Recurse into all children (args, trailing closures, etc.)
            recurse(
                node, source, scope, class_fields, method_returns,
                dir_fns, graph, out, file,
            );
        }

        _ => {
            recurse(
                node, source, scope, class_fields, method_returns,
                dir_fns, graph, out, file,
            );
        }
    }
}

fn recurse<G: Graph>(
    node: Node,
    source: &[u8],
    scope: &mut Scope,
    class_fields: &HashMap<String, HashMap<String, String>>,
    method_returns: &HashMap<(String, String), String>,
    dir_fns: &HashMap<String, HashMap<String, NodeKeys>>,
    graph: &G,
    out: &mut HashMap<(usize, usize), NodeKeys>,
    file: &str,
) {
    for i in 0..node.named_child_count() {
        if let Some(child) = node.named_child(i) {
            walk_node(
                child, source, scope, class_fields, method_returns,
                dir_fns, graph, out, file,
            );
        }
    }
}

// ── Public entry point ────────────────────────────────────────────────────────

pub fn resolve_file_calls<G: Graph>(
    source: &str,
    file: &str,
    class_fields: &HashMap<String, HashMap<String, String>>,
    method_returns: &HashMap<(String, String), String>,
    dir_fns: &HashMap<String, HashMap<String, NodeKeys>>,
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
    walk_node(
        tree.root_node(),
        src,
        &mut scope,
        class_fields,
        method_returns,
        dir_fns,
        graph,
        &mut out,
        file,
    );
    out
}
