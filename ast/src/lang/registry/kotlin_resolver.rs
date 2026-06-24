use crate::lang::asg::NodeKeys;
use crate::lang::graphs::{EdgeType, Graph, NodeType};
use crate::lang::NodeData;
use std::collections::HashMap;
use std::path::Path;
use tree_sitter::{Node, Parser};

use super::scope::{scope_bind, scope_lookup, scope_pop, scope_push, Scope};

fn make_parser() -> Option<Parser> {
    let mut parser = Parser::new();
    let lang: tree_sitter::Language = tree_sitter_kotlin_sg::LANGUAGE.into();
    parser.set_language(&lang).ok()?;
    Some(parser)
}

fn parent_dir(file: &str) -> String {
    Path::new(file)
        .parent()
        .map(|p| p.to_string_lossy().into_owned())
        .unwrap_or_default()
}

// Wrapper types whose first type argument is the dispatchable type.
const TRANSPARENT_WRAPPERS: &[&str] = &[
    "Flow",
    "StateFlow",
    "MutableStateFlow",
    "Result",
    "LiveData",
    "MutableLiveData",
];

// Primitives that carry no dispatchable class methods.
const SKIP_TYPES: &[&str] = &[
    "Unit", "Nothing", "Boolean", "Int", "Long", "Short", "Byte", "Float", "Double",
    "Char", "String", "Any",
];

// ── Type stripping ─────────────────────────────────────────────────────────────

fn strip_kotlin_type(node: Node, source: &[u8]) -> Option<String> {
    match node.kind() {
        "type_identifier" => node.utf8_text(source).ok().map(str::to_string),

        // user_type: TypeName<T>  — strip wrappers, keep outer name otherwise
        "user_type" => {
            let type_id = (0..node.named_child_count())
                .filter_map(|i| node.named_child(i))
                .find(|n| n.kind() == "type_identifier")?;
            let name = type_id.utf8_text(source).ok()?;
            if TRANSPARENT_WRAPPERS.contains(&name) {
                let type_args = (0..node.named_child_count())
                    .filter_map(|i| node.named_child(i))
                    .find(|n| n.kind() == "type_arguments")?;
                let first_arg = type_args.named_child(0)?;
                strip_kotlin_type(first_arg, source)
            } else {
                Some(name.to_string())
            }
        }

        // Type?  →  inner type
        "nullable_type" => (0..node.named_child_count())
            .filter_map(|i| node.named_child(i))
            .find(|n| matches!(n.kind(), "user_type" | "type_identifier" | "parenthesized_type"))
            .and_then(|n| strip_kotlin_type(n, source)),

        // T & Any  →  inner type
        "not_nullable_type" => (0..node.named_child_count())
            .filter_map(|i| node.named_child(i))
            .find(|n| matches!(n.kind(), "user_type" | "type_identifier" | "parenthesized_type"))
            .and_then(|n| strip_kotlin_type(n, source)),

        "parenthesized_type" => node
            .named_child(0)
            .and_then(|n| strip_kotlin_type(n, source)),

        // type_projection appears as a child of type_arguments
        "type_projection" => node
            .named_child(0)
            .and_then(|n| strip_kotlin_type(n, source)),

        _ => None,
    }
}

// ── Helpers ────────────────────────────────────────────────────────────────────

/// Return the type_identifier name of a class_declaration or object_declaration.
fn class_name_from_node(node: Node, source: &[u8]) -> Option<String> {
    (0..node.named_child_count())
        .filter_map(|i| node.named_child(i))
        .find(|n| n.kind() == "type_identifier")
        .and_then(|n| n.utf8_text(source).ok().map(str::to_string))
}

/// Collect fields declared as `val`/`var` parameters in a primary_constructor.
fn extract_fields_from_primary_ctor(ctor: Node, source: &[u8], fields: &mut HashMap<String, String>) {
    for i in 0..ctor.named_child_count() {
        let Some(param) = ctor.named_child(i) else {
            continue;
        };
        if param.kind() != "class_parameter" {
            continue;
        }
        // A class_parameter is a property only when it carries val/var.
        let has_binding = (0..param.child_count())
            .filter_map(|j| param.child(j))
            .any(|c| c.kind() == "binding_pattern_kind");
        if !has_binding {
            continue;
        }
        let Some(name_node) = (0..param.named_child_count())
            .filter_map(|j| param.named_child(j))
            .find(|n| n.kind() == "simple_identifier")
        else {
            continue;
        };
        let Ok(name) = name_node.utf8_text(source) else {
            continue;
        };
        let type_node = (0..param.named_child_count())
            .filter_map(|j| param.named_child(j))
            .find(|n| matches!(n.kind(), "user_type" | "nullable_type" | "not_nullable_type"));
        if let Some(t) = type_node.and_then(|n| strip_kotlin_type(n, source)) {
            if !SKIP_TYPES.contains(&t.as_str()) {
                fields.insert(name.to_string(), t);
            }
        }
    }
}

/// Collect fields from property_declaration nodes in a class_body.
/// Uses explicit type annotation if present, otherwise infers from constructor call.
fn extract_fields_from_class_body(body: Node, source: &[u8], fields: &mut HashMap<String, String>) {
    for i in 0..body.named_child_count() {
        let Some(child) = body.named_child(i) else {
            continue;
        };
        if child.kind() != "property_declaration" {
            continue;
        }
        let Some(var_decl) = (0..child.named_child_count())
            .filter_map(|j| child.named_child(j))
            .find(|n| n.kind() == "variable_declaration")
        else {
            continue;
        };
        let Some(name_node) = (0..var_decl.named_child_count())
            .filter_map(|j| var_decl.named_child(j))
            .find(|n| n.kind() == "simple_identifier")
        else {
            continue;
        };
        let Ok(name) = name_node.utf8_text(source) else {
            continue;
        };

        // Prefer explicit type annotation inside variable_declaration.
        let explicit = (0..var_decl.named_child_count())
            .filter_map(|j| var_decl.named_child(j))
            .find(|n| matches!(n.kind(), "user_type" | "nullable_type" | "not_nullable_type"))
            .and_then(|t| strip_kotlin_type(t, source));

        if let Some(t) = explicit {
            if !SKIP_TYPES.contains(&t.as_str()) {
                fields.insert(name.to_string(), t);
            }
            continue;
        }

        // Infer type from `= ClassName(...)` initializer.
        let inferred = (0..child.named_child_count())
            .filter_map(|j| child.named_child(j))
            .filter(|n| n.kind() == "call_expression")
            .find_map(|call| {
                let first = call.named_child(0)?;
                if first.kind() == "simple_identifier" {
                    let cname = first.utf8_text(source).ok()?;
                    if cname.chars().next().map(|c| c.is_uppercase()).unwrap_or(false) {
                        return Some(cname.to_string());
                    }
                }
                None
            });

        if let Some(t) = inferred {
            if !SKIP_TYPES.contains(&t.as_str()) {
                fields.insert(name.to_string(), t);
            }
        }
    }
}

// ── Class field extraction ─────────────────────────────────────────────────────

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
    match node.kind() {
        "class_declaration" | "object_declaration" => {
            let Some(class_name) = class_name_from_node(node, source) else {
                return;
            };
            let mut fields = HashMap::new();

            if let Some(ctor) = (0..node.named_child_count())
                .filter_map(|i| node.named_child(i))
                .find(|n| n.kind() == "primary_constructor")
            {
                extract_fields_from_primary_ctor(ctor, source, &mut fields);
            }

            if let Some(body) = (0..node.named_child_count())
                .filter_map(|i| node.named_child(i))
                .find(|n| n.kind() == "class_body")
            {
                extract_fields_from_class_body(body, source, &mut fields);
                for i in 0..body.named_child_count() {
                    if let Some(child) = body.named_child(i) {
                        walk_for_class_fields(child, source, out);
                    }
                }
            }

            if !fields.is_empty() {
                out.entry(class_name).or_default().extend(fields);
            }
        }
        _ => {
            for i in 0..node.named_child_count() {
                if let Some(child) = node.named_child(i) {
                    walk_for_class_fields(child, source, out);
                }
            }
        }
    }
}

// ── Free-function return type extraction ──────────────────────────────────────

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
        if node.kind() != "function_declaration" {
            continue;
        }
        let Some(name_node) = (0..node.named_child_count())
            .filter_map(|j| node.named_child(j))
            .find(|n| n.kind() == "simple_identifier")
        else {
            continue;
        };
        let Ok(func_name) = name_node.utf8_text(src) else {
            continue;
        };
        const SKIP_KINDS: &[&str] = &[
            "modifiers",
            "simple_identifier",
            "function_value_parameters",
            "function_body",
            "type_constraints",
            "type_parameters",
            "type_modifiers",
            "receiver_type",
        ];
        let ret_type = (0..node.named_child_count())
            .filter_map(|j| node.named_child(j))
            .find(|n| !SKIP_KINDS.contains(&n.kind()))
            .and_then(|n| strip_kotlin_type(n, src))
            .filter(|t| !SKIP_TYPES.contains(&t.as_str()));
        if let Some(rt) = ret_type {
            out.entry(func_name.to_string()).or_insert(rt);
        }
    }
    out
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
    match node.kind() {
        "class_declaration" | "object_declaration" => {
            let class_name = class_name_from_node(node, source);
            if let Some(body) = (0..node.named_child_count())
                .filter_map(|i| node.named_child(i))
                .find(|n| n.kind() == "class_body")
            {
                for i in 0..body.named_child_count() {
                    if let Some(child) = body.named_child(i) {
                        walk_for_method_returns(child, source, class_name.as_deref(), out);
                    }
                }
            }
        }
        "function_declaration" => {
            if let Some(class_name) = current_class {
                let func_name = (0..node.named_child_count())
                    .filter_map(|i| node.named_child(i))
                    .find(|n| n.kind() == "simple_identifier")
                    .and_then(|n| n.utf8_text(source).ok().map(str::to_string));

                // Return type is the first child that is not a recognised structural element.
                const SKIP_KINDS: &[&str] = &[
                    "modifiers",
                    "simple_identifier",
                    "function_value_parameters",
                    "function_body",
                    "type_constraints",
                    "type_parameters",
                    "type_modifiers",
                    "receiver_type",
                ];
                let ret_type = (0..node.named_child_count())
                    .filter_map(|i| node.named_child(i))
                    .find(|n| !SKIP_KINDS.contains(&n.kind()))
                    .and_then(|n| strip_kotlin_type(n, source))
                    .filter(|t| !SKIP_TYPES.contains(&t.as_str()));

                if let (Some(fname), Some(rt)) = (func_name, ret_type) {
                    out.entry((class_name.to_string(), fname)).or_insert(rt);
                }
            }
        }
        _ => {
            for i in 0..node.named_child_count() {
                if let Some(child) = node.named_child(i) {
                    walk_for_method_returns(child, source, current_class, out);
                }
            }
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

    // 1. Direct class lookup
    for class_nd in &graph.find_nodes_by_name(NodeType::Class, class_name) {
        if let Some(k) = find_in(class_nd) {
            return Some(k);
        }
    }

    // 2. Interface dispatch via Implements edges
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
    fn_returns: &HashMap<String, String>,
    node: Node,
    source: &[u8],
) -> Option<String> {
    match node.kind() {
        "simple_identifier" => {
            scope_lookup(scope, node.utf8_text(source).ok()?).map(str::to_string)
        }

        "this_expression" => scope_lookup(scope, "this").map(str::to_string),

        // obj.field  →  resolve obj type, look up field
        "navigation_expression" => {
            let recv = node.named_child(0)?;
            let nav_suffix = (0..node.named_child_count())
                .filter_map(|i| node.named_child(i))
                .find(|n| n.kind() == "navigation_suffix")?;
            let prop = nav_suffix
                .named_child(0)
                .filter(|n| n.kind() == "simple_identifier")
                .and_then(|n| n.utf8_text(source).ok())?;
            let recv_type = eval_expr_type(scope, class_fields, method_returns, fn_returns, recv, source)?;
            class_fields.get(&recv_type)?.get(prop).cloned()
        }

        "call_expression" => {
            let first = node.named_child(0)?;
            match first.kind() {
                // ClassName(...)  →  constructor call: type = class name
                // factoryFn()    →  free function return type from fn_returns
                "simple_identifier" => {
                    let name = first.utf8_text(source).ok()?;
                    if name.chars().next().map(|c| c.is_uppercase()).unwrap_or(false) {
                        Some(name.to_string())
                    } else {
                        fn_returns.get(name).cloned()
                    }
                }
                // obj.method(...)  →  return type of method
                "navigation_expression" => {
                    let recv = first.named_child(0)?;
                    let nav_suffix = (0..first.named_child_count())
                        .filter_map(|i| first.named_child(i))
                        .find(|n| n.kind() == "navigation_suffix")?;
                    let method_name = nav_suffix
                        .named_child(0)
                        .filter(|n| n.kind() == "simple_identifier")
                        .and_then(|n| n.utf8_text(source).ok())?;
                    let recv_type =
                        eval_expr_type(scope, class_fields, method_returns, fn_returns, recv, source)?;
                    method_returns
                        .get(&(recv_type, method_name.to_string()))
                        .cloned()
                }
                _ => None,
            }
        }

        _ => None,
    }
}

// ── Parameter binding ──────────────────────────────────────────────────────────

fn bind_parameters(params: Node, source: &[u8], scope: &mut Scope) {
    for i in 0..params.named_child_count() {
        let Some(param) = params.named_child(i) else {
            continue;
        };
        if param.kind() != "parameter" {
            continue;
        }
        let Some(name_node) = (0..param.named_child_count())
            .filter_map(|j| param.named_child(j))
            .find(|n| n.kind() == "simple_identifier")
        else {
            continue;
        };
        let Ok(name) = name_node.utf8_text(source) else {
            continue;
        };
        if name == "_" {
            continue;
        }
        let type_node = (0..param.named_child_count())
            .filter_map(|j| param.named_child(j))
            .find(|n| matches!(n.kind(), "user_type" | "nullable_type" | "not_nullable_type"));
        if let Some(t) = type_node.and_then(|n| strip_kotlin_type(n, source)) {
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
    fn_returns: &HashMap<String, String>,
    dir_fns: &HashMap<String, HashMap<String, NodeKeys>>,
    graph: &G,
    out: &mut HashMap<(usize, usize), NodeKeys>,
    file: &str,
) {
    match node.kind() {
        // Class / object entry: push scope, bind "this", seed fields.
        "class_declaration" | "object_declaration" => {
            let class_name = class_name_from_node(node, source);
            scope_push(scope);
            if let Some(ref name) = class_name {
                scope_bind(scope, "this", name);
                if let Some(fields) = class_fields.get(name.as_str()) {
                    for (field_name, field_type) in fields {
                        scope_bind(scope, field_name, field_type);
                    }
                }
            }
            if let Some(body) = (0..node.named_child_count())
                .filter_map(|i| node.named_child(i))
                .find(|n| n.kind() == "class_body")
            {
                for i in 0..body.named_child_count() {
                    if let Some(child) = body.named_child(i) {
                        walk_node(
                            child, source, scope, class_fields, method_returns,
                            fn_returns, dir_fns, graph, out, file,
                        );
                    }
                }
            }
            scope_pop(scope);
        }

        // Function / constructor: push scope, bind params, walk body.
        "function_declaration" | "anonymous_function" | "secondary_constructor" => {
            scope_push(scope);
            if let Some(params) = (0..node.named_child_count())
                .filter_map(|i| node.named_child(i))
                .find(|n| n.kind() == "function_value_parameters")
            {
                bind_parameters(params, source, scope);
            }
            if let Some(body) = (0..node.named_child_count())
                .filter_map(|i| node.named_child(i))
                .find(|n| n.kind() == "function_body")
            {
                walk_node(
                    body, source, scope, class_fields, method_returns,
                    fn_returns, dir_fns, graph, out, file,
                );
            }
            scope_pop(scope);
        }

        // Local val/var: bind name → type in scope, recurse into initializer.
        "property_declaration" => {
            if let Some(var_decl) = (0..node.named_child_count())
                .filter_map(|i| node.named_child(i))
                .find(|n| n.kind() == "variable_declaration")
            {
                let name = (0..var_decl.named_child_count())
                    .filter_map(|i| var_decl.named_child(i))
                    .find(|n| n.kind() == "simple_identifier")
                    .and_then(|n| n.utf8_text(source).ok().map(str::to_string));

                let explicit = (0..var_decl.named_child_count())
                    .filter_map(|i| var_decl.named_child(i))
                    .find(|n| {
                        matches!(n.kind(), "user_type" | "nullable_type" | "not_nullable_type")
                    })
                    .and_then(|t| strip_kotlin_type(t, source));

                // Initializer: first child that is not a structural element
                let init = (0..node.named_child_count())
                    .filter_map(|i| node.named_child(i))
                    .find(|n| {
                        !matches!(
                            n.kind(),
                            "binding_pattern_kind"
                                | "variable_declaration"
                                | "modifiers"
                                | "property_delegate"
                                | "property_getter"
                                | "property_setter"
                                | "type_constraints"
                        )
                    });

                let inferred = init.and_then(|n| {
                    eval_expr_type(scope, class_fields, method_returns, fn_returns, n, source)
                });

                let bound = explicit.or(inferred);
                if let (Some(n), Some(t)) = (&name, &bound) {
                    if !SKIP_TYPES.contains(&t.as_str()) && n != "_" {
                        scope_bind(scope, n, t);
                    }
                }

                if let Some(init_node) = init {
                    walk_node(
                        init_node, source, scope, class_fields, method_returns,
                        fn_returns, dir_fns, graph, out, file,
                    );
                }
            } else {
                recurse(node, source, scope, class_fields, method_returns, fn_returns, dir_fns, graph, out, file);
            }
        }

        // Call site: resolve, record, then recurse into all children.
        "call_expression" => {
            if let Some(first) = node.named_child(0) {
                match first.kind() {
                    // obj.method(...)
                    "navigation_expression" => {
                        let recv_node = first.named_child(0);
                        let nav_suffix = (0..first.named_child_count())
                            .filter_map(|i| first.named_child(i))
                            .find(|n| n.kind() == "navigation_suffix");

                        if let (Some(recv_node), Some(nav_suffix)) = (recv_node, nav_suffix) {
                            if let Some(method_name_node) = (0..nav_suffix.named_child_count())
                                .filter_map(|i| nav_suffix.named_child(i))
                                .find(|n| n.kind() == "simple_identifier")
                            {
                                if let Ok(method_name) = method_name_node.utf8_text(source) {
                                    if let Some(recv_type) = eval_expr_type(
                                        scope, class_fields, method_returns, fn_returns, recv_node, source,
                                    ) {
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

                    // Bare function call: PersonList(...), UserRow(...)
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
            // Recurse into all children to process nested calls (args, trailing lambdas, etc.)
            recurse(node, source, scope, class_fields, method_returns, fn_returns, dir_fns, graph, out, file);
        }

        _ => {
            recurse(node, source, scope, class_fields, method_returns, fn_returns, dir_fns, graph, out, file);
        }
    }
}

fn recurse<G: Graph>(
    node: Node,
    source: &[u8],
    scope: &mut Scope,
    class_fields: &HashMap<String, HashMap<String, String>>,
    method_returns: &HashMap<(String, String), String>,
    fn_returns: &HashMap<String, String>,
    dir_fns: &HashMap<String, HashMap<String, NodeKeys>>,
    graph: &G,
    out: &mut HashMap<(usize, usize), NodeKeys>,
    file: &str,
) {
    for i in 0..node.named_child_count() {
        if let Some(child) = node.named_child(i) {
            walk_node(
                child, source, scope, class_fields, method_returns,
                fn_returns, dir_fns, graph, out, file,
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
    fn_returns: &HashMap<String, String>,
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
        fn_returns,
        dir_fns,
        graph,
        &mut out,
        file,
    );
    out
}
