use crate::lang::asg::NodeKeys;
use crate::lang::graphs::{Graph, NodeType};
use std::collections::HashMap;
use std::path::Path;
use tree_sitter::{Node, Parser};

use super::scope::{scope_bind, scope_lookup, scope_pop, scope_push, Scope};

fn make_parser() -> Option<Parser> {
    let mut parser = Parser::new();
    let lang: tree_sitter::Language = tree_sitter_cpp::LANGUAGE.into();
    parser.set_language(&lang).ok()?;
    Some(parser)
}

fn parent_dir(file: &str) -> String {
    Path::new(file)
        .parent()
        .map(|p| p.to_string_lossy().into_owned())
        .unwrap_or_default()
}

fn strip_cpp_type(node: Node, source: &[u8]) -> Option<String> {
    match node.kind() {
        "type_identifier" => node.utf8_text(source).ok().map(str::to_string),

        "qualified_identifier" | "scoped_type_identifier" => {
            let name = node.child_by_field_name("name")?;
            strip_cpp_type(name, source)
        }

        "template_type" => {
            let name = node.child_by_field_name("name")?;
            let name_text = name.utf8_text(source).ok()?;
            if matches!(
                name_text,
                "unique_ptr" | "shared_ptr" | "optional" | "vector" | "future"
            ) {
                let args = node.child_by_field_name("arguments")?;
                let first = args.named_child(0)?;
                strip_cpp_type(first, source)
            } else {
                Some(name_text.to_string())
            }
        }

        "primitive_type" | "sized_type_specifier" | "auto" => None,

        _ => None,
    }
}

// ── Class field extraction ───────────────────────────────────────────────────

pub fn extract_class_fields(source: &str) -> HashMap<String, HashMap<String, String>> {
    let mut out: HashMap<String, HashMap<String, String>> = HashMap::new();
    let Some(mut parser) = make_parser() else {
        return out;
    };
    let Some(tree) = parser.parse(source, None) else {
        return out;
    };
    walk_for_class_fields(tree.root_node(), source.as_bytes(), &mut out);
    out
}

fn walk_for_class_fields(
    node: Node,
    source: &[u8],
    out: &mut HashMap<String, HashMap<String, String>>,
) {
    match node.kind() {
        "class_specifier" | "struct_specifier" => {
            if let Some(name_node) = node.child_by_field_name("name") {
                if let Ok(class_name) = name_node.utf8_text(source) {
                    let mut fields = HashMap::new();
                    if let Some(body) = node.child_by_field_name("body") {
                        extract_fields_from_body(body, source, &mut fields);
                        for i in 0..body.named_child_count() {
                            if let Some(child) = body.named_child(i) {
                                walk_for_class_fields(child, source, out);
                            }
                        }
                    }
                    if !fields.is_empty() {
                        out.entry(class_name.to_string())
                            .or_default()
                            .extend(fields);
                    }
                }
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

fn extract_fields_from_body(body: Node, source: &[u8], fields: &mut HashMap<String, String>) {
    for i in 0..body.named_child_count() {
        let Some(child) = body.named_child(i) else {
            continue;
        };
        if child.kind() != "field_declaration" {
            continue;
        }
        let type_node = child.child_by_field_name("type");
        let field_type = type_node.and_then(|t| strip_cpp_type(t, source));
        let Some(ref ft) = field_type else {
            continue;
        };
        if let Some(declarator) = child.child_by_field_name("declarator") {
            if let Some(name) = extract_declarator_name(declarator, source) {
                fields.insert(name, ft.clone());
            }
        }
        for j in 0..child.named_child_count() {
            if let Some(decl) = child.named_child(j) {
                if decl.kind() == "field_identifier" || decl.kind() == "identifier" {
                    if let Ok(name) = decl.utf8_text(source) {
                        fields.entry(name.to_string()).or_insert(ft.clone());
                    }
                }
            }
        }
    }
}

fn extract_declarator_name(node: Node, source: &[u8]) -> Option<String> {
    match node.kind() {
        "field_identifier" | "identifier" => node.utf8_text(source).ok().map(str::to_string),
        "pointer_declarator" | "reference_declarator" => {
            node.child_by_field_name("declarator")
                .and_then(|d| extract_declarator_name(d, source))
        }
        _ => None,
    }
}

// ── Method return type extraction ────────────────────────────────────────────

pub fn extract_method_return_types(source: &str) -> HashMap<(String, String), String> {
    let mut out = HashMap::new();
    let Some(mut parser) = make_parser() else {
        return out;
    };
    let Some(tree) = parser.parse(source, None) else {
        return out;
    };
    walk_for_method_returns(tree.root_node(), source.as_bytes(), None, &mut out);
    out
}

fn walk_for_method_returns(
    node: Node,
    source: &[u8],
    current_class: Option<&str>,
    out: &mut HashMap<(String, String), String>,
) {
    match node.kind() {
        "class_specifier" | "struct_specifier" => {
            if let Some(name_node) = node.child_by_field_name("name") {
                if let Ok(class_name) = name_node.utf8_text(source) {
                    if let Some(body) = node.child_by_field_name("body") {
                        for i in 0..body.named_child_count() {
                            if let Some(child) = body.named_child(i) {
                                walk_for_method_returns(child, source, Some(class_name), out);
                            }
                        }
                    }
                }
            }
        }
        "function_definition" | "declaration" => {
            if let Some(ret_node) = node.child_by_field_name("type") {
                if let Some(ret_type) = strip_cpp_type(ret_node, source) {
                    if let Some(declarator) = node.child_by_field_name("declarator") {
                        if let Some((class, method)) =
                            extract_qualified_method_name(declarator, source)
                        {
                            out.entry((class, method)).or_insert(ret_type.clone());
                        } else if let Some(class_name) = current_class {
                            if let Some(name) = extract_simple_fn_name(declarator, source) {
                                out.entry((class_name.to_string(), name))
                                    .or_insert(ret_type);
                            }
                        }
                    }
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

fn extract_qualified_method_name(declarator: Node, source: &[u8]) -> Option<(String, String)> {
    match declarator.kind() {
        "function_declarator" => {
            let name_node = declarator.child_by_field_name("declarator")?;
            extract_qualified_method_name(name_node, source)
        }
        "qualified_identifier" => {
            let scope_node = declarator.child_by_field_name("scope")?;
            let name_node = declarator.child_by_field_name("name")?;
            let class = scope_node.utf8_text(source).ok()?;
            let method = name_node.utf8_text(source).ok()?;
            if method.starts_with('~') {
                return None;
            }
            Some((class.to_string(), method.to_string()))
        }
        _ => None,
    }
}

fn extract_simple_fn_name(declarator: Node, source: &[u8]) -> Option<String> {
    match declarator.kind() {
        "function_declarator" => {
            let name_node = declarator.child_by_field_name("declarator")?;
            extract_simple_fn_name(name_node, source)
        }
        "identifier" | "field_identifier" => declarator.utf8_text(source).ok().map(str::to_string),
        _ => None,
    }
}

// ── Free function return type extraction ─────────────────────────────────────

pub fn extract_fn_returns(source: &str) -> HashMap<String, String> {
    let mut out = HashMap::new();
    let Some(mut parser) = make_parser() else {
        return out;
    };
    let Some(tree) = parser.parse(source, None) else {
        return out;
    };
    walk_for_fn_returns(tree.root_node(), source.as_bytes(), false, &mut out);
    out
}

fn walk_for_fn_returns(node: Node, source: &[u8], in_class: bool, out: &mut HashMap<String, String>) {
    match node.kind() {
        "class_specifier" | "struct_specifier" => {
            if let Some(body) = node.child_by_field_name("body") {
                for i in 0..body.named_child_count() {
                    if let Some(child) = body.named_child(i) {
                        walk_for_fn_returns(child, source, true, out);
                    }
                }
            }
        }
        "function_definition" | "declaration" if !in_class => {
            if let Some(ret_node) = node.child_by_field_name("type") {
                if let Some(ret_type) = strip_cpp_type(ret_node, source) {
                    if let Some(declarator) = node.child_by_field_name("declarator") {
                        if extract_qualified_method_name(declarator, source).is_none() {
                            if let Some(name) = extract_simple_fn_name(declarator, source) {
                                out.entry(name).or_insert(ret_type);
                            }
                        }
                    }
                }
            }
        }
        _ => {
            for i in 0..node.named_child_count() {
                if let Some(child) = node.named_child(i) {
                    walk_for_fn_returns(child, source, in_class, out);
                }
            }
        }
    }
}

// ── Method lookup ────────────────────────────────────────────────────────────

fn find_method_in_class<G: Graph>(
    graph: &G,
    class_name: &str,
    method_name: &str,
) -> Option<NodeKeys> {
    if let Some(n) = graph
        .find_nodes_by_name(NodeType::Function, method_name)
        .into_iter()
        .find(|n| n.meta.get("operand").map(|s| s.as_str()) == Some(class_name))
    {
        return Some(NodeKeys::from(&n));
    }
    None
}

// ── Type evaluator ───────────────────────────────────────────────────────────

fn eval_expr_type(
    scope: &Scope,
    class_fields: &HashMap<String, HashMap<String, String>>,
    method_returns: &HashMap<(String, String), String>,
    fn_returns: &HashMap<String, String>,
    node: Node,
    source: &[u8],
) -> Option<String> {
    match node.kind() {
        "identifier" => {
            let name = node.utf8_text(source).ok()?;
            scope_lookup(scope, name).map(str::to_string).or_else(|| {
                if name.chars().next().map(|c| c.is_uppercase()).unwrap_or(false) {
                    Some(name.to_string())
                } else {
                    None
                }
            })
        }

        "this" => scope_lookup(scope, "this").map(str::to_string),

        "field_expression" => {
            let obj = node.child_by_field_name("argument")?;
            let obj_type =
                eval_expr_type(scope, class_fields, method_returns, fn_returns, obj, source)?;
            let field = node.child_by_field_name("field")?.utf8_text(source).ok()?;
            class_fields
                .get(obj_type.as_str())?
                .get(field)
                .cloned()
        }

        "new_expression" => {
            let type_node = node.child_by_field_name("type")?;
            strip_cpp_type(type_node, source)
        }

        "call_expression" => {
            let func = node.child_by_field_name("function")?;
            if func.kind() == "field_expression" {
                let obj = func.child_by_field_name("argument")?;
                let obj_type = eval_expr_type(
                    scope,
                    class_fields,
                    method_returns,
                    fn_returns,
                    obj,
                    source,
                )?;
                let method = func.child_by_field_name("field")?.utf8_text(source).ok()?;
                method_returns
                    .get(&(obj_type, method.to_string()))
                    .cloned()
            } else if func.kind() == "identifier" {
                let name = func.utf8_text(source).ok()?;
                fn_returns.get(name).cloned()
            } else if func.kind() == "qualified_identifier" {
                let name = func.child_by_field_name("name")?;
                let name_text = name.utf8_text(source).ok()?;
                let scope_node = func.child_by_field_name("scope")?;
                let class_text = scope_node.utf8_text(source).ok()?;
                method_returns
                    .get(&(class_text.to_string(), name_text.to_string()))
                    .cloned()
            } else {
                None
            }
        }

        _ => None,
    }
}

// ── Parameter binding ────────────────────────────────────────────────────────

fn bind_parameters(params: Node, source: &[u8], scope: &mut Scope) {
    for i in 0..params.named_child_count() {
        let Some(param) = params.named_child(i) else {
            continue;
        };
        if param.kind() != "parameter_declaration" {
            continue;
        }
        let Some(type_node) = param.child_by_field_name("type") else {
            continue;
        };
        let Some(type_name) = strip_cpp_type(type_node, source) else {
            continue;
        };
        if let Some(decl) = param.child_by_field_name("declarator") {
            if let Some(name) = extract_declarator_name(decl, source) {
                scope_bind(scope, &name, &type_name);
            }
        }
    }
}

// ── AST walker ───────────────────────────────────────────────────────────────

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
        "class_specifier" | "struct_specifier" => {
            let class_name = node
                .child_by_field_name("name")
                .and_then(|n| n.utf8_text(source).ok().map(str::to_string));

            scope_push(scope);
            if let Some(ref name) = class_name {
                scope_bind(scope, "this", name);
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
                            child,
                            source,
                            scope,
                            class_fields,
                            method_returns,
                            fn_returns,
                            dir_fns,
                            graph,
                            out,
                            file,
                        );
                    }
                }
            }
            scope_pop(scope);
        }

        "function_definition" => {
            scope_push(scope);
            if let Some(declarator) = node.child_by_field_name("declarator") {
                if declarator.kind() == "function_declarator" {
                    if let Some(params) = declarator.child_by_field_name("parameters") {
                        bind_parameters(params, source, scope);
                    }
                }
            }
            if let Some(body) = node.child_by_field_name("body") {
                walk_node(
                    body, source, scope, class_fields, method_returns, fn_returns, dir_fns, graph,
                    out, file,
                );
            }
            scope_pop(scope);
        }

        "declaration" => {
            let type_node = node.child_by_field_name("type");
            let explicit_type = type_node.and_then(|t| strip_cpp_type(t, source));

            if let Some(declarator) = node.child_by_field_name("declarator") {
                bind_and_walk_declarator(
                    declarator,
                    &explicit_type,
                    source,
                    scope,
                    class_fields,
                    method_returns,
                    fn_returns,
                    dir_fns,
                    graph,
                    out,
                    file,
                );
            }
            for i in 0..node.named_child_count() {
                if let Some(child) = node.named_child(i) {
                    if child.kind() == "init_declarator" {
                        bind_and_walk_declarator(
                            child,
                            &explicit_type,
                            source,
                            scope,
                            class_fields,
                            method_returns,
                            fn_returns,
                            dir_fns,
                            graph,
                            out,
                            file,
                        );
                    }
                }
            }
        }

        "call_expression" => {
            if let Some(func_node) = node.child_by_field_name("function") {
                let resolved = if func_node.kind() == "field_expression" {
                    let obj_node = func_node.child_by_field_name("argument");
                    let name_node = func_node.child_by_field_name("field");
                    if let (Some(obj_node), Some(name_node)) = (obj_node, name_node) {
                        let obj_type = eval_expr_type(
                            scope,
                            class_fields,
                            method_returns,
                            fn_returns,
                            obj_node,
                            source,
                        );
                        let method_name = name_node.utf8_text(source).ok();
                        if let (Some(obj_type), Some(method_name)) = (obj_type, method_name) {
                            let pos = {
                                let p = name_node.start_position();
                                (p.row, p.column)
                            };
                            find_method_in_class(graph, &obj_type, method_name)
                                .map(|t| (pos, t))
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                } else if func_node.kind() == "identifier" {
                    if let Ok(fn_name) = func_node.utf8_text(source) {
                        let dir = parent_dir(file);
                        let pos = {
                            let p = func_node.start_position();
                            (p.row, p.column)
                        };
                        dir_fns
                            .get(&dir)
                            .and_then(|m| m.get(fn_name))
                            .map(|nk| (pos, nk.clone()))
                    } else {
                        None
                    }
                } else if func_node.kind() == "qualified_identifier" {
                    let name_node = func_node.child_by_field_name("name");
                    let scope_node = func_node.child_by_field_name("scope");
                    if let (Some(scope_node), Some(name_node)) = (scope_node, name_node) {
                        let class = scope_node.utf8_text(source).ok();
                        let method = name_node.utf8_text(source).ok();
                        if let (Some(class), Some(method)) = (class, method) {
                            let pos = {
                                let p = name_node.start_position();
                                (p.row, p.column)
                            };
                            find_method_in_class(graph, class, method).map(|t| (pos, t))
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                } else {
                    None
                };

                if let Some((pos, target)) = resolved {
                    out.entry(pos).or_insert(target);
                }
            }
            if let Some(args) = node.child_by_field_name("arguments") {
                walk_node(
                    args, source, scope, class_fields, method_returns, fn_returns, dir_fns, graph,
                    out, file,
                );
            }
        }

        _ => {
            recurse(
                node, source, scope, class_fields, method_returns, fn_returns, dir_fns, graph, out,
                file,
            );
        }
    }
}

fn bind_and_walk_declarator<G: Graph>(
    node: Node,
    explicit_type: &Option<String>,
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
        "init_declarator" => {
            let decl_node = node.child_by_field_name("declarator");
            let value_node = node.child_by_field_name("value");

            if let Some(decl) = decl_node {
                let name = extract_declarator_name(decl, source);
                let bound = if let Some(ref t) = explicit_type {
                    Some(t.clone())
                } else {
                    value_node.and_then(|v| {
                        eval_expr_type(scope, class_fields, method_returns, fn_returns, v, source)
                    })
                };
                if let (Some(name), Some(t)) = (name, bound) {
                    scope_bind(scope, &name, &t);
                }
            }

            if let Some(value) = value_node {
                walk_node(
                    value, source, scope, class_fields, method_returns, fn_returns, dir_fns, graph,
                    out, file,
                );
            }
        }
        "pointer_declarator" | "reference_declarator" => {
            if let Some(inner) = node.child_by_field_name("declarator") {
                let name = extract_declarator_name(inner, source);
                if let (Some(name), Some(ref t)) = (name, explicit_type) {
                    scope_bind(scope, &name, t);
                }
            }
        }
        "identifier" | "field_identifier" => {
            if let (Ok(name), Some(ref t)) = (node.utf8_text(source), explicit_type) {
                scope_bind(scope, name, t);
            }
        }
        _ => {}
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
                child, source, scope, class_fields, method_returns, fn_returns, dir_fns, graph,
                out, file,
            );
        }
    }
}

// ── Public entry point ───────────────────────────────────────────────────────

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
    let mut scope: Scope = vec![HashMap::new()];

    walk_node(
        tree.root_node(),
        source.as_bytes(),
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
