use crate::lang::asg::NodeKeys;
use crate::lang::graphs::{EdgeType, Graph, NodeType};
use std::collections::HashMap;
use std::path::Path;
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

fn make_parser() -> Option<Parser> {
    let mut parser = Parser::new();
    let lang: tree_sitter::Language = tree_sitter_php::LANGUAGE_PHP.into();
    parser.set_language(&lang).ok()?;
    Some(parser)
}

fn parent_dir(file: &str) -> String {
    Path::new(file)
        .parent()
        .map(|p| p.to_string_lossy().into_owned())
        .unwrap_or_default()
}

fn is_php_primitive(text: &str) -> bool {
    matches!(
        text,
        "string"
            | "int"
            | "float"
            | "bool"
            | "array"
            | "void"
            | "null"
            | "mixed"
            | "never"
            | "static"
            | "self"
            | "parent"
            | "object"
            | "callable"
            | "iterable"
            | "true"
            | "false"
    )
}

fn strip_php_type(node: Node, source: &[u8]) -> Option<String> {
    match node.kind() {
        "named_type" => node.named_child(0).and_then(|n| strip_php_type(n, source)),
        "name" => {
            let text = node.utf8_text(source).ok()?;
            if is_php_primitive(text) {
                None
            } else {
                Some(text.to_string())
            }
        }
        "qualified_name" => {
            let last = node.named_child(node.named_child_count().saturating_sub(1))?;
            strip_php_type(last, source)
        }
        "nullable_type" => node.named_child(0).and_then(|n| strip_php_type(n, source)),
        "union_type" | "intersection_type" => (0..node.named_child_count())
            .filter_map(|i| node.named_child(i))
            .find_map(|n| strip_php_type(n, source)),
        "primitive_type" => None,
        _ => None,
    }
}

// ── Class field extraction ─────────────────────────────────────────────────────

pub fn extract_class_fields(source: &str) -> HashMap<String, HashMap<String, String>> {
    let mut out = HashMap::new();
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
    src: &[u8],
    out: &mut HashMap<String, HashMap<String, String>>,
) {
    match node.kind() {
        "class_declaration" | "interface_declaration" | "trait_declaration" => {
            let Some(name_node) = node.child_by_field_name("name") else {
                recurse_fields(node, src, out);
                return;
            };
            let Ok(class_name) = name_node.utf8_text(src) else {
                recurse_fields(node, src, out);
                return;
            };
            let mut fields = HashMap::new();
            let body = node.child_by_field_name("body").or_else(|| {
                (0..node.named_child_count())
                    .filter_map(|i| node.named_child(i))
                    .find(|n| n.kind() == "declaration_list")
            });
            if let Some(body) = body {
                for i in 0..body.named_child_count() {
                    let Some(child) = body.named_child(i) else {
                        continue;
                    };
                    match child.kind() {
                        "property_declaration" => {
                            extract_property_decl_fields(child, src, &mut fields);
                        }
                        "method_declaration" => {
                            let is_ctor = child
                                .child_by_field_name("name")
                                .and_then(|n| n.utf8_text(src).ok())
                                .map(|n| n == "__construct")
                                .unwrap_or(false);
                            if is_ctor {
                                if let Some(params) = child.child_by_field_name("parameters") {
                                    extract_promoted_props(params, src, &mut fields);
                                }
                            }
                        }
                        _ => {}
                    }
                }
            }
            if !fields.is_empty() {
                out.entry(class_name.to_string()).or_default().extend(fields);
            }
        }
        _ => recurse_fields(node, src, out),
    }
}

fn extract_property_decl_fields(node: Node, src: &[u8], fields: &mut HashMap<String, String>) {
    let type_node = node.child_by_field_name("type").or_else(|| {
        (0..node.named_child_count())
            .filter_map(|i| node.named_child(i))
            .find(|n| {
                matches!(
                    n.kind(),
                    "named_type"
                        | "nullable_type"
                        | "union_type"
                        | "intersection_type"
                        | "primitive_type"
                )
            })
    });
    let Some(prop_type) = type_node.and_then(|t| strip_php_type(t, src)) else {
        return;
    };
    for i in 0..node.named_child_count() {
        let Some(elem) = node.named_child(i) else {
            continue;
        };
        if elem.kind() != "property_element" {
            continue;
        }
        let var_node = elem
            .child_by_field_name("name")
            .or_else(|| elem.named_child(0));
        if let Some(vn) = var_node {
            if let Ok(raw) = vn.utf8_text(src) {
                fields.insert(raw.trim_start_matches('$').to_string(), prop_type.clone());
            }
        }
    }
}

fn extract_promoted_props(params: Node, src: &[u8], fields: &mut HashMap<String, String>) {
    for i in 0..params.named_child_count() {
        let Some(param) = params.named_child(i) else {
            continue;
        };
        if !matches!(
            param.kind(),
            "property_promotion_parameter" | "promoted_property"
        ) {
            continue;
        }
        let type_node = param.child_by_field_name("type").or_else(|| {
            (0..param.named_child_count())
                .filter_map(|j| param.named_child(j))
                .find(|n| {
                    matches!(
                        n.kind(),
                        "named_type"
                            | "nullable_type"
                            | "union_type"
                            | "intersection_type"
                            | "primitive_type"
                    )
                })
        });
        let Some(t) = type_node.and_then(|t| strip_php_type(t, src)) else {
            continue;
        };
        let var_node = param.child_by_field_name("variable_name").or_else(|| {
            (0..param.named_child_count())
                .filter_map(|j| param.named_child(j))
                .find(|n| n.kind() == "variable_name")
        });
        if let Some(vn) = var_node {
            if let Ok(raw) = vn.utf8_text(src) {
                fields.insert(raw.trim_start_matches('$').to_string(), t);
            }
        }
    }
}

fn recurse_fields(
    node: Node,
    src: &[u8],
    out: &mut HashMap<String, HashMap<String, String>>,
) {
    for i in 0..node.named_child_count() {
        if let Some(child) = node.named_child(i) {
            walk_for_class_fields(child, src, out);
        }
    }
}

// ── Free-function return type extraction ──────────────────────────────────────

/// Extract top-level function return types from PHP source.
/// Returns { func_name → base_return_type }. Class methods are excluded.
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
        if node.kind() != "function_definition" {
            continue;
        }
        let Some(name_node) = node.child_by_field_name("name") else {
            continue;
        };
        let Ok(func_name) = name_node.utf8_text(src) else {
            continue;
        };
        let Some(ret_wrapper) = node.child_by_field_name("return_type") else {
            continue;
        };
        let type_node = ret_wrapper.named_child(0).unwrap_or(ret_wrapper);
        if let Some(ret_type) = strip_php_type(type_node, src) {
            out.entry(func_name.to_string()).or_insert(ret_type);
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
    walk_for_method_returns(tree.root_node(), source.as_bytes(), None, &mut out);
    out
}

fn walk_for_method_returns(
    node: Node,
    src: &[u8],
    current_class: Option<&str>,
    out: &mut HashMap<(String, String), String>,
) {
    match node.kind() {
        "class_declaration" | "interface_declaration" | "trait_declaration" => {
            let name = node
                .child_by_field_name("name")
                .and_then(|n| n.utf8_text(src).ok());
            let body = node.child_by_field_name("body").or_else(|| {
                (0..node.named_child_count())
                    .filter_map(|i| node.named_child(i))
                    .find(|n| n.kind() == "declaration_list")
            });
            if let (Some(class_name), Some(body)) = (name, body) {
                for i in 0..body.named_child_count() {
                    if let Some(child) = body.named_child(i) {
                        walk_for_method_returns(child, src, Some(class_name), out);
                    }
                }
            }
        }
        "method_declaration" => {
            if let Some(class_name) = current_class {
                if let Some(ret_wrapper) = node.child_by_field_name("return_type") {
                    // return_type node is `: type_hint` — the named child is the actual type
                    let type_node = ret_wrapper.named_child(0).unwrap_or(ret_wrapper);
                    if let Some(ret_type) = strip_php_type(type_node, src) {
                        if let Some(method_name) = node
                            .child_by_field_name("name")
                            .and_then(|n| n.utf8_text(src).ok())
                        {
                            out.entry((class_name.to_string(), method_name.to_string()))
                                .or_insert(ret_type);
                        }
                    }
                }
            }
        }
        _ => {
            for i in 0..node.named_child_count() {
                if let Some(child) = node.named_child(i) {
                    walk_for_method_returns(child, src, current_class, out);
                }
            }
        }
    }
}

// ── Method lookup ──────────────────────────────────────────────────────────────

fn find_method_in_class<G: Graph>(
    graph: &G,
    class_name: &str,
    method_name: &str,
) -> Option<NodeKeys> {
    // 1. Direct: Function with operand == class_name (covers classes and interfaces as Class nodes)
    if let Some(n) = graph
        .find_nodes_by_name(NodeType::Function, method_name)
        .into_iter()
        .find(|n| n.meta.get("operand").map(|s| s.as_str()) == Some(class_name))
    {
        return Some(NodeKeys::from(&n));
    }

    // 2. Trait dispatch via ParentOf edges (PHP traits implementing interfaces)
    if graph.find_nodes_by_name(NodeType::Trait, class_name).is_empty() {
        return None;
    }

    let implementing: Vec<_> = graph
        .find_nodes_with_edge_type(NodeType::Class, NodeType::Trait, EdgeType::Implements)
        .into_iter()
        .filter(|(_, trait_nd)| trait_nd.name == class_name)
        .map(|(class_nd, _)| class_nd)
        .collect();

    for class_nd in &implementing {
        if let Some(n) = graph
            .find_nodes_by_name(NodeType::Function, method_name)
            .into_iter()
            .find(|f| {
                f.meta.get("operand").map(|s| s.as_str()) == Some(class_nd.name.as_str())
            })
        {
            return Some(NodeKeys::from(&n));
        }
    }

    // 3. Fallback: method on trait itself
    if let Some(trait_nd) = graph
        .find_nodes_by_name(NodeType::Trait, class_name)
        .into_iter()
        .next()
    {
        if let Some(n) =
            graph.find_node_by_name_in_file(NodeType::Function, method_name, &trait_nd.file)
        {
            return Some(NodeKeys::from(&n));
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
    src: &[u8],
) -> Option<String> {
    match node.kind() {
        // $variable or $this — look up in scope
        "variable_name" => {
            let text = node.utf8_text(src).ok()?;
            scope_lookup(scope, text).map(str::to_string)
        }

        // $obj->property
        "member_access_expression" | "nullsafe_member_access_expression" => {
            let obj_type = eval_expr_type(
                scope,
                class_fields,
                method_returns,
                fn_returns,
                node.child_by_field_name("object")?,
                src,
            )?;
            let prop = node.child_by_field_name("name")?.utf8_text(src).ok()?;
            class_fields.get(obj_type.as_str())?.get(prop).cloned()
        }

        // $obj->method() — return type of called method
        "member_call_expression" | "nullsafe_member_call_expression" => {
            let obj_type = eval_expr_type(
                scope,
                class_fields,
                method_returns,
                fn_returns,
                node.child_by_field_name("object")?,
                src,
            )?;
            let method_name = node.child_by_field_name("name")?.utf8_text(src).ok()?;
            method_returns
                .get(&(obj_type, method_name.to_string()))
                .cloned()
        }

        // new ClassName(...)
        "object_creation_expression" => {
            let class_node = node
                .child_by_field_name("class_name")
                .or_else(|| node.named_child(0))?;
            strip_php_type(class_node, src)
        }

        // Class::method() — return type
        "scoped_call_expression" => {
            let scope_node = node.child_by_field_name("scope")?;
            let class_name = scope_node.utf8_text(src).ok()?;
            if matches!(class_name, "self" | "static" | "parent") {
                return None;
            }
            let method_name = node.child_by_field_name("name")?.utf8_text(src).ok()?;
            method_returns
                .get(&(class_name.to_string(), method_name.to_string()))
                .cloned()
        }

        // free_function() — return type via fn_returns
        "function_call_expression" => {
            let func_node = node
                .child_by_field_name("function")
                .or_else(|| node.named_child(0))?;
            if func_node.kind() == "name" {
                let func_name = func_node.utf8_text(src).ok()?;
                fn_returns.get(func_name).cloned()
            } else {
                None
            }
        }

        _ => None,
    }
}

// ── Parameter binding ──────────────────────────────────────────────────────────

fn bind_parameters(params: Node, src: &[u8], scope: &mut Scope) {
    for i in 0..params.named_child_count() {
        let Some(param) = params.named_child(i) else {
            continue;
        };
        let type_node = param.child_by_field_name("type");
        let Some(type_name) = type_node.and_then(|t| strip_php_type(t, src)) else {
            continue;
        };
        let var_node = param.child_by_field_name("variable_name").or_else(|| {
            (0..param.named_child_count())
                .filter_map(|j| param.named_child(j))
                .find(|n| n.kind() == "variable_name")
        });
        if let Some(vn) = var_node {
            if let Ok(name) = vn.utf8_text(src) {
                scope_bind(scope, name, &type_name);
            }
        }
    }
}

// ── AST walker ─────────────────────────────────────────────────────────────────

fn walk_node<G: Graph>(
    node: Node,
    src: &[u8],
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
        "class_declaration" | "interface_declaration" | "trait_declaration" => {
            let class_name = node
                .child_by_field_name("name")
                .and_then(|n| n.utf8_text(src).ok().map(str::to_string));

            scope_push(scope);
            if let Some(ref name) = class_name {
                scope_bind(scope, "$this", name);
            }
            let body = node.child_by_field_name("body").or_else(|| {
                (0..node.named_child_count())
                    .filter_map(|i| node.named_child(i))
                    .find(|n| n.kind() == "declaration_list")
            });
            if let Some(body) = body {
                for i in 0..body.named_child_count() {
                    if let Some(child) = body.named_child(i) {
                        walk_node(
                            child,
                            src,
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

        "method_declaration" | "function_definition" => {
            scope_push(scope);
            if let Some(params) = node.child_by_field_name("parameters") {
                bind_parameters(params, src, scope);
            }
            if let Some(body) = node.child_by_field_name("body") {
                walk_node(
                    body,
                    src,
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
            scope_pop(scope);
        }

        // $var = expr
        "assignment_expression" => {
            let left = node.child_by_field_name("left");
            let right = node.child_by_field_name("right");
            if let (Some(left), Some(right)) = (left, right) {
                if left.kind() == "variable_name" {
                    if let Ok(var_name) = left.utf8_text(src) {
                        if let Some(t) =
                            eval_expr_type(scope, class_fields, method_returns, fn_returns, right, src)
                        {
                            scope_bind(scope, var_name, &t);
                        }
                    }
                }
                walk_node(
                    right,
                    src,
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

        // $obj->method(...)
        "member_call_expression" | "nullsafe_member_call_expression" => {
            let obj_node = node.child_by_field_name("object");
            let name_node = node.child_by_field_name("name");
            if let (Some(obj_node), Some(name_node)) = (obj_node, name_node) {
                let obj_type =
                    eval_expr_type(scope, class_fields, method_returns, fn_returns, obj_node, src);
                let method_name = name_node.utf8_text(src).ok();
                if let (Some(obj_type), Some(method_name)) = (obj_type, method_name) {
                    let pos = {
                        let p = name_node.start_position();
                        (p.row, p.column)
                    };
                    if let Some(target) = find_method_in_class(graph, &obj_type, method_name) {
                        out.entry(pos).or_insert(target);
                    }
                }
            }
            if let Some(args) = node.child_by_field_name("arguments") {
                walk_node(
                    args,
                    src,
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

        // Class::method(...) — static calls
        "scoped_call_expression" => {
            let scope_node = node.child_by_field_name("scope");
            let name_node = node.child_by_field_name("name");
            if let (Some(scope_node), Some(name_node)) = (scope_node, name_node) {
                if let Ok(class_name) = scope_node.utf8_text(src) {
                    if !matches!(class_name, "self" | "static" | "parent") {
                        if let Ok(method_name) = name_node.utf8_text(src) {
                            let pos = {
                                let p = name_node.start_position();
                                (p.row, p.column)
                            };
                            if let Some(target) =
                                find_method_in_class(graph, class_name, method_name)
                            {
                                out.entry(pos).or_insert(target);
                            }
                        }
                    }
                }
            }
            if let Some(args) = node.child_by_field_name("arguments") {
                walk_node(
                    args,
                    src,
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

        // new ClassName(...)
        "object_creation_expression" => {
            if let Some(args) = node.child_by_field_name("arguments") {
                walk_node(
                    args,
                    src,
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

        // bare function call: func(...)
        "function_call_expression" => {
            let func_node = node
                .child_by_field_name("function")
                .or_else(|| node.named_child(0));
            if let Some(func_node) = func_node {
                if func_node.kind() == "name" {
                    if let Ok(func_name) = func_node.utf8_text(src) {
                        let dir = parent_dir(file);
                        let pos = {
                            let p = func_node.start_position();
                            (p.row, p.column)
                        };
                        if let Some(target) =
                            dir_fns.get(&dir).and_then(|m| m.get(func_name))
                        {
                            out.entry(pos).or_insert(target.clone());
                        }
                    }
                }
            }
            if let Some(args) = node.child_by_field_name("arguments") {
                walk_node(
                    args,
                    src,
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

        _ => {
            for i in 0..node.named_child_count() {
                if let Some(child) = node.named_child(i) {
                    walk_node(
                        child,
                        src,
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
}

// ── Public entry point ─────────────────────────────────────────────────────────

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
