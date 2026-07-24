use crate::lang::asg::NodeKeys;
use crate::lang::graphs::Graph;
use std::collections::HashMap;
use std::path::Path;
use tree_sitter::{Node, Parser};

use super::scope::{scope_bind, scope_lookup, scope_pop, scope_push, Scope};

fn make_parser() -> Option<Parser> {
    let mut parser = Parser::new();
    let lang: tree_sitter::Language = tree_sitter_c::LANGUAGE.into();
    parser.set_language(&lang).ok()?;
    Some(parser)
}

fn parent_dir(file: &str) -> String {
    Path::new(file)
        .parent()
        .map(|p| p.to_string_lossy().into_owned())
        .unwrap_or_default()
}

fn strip_c_type(node: Node, source: &[u8]) -> Option<String> {
    match node.kind() {
        "type_identifier" => node.utf8_text(source).ok().map(str::to_string),

        "struct_specifier" => {
            node.child_by_field_name("name")
                .and_then(|n| n.utf8_text(source).ok().map(str::to_string))
        }

        "primitive_type" | "sized_type_specifier" => None,

        _ => None,
    }
}

// ── Struct field extraction ──────────────────────────────────────────────────

pub fn extract_struct_fields(source: &str) -> HashMap<String, HashMap<String, String>> {
    let mut out: HashMap<String, HashMap<String, String>> = HashMap::new();
    let Some(mut parser) = make_parser() else {
        return out;
    };
    let Some(tree) = parser.parse(source, None) else {
        return out;
    };
    walk_for_struct_fields(tree.root_node(), source.as_bytes(), &mut out);
    out
}

fn walk_for_struct_fields(
    node: Node,
    source: &[u8],
    out: &mut HashMap<String, HashMap<String, String>>,
) {
    match node.kind() {
        "struct_specifier" => {
            if let Some(name_node) = node.child_by_field_name("name") {
                if let Ok(struct_name) = name_node.utf8_text(source) {
                    let mut fields = HashMap::new();
                    if let Some(body) = node.child_by_field_name("body") {
                        for i in 0..body.named_child_count() {
                            let Some(child) = body.named_child(i) else {
                                continue;
                            };
                            if child.kind() != "field_declaration" {
                                continue;
                            }
                            let type_node = child.child_by_field_name("type");
                            let field_type = type_node.and_then(|t| strip_c_type(t, source));
                            let Some(ref ft) = field_type else {
                                continue;
                            };
                            if let Some(declarator) = child.child_by_field_name("declarator") {
                                if let Some(name) = extract_declarator_name(declarator, source) {
                                    fields.insert(name, ft.clone());
                                }
                            }
                        }
                    }
                    if !fields.is_empty() {
                        out.entry(struct_name.to_string())
                            .or_default()
                            .extend(fields);
                    }
                }
            }
        }
        "type_definition" => {
            // typedef struct { ... } TypeName;
            for i in 0..node.named_child_count() {
                if let Some(child) = node.named_child(i) {
                    if child.kind() == "struct_specifier" {
                        let mut fields = HashMap::new();
                        if let Some(body) = child.child_by_field_name("body") {
                            for j in 0..body.named_child_count() {
                                let Some(field_decl) = body.named_child(j) else {
                                    continue;
                                };
                                if field_decl.kind() != "field_declaration" {
                                    continue;
                                }
                                let type_node = field_decl.child_by_field_name("type");
                                let field_type = type_node.and_then(|t| strip_c_type(t, source));
                                let Some(ref ft) = field_type else {
                                    continue;
                                };
                                if let Some(declarator) =
                                    field_decl.child_by_field_name("declarator")
                                {
                                    if let Some(name) =
                                        extract_declarator_name(declarator, source)
                                    {
                                        fields.insert(name, ft.clone());
                                    }
                                }
                            }
                        }
                        if !fields.is_empty() {
                            // use the typedef name (last type_identifier child of type_definition)
                            if let Some(typedef_name) = node
                                .child_by_field_name("declarator")
                                .and_then(|d| extract_declarator_name(d, source))
                            {
                                out.entry(typedef_name)
                                    .or_default()
                                    .extend(fields.clone());
                            }
                            // also store under struct name if it has one
                            if let Some(name_node) = child.child_by_field_name("name") {
                                if let Ok(name) = name_node.utf8_text(source) {
                                    out.entry(name.to_string())
                                        .or_default()
                                        .extend(fields);
                                }
                            }
                        }
                    }
                }
            }
        }
        _ => {
            for i in 0..node.named_child_count() {
                if let Some(child) = node.named_child(i) {
                    walk_for_struct_fields(child, source, out);
                }
            }
        }
    }
}

fn extract_declarator_name(node: Node, source: &[u8]) -> Option<String> {
    match node.kind() {
        "field_identifier" | "identifier" | "type_identifier" => {
            node.utf8_text(source).ok().map(str::to_string)
        }
        "pointer_declarator" => node
            .child_by_field_name("declarator")
            .and_then(|d| extract_declarator_name(d, source)),
        _ => None,
    }
}

// ── Function return type extraction ──────────────────────────────────────────

pub fn extract_fn_returns(source: &str) -> HashMap<String, String> {
    let mut out = HashMap::new();
    let Some(mut parser) = make_parser() else {
        return out;
    };
    let Some(tree) = parser.parse(source, None) else {
        return out;
    };
    walk_for_fn_returns(tree.root_node(), source.as_bytes(), &mut out);
    out
}

fn walk_for_fn_returns(node: Node, source: &[u8], out: &mut HashMap<String, String>) {
    match node.kind() {
        "function_definition" | "declaration" => {
            if let Some(ret_node) = node.child_by_field_name("type") {
                if let Some(ret_type) = strip_c_type(ret_node, source) {
                    if let Some(declarator) = node.child_by_field_name("declarator") {
                        if let Some(name) = extract_fn_decl_name(declarator, source) {
                            out.entry(name).or_insert(ret_type);
                        }
                    }
                }
            }
        }
        _ => {
            for i in 0..node.named_child_count() {
                if let Some(child) = node.named_child(i) {
                    walk_for_fn_returns(child, source, out);
                }
            }
        }
    }
}

fn extract_fn_decl_name(declarator: Node, source: &[u8]) -> Option<String> {
    match declarator.kind() {
        "function_declarator" => {
            let name_node = declarator.child_by_field_name("declarator")?;
            extract_fn_decl_name(name_node, source)
        }
        "pointer_declarator" => {
            let inner = declarator.child_by_field_name("declarator")?;
            extract_fn_decl_name(inner, source)
        }
        "identifier" => declarator.utf8_text(source).ok().map(str::to_string),
        _ => None,
    }
}

// ── Type evaluator ───────────────────────────────────────────────────────────

fn eval_expr_type(
    scope: &Scope,
    struct_fields: &HashMap<String, HashMap<String, String>>,
    fn_returns: &HashMap<String, String>,
    node: Node,
    source: &[u8],
) -> Option<String> {
    match node.kind() {
        "identifier" => {
            let name = node.utf8_text(source).ok()?;
            scope_lookup(scope, name).map(str::to_string)
        }

        "field_expression" => {
            let obj = node.child_by_field_name("argument")?;
            let obj_type = eval_expr_type(scope, struct_fields, fn_returns, obj, source)?;
            let field = node.child_by_field_name("field")?.utf8_text(source).ok()?;
            struct_fields.get(obj_type.as_str())?.get(field).cloned()
        }

        "call_expression" => {
            let func = node.child_by_field_name("function")?;
            if func.kind() == "identifier" {
                let name = func.utf8_text(source).ok()?;
                fn_returns.get(name).cloned()
            } else {
                None
            }
        }

        "cast_expression" => {
            let type_node = node.child_by_field_name("type")?;
            strip_c_type(type_node, source)
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
        let Some(type_name) = strip_c_type(type_node, source) else {
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
    struct_fields: &HashMap<String, HashMap<String, String>>,
    fn_returns: &HashMap<String, String>,
    dir_fns: &HashMap<String, HashMap<String, NodeKeys>>,
    graph: &G,
    out: &mut HashMap<(usize, usize), NodeKeys>,
    file: &str,
) {
    match node.kind() {
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
                    body, source, scope, struct_fields, fn_returns, dir_fns, graph, out, file,
                );
            }
            scope_pop(scope);
        }

        "declaration" => {
            let type_node = node.child_by_field_name("type");
            let explicit_type = type_node.and_then(|t| strip_c_type(t, source));

            if let Some(declarator) = node.child_by_field_name("declarator") {
                bind_and_walk_declarator(
                    declarator,
                    &explicit_type,
                    source,
                    scope,
                    struct_fields,
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
                            struct_fields,
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
                if func_node.kind() == "identifier" {
                    if let Ok(fn_name) = func_node.utf8_text(source) {
                        let dir = parent_dir(file);
                        let pos = {
                            let p = func_node.start_position();
                            (p.row, p.column)
                        };
                        if let Some(nk) = dir_fns
                            .get(&dir)
                            .and_then(|m| m.get(fn_name))
                        {
                            out.entry(pos).or_insert(nk.clone());
                        }
                    }
                }
            }
            if let Some(args) = node.child_by_field_name("arguments") {
                walk_node(
                    args, source, scope, struct_fields, fn_returns, dir_fns, graph, out, file,
                );
            }
        }

        _ => {
            for i in 0..node.named_child_count() {
                if let Some(child) = node.named_child(i) {
                    walk_node(
                        child, source, scope, struct_fields, fn_returns, dir_fns, graph, out, file,
                    );
                }
            }
        }
    }
}

fn bind_and_walk_declarator<G: Graph>(
    node: Node,
    explicit_type: &Option<String>,
    source: &[u8],
    scope: &mut Scope,
    struct_fields: &HashMap<String, HashMap<String, String>>,
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
                        eval_expr_type(scope, struct_fields, fn_returns, v, source)
                    })
                };
                if let (Some(name), Some(t)) = (name, bound) {
                    scope_bind(scope, &name, &t);
                }
            }

            if let Some(value) = value_node {
                walk_node(
                    value, source, scope, struct_fields, fn_returns, dir_fns, graph, out, file,
                );
            }
        }
        "pointer_declarator" => {
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

// ── Public entry point ───────────────────────────────────────────────────────

pub fn resolve_file_calls<G: Graph>(
    source: &str,
    file: &str,
    struct_fields: &HashMap<String, HashMap<String, String>>,
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
        struct_fields,
        fn_returns,
        dir_fns,
        graph,
        &mut out,
        file,
    );
    out
}
