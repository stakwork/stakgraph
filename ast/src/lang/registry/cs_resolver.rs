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
    let lang: tree_sitter::Language = tree_sitter_c_sharp::LANGUAGE.into();
    parser.set_language(&lang).ok()?;
    Some(parser)
}

fn parent_dir(file: &str) -> String {
    Path::new(file)
        .parent()
        .map(|p| p.to_string_lossy().into_owned())
        .unwrap_or_default()
}

// ── Type stripping ─────────────────────────────────────────────────────────────

/// Strip wrappers and return the base class name suitable for method dispatch.
/// Returns None for primitives, void, and types that carry no dispatchable methods.
fn strip_cs_type(node: Node, source: &[u8]) -> Option<String> {
    match node.kind() {
        "identifier" => node.utf8_text(source).ok().map(str::to_string),

        // Task<T> / ValueTask<T> — unwrap to T; other generics keep outer name.
        "generic_name" => {
            let name_node = node.named_child(0)?;
            let name = name_node.utf8_text(source).ok()?;
            if matches!(name, "Task" | "ValueTask") {
                // named_child(1) is the type_argument_list; its named_child(0) is the type.
                let type_args = node.named_child(1)?;
                let first_arg = type_args.named_child(0)?;
                strip_cs_type(first_arg, source)
            } else {
                Some(name.to_string())
            }
        }

        // Product? → Product
        "nullable_type" => {
            let inner = node.child_by_field_name("type")?;
            strip_cs_type(inner, source)
        }

        // Namespace.ClassName → take the last segment
        "qualified_name" => {
            let name_node = node.child_by_field_name("name")?;
            strip_cs_type(name_node, source)
        }

        // void, int, string, bool, etc. — not dispatchable
        "predefined_type" => None,

        // T[] — not useful for class dispatch
        "array_type" => None,

        _ => None,
    }
}

// ── Class field extraction ─────────────────────────────────────────────────────

/// Extract `class_name → { field_name → base_type }` from a C# source file.
/// Covers both class field declarations and interface member declarations.
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

fn walk_for_class_fields(node: Node, source: &[u8], out: &mut HashMap<String, HashMap<String, String>>) {
    match node.kind() {
        "class_declaration" | "interface_declaration" | "record_declaration" => {
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
                        out.entry(class_name.to_string()).or_default().extend(fields);
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
        // field_declaration children: [modifier*, variable_declaration]
        let Some(var_decl) = (0..child.named_child_count())
            .filter_map(|j| child.named_child(j))
            .find(|n| n.kind() == "variable_declaration")
        else {
            continue;
        };
        let Some(type_node) = var_decl.child_by_field_name("type") else {
            continue;
        };
        let Some(field_type) = strip_cs_type(type_node, source) else {
            continue;
        };
        for j in 0..var_decl.named_child_count() {
            let Some(decl) = var_decl.named_child(j) else {
                continue;
            };
            if decl.kind() != "variable_declarator" {
                continue;
            }
            if let Some(name_node) = decl.child_by_field_name("name") {
                if let Ok(name) = name_node.utf8_text(source) {
                    fields.insert(name.to_string(), field_type.clone());
                }
            }
        }
    }
}

// ── Method return type extraction ──────────────────────────────────────────────

/// Extract `(class_name, method_name) → return_type` for every non-void method
/// in both class and interface declarations.
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
        "class_declaration" | "interface_declaration" | "record_declaration" => {
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
        // C# uses `returns` field for the return type (unlike Java which uses `type`)
        "method_declaration" => {
            if let Some(class_name) = current_class {
                if let Some(ret_node) = node.child_by_field_name("returns") {
                    if let Some(ret_type) = strip_cs_type(ret_node, source) {
                        if let Some(name_node) = node.child_by_field_name("name") {
                            if let Ok(method_name) = name_node.utf8_text(source) {
                                out.entry((class_name.to_string(), method_name.to_string()))
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

// ── Method lookup ──────────────────────────────────────────────────────────────

/// Find the Function node for `method_name` on `class_name`.
///
/// C# sets `operand` metadata on class methods (via get_operand in csharp.rs query),
/// so we can use operand-based lookup — the same approach as ts_resolver.
///
/// Interface dispatch: when class_name is a Trait (C# interface), we traverse
/// Implements edges to find concrete implementing classes.
fn find_method_in_class<G: Graph>(graph: &G, class_name: &str, method_name: &str) -> Option<NodeKeys> {
    // 1. Strict: Function with operand == class_name
    if let Some(n) = graph
        .find_nodes_by_name(NodeType::Function, method_name)
        .into_iter()
        .find(|n| n.meta.get("operand").map(|s| s.as_str()) == Some(class_name))
    {
        return Some(NodeKeys::from(&n));
    }

    // 2. Interface dispatch: class_name is a known Trait → find implementing classes.
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
            .find(|f| f.meta.get("operand").map(|s| s.as_str()) == Some(class_nd.name.as_str()))
        {
            return Some(NodeKeys::from(&n));
        }
    }

    // 3. Fallback: interface method itself (for interfaces with no implementation in the graph)
    if let Some(trait_nd) = graph.find_nodes_by_name(NodeType::Trait, class_name).into_iter().next() {
        if let Some(n) = graph.find_node_by_name_in_file(NodeType::Function, method_name, &trait_nd.file) {
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
    node: Node,
    source: &[u8],
) -> Option<String> {
    match node.kind() {
        // Bare identifier: local var, parameter, or field seeded from class scope
        "identifier" => scope_lookup(scope, node.utf8_text(source).ok()?).map(str::to_string),

        // `this` keyword → current class type from scope
        "this" => scope_lookup(scope, "this").map(str::to_string),

        // obj.Field or obj.Property
        "member_access_expression" => {
            let obj_type = eval_expr_type(
                scope,
                class_fields,
                method_returns,
                node.child_by_field_name("expression")?,
                source,
            )?;
            let prop = node.child_by_field_name("name")?.utf8_text(source).ok()?;
            class_fields.get(obj_type.as_str())?.get(prop).cloned()
        }

        // new SomeClass(...) or new SomeClass { ... }
        "object_creation_expression" => {
            let type_node = node.child_by_field_name("type")?;
            strip_cs_type(type_node, source)
        }

        // obj.Method() — return type of the called method
        "invocation_expression" => {
            let func_node = node.child_by_field_name("function")?;
            if func_node.kind() == "member_access_expression" {
                let obj_type = eval_expr_type(
                    scope,
                    class_fields,
                    method_returns,
                    func_node.child_by_field_name("expression")?,
                    source,
                )?;
                let method_name = func_node.child_by_field_name("name")?.utf8_text(source).ok()?;
                method_returns.get(&(obj_type, method_name.to_string())).cloned()
            } else {
                None
            }
        }

        // await expr — transparent
        "await_expression" => {
            node.named_child(0)
                .and_then(|inner| eval_expr_type(scope, class_fields, method_returns, inner, source))
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
        let Some(type_node) = param.child_by_field_name("type") else {
            continue;
        };
        let Some(type_name) = strip_cs_type(type_node, source) else {
            continue;
        };
        if let Some(name_node) = param.child_by_field_name("name") {
            if let Ok(name) = name_node.utf8_text(source) {
                scope_bind(scope, name, &type_name);
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
        // Class / record entry: push scope, bind "this", seed fields.
        "class_declaration" | "interface_declaration" | "record_declaration" => {
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
                        walk_node(child, source, scope, class_fields, method_returns, dir_fns, graph, out, file);
                    }
                }
            }
            scope_pop(scope);
        }

        // Method / constructor: push scope, bind parameters, walk body.
        "method_declaration" | "constructor_declaration" => {
            scope_push(scope);
            if let Some(params) = node.child_by_field_name("parameters") {
                bind_parameters(params, source, scope);
            }
            if let Some(body) = node.child_by_field_name("body") {
                walk_node(body, source, scope, class_fields, method_returns, dir_fns, graph, out, file);
            }
            scope_pop(scope);
        }

        // Local variable declaration: bind the declared type (or infer via eval for `var`).
        "local_declaration_statement" => {
            // Find the variable_declaration child (past any modifiers).
            let Some(var_decl) = (0..node.named_child_count())
                .filter_map(|i| node.named_child(i))
                .find(|n| n.kind() == "variable_declaration")
            else {
                recurse(node, source, scope, class_fields, method_returns, dir_fns, graph, out, file);
                return;
            };

            let type_node = var_decl.child_by_field_name("type");
            // explicit_type is None when the keyword is `var` (predefined_type) or a primitive.
            let explicit_type = type_node.and_then(|t| strip_cs_type(t, source));

            for j in 0..var_decl.named_child_count() {
                let Some(decl) = var_decl.named_child(j) else {
                    continue;
                };
                if decl.kind() != "variable_declarator" {
                    continue;
                }
                let Some(name_node) = decl.child_by_field_name("name") else {
                    continue;
                };
                let Ok(name) = name_node.utf8_text(source) else {
                    continue;
                };

                // Bind type: prefer explicit annotation, then eval initializer (for `var`).
                let bound = if let Some(ref t) = explicit_type {
                    Some(t.clone())
                } else {
                    // var keyword or primitives — initializer is named_child(1) of the declarator
                    decl.named_child(1).and_then(|init| {
                        eval_expr_type(scope, class_fields, method_returns, init, source)
                    })
                };
                if let Some(t) = bound {
                    scope_bind(scope, name, &t);
                }

                // Recurse into initializer to capture nested calls.
                if let Some(init) = decl.named_child(1) {
                    walk_node(init, source, scope, class_fields, method_returns, dir_fns, graph, out, file);
                }
            }
        }

        // Method call: resolve via type, record position, recurse into args.
        "invocation_expression" => {
            if let Some(func_node) = node.child_by_field_name("function") {
                let resolved = if func_node.kind() == "member_access_expression" {
                    // obj.Method(...)
                    let obj_node = func_node.child_by_field_name("expression");
                    let name_node = func_node.child_by_field_name("name");
                    if let (Some(obj_node), Some(name_node)) = (obj_node, name_node) {
                        let obj_type =
                            eval_expr_type(scope, class_fields, method_returns, obj_node, source);
                        let method_name = name_node.utf8_text(source).ok();
                        if let (Some(obj_type), Some(method_name)) = (obj_type, method_name) {
                            let pos = {
                                let p = name_node.start_position();
                                (p.row, p.column)
                            };
                            let target = find_method_in_class(graph, &obj_type, method_name);
                            target.map(|t| (pos, t))
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                } else if func_node.kind() == "identifier" {
                    // Bare function call — same-dir fallback
                    if let Ok(method_name) = func_node.utf8_text(source) {
                        let dir = parent_dir(file);
                        let pos = {
                            let p = func_node.start_position();
                            (p.row, p.column)
                        };
                        dir_fns.get(&dir)
                            .and_then(|m| m.get(method_name))
                            .map(|nk| (pos, nk.clone()))
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
            // Recurse into arguments.
            if let Some(args) = node.child_by_field_name("arguments") {
                walk_node(args, source, scope, class_fields, method_returns, dir_fns, graph, out, file);
            }
        }

        // object_creation_expression: recurse into args / initializer for nested calls.
        "object_creation_expression" => {
            if let Some(args) = node.child_by_field_name("arguments") {
                walk_node(args, source, scope, class_fields, method_returns, dir_fns, graph, out, file);
            }
            if let Some(init) = node.child_by_field_name("initializer") {
                walk_node(init, source, scope, class_fields, method_returns, dir_fns, graph, out, file);
            }
        }

        _ => {
            recurse(node, source, scope, class_fields, method_returns, dir_fns, graph, out, file);
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
            walk_node(child, source, scope, class_fields, method_returns, dir_fns, graph, out, file);
        }
    }
}

// ── Public entry point ────────────────────────────────────────────────────────

/// Walk a C# source file and return a map from (row, col) of each resolved
/// call-site's name position to the target NodeKeys.
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
