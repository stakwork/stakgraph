use crate::lang::asg::NodeKeys;
use crate::lang::graphs::{EdgeType, Graph, NodeType};
use crate::lang::NodeData;
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
    let lang: tree_sitter::Language = tree_sitter_java::LANGUAGE.into();
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

/// Strip generic/array wrappers and return the base type name.
/// Returns None for primitives, void, wildcards, and unknown node kinds.
fn strip_java_type(node: Node, source: &[u8]) -> Option<String> {
    match node.kind() {
        "type_identifier" => node.utf8_text(source).ok().map(str::to_string),

        // e.g. java.util.List — take the rightmost segment
        "scoped_type_identifier" => node
            .child_by_field_name("name")
            .and_then(|n| n.utf8_text(source).ok())
            .map(str::to_string),

        // List<Person>, Optional<T> — keep the outer type name only
        "generic_type" => node
            .child_by_field_name("name")
            .and_then(|n| strip_java_type(n, source)),

        // Person[] — strip to element type
        "array_type" => node
            .child_by_field_name("element")
            .and_then(|n| strip_java_type(n, source)),

        // primitives and void carry no class methods we'd dispatch on
        "integral_type"
        | "boolean_type"
        | "floating_point_type"
        | "void_type"
        | "wildcard" => None,

        _ => None,
    }
}

// ── Class field extraction ─────────────────────────────────────────────────────

fn walk_class_body_for_fields(body: Node, source: &[u8], fields: &mut HashMap<String, String>) {
    for i in 0..body.named_child_count() {
        let Some(child) = body.named_child(i) else {
            continue;
        };
        if child.kind() != "field_declaration" {
            continue;
        }
        let Some(type_node) = child.child_by_field_name("type") else {
            continue;
        };
        let Some(field_type) = strip_java_type(type_node, source) else {
            continue;
        };
        for j in 0..child.named_child_count() {
            let Some(decl) = child.named_child(j) else {
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

/// Extract `(class_name, method_name) → return_type` for every non-void, non-primitive
/// method declared in the source file.  Used to evaluate the type of a chained
/// method call: `new Foo().bar().baz()` → resolve `bar` return type → resolve `baz`.
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
        "class_declaration" | "record_declaration" => {
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
        "method_declaration" => {
            if let Some(class_name) = current_class {
                if let Some(type_node) = node.child_by_field_name("type") {
                    if let Some(ret_type) = strip_java_type(type_node, source) {
                        if let Some(name_node) = node.child_by_field_name("name") {
                            if let Ok(method_name) = name_node.utf8_text(source) {
                                // First declaration wins; overloads are ignored.
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

/// Extract `class_name → { field_name → base_type }` from a Java source file.
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
        "class_declaration" | "interface_declaration" | "record_declaration" => {
            if let Some(name_node) = node.child_by_field_name("name") {
                if let Ok(class_name) = name_node.utf8_text(source) {
                    let mut fields = HashMap::new();
                    if let Some(body) = node.child_by_field_name("body") {
                        walk_class_body_for_fields(body, source, &mut fields);
                        // Recurse for nested classes inside the body
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

// ── Class-based method resolution ──────────────────────────────────────────────

/// Find the Function node for `method_name` on class `class_name`.
///
/// Java functions do not carry an `operand` meta key (unlike Rust/Go methods).
/// We locate the Class (or Trait) node in the graph, then find the Function
/// node in the same file whose byte-offset start is ≥ the class node's start.
/// When multiple classes share a file, taking the function closest to — but not
/// before — the class declaration gives us the right method.
///
/// Interface dispatch: when `class_name` maps to a Trait (interface), we
/// traverse `Implements` edges to find the concrete implementing class and
/// search for the method there instead.
fn find_method_in_class<G: Graph>(
    graph: &G,
    class_name: &str,
    method_name: &str,
) -> Option<NodeKeys> {
    // Helper: given a class NodeData, find the first Function named `method_name`
    // that appears in the same file at or after the class declaration start.
    let find_in_class = |class_nd: &NodeData| -> Option<NodeKeys> {
        let mut candidates: Vec<_> = graph
            .find_nodes_by_name(NodeType::Function, method_name)
            .into_iter()
            .filter(|f| f.file == class_nd.file && f.start >= class_nd.start)
            .collect();
        candidates.sort_by_key(|f| f.start);
        candidates.into_iter().next().map(|nd| NodeKeys::from(&nd))
    };

    // 1. Direct Class lookup.
    let class_nodes = graph.find_nodes_by_name(NodeType::Class, class_name);
    for class_nd in &class_nodes {
        if let Some(keys) = find_in_class(class_nd) {
            return Some(keys);
        }
    }

    // 2. Interface dispatch: check whether class_name is a Trait (Java interface),
    //    then find classes that implement it and search them for the method.
    if graph
        .find_nodes_by_name(NodeType::Trait, class_name)
        .is_empty()
    {
        return None;
    }

    let implementing: Vec<_> = graph
        .find_nodes_with_edge_type(NodeType::Class, NodeType::Trait, EdgeType::Implements)
        .into_iter()
        .filter(|(_, trait_nd)| trait_nd.name == class_name)
        .map(|(class_nd, _)| class_nd)
        .collect();

    for class_nd in &implementing {
        if let Some(keys) = find_in_class(class_nd) {
            return Some(keys);
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
        // `this` is its own keyword node in tree-sitter-java (not an identifier)
        "this" => scope_lookup(scope, "this").map(str::to_string),

        // Bare name: local var, parameter, or class field (seeded into scope on class entry)
        "identifier" => {
            scope_lookup(scope, node.utf8_text(source).ok()?).map(str::to_string)
        }

        // this.gateway  or  obj.field
        "field_access" => {
            let obj_type = eval_expr_type(
                scope,
                class_fields,
                method_returns,
                node.child_by_field_name("object")?,
                source,
            )?;
            let field = node
                .child_by_field_name("field")?
                .utf8_text(source)
                .ok()?;
            class_fields.get(&obj_type)?.get(field).cloned()
        }

        // new SomeClass(...) → type is SomeClass
        "object_creation_expression" => {
            let type_node = node.child_by_field_name("type")?;
            strip_java_type(type_node, source)
        }

        // (SomeType) expr → use the declared cast type
        "cast_expression" => {
            let type_node = node.child_by_field_name("type")?;
            strip_java_type(type_node, source)
        }

        // (expr) — transparent
        "parenthesized_expression" => node
            .named_child(0)
            .and_then(|inner| eval_expr_type(scope, class_fields, method_returns, inner, source)),

        // Chained call: resolve the receiver's type, then look up the method's return type.
        // e.g. builder.withName("x").withEmail("y") — after withName resolves to PersonBuilder,
        // look up (PersonBuilder, withEmail) → PersonBuilder.
        "method_invocation" => {
            let receiver_type = if let Some(obj) = node.child_by_field_name("object") {
                eval_expr_type(scope, class_fields, method_returns, obj, source)?
            } else {
                scope_lookup(scope, "this")?.to_string()
            };
            let method_name = node.child_by_field_name("name")?.utf8_text(source).ok()?;
            method_returns
                .get(&(receiver_type, method_name.to_string()))
                .cloned()
        }

        _ => None,
    }
}

// ── Callee resolution ──────────────────────────────────────────────────────────

/// Resolve a `method_invocation` node.
/// Returns (target NodeKeys, (row, col) of the method name token).
/// The position must match the @FUNCTION_NAME capture in the Java tree-sitter
/// query, which is `name: (identifier)` inside `method_invocation`.
fn resolve_method_call<G: Graph>(
    scope: &Scope,
    class_fields: &HashMap<String, HashMap<String, String>>,
    method_returns: &HashMap<(String, String), String>,
    pkg_fns: &HashMap<String, HashMap<String, NodeKeys>>,
    graph: &G,
    node: Node,
    source: &[u8],
    file: &str,
) -> Option<(NodeKeys, (usize, usize))> {
    let name_node = node.child_by_field_name("name")?;
    let method_name = name_node.utf8_text(source).ok()?;
    let pos = {
        let p = name_node.start_position();
        (p.row, p.column)
    };

    let target = if let Some(object_node) = node.child_by_field_name("object") {
        let obj_type =
            eval_expr_type(scope, class_fields, method_returns, object_node, source)?;
        find_method_in_class(graph, &obj_type, method_name)?
    } else {
        if let Some(nd) = graph.find_node_by_name_in_file(NodeType::Function, method_name, file) {
            NodeKeys::from(&nd)
        } else {
            let dir = parent_dir(file);
            pkg_fns.get(&dir)?.get(method_name).cloned()?
        }
    };

    Some((target, pos))
}

/// Resolve an `object_creation_expression` (`new SomeClass(args)`).
/// Returns (constructor NodeKeys, position of class name token).
/// The position must match the `(type_identifier) @FUNCTION_NAME` capture.
fn resolve_constructor<G: Graph>(
    graph: &G,
    node: Node,
    source: &[u8],
    file: &str,
) -> Option<(NodeKeys, (usize, usize))> {
    let type_node = node.child_by_field_name("type")?;

    // Find the innermost type_identifier token — that is what @FUNCTION_NAME captures.
    let (type_name, pos) = match type_node.kind() {
        "type_identifier" => {
            let p = type_node.start_position();
            let name = type_node.utf8_text(source).ok()?.to_string();
            (name, (p.row, p.column))
        }
        "scoped_type_identifier" => {
            // The query captures the inner (type_identifier) at the last segment.
            let inner = type_node
                .named_child(type_node.named_child_count().saturating_sub(1))?;
            let p = inner.start_position();
            let name = inner.utf8_text(source).ok()?.to_string();
            (name, (p.row, p.column))
        }
        "generic_type" => {
            let name_node = type_node.child_by_field_name("name")?;
            match name_node.kind() {
                "type_identifier" => {
                    let p = name_node.start_position();
                    let name = name_node.utf8_text(source).ok()?.to_string();
                    (name, (p.row, p.column))
                }
                _ => return None,
            }
        }
        _ => return None,
    };

    // Constructors are Function nodes with the same name as the class.
    // Use the same class-position disambiguation as regular methods.
    let target = find_method_in_class(graph, &type_name, &type_name)
        .or_else(|| {
            // Fallback: constructor defined in the same file (e.g. local inner class)
            graph
                .find_node_by_name_in_file(NodeType::Function, &type_name, file)
                .map(|n| NodeKeys::from(&n))
        })?;

    Some((target, pos))
}

// ── Parameter binding ──────────────────────────────────────────────────────────

fn bind_formal_parameters(params: Node, source: &[u8], scope: &mut Scope) {
    for i in 0..params.named_child_count() {
        let Some(param) = params.named_child(i) else {
            continue;
        };
        match param.kind() {
            "formal_parameter" | "spread_parameter" => {
                let Some(type_node) = param.child_by_field_name("type") else {
                    continue;
                };
                let Some(type_name) = strip_java_type(type_node, source) else {
                    continue;
                };
                let name_opt = param
                    .child_by_field_name("name")
                    .and_then(|n| n.utf8_text(source).ok().map(str::to_string))
                    .or_else(|| {
                        param
                            .child_by_field_name("declarator")
                            .and_then(|d| d.child_by_field_name("name"))
                            .and_then(|n| n.utf8_text(source).ok().map(str::to_string))
                    });
                if let Some(name) = name_opt {
                    if name != "_" {
                        scope_bind(scope, &name, &type_name);
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
    class_fields: &HashMap<String, HashMap<String, String>>,
    method_returns: &HashMap<(String, String), String>,
    pkg_fns: &HashMap<String, HashMap<String, NodeKeys>>,
    graph: &G,
    out: &mut HashMap<(usize, usize), NodeKeys>,
    file: &str,
) {
    match node.kind() {
        // Class entry: push a scope frame, bind "this" to the class name,
        // and seed all instance fields so bare field names resolve (e.g. `repository.findById`).
        "class_declaration" | "record_declaration" | "enum_declaration" => {
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
                        walk_node(child, source, scope, class_fields, method_returns, pkg_fns, graph, out, file);
                    }
                }
            }
            scope_pop(scope);
        }

        // Method and constructor: push scope, bind parameters, walk body.
        "method_declaration" | "constructor_declaration" => {
            scope_push(scope);
            if let Some(params) = node.child_by_field_name("parameters") {
                bind_formal_parameters(params, source, scope);
            }
            if let Some(body) = node.child_by_field_name("body") {
                walk_node(body, source, scope, class_fields, method_returns, pkg_fns, graph, out, file);
            }
            scope_pop(scope);
        }

        // Lambda: push scope, bind typed parameters when present, walk body.
        "lambda_expression" => {
            scope_push(scope);
            if let Some(params) = node.child_by_field_name("parameters") {
                if params.kind() == "formal_parameters" {
                    bind_formal_parameters(params, source, scope);
                }
            }
            if let Some(body) = node.child_by_field_name("body") {
                walk_node(body, source, scope, class_fields, method_returns, pkg_fns, graph, out, file);
            }
            scope_pop(scope);
        }

        // Local variable: bind declared type into current scope frame, then recurse into initialiser.
        "local_variable_declaration" => {
            let type_name = node
                .child_by_field_name("type")
                .and_then(|t| strip_java_type(t, source));

            for i in 0..node.named_child_count() {
                let Some(child) = node.named_child(i) else {
                    continue;
                };
                if child.kind() != "variable_declarator" {
                    continue;
                }
                if let Some(ref t) = type_name {
                    if let Some(name_node) = child.child_by_field_name("name") {
                        if let Ok(name) = name_node.utf8_text(source) {
                            if name != "_" {
                                scope_bind(scope, name, t);
                            }
                        }
                    }
                }
                if let Some(val) = child.child_by_field_name("value") {
                    walk_node(val, source, scope, class_fields, method_returns, pkg_fns, graph, out, file);
                }
            }
        }

        // Method call: resolve, record position, recurse into receiver and arguments.
        "method_invocation" => {
            if let Some((target, pos)) =
                resolve_method_call(scope, class_fields, method_returns, pkg_fns, graph, node, source, file)
            {
                out.entry(pos).or_insert(target);
            }
            if let Some(obj) = node.child_by_field_name("object") {
                walk_node(obj, source, scope, class_fields, method_returns, pkg_fns, graph, out, file);
            }
            if let Some(args) = node.child_by_field_name("arguments") {
                walk_node(args, source, scope, class_fields, method_returns, pkg_fns, graph, out, file);
            }
        }

        // Constructor call: resolve, record position, recurse into arguments.
        "object_creation_expression" => {
            if let Some((target, pos)) = resolve_constructor(graph, node, source, file) {
                out.entry(pos).or_insert(target);
            }
            if let Some(args) = node.child_by_field_name("arguments") {
                walk_node(args, source, scope, class_fields, method_returns, pkg_fns, graph, out, file);
            }
        }

        _ => {
            for i in 0..node.named_child_count() {
                if let Some(child) = node.named_child(i) {
                    walk_node(child, source, scope, class_fields, method_returns, pkg_fns, graph, out, file);
                }
            }
        }
    }
}

// ── Public entry point ────────────────────────────────────────────────────────

/// Walk a Java source file and return a map from (row, col) of each resolved
/// call-site's name position to the target NodeKeys.
pub fn resolve_file_calls<G: Graph>(
    source: &str,
    file: &str,
    class_fields: &HashMap<String, HashMap<String, String>>,
    method_returns: &HashMap<(String, String), String>,
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
    let mut scope: Scope = vec![HashMap::new()];

    walk_node(
        tree.root_node(),
        src,
        &mut scope,
        class_fields,
        method_returns,
        pkg_fns,
        graph,
        &mut out,
        file,
    );
    out
}
