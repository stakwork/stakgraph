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

fn base_type(type_str: &str) -> &str {
    type_str.split('@').next().unwrap_or(type_str)
}

fn make_parser() -> Option<Parser> {
    let mut parser = Parser::new();
    let lang: tree_sitter::Language = tree_sitter_python::LANGUAGE.into();
    parser.set_language(&lang).ok()?;
    Some(parser)
}

// ── extraction helpers ────────────────────────────────────────────────────────

/// Extract class field types from `__init__` bodies.
/// Handles:
///   self.field: TypeName = ...      → explicit annotation wins
///   self.field = ClassName()        → constructor type
///   self.field = param_name         → follow __init__ param annotation
///
/// Returns class_name → { field_name → type_name }.
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

fn walk_classes(node: Node, source: &[u8], out: &mut HashMap<String, HashMap<String, String>>) {
    if matches!(node.kind(), "class_definition" | "decorated_definition") {
        let class_node = if node.kind() == "decorated_definition" {
            node.child_by_field_name("definition").unwrap_or(node)
        } else {
            node
        };
        if class_node.kind() == "class_definition" {
            if let Some(name_node) = class_node.child_by_field_name("name") {
                if let Ok(class_name) = name_node.utf8_text(source) {
                    let fields = extract_fields_from_class(class_node, source);
                    if !fields.is_empty() {
                        out.entry(class_name.to_string()).or_default().extend(fields);
                    }
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

fn extract_fields_from_class(
    class_node: Node,
    source: &[u8],
) -> HashMap<String, String> {
    let mut fields: HashMap<String, String> = HashMap::new();
    let Some(body) = class_node.child_by_field_name("body") else {
        return fields;
    };

    // Find __init__ and collect its param types so we can resolve
    // `self.field = param` assignments.
    let mut init_params: HashMap<String, String> = HashMap::new();
    let mut init_body: Option<Node> = None;

    for i in 0..body.named_child_count() {
        let Some(member) = body.named_child(i) else { continue };
        let fn_node = if member.kind() == "decorated_definition" {
            member.child_by_field_name("definition").unwrap_or(member)
        } else {
            member
        };
        if fn_node.kind() != "function_definition" {
            continue;
        }
        let Some(name_node) = fn_node.child_by_field_name("name") else { continue };
        let Ok(fn_name) = name_node.utf8_text(source) else { continue };
        if fn_name != "__init__" {
            continue;
        }
        // Collect param type annotations (skip `self`)
        if let Some(params) = fn_node.child_by_field_name("parameters") {
            let mut first = true;
            for j in 0..params.named_child_count() {
                let Some(param) = params.named_child(j) else { continue };
                if first {
                    first = false;
                    continue; // skip self
                }
                let (pname, ptype) = extract_param_annotation(param, source);
                if let (Some(n), Some(t)) = (pname, ptype) {
                    init_params.insert(n, t);
                }
            }
        }
        init_body = fn_node.child_by_field_name("body");
        break;
    }

    let Some(body_node) = init_body else {
        return fields;
    };

    for i in 0..body_node.named_child_count() {
        let Some(stmt) = body_node.named_child(i) else { continue };
        if stmt.kind() != "expression_statement" {
            continue;
        }
        let Some(inner) = stmt.named_child(0) else { continue };
        if inner.kind() != "assignment" {
            continue;
        }
        let Some(left) = inner.child_by_field_name("left") else { continue };
        let Some(right) = inner.child_by_field_name("right") else { continue };

        // left must be `self.field`
        if left.kind() != "attribute" {
            continue;
        }
        let Some(obj) = left.child_by_field_name("object") else { continue };
        let Ok(obj_text) = obj.utf8_text(source) else { continue };
        if obj_text != "self" {
            continue;
        }
        let Some(attr) = left.child_by_field_name("attribute") else { continue };
        let Ok(field_name) = attr.utf8_text(source) else { continue };

        // Explicit type annotation on the assignment wins
        if let Some(type_node) = inner.child_by_field_name("type") {
            if let Some(t) = extract_type_annotation(type_node, source) {
                fields.insert(field_name.to_string(), t);
                continue;
            }
        }

        // self.field = ClassName()  →  ClassName
        if right.kind() == "call" {
            if let Some(func) = right.child_by_field_name("function") {
                if func.kind() == "identifier" {
                    if let Ok(ctor) = func.utf8_text(source) {
                        fields.insert(field_name.to_string(), ctor.to_string());
                        continue;
                    }
                }
            }
        }

        // self.field = param_name  →  follow __init__ param annotation
        if right.kind() == "identifier" {
            if let Ok(rhs_name) = right.utf8_text(source) {
                if let Some(t) = init_params.get(rhs_name) {
                    fields.insert(field_name.to_string(), t.clone());
                }
            }
        }
    }

    fields
}

fn extract_param_annotation(param: Node, source: &[u8]) -> (Option<String>, Option<String>) {
    let kind = param.kind();

    let name: Option<String> = match kind {
        // `service: UserService` — first named child is the identifier
        "typed_parameter" => param
            .named_child(0)
            .filter(|n| n.kind() == "identifier")
            .and_then(|n| n.utf8_text(source).ok())
            .map(|s| s.to_string()),
        // plain `x` or `x=default` — node itself or `name` field
        "identifier" => param.utf8_text(source).ok().map(|s| s.to_string()),
        "default_parameter" => param
            .child_by_field_name("name")
            .and_then(|n| n.utf8_text(source).ok())
            .map(|s| s.to_string()),
        _ => None,
    };

    let type_ann = param
        .child_by_field_name("type")
        .and_then(|t| extract_type_annotation(t, source));

    (name, type_ann)
}

fn extract_type_annotation(type_node: Node, source: &[u8]) -> Option<String> {
    // type node wraps the actual type expression
    let inner = if type_node.named_child_count() > 0 {
        type_node.named_child(0)?
    } else {
        type_node
    };
    // Skip generic / Optional / Union annotations — just take plain identifiers
    if inner.kind() == "identifier" {
        inner.utf8_text(source).ok().map(|s| s.to_string())
    } else {
        None
    }
}

/// Extract module-level `x = ClassName()` and `x: TypeName = ...` bindings.
pub fn extract_top_level_vars(source: &str) -> HashMap<String, String> {
    let mut out = HashMap::new();
    let Some(mut parser) = make_parser() else { return out };
    let Some(tree) = parser.parse(source, None) else { return out };
    let src = source.as_bytes();
    let root = tree.root_node();

    for i in 0..root.named_child_count() {
        let Some(stmt) = root.named_child(i) else { continue };
        let inner = if stmt.kind() == "decorated_definition" {
            stmt.named_child(0).unwrap_or(stmt)
        } else {
            stmt
        };
        if inner.kind() != "expression_statement" {
            continue;
        }
        let Some(assign) = inner.named_child(0) else { continue };
        if assign.kind() != "assignment" {
            continue;
        }
        let Some(left) = assign.child_by_field_name("left") else { continue };
        let Some(right) = assign.child_by_field_name("right") else { continue };
        let Ok(var_name) = left.utf8_text(src) else { continue };
        if left.kind() != "identifier" {
            continue;
        }

        // Explicit type annotation
        if let Some(type_node) = assign.child_by_field_name("type") {
            if let Some(t) = extract_type_annotation(type_node, src) {
                out.entry(var_name.to_string()).or_insert(t);
                continue;
            }
        }

        // x = ClassName()
        if right.kind() == "call" {
            if let Some(func) = right.child_by_field_name("function") {
                if func.kind() == "identifier" {
                    if let Ok(ctor) = func.utf8_text(src) {
                        out.entry(var_name.to_string()).or_insert_with(|| ctor.to_string());
                    }
                }
            }
        }
    }
    out
}

/// Extract top-level function return type annotations.
/// Returns func_name → return_type for non-generic plain-identifier return types.
pub fn extract_fn_returns(source: &str) -> HashMap<String, String> {
    let mut out = HashMap::new();
    let Some(mut parser) = make_parser() else { return out };
    let Some(tree) = parser.parse(source, None) else { return out };
    let src = source.as_bytes();
    let root = tree.root_node();

    for i in 0..root.named_child_count() {
        let Some(stmt) = root.named_child(i) else { continue };
        let inner = if stmt.kind() == "decorated_definition" {
            stmt.child_by_field_name("definition").unwrap_or(stmt)
        } else {
            stmt
        };
        if inner.kind() != "function_definition" {
            continue;
        }
        let Some(name_node) = inner.child_by_field_name("name") else { continue };
        let Ok(fn_name) = name_node.utf8_text(src) else { continue };
        let Some(ret_node) = inner.child_by_field_name("return_type") else { continue };
        if let Some(t) = extract_type_annotation(ret_node, src) {
            out.insert(fn_name.to_string(), t);
        }
    }
    out
}

// ── expression type evaluator ─────────────────────────────────────────────────

fn eval_expr_type(
    scope: &Scope,
    class_fields: &HashMap<String, HashMap<String, String>>,
    fn_returns: &HashMap<String, (String, String)>,
    node: Node,
    source: &[u8],
) -> Option<String> {
    match node.kind() {
        "identifier" => scope_lookup(scope, node.utf8_text(source).ok()?).map(|s| s.to_string()),
        "attribute" => {
            let obj_type = eval_expr_type(
                scope,
                class_fields,
                fn_returns,
                node.child_by_field_name("object")?,
                source,
            )?;
            let attr = node.child_by_field_name("attribute")?.utf8_text(source).ok()?;
            class_fields.get(base_type(&obj_type))?.get(attr).cloned()
        }
        "call" => {
            let func = node.child_by_field_name("function")?;
            if func.kind() == "identifier" {
                let func_name = func.utf8_text(source).ok()?;
                fn_returns
                    .get(func_name)
                    .map(|(ret_type, def_file)| format!("{}@{}", ret_type, def_file))
            } else {
                None
            }
        }
        "await" => {
            for i in 0..node.named_child_count() {
                if let Some(child) = node.named_child(i) {
                    if let Some(t) = eval_expr_type(scope, class_fields, fn_returns, child, source)
                    {
                        return Some(t);
                    }
                }
            }
            None
        }
        _ => None,
    }
}

// ── callee resolution ─────────────────────────────────────────────────────────

fn resolve_callee<G: Graph>(
    scope: &Scope,
    class_fields: &HashMap<String, HashMap<String, String>>,
    fn_returns: &HashMap<String, (String, String)>,
    graph: &G,
    node: Node,
    source: &[u8],
) -> Option<NodeKeys> {
    if node.kind() != "attribute" {
        return None;
    }
    let raw_type = eval_expr_type(
        scope,
        class_fields,
        fn_returns,
        node.child_by_field_name("object")?,
        source,
    )?;
    let method_name = node.child_by_field_name("attribute")?.utf8_text(source).ok()?;

    let (receiver_type, file_hint): (&str, Option<&str>) = if let Some(idx) = raw_type.find('@') {
        (&raw_type[..idx], Some(&raw_type[idx + 1..]))
    } else {
        (raw_type.as_str(), None)
    };

    // Strict: method with matching operand (class methods stored with operand == class name)
    if let Some(n) = graph
        .find_nodes_by_name(NodeType::Function, method_name)
        .into_iter()
        .find(|n| n.meta.get("operand").map(|s| s.as_str()) == Some(receiver_type))
    {
        return Some(NodeKeys::from(&n));
    }

    // File-hint: when type came from fn_returns, search in the defining file
    if let Some(file) = file_hint {
        if let Some(n) = graph.find_node_by_name_in_file(NodeType::Function, method_name, file) {
            return Some(NodeKeys::from(&n));
        }
    }

    // DataModel / Trait fallback
    if file_hint.is_none() {
        let anchor = graph
            .find_nodes_by_name(NodeType::DataModel, receiver_type)
            .into_iter()
            .next()
            .or_else(|| {
                graph
                    .find_nodes_by_name(NodeType::Trait, receiver_type)
                    .into_iter()
                    .next()
            })?;
        return graph
            .find_node_by_name_in_file(NodeType::Function, method_name, &anchor.file)
            .map(|n| NodeKeys::from(&n));
    }

    None
}

// ── AST walker ────────────────────────────────────────────────────────────────

fn try_bind_assignment(
    node: Node,
    source: &[u8],
    scope: &mut Scope,
    class_fields: &HashMap<String, HashMap<String, String>>,
    fn_returns: &HashMap<String, (String, String)>,
) {
    if node.kind() != "assignment" {
        return;
    }
    let Some(left) = node.child_by_field_name("left") else { return };
    let Some(right) = node.child_by_field_name("right") else { return };
    if left.kind() != "identifier" {
        return;
    }
    let Ok(name) = left.utf8_text(source) else { return };

    // Explicit annotation: x: TypeName = ...
    if let Some(type_node) = node.child_by_field_name("type") {
        if let Some(t) = extract_type_annotation(type_node, source) {
            scope_bind(scope, name, &t);
            return;
        }
    }

    // x = ClassName()
    if right.kind() == "call" {
        if let Some(func) = right.child_by_field_name("function") {
            if func.kind() == "identifier" {
                if let Ok(ctor) = func.utf8_text(source) {
                    scope_bind(scope, name, ctor);
                    return;
                }
            }
        }
    }

    // x = expr  — infer via eval
    if let Some(t) = eval_expr_type(scope, class_fields, fn_returns, right, source) {
        scope_bind(scope, name, &t);
    }
}

fn bind_params(params_node: Node, source: &[u8], scope: &mut Scope, skip_first: bool) {
    let mut idx = 0u32;
    for i in 0..params_node.named_child_count() {
        let Some(param) = params_node.named_child(i) else { continue };
        // Skip list_splat_pattern / dictionary_splat_pattern / comment
        if matches!(
            param.kind(),
            "list_splat_pattern" | "dictionary_splat_pattern" | "comment"
        ) {
            continue;
        }
        if skip_first && idx == 0 {
            idx += 1;
            continue;
        }
        idx += 1;
        let (name, type_ann) = extract_param_annotation(param, source);
        if let (Some(n), Some(t)) = (name, type_ann) {
            scope_bind(scope, &n, &t);
        }
    }
}

fn walk_node<G: Graph>(
    node: Node,
    source: &[u8],
    scope: &mut Scope,
    enclosing_class: Option<&str>,
    class_fields: &HashMap<String, HashMap<String, String>>,
    fn_returns: &HashMap<String, (String, String)>,
    graph: &G,
    out: &mut HashMap<(usize, usize), NodeKeys>,
) {
    match node.kind() {
        "class_definition" => {
            let class_name = node
                .child_by_field_name("name")
                .and_then(|n| n.utf8_text(source).ok())
                .map(|s| s.to_string());
            if let Some(body) = node.child_by_field_name("body") {
                for i in 0..body.named_child_count() {
                    if let Some(child) = body.named_child(i) {
                        walk_node(
                            child,
                            source,
                            scope,
                            class_name.as_deref(),
                            class_fields,
                            fn_returns,
                            graph,
                            out,
                        );
                    }
                }
            }
        }
        "decorated_definition" => {
            let inner = node.child_by_field_name("definition").unwrap_or(node);
            walk_node(
                inner,
                source,
                scope,
                enclosing_class,
                class_fields,
                fn_returns,
                graph,
                out,
            );
        }
        "function_definition" => {
            scope_push(scope);
            // Bind self/cls → enclosing class type
            if let (Some(cls), Some(params)) =
                (enclosing_class, node.child_by_field_name("parameters"))
            {
                // First named child is the self/cls identifier
                if let Some(first_param) = params.named_child(0) {
                    if first_param.kind() == "identifier" {
                        if let Ok(self_name) = first_param.utf8_text(source) {
                            scope_bind(scope, self_name, cls);
                        }
                    }
                }
                // Remaining params
                bind_params(params, source, scope, true);
            } else if let Some(params) = node.child_by_field_name("parameters") {
                bind_params(params, source, scope, false);
            }
            if let Some(body) = node.child_by_field_name("body") {
                walk_node(
                    body,
                    source,
                    scope,
                    enclosing_class,
                    class_fields,
                    fn_returns,
                    graph,
                    out,
                );
            }
            scope_pop(scope);
        }
        "call" => {
            if let Some(func_node) = node.child_by_field_name("function") {
                if let Some(target) =
                    resolve_callee(scope, class_fields, fn_returns, graph, func_node, source)
                {
                    // Key by the attribute identifier position (the method name)
                    let pos = func_node
                        .child_by_field_name("attribute")
                        .map(|a| a.start_position())
                        .unwrap_or_else(|| func_node.start_position());
                    out.insert((pos.row, pos.column), target);
                }
                // Walk arguments too (calls can be nested)
                if let Some(args) = node.child_by_field_name("arguments") {
                    walk_node(
                        args,
                        source,
                        scope,
                        enclosing_class,
                        class_fields,
                        fn_returns,
                        graph,
                        out,
                    );
                }
            }
        }
        "expression_statement" => {
            if let Some(inner) = node.named_child(0) {
                if inner.kind() == "assignment" {
                    try_bind_assignment(inner, source, scope, class_fields, fn_returns);
                }
                // Still walk for nested calls
                walk_node(
                    inner,
                    source,
                    scope,
                    enclosing_class,
                    class_fields,
                    fn_returns,
                    graph,
                    out,
                );
            }
        }
        _ => {
            for i in 0..node.named_child_count() {
                if let Some(child) = node.named_child(i) {
                    walk_node(
                        child,
                        source,
                        scope,
                        enclosing_class,
                        class_fields,
                        fn_returns,
                        graph,
                        out,
                    );
                }
            }
        }
    }
}

// ── public entry point ────────────────────────────────────────────────────────

/// Walk a Python source file and return a map from
/// (row, col) of each resolved method-call's attribute identifier to the target NodeKeys.
pub fn resolve_file_calls<G: Graph>(
    source: &str,
    file: &str,
    class_fields: &HashMap<String, HashMap<String, String>>,
    fn_returns: &HashMap<String, (String, String)>,
    import_sources: &HashMap<(String, String), String>,
    var_types: &HashMap<(String, String), String>,
    graph: &G,
) -> HashMap<(usize, usize), NodeKeys> {
    let mut out = HashMap::new();
    let Some(mut parser) = make_parser() else { return out };
    let Some(tree) = parser.parse(source, None) else { return out };
    let src = source.as_bytes();

    // Seed root scope from imports
    let mut scope: Scope = vec![HashMap::new()];
    for ((caller_file, name), source_file) in import_sources {
        if caller_file != file {
            continue;
        }
        let type_name = lookup_with_extensions(source_file, name, var_types);
        if let Some(t) = type_name {
            scope_bind(&mut scope, name, t.as_str());
        }
    }

    walk_node(
        tree.root_node(),
        src,
        &mut scope,
        None,
        class_fields,
        fn_returns,
        graph,
        &mut out,
    );
    out
}

fn lookup_with_extensions(
    base: &str,
    name: &str,
    var_types: &HashMap<(String, String), String>,
) -> Option<String> {
    for ext in &["", ".py"] {
        let key = (format!("{}{}", base, ext), name.to_string());
        if let Some(v) = var_types.get(&key) {
            return Some(v.clone());
        }
    }
    None
}

/// Normalize a relative import path like `../services` relative to `current_file`.
pub(crate) fn resolve_import_relative(current_file: &str, import_path: &str) -> String {
    use std::path::{Component, Path, PathBuf};
    if !import_path.starts_with('.') {
        return import_path.to_string();
    }
    let base_dir = Path::new(current_file).parent().unwrap_or(Path::new(""));
    let joined = base_dir.join(import_path);
    let mut normalized = PathBuf::new();
    for component in joined.components() {
        match component {
            Component::ParentDir => { normalized.pop(); }
            Component::CurDir => {}
            c => normalized.push(c),
        }
    }
    normalized.display().to_string()
}
