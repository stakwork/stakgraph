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

/// Strip the `@file` hint from a type string produced by `eval_expr_type`.
fn base_type(type_str: &str) -> &str {
    type_str.split('@').next().unwrap_or(type_str)
}

fn eval_expr_type(
    scope: &Scope,
    class_fields: &HashMap<String, HashMap<String, String>>,
    fn_returns: &HashMap<String, (String, String)>,
    node: Node,
    source: &[u8],
) -> Option<String> {
    match node.kind() {
        "identifier" => scope_lookup(scope, node.utf8_text(source).ok()?).map(|s| s.to_string()),
        "member_expression" => {
            let obj_type =
                eval_expr_type(scope, class_fields, fn_returns, node.child_by_field_name("object")?, source)?;
            let prop = node.child_by_field_name("property")?.utf8_text(source).ok()?;
            // Strip any @file hint before looking up class fields.
            class_fields.get(base_type(&obj_type))?.get(prop).cloned()
        }
        "new_expression" => node
            .child_by_field_name("constructor")?
            .utf8_text(source)
            .ok()
            .map(|s| s.to_string()),
        "call_expression" => {
            let func_node = node.child_by_field_name("function")?;
            if func_node.kind() == "identifier" {
                let func_name = func_node.utf8_text(source).ok()?;
                fn_returns.get(func_name).map(|(ret_type, def_file)| {
                    format!("{}@{}", ret_type, def_file)
                })
            } else if func_node.kind() == "member_expression" {
                // obj.method() — look up return type from class_fields["ClassName"]["method()"]
                let obj_type = eval_expr_type(
                    scope,
                    class_fields,
                    fn_returns,
                    func_node.child_by_field_name("object")?,
                    source,
                )?;
                let method_name = func_node
                    .child_by_field_name("property")?
                    .utf8_text(source)
                    .ok()?;
                class_fields
                    .get(base_type(&obj_type))?
                    .get(&format!("{}()", method_name))
                    .cloned()
            } else {
                None
            }
        }
        "await_expression" => {
            for i in 0..node.named_child_count() {
                if let Some(child) = node.named_child(i) {
                    if let Some(t) = eval_expr_type(scope, class_fields, fn_returns, child, source) {
                        return Some(t);
                    }
                }
            }
            None
        }
        _ => None,
    }
}

fn resolve_callee<G: Graph>(
    scope: &Scope,
    class_fields: &HashMap<String, HashMap<String, String>>,
    fn_returns: &HashMap<String, (String, String)>,
    graph: &G,
    node: Node,
    source: &[u8],
    file: &str,
    import_sources: &HashMap<(String, String), String>,
) -> Option<NodeKeys> {
    if node.kind() == "identifier" {
        let fname = node.utf8_text(source).ok()?;
        if let Some(n) = graph.find_node_by_name_in_file(NodeType::Function, fname, file) {
            return Some(NodeKeys::from(&n));
        }
        if let Some(src_file) = import_sources.get(&(file.to_string(), fname.to_string())) {
            let resolved = resolve_import_relative(file, src_file);
            for ext in &["", ".ts", ".tsx"] {
                let full = format!("{}{}", resolved, ext);
                if let Some(n) = graph.find_node_by_name_in_file(NodeType::Function, fname, &full) {
                    return Some(NodeKeys::from(&n));
                }
            }
        }
        return None;
    }
    if node.kind() != "member_expression" {
        return None;
    }
    let raw_type =
        eval_expr_type(scope, class_fields, fn_returns, node.child_by_field_name("object")?, source)?;
    let method_name = node
        .child_by_field_name("property")?
        .utf8_text(source)
        .ok()?;

    // Parse optional "@file" hint encoded by the call_expression arm of eval_expr_type.
    let (receiver_type, file_hint): (&str, Option<&str>) = if let Some(idx) = raw_type.find('@') {
        (&raw_type[..idx], Some(&raw_type[idx + 1..]))
    } else {
        (raw_type.as_str(), None)
    };

    // Strict: function with operand == receiver_type (class methods)
    if let Some(n) = graph
        .find_nodes_by_name(NodeType::Function, method_name)
        .into_iter()
        .find(|n| n.meta.get("operand").map(|s| s.as_str()) == Some(receiver_type))
    {
        return Some(NodeKeys::from(&n));
    }

    // File-hint lookup: when the type came from a fn_returns entry, search directly in the
    // defining file to avoid collisions with same-named types in other files.
    if let Some(file) = file_hint {
        if let Some(n) = graph.find_node_by_name_in_file(NodeType::Function, method_name, file) {
            return Some(NodeKeys::from(&n));
        }
    }

    // DataModel/Trait fallback when no file hint is available.
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

fn try_bind_declaration(
    node: Node,
    source: &[u8],
    scope: &mut Scope,
    class_fields: &HashMap<String, HashMap<String, String>>,
    fn_returns: &HashMap<String, (String, String)>,
) {
    for i in 0..node.named_child_count() {
        let Some(child) = node.named_child(i) else {
            continue;
        };
        if child.kind() != "variable_declarator" {
            continue;
        }
        let Some(name_node) = child.child_by_field_name("name") else {
            continue;
        };
        let Ok(name) = name_node.utf8_text(source) else {
            continue;
        };
        if let Some(type_node) = child.child_by_field_name("type") {
            if let Some(inner) = type_node.named_child(0) {
                if let Ok(type_str) = inner.utf8_text(source) {
                    scope_bind(scope, name, type_str);
                    continue;
                }
            }
        }
        if let Some(value_node) = child.child_by_field_name("value") {
            if let Some(type_name) = eval_expr_type(scope, class_fields, fn_returns, value_node, source) {
                scope_bind(scope, name, &type_name);
            }
        }
    }
}

fn bind_params(params_node: Node, source: &[u8], scope: &mut Scope) {
    for i in 0..params_node.named_child_count() {
        let Some(param) = params_node.named_child(i) else {
            continue;
        };
        if !matches!(param.kind(), "required_parameter" | "optional_parameter") {
            continue;
        }
        let Some(name_node) = param.child_by_field_name("pattern") else {
            continue;
        };
        let Ok(name) = name_node.utf8_text(source) else {
            continue;
        };
        if let Some(type_node) = param.child_by_field_name("type") {
            if let Some(inner) = type_node.named_child(0) {
                if let Ok(type_str) = inner.utf8_text(source) {
                    scope_bind(scope, name, type_str);
                }
            }
        }
    }
}

fn walk_node<G: Graph>(
    node: Node,
    source: &[u8],
    scope: &mut Scope,
    class_fields: &HashMap<String, HashMap<String, String>>,
    fn_returns: &HashMap<String, (String, String)>,
    graph: &G,
    out: &mut HashMap<(usize, usize), NodeKeys>,
    file: &str,
    import_sources: &HashMap<(String, String), String>,
) {
    match node.kind() {
        "call_expression" => {
            if let Some(func_node) = node.child_by_field_name("function") {
                if let Some(target) = resolve_callee(scope, class_fields, fn_returns, graph, func_node, source, file, import_sources)
                {
                    let pos = func_node
                        .child_by_field_name("property")
                        .map(|p| p.start_position())
                        .unwrap_or_else(|| func_node.start_position());
                    out.insert((pos.row, pos.column), target);
                }
                if let Some(args) = node.child_by_field_name("arguments") {
                    walk_node(args, source, scope, class_fields, fn_returns, graph, out, file, import_sources);
                }
            }
        }
        "lexical_declaration" | "variable_declaration" => {
            try_bind_declaration(node, source, scope, class_fields, fn_returns);
            for i in 0..node.named_child_count() {
                if let Some(child) = node.named_child(i) {
                    walk_node(child, source, scope, class_fields, fn_returns, graph, out, file, import_sources);
                }
            }
        }
        "function_declaration"
        | "function"
        | "arrow_function"
        | "method_definition"
        | "generator_function"
        | "generator_function_declaration" => {
            scope_push(scope);
            if let Some(params) = node.child_by_field_name("parameters") {
                bind_params(params, source, scope);
            }
            if let Some(body) = node.child_by_field_name("body") {
                walk_node(body, source, scope, class_fields, fn_returns, graph, out, file, import_sources);
            }
            scope_pop(scope);
        }
        "class_declaration" | "class" => {
            if let Some(body) = node.child_by_field_name("body") {
                for i in 0..body.named_child_count() {
                    if let Some(child) = body.named_child(i) {
                        walk_node(child, source, scope, class_fields, fn_returns, graph, out, file, import_sources);
                    }
                }
            }
        }
        "jsx_element" => {
            if let Some(open_tag) = node.child_by_field_name("open_tag") {
                if let Some(name_node) = open_tag.child_by_field_name("name") {
                    if name_node.kind() == "identifier" {
                        if let Some(target) = resolve_callee(scope, class_fields, fn_returns, graph, name_node, source, file, import_sources) {
                            let pos = name_node.start_position();
                            out.insert((pos.row, pos.column), target);
                        }
                    }
                }
            }
            for i in 0..node.named_child_count() {
                if let Some(child) = node.named_child(i) {
                    walk_node(child, source, scope, class_fields, fn_returns, graph, out, file, import_sources);
                }
            }
        }
        "jsx_self_closing_element" => {
            if let Some(name_node) = node.child_by_field_name("name") {
                if name_node.kind() == "identifier" {
                    if let Some(target) = resolve_callee(scope, class_fields, fn_returns, graph, name_node, source, file, import_sources) {
                        let pos = name_node.start_position();
                        out.insert((pos.row, pos.column), target);
                    }
                }
            }
            for i in 0..node.named_child_count() {
                if let Some(child) = node.named_child(i) {
                    walk_node(child, source, scope, class_fields, fn_returns, graph, out, file, import_sources);
                }
            }
        }
        _ => {
            for i in 0..node.named_child_count() {
                if let Some(child) = node.named_child(i) {
                    walk_node(child, source, scope, class_fields, fn_returns, graph, out, file, import_sources);
                }
            }
        }
    }
}

fn make_parser() -> Option<Parser> {
    let mut parser = Parser::new();
    let lang: tree_sitter::Language = tree_sitter_typescript::LANGUAGE_TSX.into();
    parser.set_language(&lang).ok()?;
    Some(parser)
}

/// Walk a TypeScript/TSX source file and return a map from
/// (row, col) of each method-call's property identifier to the target NodeKeys.
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
    let Some(mut parser) = make_parser() else {
        return out;
    };
    let Some(tree) = parser.parse(source, None) else {
        return out;
    };
    let src = source.as_bytes();

    let mut scope: Scope = vec![HashMap::new()];
    for ((caller_file, name), source_file) in import_sources {
        if caller_file != file {
            continue;
        }
        // Try direct lookup; fall back to resolving relative import path
        let type_name = lookup_with_extensions(source_file, name, var_types).or_else(|| {
            let resolved = resolve_import_relative(file, source_file);
            lookup_with_extensions(&resolved, name, var_types)
        });
        if let Some(t) = type_name {
            scope_bind(&mut scope, name, t.as_str());
        }
    }

    walk_node(tree.root_node(), src, &mut scope, class_fields, fn_returns, graph, &mut out, file, import_sources);
    out
}

fn lookup_with_extensions(
    base: &str,
    name: &str,
    var_types: &HashMap<(String, String), String>,
) -> Option<String> {
    for ext in &["", ".ts", ".tsx"] {
        let key = (format!("{}{}", base, ext), name.to_string());
        if let Some(v) = var_types.get(&key) {
            return Some(v.clone());
        }
    }
    None
}

/// Normalize a relative import path like "../../lib/api/client" relative to
/// the directory of `current_file`, returning the canonical relative path.
pub(crate) fn resolve_import_relative(current_file: &str, import_path: &str) -> String {
    use std::path::{Component, Path, PathBuf};
    if !import_path.starts_with('.') {
        return import_path.to_string();
    }
    let base_dir = Path::new(current_file)
        .parent()
        .unwrap_or(Path::new(""));
    let joined = base_dir.join(import_path);
    let mut normalized = PathBuf::new();
    for component in joined.components() {
        match component {
            Component::ParentDir => {
                normalized.pop();
            }
            Component::CurDir => {}
            c => normalized.push(c),
        }
    }
    normalized.display().to_string()
}

/// Extract top-level `const/let x = new Class()` var types from a TS/TSX source.
/// Returns { var_name → constructor_type } for unannotated new-expression assignments.
pub fn extract_top_level_vars(source: &str) -> HashMap<String, String> {
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
        let Some(stmt) = root.named_child(i) else {
            continue;
        };
        // unwrap export_statement to get the inner declaration
        let decl = if stmt.kind() == "export_statement" {
            stmt.named_child(0).unwrap_or(stmt)
        } else {
            stmt
        };
        if !matches!(decl.kind(), "lexical_declaration" | "variable_declaration") {
            continue;
        }
        for j in 0..decl.named_child_count() {
            let Some(declarator) = decl.named_child(j) else {
                continue;
            };
            if declarator.kind() != "variable_declarator" {
                continue;
            }
            let Some(name_node) = declarator.child_by_field_name("name") else {
                continue;
            };
            let Some(value_node) = declarator.child_by_field_name("value") else {
                continue;
            };
            if value_node.kind() != "new_expression" {
                continue;
            }
            let Some(ctor_node) = value_node.child_by_field_name("constructor") else {
                continue;
            };
            if let (Ok(var_name), Ok(type_name)) =
                (name_node.utf8_text(src), ctor_node.utf8_text(src))
            {
                out.insert(var_name.to_string(), type_name.to_string());
            }
        }
    }
    out
}

/// Extract named function return types from a TS/TSX source file.
/// Returns { func_name → return_type } for top-level and exported function declarations
/// and arrow-function variable declarators that have an explicit return type annotation.
/// Only emits non-generic return types (no `<`).
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
        let Some(stmt) = root.named_child(i) else { continue };
        let inner = if stmt.kind() == "export_statement" {
            stmt.named_child(0).unwrap_or(stmt)
        } else {
            stmt
        };
        match inner.kind() {
            "function_declaration" => {
                let Some(name_node) = inner.child_by_field_name("name") else { continue };
                let Some(ret_node) = inner.child_by_field_name("return_type") else { continue };
                if let (Ok(name), Ok(ret)) = (name_node.utf8_text(src), ret_node.utf8_text(src)) {
                    let clean = ret.trim_start_matches(':').trim().to_string();
                    let unwrapped = if clean.starts_with("Promise<") && clean.ends_with('>') {
                        let inner = &clean[8..clean.len() - 1];
                        // Skip single-char type parameters like Promise<T>
                        if inner.len() > 1 { inner.to_string() } else { clean }
                    } else {
                        clean
                    };
                    if !unwrapped.is_empty() && !unwrapped.contains('<') {
                        out.insert(name.to_string(), unwrapped);
                    }
                }
            }
            "lexical_declaration" | "variable_declaration" => {
                for j in 0..inner.named_child_count() {
                    let Some(decl) = inner.named_child(j) else { continue };
                    if decl.kind() != "variable_declarator" { continue }
                    let Some(name_node) = decl.child_by_field_name("name") else { continue };
                    let Some(val) = decl.child_by_field_name("value") else { continue };
                    if !matches!(val.kind(), "arrow_function" | "function_expression") { continue }
                    let Some(ret_node) = val.child_by_field_name("return_type") else { continue };
                    if let (Ok(name), Ok(ret)) = (name_node.utf8_text(src), ret_node.utf8_text(src)) {
                        let clean = ret.trim_start_matches(':').trim().to_string();
                        if !clean.is_empty() && !clean.contains('<') {
                            out.insert(name.to_string(), clean);
                        }
                    }
                }
            }
            _ => {}
        }
    }
    out
}

/// Extract class field types from a TypeScript/TSX source file.
/// Returns class_name → { field_name → constructor_type }.
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

fn walk_classes(
    node: Node,
    source: &[u8],
    out: &mut HashMap<String, HashMap<String, String>>,
) {
    if matches!(node.kind(), "class_declaration" | "class") {
        if let (Some(name_node), Some(body_node)) = (
            node.child_by_field_name("name"),
            node.child_by_field_name("body"),
        ) {
            if let Ok(class_name) = name_node.utf8_text(source) {
                let mut fields: HashMap<String, String> = HashMap::new();
                for i in 0..body_node.named_child_count() {
                    let Some(member) = body_node.named_child(i) else {
                        continue;
                    };
                    if member.kind() == "public_field_definition" {
                        // Field initialised with a constructor: `users = new UsersAPI()`
                        let Some(fname_node) = member.child_by_field_name("name") else {
                            continue;
                        };
                        let Some(value_node) = member.child_by_field_name("value") else {
                            continue;
                        };
                        if value_node.kind() != "new_expression" {
                            continue;
                        }
                        let Some(ctor_node) = value_node.child_by_field_name("constructor") else {
                            continue;
                        };
                        if let (Ok(field_name), Ok(type_name)) = (
                            fname_node.utf8_text(source),
                            ctor_node.utf8_text(source),
                        ) {
                            fields.insert(field_name.to_string(), type_name.to_string());
                        }
                    } else if member.kind() == "method_definition" {
                        // Method with explicit return type: stored as "name()" → return_type
                        // so call_expression eval can look them up without a new parameter.
                        let Some(mname_node) = member.child_by_field_name("name") else {
                            continue;
                        };
                        let Some(ret_node) = member.child_by_field_name("return_type") else {
                            continue;
                        };
                        if let (Ok(method_name), Ok(ret)) =
                            (mname_node.utf8_text(source), ret_node.utf8_text(source))
                        {
                            let clean = ret.trim_start_matches(':').trim().to_string();
                            let unwrapped =
                                if clean.starts_with("Promise<") && clean.ends_with('>') {
                                    let inner = &clean[8..clean.len() - 1];
                                    if inner.len() > 1 {
                                        inner.to_string()
                                    } else {
                                        clean
                                    }
                                } else {
                                    clean
                                };
                            if !unwrapped.is_empty() && !unwrapped.contains('<') {
                                fields.insert(format!("{}()", method_name), unwrapped);
                            }
                        }
                    }
                }
                if !fields.is_empty() {
                    out.entry(class_name.to_string()).or_default().extend(fields);
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

