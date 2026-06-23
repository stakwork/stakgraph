use super::scope::{scope_bind, scope_lookup, scope_pop, scope_push, Scope};
use crate::lang::asg::NodeKeys;
use crate::lang::graphs::{Graph, NodeType};
use std::collections::HashMap;
use std::path::Path;
use tree_sitter::{Node, Parser};

fn make_parser() -> Option<Parser> {
    let mut parser = Parser::new();
    let lang: tree_sitter::Language = tree_sitter_ruby::LANGUAGE.into();
    parser.set_language(&lang).ok()?;
    Some(parser)
}

fn parent_dir(file: &str) -> String {
    Path::new(file)
        .parent()
        .map(|p| p.to_string_lossy().into_owned())
        .unwrap_or_default()
}

// ── Method lookup ──────────────────────────────────────────────────────────────

fn find_method_in_class<G: Graph>(
    graph: &G,
    class_name: &str,
    method_name: &str,
) -> Option<NodeKeys> {
    graph
        .find_nodes_by_name(NodeType::Function, method_name)
        .into_iter()
        .find(|n| n.meta.get("operand").map(|s| s.as_str()) == Some(class_name))
        .map(|n| NodeKeys::from(&n))
}

// ── Type evaluator ─────────────────────────────────────────────────────────────

// Ruby has no type annotations, so resolution is limited to:
// - direct constant (ClassName)
// - local variable previously bound via ClassName.new / ClassName.find / etc.
// - ClassName.new(...) call expression itself
fn eval_expr_type(scope: &Scope, node: Node, src: &[u8]) -> Option<String> {
    match node.kind() {
        "constant" => node.utf8_text(src).ok().map(str::to_string),
        "identifier" => scope_lookup(scope, node.utf8_text(src).ok()?).map(str::to_string),
        "call" => {
            let receiver = node.child_by_field_name("receiver")?;
            let method = node.child_by_field_name("method")?;
            // ClassName.new(...) → type is ClassName
            if receiver.kind() == "constant"
                && method.utf8_text(src).ok()? == "new"
            {
                receiver.utf8_text(src).ok().map(str::to_string)
            } else {
                None
            }
        }
        _ => None,
    }
}

// ── AST walker ─────────────────────────────────────────────────────────────────

// ActiveRecord class-level finders: when called on a constant, bind the left-hand
// variable to that constant's type so chained calls can be resolved.
const BINDING_METHODS: &[&str] = &["new", "find", "find_by", "find_by!", "create", "first", "last", "build"];

fn walk_node<G: Graph>(
    node: Node,
    src: &[u8],
    scope: &mut Scope,
    dir_fns: &HashMap<String, HashMap<String, NodeKeys>>,
    graph: &G,
    out: &mut HashMap<(usize, usize), NodeKeys>,
    file: &str,
) {
    match node.kind() {
        "class" | "module" => {
            let class_name = node
                .child_by_field_name("name")
                .and_then(|n| n.utf8_text(src).ok().map(str::to_string));
            scope_push(scope);
            if let Some(ref name) = class_name {
                scope_bind(scope, "self", name);
            }
            // body_statement is a named child (no field name in tree-sitter-ruby)
            for i in 0..node.named_child_count() {
                if let Some(child) = node.named_child(i) {
                    if child.kind() == "body_statement" {
                        walk_node(child, src, scope, dir_fns, graph, out, file);
                    }
                }
            }
            scope_pop(scope);
        }

        "method" | "singleton_method" => {
            scope_push(scope);
            if let Some(body) = node.child_by_field_name("body").or_else(|| {
                (0..node.named_child_count())
                    .filter_map(|i| node.named_child(i))
                    .find(|n| n.kind() == "body_statement")
            }) {
                walk_node(body, src, scope, dir_fns, graph, out, file);
            }
            scope_pop(scope);
        }

        "assignment" => {
            let left = node.child_by_field_name("left");
            let right = node.child_by_field_name("right");
            if let (Some(left), Some(right)) = (left, right) {
                // Bind local variable when assigned from a typed expression.
                if left.kind() == "identifier" {
                    if let Ok(var_name) = left.utf8_text(src) {
                        // ClassName.new / ClassName.find / etc. → bind var → ClassName
                        let type_name = if right.kind() == "call" {
                            let recv = right.child_by_field_name("receiver");
                            let meth = right.child_by_field_name("method");
                            match (recv, meth) {
                                (Some(r), Some(m))
                                    if r.kind() == "constant"
                                        && m.utf8_text(src)
                                            .map(|s| BINDING_METHODS.contains(&s))
                                            .unwrap_or(false) =>
                                {
                                    r.utf8_text(src).ok().map(str::to_string)
                                }
                                _ => None,
                            }
                        } else {
                            eval_expr_type(scope, right, src)
                        };
                        if let Some(t) = type_name {
                            scope_bind(scope, var_name, &t);
                        }
                    }
                }
                // Always recurse into right side for nested calls.
                walk_node(right, src, scope, dir_fns, graph, out, file);
            }
        }

        "call" => {
            let receiver_node = node.child_by_field_name("receiver");
            let method_node = node.child_by_field_name("method");

            if let Some(method_node) = method_node {
                let method_name = method_node.utf8_text(src).ok();

                let resolved = if let Some(recv) = receiver_node {
                    // Typed receiver: constant or scope-tracked variable.
                    eval_expr_type(scope, recv, src).and_then(|class_name| {
                        method_name.and_then(|mn| {
                            let pos = {
                                let p = method_node.start_position();
                                (p.row, p.column)
                            };
                            find_method_in_class(graph, &class_name, mn).map(|t| (pos, t))
                        })
                    })
                } else if let Some(mn) = method_name {
                    // Bare function call → same-directory fallback.
                    let dir = parent_dir(file);
                    let pos = {
                        let p = method_node.start_position();
                        (p.row, p.column)
                    };
                    dir_fns.get(&dir).and_then(|m| m.get(mn)).map(|nk| (pos, nk.clone()))
                } else {
                    None
                };

                if let Some((pos, target)) = resolved {
                    out.entry(pos).or_insert(target);
                }
            }

            // Recurse into arguments and block.
            if let Some(args) = node.child_by_field_name("arguments") {
                walk_node(args, src, scope, dir_fns, graph, out, file);
            }
            if let Some(block) = node.child_by_field_name("block") {
                walk_node(block, src, scope, dir_fns, graph, out, file);
            }
        }

        _ => {
            for i in 0..node.named_child_count() {
                if let Some(child) = node.named_child(i) {
                    walk_node(child, src, scope, dir_fns, graph, out, file);
                }
            }
        }
    }
}

// ── Public entry point ─────────────────────────────────────────────────────────

pub fn resolve_file_calls<G: Graph>(
    source: &str,
    file: &str,
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
    walk_node(tree.root_node(), src, &mut scope, dir_fns, graph, &mut out, file);
    out
}
