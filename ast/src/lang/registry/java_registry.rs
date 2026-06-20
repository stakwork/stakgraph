use super::{java_resolver, Registry};
use crate::lang::asg::NodeKeys;
use crate::lang::graphs::{Graph, NodeType};
use std::collections::HashMap;
use std::path::Path;

fn parent_dir(file: &str) -> String {
    Path::new(file)
        .parent()
        .map(|p| p.to_string_lossy().into_owned())
        .unwrap_or_default()
}

pub struct JavaRegistry {
    /// class_name → { field_name → base_type }
    class_fields: HashMap<String, HashMap<String, String>>,
    /// (class_name, method_name) → return type for method-chain resolution
    method_returns: HashMap<(String, String), String>,
    /// dir → { fn_name → NodeKeys } for same-package bare-name resolution
    pkg_fns: HashMap<String, HashMap<String, NodeKeys>>,
    /// (file, row, col) → pre-resolved target NodeKeys
    resolved: HashMap<(String, usize, usize), NodeKeys>,
}

impl JavaRegistry {
    pub fn new(graph: &impl Graph, filez: &[(String, String)]) -> Self {
        let mut reg = JavaRegistry {
            class_fields: HashMap::new(),
            method_returns: HashMap::new(),
            pkg_fns: HashMap::new(),
            resolved: HashMap::new(),
        };

        // Pass 1: index Function nodes from the graph for same-package bare-name fallback.
        for (node_type, node_data) in graph.iter_all_nodes() {
            if !node_data.file.ends_with(".java") {
                continue;
            }
            if *node_type != NodeType::Function {
                continue;
            }
            let dir = parent_dir(&node_data.file);
            reg.pkg_fns
                .entry(dir)
                .or_default()
                .entry(node_data.name.clone())
                .or_insert_with(|| NodeKeys::from(node_data));
        }

        // Pass 1.5: extract class field types and method return types from source AST.
        for (file, source) in filez {
            if !file.ends_with(".java") {
                continue;
            }
            let fields = java_resolver::extract_class_fields(source);
            for (class_name, field_map) in fields {
                reg.class_fields
                    .entry(class_name)
                    .or_default()
                    .extend(field_map);
            }
            let returns = java_resolver::extract_method_return_types(source);
            for (key, ret) in returns {
                reg.method_returns.entry(key).or_insert(ret);
            }
        }

        // Pass 2: pre-resolve all call sites per file.
        let all_resolved: Vec<((String, usize, usize), NodeKeys)> = filez
            .iter()
            .filter(|(f, _)| f.ends_with(".java"))
            .flat_map(|(file, source)| {
                java_resolver::resolve_file_calls(
                    source,
                    file,
                    &reg.class_fields,
                    &reg.method_returns,
                    &reg.pkg_fns,
                    graph,
                )
                .into_iter()
                .map(|((row, col), nk)| ((file.clone(), row, col), nk))
            })
            .collect();
        reg.resolved.extend(all_resolved);

        reg
    }
}

impl Registry for JavaRegistry {
    fn resolve_type(&self, _file: &str, _var_name: &str) -> Option<&str> {
        None
    }

    fn resolve_method(&self, _type_name: &str, _method_name: &str) -> Option<&str> {
        None
    }

    fn resolve_call_at(&self, file: &str, row: usize, col: usize) -> Option<NodeKeys> {
        self.resolved
            .get(&(file.to_string(), row, col))
            .cloned()
    }
}
