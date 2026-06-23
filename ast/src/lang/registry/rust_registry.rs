use super::{rust_resolver, Registry};
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

pub struct RustRegistry {
    /// (operand_type, method_name) → defining file
    methods_idx: HashMap<(String, String), String>,
    /// struct_name → { field_name → base_type }
    struct_fields: HashMap<String, HashMap<String, String>>,
    /// dir → { fn_name → NodeKeys } for free functions in the same crate directory
    pkg_fns: HashMap<String, HashMap<String, NodeKeys>>,
    /// (file, row, col) → pre-resolved target NodeKeys
    resolved: HashMap<(String, usize, usize), NodeKeys>,
}

impl RustRegistry {
    pub fn new(graph: &impl Graph, filez: &[(String, String)]) -> Self {
        let mut reg = RustRegistry {
            methods_idx: HashMap::new(),
            struct_fields: HashMap::new(),
            pkg_fns: HashMap::new(),
            resolved: HashMap::new(),
        };

        // Pass 1: scan graph nodes for functions.
        for (node_type, node_data) in graph.iter_all_nodes() {
            let file = &node_data.file;
            if !file.ends_with(".rs") {
                continue;
            }
            if *node_type != NodeType::Function {
                continue;
            }
            let dir = parent_dir(file);

            // Index free functions and methods by directory (crate module scope).
            reg.pkg_fns
                .entry(dir)
                .or_default()
                .entry(node_data.name.clone())
                .or_insert_with(|| NodeKeys::from(node_data));

            // Index methods by (operand_type, method_name) → file.
            if let Some(operand) = node_data.meta.get("operand") {
                reg.methods_idx
                    .entry((operand.clone(), node_data.name.clone()))
                    .or_insert_with(|| file.clone());
            }
        }

        // Pass 1.5: extract struct field types from source.
        for (file, source) in filez {
            if !file.ends_with(".rs") {
                continue;
            }
            let fields = rust_resolver::extract_struct_fields(source);
            for (struct_name, field_map) in fields {
                reg.struct_fields
                    .entry(struct_name)
                    .or_default()
                    .extend(field_map);
            }
        }

        // Pass 2: pre-resolve all call sites per file.
        let all_resolved: Vec<((String, usize, usize), NodeKeys)> = filez
            .iter()
            .filter(|(f, _)| f.ends_with(".rs"))
            .flat_map(|(file, source)| {
                rust_resolver::resolve_file_calls(
                    source,
                    file,
                    &reg.struct_fields,
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

impl Registry for RustRegistry {
    fn resolve_type(&self, _file: &str, _var_name: &str) -> Option<&str> {
        None
    }

    fn resolve_method(&self, type_name: &str, method_name: &str) -> Option<&str> {
        self.methods_idx
            .get(&(type_name.to_string(), method_name.to_string()))
            .map(|s| s.as_str())
    }

    fn resolve_field(&self, type_name: &str, field_name: &str) -> Option<&str> {
        self.struct_fields
            .get(type_name)?
            .get(field_name)
            .map(|s| s.as_str())
    }

    fn resolve_call_at(&self, file: &str, row: usize, col: usize) -> Option<NodeKeys> {
        self.resolved
            .get(&(file.to_string(), row, col))
            .cloned()
    }
}
