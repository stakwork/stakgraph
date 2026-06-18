use super::{go_resolver, Registry};
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

pub struct GoRegistry {
    var_types: HashMap<(String, String), String>,
    methods: HashMap<(String, String), String>,
    struct_fields: HashMap<String, HashMap<String, String>>,
    pkg_fns: HashMap<String, HashMap<String, NodeKeys>>,
    resolved: HashMap<(String, usize, usize), NodeKeys>,
}

impl GoRegistry {
    pub fn new(graph: &impl Graph, filez: &[(String, String)]) -> Self {
        let mut reg = GoRegistry {
            var_types: HashMap::new(),
            methods: HashMap::new(),
            struct_fields: HashMap::new(),
            pkg_fns: HashMap::new(),
            resolved: HashMap::new(),
        };

        // Pass 1: scan graph nodes for functions, methods, and typed vars
        for (node_type, node_data) in graph.iter_all_nodes() {
            let file = &node_data.file;
            if !file.ends_with(".go") {
                continue;
            }
            let dir = parent_dir(file);
            match node_type {
                NodeType::Function => {
                    // Index by package directory for same-package identifier resolution
                    reg.pkg_fns
                        .entry(dir.clone())
                        .or_default()
                        .entry(node_data.name.clone())
                        .or_insert_with(|| NodeKeys::from(node_data));
                    // Index by (receiver_type, method_name) for selector_expression resolution
                    if let Some(operand) = node_data.meta.get("operand") {
                        reg.methods.insert(
                            (operand.clone(), node_data.name.clone()),
                            file.clone(),
                        );
                    }
                }
                NodeType::Var | NodeType::Instance => {
                    if let Some(dt) = &node_data.data_type {
                        reg.var_types
                            .insert((file.clone(), node_data.name.clone()), dt.clone());
                    }
                }
                _ => {}
            }
        }

        // Pass 1.5: extract struct field types from source files
        for (file, source) in filez {
            if !file.ends_with(".go") {
                continue;
            }
            let fields = go_resolver::extract_struct_fields(source);
            for (struct_name, field_map) in fields {
                reg.struct_fields
                    .entry(struct_name)
                    .or_default()
                    .extend(field_map);
            }
        }

        // Pass 2: pre-resolve call sites per file
        let all_resolved: Vec<((String, usize, usize), NodeKeys)> = filez
            .iter()
            .filter(|(f, _)| f.ends_with(".go"))
            .flat_map(|(file, source)| {
                go_resolver::resolve_file_calls(
                    source,
                    file,
                    &reg.struct_fields,
                    &reg.pkg_fns,
                    &reg.var_types,
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

impl Registry for GoRegistry {
    fn resolve_type(&self, file: &str, var_name: &str) -> Option<&str> {
        self.var_types
            .get(&(file.to_string(), var_name.to_string()))
            .map(|s| s.as_str())
    }

    fn resolve_method(&self, type_name: &str, method_name: &str) -> Option<&str> {
        self.methods
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
