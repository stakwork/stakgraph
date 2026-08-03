use super::{py_resolver, Registry};
use crate::lang::asg::NodeKeys;
use crate::lang::call_finder::IMPORT_CACHE;
use crate::lang::graphs::{Graph, NodeType};
use std::collections::HashMap;

pub struct PythonRegistry {
    pub(super) var_types: HashMap<(String, String), String>,
    methods: HashMap<(String, String), String>,
    pub(super) import_sources: HashMap<(String, String), String>,
    pub(super) class_fields: HashMap<String, HashMap<String, String>>,
    method_returns: HashMap<(String, String), String>,
    resolved: HashMap<(String, usize, usize), NodeKeys>,
}

impl PythonRegistry {
    pub fn new(graph: &impl Graph, filez: &[(String, String)]) -> Self {
        let mut reg = PythonRegistry {
            var_types: HashMap::new(),
            methods: HashMap::new(),
            import_sources: HashMap::new(),
            class_fields: HashMap::new(),
            method_returns: HashMap::new(),
            resolved: HashMap::new(),
        };

        // Pass 1: scan graph nodes
        for (node_type, node_data) in graph.iter_all_nodes() {
            let file = &node_data.file;
            if !file.ends_with(".py") {
                continue;
            }
            match node_type {
                NodeType::Function => {
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

        // Populate import_sources from IMPORT_CACHE
        for (file, _) in filez {
            if !file.ends_with(".py") {
                continue;
            }
            if let Some(entry) = IMPORT_CACHE.get(file) {
                if let Some(imports) = entry.value() {
                    for (source_file, names) in imports {
                        for name in names {
                            reg.import_sources.insert(
                                (file.clone(), name.clone()),
                                source_file.clone(),
                            );
                        }
                    }
                }
            }
        }

        // Pass 1.5: extract class fields and top-level vars from source
        for (file, source) in filez {
            if !file.ends_with(".py") {
                continue;
            }
            let fields = py_resolver::extract_class_fields(source);
            for (class_name, field_map) in fields {
                reg.class_fields
                    .entry(class_name)
                    .or_default()
                    .extend(field_map);
            }
            let top_vars = py_resolver::extract_top_level_vars(source);
            for (var_name, type_name) in top_vars {
                reg.var_types
                    .entry((file.clone(), var_name))
                    .or_insert(type_name);
            }
            let method_rets = py_resolver::extract_method_return_types(source);
            for (key, ret) in method_rets {
                reg.method_returns.entry(key).or_insert(ret);
            }
        }

        // Build fn_returns: func_name → (return_type, defining_file)
        let fn_returns: HashMap<String, (String, String)> = filez
            .iter()
            .filter(|(f, _)| f.ends_with(".py"))
            .flat_map(|(file, source)| {
                py_resolver::extract_fn_returns(source)
                    .into_iter()
                    .map(|(func_name, ret_type)| (func_name, (ret_type, file.clone())))
            })
            .collect();

        // Pass 2: pre-resolve call sites per file
        let all_resolved: Vec<((String, usize, usize), NodeKeys)> = filez
            .iter()
            .filter(|(f, _)| f.ends_with(".py"))
            .flat_map(|(file, source)| {
                py_resolver::resolve_file_calls(
                    source,
                    file,
                    &reg.class_fields,
                    &reg.method_returns,
                    &fn_returns,
                    &reg.import_sources,
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

impl Registry for PythonRegistry {
    fn resolve_type(&self, file: &str, var_name: &str) -> Option<&str> {
        if let Some(t) = self.var_types.get(&(file.to_string(), var_name.to_string())) {
            return Some(t.as_str());
        }
        if let Some(source_file) = self
            .import_sources
            .get(&(file.to_string(), var_name.to_string()))
        {
            for ext in &["", ".py"] {
                let key = (format!("{}{}", source_file, ext), var_name.to_string());
                if let Some(t) = self.var_types.get(&key) {
                    return Some(t.as_str());
                }
            }
        }
        None
    }

    fn resolve_method(&self, type_name: &str, method_name: &str) -> Option<&str> {
        self.methods
            .get(&(type_name.to_string(), method_name.to_string()))
            .map(|s| s.as_str())
    }

    fn resolve_field(&self, type_name: &str, field_name: &str) -> Option<&str> {
        self.class_fields
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
