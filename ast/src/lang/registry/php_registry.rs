use super::{php_resolver, Registry};
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

pub struct PhpRegistry {
    var_types: HashMap<(String, String), String>,
    methods: HashMap<(String, String), String>,
    class_fields: HashMap<String, HashMap<String, String>>,
    method_returns: HashMap<(String, String), String>,
    fn_returns: HashMap<String, String>,
    dir_fns: HashMap<String, HashMap<String, NodeKeys>>,
    resolved: HashMap<(String, usize, usize), NodeKeys>,
}

impl PhpRegistry {
    pub fn new(graph: &impl Graph, filez: &[(String, String)]) -> Self {
        let mut reg = PhpRegistry {
            var_types: HashMap::new(),
            methods: HashMap::new(),
            class_fields: HashMap::new(),
            method_returns: HashMap::new(),
            fn_returns: HashMap::new(),
            dir_fns: HashMap::new(),
            resolved: HashMap::new(),
        };

        // Pass 1: index graph nodes — functions, methods, and typed variables.
        for (node_type, node_data) in graph.iter_all_nodes() {
            if !node_data.file.ends_with(".php") {
                continue;
            }
            match node_type {
                NodeType::Function => {
                    let dir = parent_dir(&node_data.file);
                    reg.dir_fns
                        .entry(dir)
                        .or_default()
                        .entry(node_data.name.clone())
                        .or_insert_with(|| NodeKeys::from(node_data));
                    if let Some(operand) = node_data.meta.get("operand") {
                        reg.methods.insert(
                            (operand.clone(), node_data.name.clone()),
                            node_data.file.clone(),
                        );
                    }
                }
                NodeType::Var | NodeType::Instance => {
                    if let Some(dt) = &node_data.data_type {
                        reg.var_types
                            .insert((node_data.file.clone(), node_data.name.clone()), dt.clone());
                    }
                }
                _ => {}
            }
        }

        // Pass 1.5: extract class field types and method return types.
        for (file, source) in filez {
            if !file.ends_with(".php") {
                continue;
            }
            let fields = php_resolver::extract_class_fields(source);
            for (class_name, field_map) in fields {
                reg.class_fields
                    .entry(class_name)
                    .or_default()
                    .extend(field_map);
            }
            let returns = php_resolver::extract_method_return_types(source);
            for (key, ret) in returns {
                reg.method_returns.entry(key).or_insert(ret);
            }
            let fn_rets = php_resolver::extract_fn_returns(source);
            for (key, ret) in fn_rets {
                reg.fn_returns.entry(key).or_insert(ret);
            }
        }

        // Pass 2: pre-resolve all call sites per file.
        let all_resolved: Vec<((String, usize, usize), NodeKeys)> = filez
            .iter()
            .filter(|(f, _)| f.ends_with(".php"))
            .flat_map(|(file, source)| {
                php_resolver::resolve_file_calls(
                    source,
                    file,
                    &reg.class_fields,
                    &reg.method_returns,
                    &reg.fn_returns,
                    &reg.dir_fns,
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

impl Registry for PhpRegistry {
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
