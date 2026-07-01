use super::{c_resolver, Registry};
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

pub struct CRegistry {
    var_types: HashMap<(String, String), String>,
    struct_fields: HashMap<String, HashMap<String, String>>,
    fn_returns: HashMap<String, String>,
    dir_fns: HashMap<String, HashMap<String, NodeKeys>>,
    resolved: HashMap<(String, usize, usize), NodeKeys>,
}

impl CRegistry {
    pub fn new(graph: &impl Graph, filez: &[(String, String)]) -> Self {
        let mut reg = CRegistry {
            var_types: HashMap::new(),
            struct_fields: HashMap::new(),
            fn_returns: HashMap::new(),
            dir_fns: HashMap::new(),
            resolved: HashMap::new(),
        };

        for (node_type, node_data) in graph.iter_all_nodes() {
            if !(node_data.file.ends_with(".c") || node_data.file.ends_with(".h")) {
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

        for (file, source) in filez {
            if !(file.ends_with(".c") || file.ends_with(".h")) {
                continue;
            }
            let fields = c_resolver::extract_struct_fields(source);
            for (struct_name, field_map) in fields {
                reg.struct_fields
                    .entry(struct_name)
                    .or_default()
                    .extend(field_map);
            }
            let fn_rets = c_resolver::extract_fn_returns(source);
            for (key, ret) in fn_rets {
                reg.fn_returns.entry(key).or_insert(ret);
            }
        }

        let all_resolved: Vec<((String, usize, usize), NodeKeys)> = filez
            .iter()
            .filter(|(f, _)| f.ends_with(".c") || f.ends_with(".h"))
            .flat_map(|(file, source)| {
                c_resolver::resolve_file_calls(
                    source,
                    file,
                    &reg.struct_fields,
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

impl Registry for CRegistry {
    fn resolve_type(&self, file: &str, var_name: &str) -> Option<&str> {
        self.var_types
            .get(&(file.to_string(), var_name.to_string()))
            .map(|s| s.as_str())
    }

    fn resolve_method(&self, _type_name: &str, _method_name: &str) -> Option<&str> {
        None
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
