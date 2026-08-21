use super::{ruby_resolver, Registry};
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

pub struct RubyRegistry {
    dir_fns: HashMap<String, HashMap<String, NodeKeys>>,
    resolved: HashMap<(String, usize, usize), NodeKeys>,
}

impl RubyRegistry {
    pub fn new(graph: &impl Graph, filez: &[(String, String)]) -> Self {
        let mut reg = RubyRegistry {
            dir_fns: HashMap::new(),
            resolved: HashMap::new(),
        };

        // Pass 1: index Function nodes by directory for bare-name fallback.
        for (node_type, node_data) in graph.iter_all_nodes() {
            if !node_data.file.ends_with(".rb") {
                continue;
            }
            if *node_type != NodeType::Function {
                continue;
            }
            let dir = parent_dir(&node_data.file);
            reg.dir_fns
                .entry(dir)
                .or_default()
                .entry(node_data.name.clone())
                .or_insert_with(|| NodeKeys::from(node_data));
        }

        // Pass 2: pre-resolve all call sites per file.
        // Skip spec/test files: the default resolver handles those and must
        // preserve class_call edges that test annotations rely on.
        let all_resolved: Vec<((String, usize, usize), NodeKeys)> = filez
            .iter()
            .filter(|(f, _)| {
                f.ends_with(".rb")
                    && !f.contains("/spec/")
                    && !f.contains("/test/")
                    && !f.ends_with("_spec.rb")
                    && !f.ends_with("_test.rb")
            })
            .flat_map(|(file, source)| {
                ruby_resolver::resolve_file_calls(source, file, &reg.dir_fns, graph)
                    .into_iter()
                    .map(|((row, col), nk)| ((file.clone(), row, col), nk))
            })
            .collect();
        reg.resolved.extend(all_resolved);

        reg
    }
}

impl Registry for RubyRegistry {
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
