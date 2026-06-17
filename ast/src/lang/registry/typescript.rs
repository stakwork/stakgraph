use super::{ts_resolver, Registry};
use crate::lang::asg::NodeKeys;
use crate::lang::call_finder::IMPORT_CACHE;
use crate::lang::graphs::{Graph, NodeType};
use std::collections::HashMap;

pub struct TypeScriptRegistry {
    pub(super) var_types: HashMap<(String, String), String>,
    type_defs: HashMap<String, String>,
    methods: HashMap<(String, String), String>,
    return_types: HashMap<(String, String), String>,
    pub(super) import_sources: HashMap<(String, String), String>,
    pub(super) class_fields: HashMap<String, HashMap<String, String>>,
    resolved: HashMap<(String, usize, usize), NodeKeys>,
}

impl TypeScriptRegistry {
    pub fn new(graph: &impl Graph, filez: &[(String, String)]) -> Self {
        let mut reg = TypeScriptRegistry {
            var_types: HashMap::new(),
            type_defs: HashMap::new(),
            methods: HashMap::new(),
            return_types: HashMap::new(),
            import_sources: HashMap::new(),
            class_fields: HashMap::new(),
            resolved: HashMap::new(),
        };

        // Pass 1: scan graph nodes
        for (node_type, node_data) in graph.iter_all_nodes() {
            let file = &node_data.file;
            if !file.ends_with(".ts") && !file.ends_with(".tsx") {
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
                    if let Some(ret) = node_data.meta.get("return_type") {
                        reg.return_types
                            .insert((file.clone(), node_data.name.clone()), ret.clone());
                    }
                    if let Some(pt_json) = node_data.meta.get("param_types") {
                        if let Ok(params) =
                            serde_json::from_str::<Vec<HashMap<String, String>>>(pt_json)
                        {
                            for param in params {
                                if let (Some(name), Some(type_str)) =
                                    (param.get("name"), param.get("type"))
                                {
                                    reg.var_types.insert(
                                        (file.clone(), name.clone()),
                                        type_str.clone(),
                                    );
                                }
                            }
                        }
                    }
                }
                NodeType::Var | NodeType::Instance => {
                    if let Some(dt) = &node_data.data_type {
                        reg.var_types
                            .insert((file.clone(), node_data.name.clone()), dt.clone());
                    }
                }
                NodeType::Class | NodeType::DataModel | NodeType::Trait => {
                    reg.type_defs.insert(node_data.name.clone(), file.clone());
                }
                _ => {}
            }
        }

        for (file, _) in filez {
            if !file.ends_with(".ts") && !file.ends_with(".tsx") {
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

        // Pass 1.5: extract class field types and top-level new-expression vars from source files
        for (file, source) in filez {
            if !file.ends_with(".ts") && !file.ends_with(".tsx") {
                continue;
            }
            let fields = ts_resolver::extract_class_fields(source);
            for (class_name, field_map) in fields {
                reg.class_fields
                    .entry(class_name)
                    .or_default()
                    .extend(field_map);
            }
            // capture unannotated `const x = new Class()` so cross-file scope seeding works
            let top_vars = ts_resolver::extract_top_level_vars(source);
            for (var_name, type_name) in top_vars {
                reg.var_types
                    .entry((file.clone(), var_name))
                    .or_insert(type_name);
            }
        }

        // Pass 2: pre-resolve call sites per file
        let all_resolved: Vec<((String, usize, usize), NodeKeys)> = filez
            .iter()
            .filter(|(f, _)| f.ends_with(".ts") || f.ends_with(".tsx"))
            .flat_map(|(file, source)| {
                ts_resolver::resolve_file_calls(
                    source,
                    file,
                    &reg.class_fields,
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

impl Registry for TypeScriptRegistry {
    fn resolve_type(&self, file: &str, var_name: &str) -> Option<&str> {
        if let Some(t) = self
            .var_types
            .get(&(file.to_string(), var_name.to_string()))
        {
            return Some(t.as_str());
        }
        // Cross-file: follow import_sources to the defining file
        if let Some(source_file) = self
            .import_sources
            .get(&(file.to_string(), var_name.to_string()))
        {
            for ext in &["", ".ts", ".tsx"] {
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
