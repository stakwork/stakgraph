use crate::lang::call_finder::IMPORT_CACHE;
use crate::lang::graphs::{Graph, NodeType};
use std::collections::HashMap;
use super::Registry;

pub struct TypeScriptRegistry {
    var_types: HashMap<(String, String), String>,
    type_defs: HashMap<String, String>,
    return_types: HashMap<(String, String), String>,
    import_sources: HashMap<(String, String), String>,
}

impl TypeScriptRegistry {
    pub fn new(graph: &impl Graph) -> Self {
        let mut reg = TypeScriptRegistry {
            var_types: HashMap::new(),
            type_defs: HashMap::new(),
            return_types: HashMap::new(),
            import_sources: HashMap::new(),
        };

        for (node_type, node_data) in graph.iter_all_nodes() {
            let file = &node_data.file;
            if !file.ends_with(".ts") && !file.ends_with(".tsx") {
                continue;
            }
            match node_type {
                NodeType::Function => {
                    if let Some(ret) = node_data.meta.get("return_type") {
                        reg.return_types.insert(
                            (file.clone(), node_data.name.clone()),
                            ret.clone(),
                        );
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

        for entry in IMPORT_CACHE.iter() {
            let caller_file = entry.key().clone();
            if !caller_file.ends_with(".ts") && !caller_file.ends_with(".tsx") {
                continue;
            }
            if let Some(imports) = entry.value() {
                for (source_file, names) in imports {
                    for name in names {
                        reg.import_sources.insert(
                            (caller_file.clone(), name.clone()),
                            source_file.clone(),
                        );
                    }
                }
            }
        }

        reg
    }
}

impl Registry for TypeScriptRegistry {
    fn resolve_type(&self, file: &str, var_name: &str) -> Option<&str> {
        self.var_types
            .get(&(file.to_string(), var_name.to_string()))
            .map(|s| s.as_str())
    }

    fn resolve_method(&self, type_name: &str, _method_name: &str) -> Option<&str> {
        self.type_defs.get(type_name).map(|s| s.as_str())
    }
}
