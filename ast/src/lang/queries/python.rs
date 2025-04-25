use super::super::*;
use super::consts::*;
use anyhow::{Context, Result};
use tree_sitter::{Language, Node as TreeNode, Parser, Query, Tree};

pub struct Python(Language);

impl Python {
    pub fn new() -> Self {
        Python(tree_sitter_python::LANGUAGE.into())
    }
}

impl Stack for Python {
    fn q(&self, q: &str, nt: &NodeType) -> Query {
        if matches!(nt, NodeType::Library) {
            Query::new(&tree_sitter_bash::LANGUAGE.into(), q).unwrap()
        } else {
            Query::new(&self.0, q).unwrap()
        }
    }
    fn parse(&self, code: &str, nt: &NodeType) -> Result<Tree> {
        let mut parser = Parser::new();
        if matches!(nt, NodeType::Library) {
            parser.set_language(&tree_sitter_bash::LANGUAGE.into())?;
        } else {
            parser.set_language(&self.0)?;
        }
        Ok(parser.parse(code, None).context("failed to parse")?)
    }
    fn lib_query(&self) -> Option<String> {
        Some(format!(
            "(variable_assignment
	name: (variable_name) @{LIBRARY_NAME}
    value: (word) @{LIBRARY_VERSION}
) @{LIBRARY}"
        ))
    }
    fn imports_query(&self) -> Option<String> {
        Some(format!(
            "(module
                [(import_statement)+ (import_from_statement)+] @{IMPORTS}
            )"
        ))
    }
    fn class_definition_query(&self) -> String {
        format!(
            "(class_definition
                name: (identifier) @{CLASS_NAME}
            ) @{CLASS_DEFINITION}"
        )
    }
    // this captures both
    fn function_definition_query(&self) -> String {
        format!(
            "(function_definition
                name: (identifier) @{FUNCTION_NAME}
                parameters: (parameters) @{ARGUMENTS}
            ) @{FUNCTION_DEFINITION}"
        )
    }
    fn find_function_parent(
        &self,
        node: TreeNode,
        code: &str,
        file: &str,
        func_name: &str,
        _callback: &dyn Fn(&str) -> Option<NodeData>,
        _parent_type: Option<&str>,
    ) -> Result<Option<Operand>> {
        let mut parent = node.parent();
        while parent.is_some() && parent.unwrap().kind().to_string() != "class_definition" {
            parent = parent.unwrap().parent();
        }
        let parent_of = match parent {
            Some(p) => {
                let query = self.q(&self.identifier_query(), &NodeType::Class);
                match query_to_ident(query, p, code)? {
                    Some(parent_name) => Some(Operand {
                        source: NodeKeys::new(&parent_name, file, p.start_position().row),
                        target: NodeKeys::new(func_name, file, node.start_position().row),
                    }),
                    None => None,
                }
            }
            None => None,
        };
        Ok(parent_of)
    }
    fn function_call_query(&self) -> String {
        format!(
            "(call
                function: [
                    (identifier) @{FUNCTION_NAME}
                    (attribute object: (identifier) @{OPERAND}
                        attribute: (identifier) @{FUNCTION_NAME}
                    )
                ]
            ) @{FUNCTION_CALL}"
        )
    }

    fn endpoint_finders(&self) -> Vec<String> {
        vec![
            format!(
                r#"(decorated_definition
                    (decorator
                        (call
                            function: (attribute
                                attribute: (identifier) @{ENDPOINT_VERB} (#match? @{ENDPOINT_VERB} "^get$|^post$|^put$|^delete$")
                            )
                            arguments: (argument_list
                                (string) @{ENDPOINT}
                            )
                    )
                    ) @{ROUTE}
                    definition: (function_definition
                        name: (identifier) @{HANDLER}
                    )
                )"#
            ),
            //Flask style
            format!(
                r#"(decorated_definition
                    (decorator
                        (call
                            function: (attribute
                                object: (_)
                                attribute: (identifier) @route_func (#eq? @route_func "route")
                            )
                            arguments: (argument_list
                                (string) @{ENDPOINT}
                                .
                                (keyword_argument
                                    name: (identifier) @method_kw (#eq? @method_kw "methods")
                                    value: (list
                                        (string) @{ENDPOINT_VERB}
                                    )
                                )?
                            )
                        )
                    ) @{ROUTE}
                    definition: (function_definition
                        name: (identifier) @{HANDLER}
                    )
                )"#
            ),
            format!(
                r#"(assignment
                    (identifier) @url_patterns (#match? @url_patterns "^urlpatterns$")
                    (list
                        (call
                            (identifier) @path_func (#match? @path_func "^path$|^re_path$")
                            (argument_list
                                (string) @{ENDPOINT}
                                .
                                [
                                    (attribute
                                        (identifier)
                                        (identifier) @{HANDLER}
                                    )
                                    (identifier) @{HANDLER}
                                ]
                                .
                                (keyword_argument
                                    (identifier) @name_kw (#eq? @name_kw "name")
                                    (string) @route_name
                                )?
                            )
                        ) @{ROUTE}
                    )
                )"#
            ),
        ]
    }

    fn add_endpoint_verb(&self, nd: &mut NodeData, call: &Option<String>) {
        if nd.meta.get("verb").is_some() {
            return;
        }

        if let Some(c) = call {
            let verb = c.to_uppercase();
            if !verb.is_empty() {
                nd.add_verb(&verb);
                return;
            }
        }

        if let Some(handler) = nd.meta.get("handler").cloned() {
            let method_name = if handler.contains('.') {
                handler.split('.').last().unwrap_or(&handler)
            } else {
                &handler
            };

            let verb = if method_name.starts_with("get_") || method_name == "index" {
                "GET"
            } else if method_name.starts_with("post_") || method_name.starts_with("create_") {
                "POST"
            } else if method_name.starts_with("delete_") || method_name.starts_with("remove_") {
                "DELETE"
            } else if method_name.starts_with("put_") || method_name.starts_with("update_") {
                "PUT"
            } else {
                // Default to GET if no specific verb can be inferred
                "GET"
            };

            nd.add_verb(verb);
        } else {
            nd.add_verb("GET");
        }
    }
    fn data_model_query(&self) -> Option<String> {
        Some(format!(
            "(class_definition
                name: (identifier) @{STRUCT_NAME}
             ) @{STRUCT}"
        ))
    }

    fn data_model_within_query(&self) -> Option<String> {
        Some(format!(
            r#"[
                (assignment
                    (call
                        function: (identifier) @{STRUCT_NAME} (#match? @{STRUCT_NAME} "^[A-Z].*")
                    )
                )
                (call
                    arguments: (argument_list
                        (identifier) @{STRUCT_NAME} (#match? @{STRUCT_NAME} "^[A-Z].*")
                    )
                )
                (call
                    function: (identifier) @{STRUCT_NAME} (#match? @{STRUCT_NAME} "^[A-Z].*")
                )
                (attribute
                    object: (identifier) @{STRUCT_NAME} (#match? @{STRUCT_NAME} "^[A-Z].*")
                )
            ]"#
        ))
    }

    fn is_test(&self, func_name: &str, _func_file: &str) -> bool {
        func_name.starts_with("test_")
    }
    fn handler_finder(
        &self,
        endpoint: NodeData,
        find_fn: &dyn Fn(&str, &str) -> Option<NodeData>,
        find_fns_in: &dyn Fn(&str) -> Vec<NodeData>,
        _handler_params: HandlerParams,
    ) -> Vec<(NodeData, Option<Edge>)> {
        if let Some(handler) = endpoint.meta.get("handler") {
            if handler.contains('.') {
                //Django Style
                let parts: Vec<&str> = handler.split('.').collect();
                let module_name = parts[0];
                let function_name = parts[1];

                let dir_path = endpoint.file.rsplitn(2, '/').nth(1).unwrap_or("");

                let possible_module_paths = vec![
                    format!("{}/{}.py", dir_path, module_name), // standard module
                    format!("{}/{}/views.py", dir_path, module_name), // app views
                ];

                for module_path in possible_module_paths {
                    if let Some(nd) = find_fn(function_name, &module_path) {
                        let edge = Edge::handler(&endpoint, &nd);
                        return vec![(endpoint, Some(edge))];
                    }
                }
                let py_files = find_fns_in("py");
                for func in py_files {
                    if func.name == function_name {
                        let edge = Edge::handler(&endpoint, &func);
                        return vec![(endpoint, Some(edge))];
                    }
                }
            } else {
                // Flask/FastAPI Style
                if let Some(nd) = find_fn(handler, &endpoint.file) {
                    let edge = Edge::handler(&endpoint, &nd);
                    return vec![(endpoint, Some(edge))];
                }
            }
        }

        vec![(endpoint, None)]
    }

    fn use_handler_finder(&self) -> bool {
        true
    }
}
