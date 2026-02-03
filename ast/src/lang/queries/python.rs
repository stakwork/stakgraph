use super::super::*;
use super::consts::*;
use shared::error::{Context, Result};
use tree_sitter::{Language, Node as TreeNode, Parser, Query, Tree};

pub struct Python(Language);

impl Default for Python {
    fn default() -> Self {
        Self::new()
    }
}

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
        parser.parse(code, None).context("failed to parse")
    }

    fn lib_query(&self) -> Option<String> {
        Some(format!(
            r#"[
            (word) @{LIBRARY_NAME}
                ] @{LIBRARY}
             "#
        ))
    }

    fn imports_query(&self) -> Option<String> {
        Some(format!(
            r#"
             (module
             [
                (import_from_statement
                    module_name: (relative_import)? @{IMPORTS_FROM}
                    module_name: (dotted_name)? @{IMPORTS_FROM}
                    name: (dotted_name) @{IMPORTS_NAME}

                )+
                (import_statement)+
                (future_import_statement)?
            ]@{IMPORTS}
            )
            "#
        ))
    }
    fn variables_query(&self) -> Option<String> {
        Some(format!(
            r#"
            (module
                (expression_statement
                    (assignment
                        left: (identifier) @{VARIABLE_NAME}
                        type: (type)? @{VARIABLE_TYPE}
                        right: (_) @{VARIABLE_VALUE}
                    )
                )@{VARIABLE_DECLARATION}
            )
            "#
        ))
    }
    // To differentiate classes and Data Models, we make sure that every class has at least one method in it's body... does that work?
    fn class_definition_query(&self) -> String {
        format!(
            "
            (
                (decorator)* @{ATTRIBUTES}
                .
                (class_definition
                    name: (identifier) @{CLASS_NAME}
                    superclasses: (argument_list
                        (identifier)? @{CLASS_PARENT}
                    )?
                    body: (block
                        (expression_statement
                            (string) @{CLASS_COMMENT}
                        )?
                        [
                            (function_definition)+
                            (decorated_definition)+
                        ]
                    )
                )@{CLASS_DEFINITION}
            )
            "
        )
    }
    // this captures both
    fn function_definition_query(&self) -> String {
        format!(
            r#"
            (
              (decorator)* @{ATTRIBUTES}
              .
              (function_definition
                name: (identifier) @{FUNCTION_NAME}
                parameters: (parameters) @{ARGUMENTS}
                return_type: (type)? @{RETURN_TYPES}
                body: (block
                    (expression_statement
                        (string) @{FUNCTION_COMMENT}
                    )?
                )?
              ) @{FUNCTION_DEFINITION}
            )
            "#
        )
    }
    fn comment_query(&self) -> Option<String> {
        Some(format!(r#"(comment)+ @{FUNCTION_COMMENT}"#))
    }
    fn endpoint_comment_query(&self) -> Option<String> {
        Some(format!(r#"(comment)+ @{ENDPOINT_COMMENT}"#))
    }
    fn var_comment_query(&self) -> Option<String> {
        Some(format!(r#"(comment)+ @{VAR_COMMENT}"#))
    }
    fn test_query(&self) -> Option<String> {
        Some(format!(
            r#"[
            (function_definition
                name: (identifier) @{FUNCTION_NAME}
                (#match? @{FUNCTION_NAME} "^test_")
            ) @{FUNCTION_DEFINITION}

            (class_definition
                body: (block
                    (function_definition
                        name: (identifier) @{FUNCTION_NAME}
                        (#match? @{FUNCTION_NAME} "^test_")
                    ) @{FUNCTION_DEFINITION}
                )
            )
            ]"#
        ))
    }

    fn e2e_test_query(&self) -> Option<String> {
        Some(format!(
            r#"[
            (function_definition
                name: (identifier) @{FUNCTION_NAME}
                parameters: (parameters
                    (typed_parameter
                        type: (type (identifier) @type (#match? @type "Page"))
                    )
                )
            ) @{FUNCTION_DEFINITION}
            ]"#
        ))
    }

    fn classify_test(&self, _name: &str, file: &str, body: &str) -> NodeType {
        let f = file.replace('\\', "/").to_lowercase();
        let b = body.to_lowercase();

        // 1. Path based (strongest signal)
        if f.contains("/tests/e2e/") || f.contains("/test/e2e/") || f.contains("/e2e/") {
            return NodeType::E2eTest;
        }
        if f.contains("/tests/integration/")
            || f.contains("/test/integration/")
            || f.contains("/integration/")
        {
            return NodeType::IntegrationTest;
        }
        if f.contains("/tests/unit/") || f.contains("/test/unit/") || f.contains("/unit/") {
            return NodeType::UnitTest;
        }

        // 2. Content based
        let has_playwright =
            b.contains("playwright") || b.contains("page.goto") || b.contains("expect(page");
        let has_selenium = b.contains("selenium") || b.contains("webdriver");

        if has_playwright || has_selenium {
            return NodeType::E2eTest;
        }

        let has_requests = b.contains("requests.")
            || b.contains("testclient")
            || b.contains("client.get")
            || b.contains("client.post");
        if has_requests {
            return NodeType::IntegrationTest;
        }

        NodeType::UnitTest
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
        while parent.is_some() && parent.unwrap().kind() != "class_definition" {
            parent = parent.unwrap().parent();
        }
        let parent_of = match parent {
            Some(p) => {
                let query = self.q(&self.identifier_query(), &NodeType::Class);
                query_to_ident(query, p, code)?.map(|parent_name| Operand {
                    source: NodeKeys::new(&parent_name, file, p.start_position().row),
                    target: NodeKeys::new(func_name, file, node.start_position().row),
                })
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

    fn add_endpoint_verb(&self, nd: &mut NodeData, call: &Option<String>) -> Option<String> {
        if nd.meta.contains_key("verb") {
            return None;
        }

        if let Some(c) = call {
            let verb = c.to_uppercase();
            if !verb.is_empty() {
                nd.add_verb(&verb);
                return None;
            }
        }

        if let Some(handler) = nd.meta.get("handler").cloned() {
            let method_name = if handler.contains('.') {
                handler.split('.').next_back().unwrap_or(&handler)
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
        None
    }
    // Now we need to exclude class definitinos that have methods in their body
    fn data_model_query(&self) -> Option<String> {
        Some(format!(
            "(class_definition
                name: (identifier) @{STRUCT_NAME}
                body: (block
                    (expression_statement
                        (string) @{STRUCT_COMMENT}
                    )?
                )?
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

    // A trait in python is an ABC class that inherits from ABC
    fn trait_query(&self) -> Option<String> {
        Some(format!(
            r#"
            (class_definition
                name: (identifier)@{TRAIT_NAME}
                superclasses: (argument_list
                    (identifier) @parent (#eq? @parent "ABC")
                )
                body: (block
                    (expression_statement
                        (string) @{TRAIT_COMMENT}
                    )?
                )
            )@{TRAIT}
            "#
        ))
    }

    fn trait_comment_query(&self) -> Option<String> {
        Some(format!(r#"(comment) @{TRAIT_COMMENT}"#))
    }

    fn implements_query(&self) -> Option<String> {
        Some(format!(
            r#"
            (class_definition
                name: (identifier)@{CLASS_NAME}
                superclasses: (argument_list
                    (identifier) @{TRAIT_NAME}
                )
            )@{IMPLEMENTS}  
            "#
        ))
    }

    fn is_test(&self, func_name: &str, file: &str, _func_body: &str) -> bool {
        func_name.starts_with("test_")
            || file.contains("/tests/")
            || file.contains("/test/")
            || file.ends_with("_test.py")
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

                let dir_path = endpoint.file.rsplit_once('/').map(|x| x.0).unwrap_or("");

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

    fn clean_graph(&self, callback: &mut dyn FnMut(NodeType, NodeType, &str)) {
        callback(NodeType::DataModel, NodeType::Class, "deduplicate");
    }
}
