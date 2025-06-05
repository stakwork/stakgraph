use super::super::*;
use super::consts::*;
use anyhow::{Context, Result};
use toml::Toml;
use tree_sitter::{Language, Parser, Query, Tree};
pub struct Rust(Language);

impl Rust {
    pub fn new() -> Self {
        Rust(tree_sitter_rust::LANGUAGE.into())
    }
}

impl Stack for Rust {
    fn q(&self, q: &str, nt: &NodeType) -> Query {
        if matches!(nt, NodeType::Library) {
            Query::new(&tree_sitter_toml_ng::LANGUAGE.into(), q).unwrap()
        } else {
            Query::new(&self.0, q).unwrap()
        }
    }

    fn parse(&self, code: &str, nt: &NodeType) -> Result<Tree> {
        let mut parser = Parser::new();

        if matches!(nt, NodeType::Library) {
            parser.set_language(&tree_sitter_toml_ng::LANGUAGE.into())?;
        } else {
            parser.set_language(&self.0)?;
        }

        Ok(parser.parse(code, None).context("failed to parse")?)
    }

    fn lib_query(&self) -> Option<String> {
        Toml::new().lib_query()
    }

    fn imports_query(&self) -> Option<String> {
        Some(format!(
            r#"
           (use_declaration 
                    argument: (scoped_use_list
                        path:(scoped_identifier
                            path: (crate)
                            name: (identifier)@{IMPORTS_FROM}
                        )?
                        name: (identifier)? @{IMPORTS_NAME}
                        path: (crate)?
                        list: (use_list
                                (identifier)?@{IMPORTS_NAME}
                                (scoped_identifier
                                    path: (identifier) @{IMPORTS_FROM}
                                    name: (identifier) @{IMPORTS_NAME}
                                )?
                        )?
                    )?
                argument:(scoped_identifier
                        path: (super)?@{IMPORTS_FROM}
                        path: (scoped_identifier
                            path:(_)
                            name:(identifier)@{IMPORTS_FROM}
                        )?
                        name: (identifier) @{IMPORTS_NAME}
                        
                        
                    )?
                )@{IMPORTS}

        "#
        ))
    }
    fn variables_query(&self) -> Option<String> {
        Some(format!(
            r#"
                (source_file
                    (const_item
                        name: (identifier) @{VARIABLE_NAME}
                        type: (_)? @{VARIABLE_TYPE}
                        value: (_) @{VARIABLE_VALUE}
                    )?@{VARIABLE_DECLARATION}
                    (static_item
                        name: (identifier) @{VARIABLE_NAME}
                        type: (_)? @{VARIABLE_TYPE}
                        value: (_) @{VARIABLE_VALUE}
                    )?@{VARIABLE_DECLARATION}
                    (let_declaration
                        pattern: (identifier) @{VARIABLE_NAME}
                        type: (_)? @{VARIABLE_TYPE}
                        value: (_) @{VARIABLE_VALUE}
                    )?@{VARIABLE_DECLARATION}
                )
                "#
        ))
    }
    fn trait_query(&self) -> Option<String> {
        Some(
            r#"
            (trait_item
                name: (type_identifier) @trait-name
                body: (declaration_list)
            ) @trait
            "#
            .to_string(),
        )
    }

    fn class_definition_query(&self) -> String {
        format!(
            r#"
           (struct_item
                name: (type_identifier) @class-name
            ) @class-definition
            "#
        )
    }

    fn function_definition_query(&self) -> String {
        format!(
            r#"
            (function_item
              name: (identifier) @{FUNCTION_NAME}
              parameters: (parameters) @{ARGUMENTS}
              return_type: (type_identifier)? @{RETURN_TYPES}
              body: (block)? @function.body) @{FUNCTION_DEFINITION}
              
            (function_signature_item
              name: (identifier) @{FUNCTION_NAME}
              parameters: (parameters) @{ARGUMENTS}
              return_type: (type_identifier)? @{RETURN_TYPES}) @{FUNCTION_DEFINITION}
            
            (impl_item
              type: (_) @{PARENT_TYPE}
              body: (declaration_list
                (function_item
                  name: (identifier) @{FUNCTION_NAME}
                  parameters: (parameters) @{ARGUMENTS}
                  return_type: (type_identifier)? @{RETURN_TYPES}
                  body: (block)? @method.body) @method)) @impl
            "#
        )
    }
    fn function_call_query(&self) -> String {
        format!(
            r#"
                (call_expression
                    function: [
                        (identifier) @FUNCTION_NAME
                        ;; module method
                        (scoped_identifier
                            path: (identifier) @PARENT_NAME
                            name: (identifier) @FUNCTION_NAME
                        )
                        ;; chained call
                        (field_expression
                            field: (field_identifier) @FUNCTION_NAME
                        )
                    ]
                    arguments: (arguments) @ARGUMENTS
                ) @FUNCTION_CALL
                "#
        )
    }

    fn endpoint_finders(&self) -> Vec<String> {
        vec![
            format!(
                r#"
                (call_expression
                    (arguments
                        (string_literal) @endpoint
                        (call_expression
                            function: (identifier) @verb (#match? @verb "^get$|^post$|^put$|^delete$")
                            arguments: (arguments
                                (identifier) @handler
                            )
                        )
                    )
                ) @route
                "#
            ),
            // Method-specific routes (.get("/path", handler))
            format!(
                r#"
                (call_expression
                    function: (field_expression
                        field: (field_identifier) @http_method (#match? @http_method "^get$|^post$|^put$|^delete$")
                    )
                    arguments: (arguments
                        (string_literal) @endpoint
                        (identifier) @handler
                    )
                ) @direct_method_route
        "#
            ),
            // Nested routes (.nest("/base", Router...))
            format!(
                r#"
                (call_expression
                    function: (field_expression
                        field: (field_identifier) @nest_method (#eq? @nest_method "nest")
                    )
                    arguments: (arguments
                        (string_literal) @base_path
                        (_) @nested_router
                    )
                ) @nested_route
                "#
            ),
            // Actix endpoint finder (#[get("/path")])
            format!(
                r#"
                 (
                    (attribute_item
                        (attribute
                        (identifier) @http_method (#match? @http_method "^get$|^post$|^put$|^delete$")
                        arguments: (token_tree
                            (string_literal) @endpoint . (_)*
                        )
                        )
                    )
                    .
                    (function_item
                        name: (identifier) @handler
                    )
                ) @route_with_handler
            "#,
            ),
        ]
    }

    fn data_model_query(&self) -> Option<String> {
        Some(format!(
            r#"
                (struct_item
                    name: (type_identifier) @struct-name
                ) @struct
            "#
        ))
    }
    fn data_model_within_query(&self) -> Option<String> {
        Some(format!(
            r#"
                (type_identifier) @struct-name
            "#,
        ))
    }

    fn add_endpoint_verb(&self, endpoint: &mut NodeData, call: &Option<String>) {
        if let Some(verb) = endpoint.meta.remove("http_method") {
            endpoint.add_verb(&verb);
            return;
        }

        if let Some(call_text) = call {
            if call_text.contains(".get(") || call_text.contains("get(") {
                endpoint.add_verb("GET");
                return;
            } else if call_text.contains(".post(") || call_text.contains("post(") {
                endpoint.add_verb("POST");
                return;
            } else if call_text.contains(".put(") || call_text.contains("put(") {
                endpoint.add_verb("PUT");
                return;
            } else if call_text.contains(".delete(") || call_text.contains("delete(") {
                endpoint.add_verb("DELETE");
                return;
            }
        }

        if let Some(handler) = endpoint.meta.get("handler") {
            let handler_lower = handler.to_lowercase();
            if handler_lower.starts_with("get_") {
                endpoint.add_verb("GET");
            } else if handler_lower.starts_with("post_") || handler_lower.starts_with("create_") {
                endpoint.add_verb("POST");
            } else if handler_lower.starts_with("put_") || handler_lower.starts_with("update_") {
                endpoint.add_verb("PUT");
            } else if handler_lower.starts_with("delete_") || handler_lower.starts_with("remove_") {
                endpoint.add_verb("DELETE");
            }
        }

        // Default to GET if no verb is found
        if !endpoint.meta.contains_key("verb") {
            println!(
                "WARNING: No verb detected for endpoint {}. Using GET as default.",
                endpoint.name
            );
            endpoint.add_verb("GET");
        }
    }

    fn clean_graph(&self, callback: &mut dyn FnMut(NodeType, NodeType, &str)) {
        callback(NodeType::Class, NodeType::Function, "operand");
    }
    fn resolve_import_path(&self, import_path: &str, _current_file: &str) -> String {
        let mut path = import_path.to_string();
        path = path.replace("::", "/");
        path
    }
}
