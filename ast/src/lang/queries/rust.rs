use super::super::*;
use super::consts::*;
use shared::error::{Context, Result};
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
        Some(format!(
            r#"(document
          (table 
            (bare_key) @section (#eq? @section "dependencies")
            (pair 
              (bare_key) @{LIBRARY_NAME} 
              (#not-eq? @{LIBRARY_NAME} "version")
              [
                ; Simple version string: package = "1.0.0"
                (string) @{LIBRARY_VERSION}
                
                ; Inline table with version
                (inline_table
                  (pair
                    (bare_key) @version_key (#eq? @version_key "version")
                    (string) @{LIBRARY_VERSION}
                  )
                )
              ]
            ) @{LIBRARY}
          )
        )"#
        ))
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
           [
                (struct_item
                    name: (type_identifier) @class-name
                )
                (enum_item
                    name: (type_identifier) @class-name
                )
            ]@class-definition
            "#
        )
    }

    fn implements_query(&self) -> Option<String> {
        Some(
            r#"
        (impl_item
            trait: (type_identifier)? @trait-name
            type: (type_identifier) @class-name
            body: (declaration_list)?
        ) @implements
        "#
            .to_string(),
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
    fn comment_query(&self) -> Option<String> {
        Some(format!(
            r#"
            [
              (line_comment)+
              (block_comment)+
            ] @{FUNCTION_COMMENT}
        "#
        ))
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
            // Actix/Rocket endpoint finder (#[get("/path")] or #[post("/path", data = "...")])
            format!(
                r#"
                 (
                    (attribute_item
                        (attribute
                        (identifier) @http_method (#match? @http_method "^get$|^post$|^put$|^delete$")
                        arguments: (token_tree
                            (string_literal) @endpoint (#match? @endpoint "^\"\/")
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
                [
                    (struct_item
                        name: (type_identifier) @struct-name
                    )
                    (enum_item
                        name: (type_identifier) @struct-name
                    )
                ]@struct
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

    fn test_query(&self) -> Option<String> {
        Some(format!(
            r#"
            (
                (attribute_item
                    (attribute
                        [
                            (identifier) @test_attr (#match? @test_attr "^(test|bench|rstest|proptest|quickcheck|wasm_bindgen_test)$")
                            (scoped_identifier
                                name: (identifier) @test_method (#match? @test_method "^(test|rstest|quickcheck)$")
                            )
                        ]
                    )
                )
                .
                (function_item
                    name: (identifier) @{FUNCTION_NAME}
                    parameters: (parameters) @{ARGUMENTS}
                    return_type: (_)? @{RETURN_TYPES}
                    body: (block)? @function.body
                ) @{FUNCTION_DEFINITION}
            )
            "#
        ))
    }

    fn add_endpoint_verb(&self, endpoint: &mut NodeData, call: &Option<String>) -> Option<String> {
        if let Some(verb) = endpoint.meta.remove("http_method") {
            endpoint.add_verb(&verb);
            return None;
        }

        if let Some(call_text) = call {
            if call_text.contains(".get(") || call_text.contains("get(") {
                endpoint.add_verb("GET");
                return None;
            } else if call_text.contains(".post(") || call_text.contains("post(") {
                endpoint.add_verb("POST");
                return None;
            } else if call_text.contains(".put(") || call_text.contains("put(") {
                endpoint.add_verb("PUT");
                return None;
            } else if call_text.contains(".delete(") || call_text.contains("delete(") {
                endpoint.add_verb("DELETE");
                return None;
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
        None
    }

    fn resolve_import_path(&self, import_path: &str, _current_file: &str) -> String {
        let mut path = import_path.to_string();
        path = path.replace("::", "/");
        path
    }
    fn filter_by_implements(&self) -> bool {
        true
    }

    fn is_test_file(&self, filename: &str) -> bool {
        // Simplified: only used for test classification, not identification
        let normalized = filename.replace('\\', "/");
        normalized.contains("/tests/") || normalized.contains("/benches/")
    }

    fn is_test(&self, func_name: &str, func_file: &str) -> bool {
        let Ok(code) = std::fs::read_to_string(func_file) else {
            return false;
        };
        
        let test_patterns = [
            format!("#[test"),
            format!("#[tokio::test"),
            format!("#[actix_rt::test"),
            format!("#[actix_web::test"),
            format!("#[rstest"),
            format!("#[rstest("),
            format!("#[proptest"),
            format!("#[quickcheck"),
            format!("#[wasm_bindgen_test"),
            format!("#[bench"),
        ];
        
        let fn_pattern = format!("fn {}(", func_name);
        if let Some(fn_pos) = code.find(&fn_pattern) {
            // Get code before function (up to 100 chars back to catch attributes)
            let start = fn_pos.saturating_sub(100);
            let context = &code[start..fn_pos];

            for pattern in &test_patterns {
                if context.contains(pattern) {
                    return true;
                }
            }
        }
        
        false
    }

    fn classify_test(&self, name: &str, file: &str, body: &str) -> NodeType {
        let f = file.replace('\\', "/");
        let fname = f.rsplit('/').next().unwrap_or(&f).to_lowercase();
        let name_lower = name.to_lowercase();
        
        if f.contains("/tests/e2e/")
            || f.contains("/e2e/")
            || fname.starts_with("e2e_")
            || fname.contains("e2e.rs")
            || name_lower.starts_with("e2e_")
            || name_lower.contains("_e2e_")
            || name_lower.contains("end_to_end")
        {
            return NodeType::E2eTest;
        }
        
        if f.contains("/tests/integration/")
            || fname.starts_with("integration_")
            || fname.contains("integration.rs")
            || name_lower.starts_with("integration_")
            || name_lower.contains("_integration_")
        {
            return NodeType::IntegrationTest;
        }
        
        if f.contains("/tests/") && !f.contains("/src/") {
            return NodeType::IntegrationTest;
        }
        
        let body_l = body.to_lowercase();
        let http_markers = [
            "reqwest::",
            "hyper::client",
            "actix_web::test",
            "rocket::local",
            ".get(",
            ".post(",
            "http://",
            "https://",
        ];
        
        if http_markers.iter().any(|m| body_l.contains(m)) {
            return NodeType::IntegrationTest;
        }
        
        NodeType::UnitTest
    }
}
