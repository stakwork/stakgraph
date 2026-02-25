use super::super::*;
use super::consts::*;
use shared::error::{Context, Result};
use tree_sitter::{Language, Node as TreeNode, Parser, Query, Tree};

pub struct Kotlin(Language);

impl Default for Kotlin {
    fn default() -> Self {
        Self::new()
    }
}

impl Kotlin {
    pub fn new() -> Self {
        Kotlin(tree_sitter_kotlin_sg::LANGUAGE.into())
    }
}

impl Stack for Kotlin {
    fn should_skip_function_call(&self, called: &str, operand: &Option<String>) -> bool {
        super::skips::java::should_skip(called, operand)
    }
    fn identifier_query(&self) -> String {
        "(simple_identifier) @identifier\n(identifier) @identifier".to_string()
    }

    fn q(&self, q: &str, _nt: &NodeType) -> Query {
        match Query::new(&self.0, q) {
            Ok(query) => query,
            Err(err) => panic!("Failed to compile Kotlin query '{}': {}", q, err),
        }
    }

    fn parse(&self, code: &str, _nt: &NodeType) -> Result<Tree> {
        let mut parser = Parser::new();
        parser.set_language(&self.0)?;

        parser.parse(code, None).context("failed to parse")
    }

    fn lib_query(&self) -> Option<String> {
        Some(format!(
            r#"
            (call_expression
                (simple_identifier) @{LIBRARY_NAME}
            )@{LIBRARY}
            "#
        ))
    }
    fn imports_query(&self) -> Option<String> {
        Some(format!(
            r#"
                (package_header
                    (identifier) 
                )@{IMPORTS}
                (import_list
                    (import_header
                        (identifier) @{IMPORTS_NAME} @{IMPORTS_FROM}
                    )@{IMPORTS}
                )
            "#
        ))
    }

    fn variables_query(&self) -> Option<String> {
        Some(format!(
            r#"
            (source_file
            (property_declaration
                (modifiers)?@modifiers
                    (binding_pattern_kind)
                (variable_declaration
                    (user_type)?@{VARIABLE_TYPE}
                    )@{VARIABLE_NAME}
                (_)?@{VARIABLE_VALUE}
            )@{VARIABLE_DECLARATION}
            )
            "#
        ))
    }

    fn class_definition_query(&self) -> String {
        format!(
            r#"
            (class_declaration
                (modifiers (annotation)*)? @{ATTRIBUTES}
                (type_identifier)@{CLASS_NAME}
                (delegation_specifier
                    (constructor_invocation
                        (user_type)@{CLASS_PARENT}
                    )
                )?
                
            )@{CLASS_DEFINITION}
        "#
        )
    }

    fn function_call_query(&self) -> String {
        format!(
            "
             (call_expression
        	    (simple_identifier) @{FUNCTION_NAME}
             )@{FUNCTION_CALL}
            (call_expression
                (navigation_expression
                    (simple_identifier) @{OPERAND}
                    (navigation_suffix
                        (simple_identifier) @{FUNCTION_NAME}
                    )
                )
            )@{FUNCTION_CALL}
        "
        )
    }

    //GIVEN
    fn function_definition_query(&self) -> String {
        format!(
            "(
                (class_declaration
                    (type_identifier)? @{PARENT_TYPE}
                    (class_body
                    (function_declaration
                        (modifiers (annotation)*)? @{ATTRIBUTES}
                        (simple_identifier) @{FUNCTION_NAME}
                        (function_value_parameters) @{ARGUMENTS}
                    ) @{FUNCTION_DEFINITION}
                    )
                )
                )

                (
                (function_declaration
                    (modifiers (annotation)*)? @{ATTRIBUTES}
                    (simple_identifier) @{FUNCTION_NAME}
                    (function_value_parameters) @{ARGUMENTS}
                ) @{FUNCTION_DEFINITION}
                )
            "
        )
    }
    fn comment_query(&self) -> Option<String> {
        Some(format!(
            r#"
             [
                (line_comment)+
                (multiline_comment)+
            ] @{FUNCTION_COMMENT}
        "#
        ))
    }

    fn find_function_parent(
        &self,
        node: TreeNode,
        _code: &str,
        file: &str,
        func_name: &str,
        find_class: &dyn Fn(&str) -> Option<(NodeData, NodeType)>,
        parent_type: Option<&str>,
    ) -> Result<Option<Operand>> {
        let Some(parent_type) = parent_type else {
            return Ok(None);
        };
        let nodedata = find_class(parent_type);
        Ok(match nodedata {
            Some((class, source_type)) => Some(Operand {
                source: NodeKeys::new(&class.name, &class.file, class.start),
                target: NodeKeys::new(func_name, file, node.start_position().row),
                source_type,
            }),
            None => None,
        })
    }

    fn request_finder(&self) -> Option<String> {
        Some(format!(
            r#"
        (call_expression
            (navigation_expression
                (call_expression
                    (navigation_expression
                        (call_expression
                            (navigation_expression
                                (call_expression
                                    (navigation_expression
                                        (simple_identifier)  @client_var (#eq? @client_var "Request")
                                        (navigation_suffix
                                            (simple_identifier) @builder_method (#eq? @builder_method "Builder")
                                        ) 
                                    )
                                )
                                (navigation_suffix
                                    (simple_identifier) @url_method
                                )

                            )
                            (call_suffix
                                (value_arguments
                                    (value_argument
                                        [(simple_identifier) @{ENDPOINT}
                                        (string_literal) @{ENDPOINT}]
                                    )
                                )
                            )
                        )
                        (navigation_suffix
                    (simple_identifier) @{REQUEST_CALL} (#match? @{REQUEST_CALL} "^get$|^post$|^put$|^delete$")
                )
                    )
                )
                (navigation_suffix
                    (simple_identifier) @build_method
                )
            )
        ) @{ROUTE}
        
        (function_declaration
            (modifiers
                (annotation
                    (constructor_invocation
                        (user_type
                            (type_identifier) @{REQUEST_CALL}
                        )
                        (value_arguments
                            (value_argument
                                (string_literal
                                    (string_content) @{ENDPOINT}
                                )
                            )
                        )
                    )
                    (#match? @{REQUEST_CALL} "^GET$|^POST$|^PUT$|^DELETE$")
                )
            )
        ) @{ROUTE}
        "#
        ))
    }

    fn add_endpoint_verb(&self, inst: &mut NodeData, call: &Option<String>) -> Option<String> {
        if let Some(c) = call {
            let verb = match c.to_uppercase().as_str() {
                "GET" => "GET",
                "POST" => "POST",
                "PUT" => "PUT",
                "DELETE" => "DELETE",
                _ => "",
            };

            if !verb.is_empty() {
                inst.meta.insert("verb".to_string(), verb.to_string());
            }
        }
        None
    }

    fn data_model_query(&self) -> Option<String> {
        Some(format!(
            "(class_declaration
                (type_identifier) @{STRUCT_NAME}
            ) @{STRUCT}"
        ))
    }

    fn data_model_path_filter(&self) -> Option<String> {
        Some("models".to_string())
    }

    fn data_model_within_query(&self) -> Option<String> {
        Some(format!(
            r#"
                (variable_declaration
                    (simple_identifier) @{STRUCT_NAME} 
                )@{STRUCT}
                (call_expression
                    (simple_identifier) @{STRUCT_NAME}
                )@{STRUCT}
            "#
        ))
    }

    fn test_query(&self) -> Option<String> {
        Some(format!(
            r#"
            (function_declaration
                (modifiers
                    (annotation
                        (constructor_invocation
                            (user_type
                                (type_identifier) @test_annotation
                                (#match? @test_annotation "^Test$")
                            )
                        )
                    )
                )
                (simple_identifier) @{FUNCTION_NAME}
            ) @{FUNCTION_DEFINITION}
            "#
        ))
    }

    fn integration_test_query(&self) -> Option<String> {
        // Same as test_query - will use path-based classification
        Some(format!(
            r#"
            (function_declaration
                (modifiers
                    (annotation
                        (constructor_invocation
                            (user_type
                                (type_identifier) @test_annotation
                                (#match? @test_annotation "^Test$")
                            )
                        )
                    )
                )
                (simple_identifier) @{FUNCTION_NAME}
            ) @{FUNCTION_DEFINITION}
            "#
        ))
    }

    fn is_test_file(&self, path: &str) -> bool {
        let normalized = path.replace("\\", "/");
        normalized.contains("/test/") 
            || normalized.contains("/androidTest/")
            || normalized.ends_with("Test.kt")
            || normalized.ends_with("Tests.kt")
            || normalized.ends_with("_test.kt")
            || normalized.ends_with(".test.kt")
    }

    fn is_test(&self, func_name: &str, _func_file: &str, func_body: &str) -> bool {
        func_name.starts_with("test") 
            || func_body.contains("@Test") 
            || func_body.contains("@org.junit.Test")
    }

    fn tests_are_functions(&self) -> bool {
        true
    }

    fn classify_test(&self, _name: &str, file: &str, _body: &str) -> NodeType {
        let normalized = file.replace("\\", "/");
        
        if normalized.contains("/androidTest/") {
            return NodeType::IntegrationTest;
        }

        if normalized.contains("/test/") {
            return NodeType::UnitTest;
        }

        NodeType::UnitTest
    }

    fn resolve_import_name(&self, import_name: &str) -> String {
        let import_name = import_name.to_string();
        let name = import_name
            .split('.')
            .next_back()
            .unwrap_or(&import_name)
            .to_string();
        name
    }

    fn resolve_import_path(&self, import_path: &str, _current_file: &str) -> String {
        let import_path = import_path.to_string();

        let parts: Vec<&str> = import_path.split('.').collect();
        if parts.len() > 2 {
            parts[..parts.len() - 2].join("/")
        } else {
            import_path
        }
    }
}
