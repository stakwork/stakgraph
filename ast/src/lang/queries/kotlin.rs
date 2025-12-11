use super::super::*;
use super::consts::*;
use shared::error::{Context, Result};
use tree_sitter::{Language, Node as TreeNode, Parser, Query, Tree};

pub struct Kotlin(Language);

impl Kotlin {
    pub fn new() -> Self {
        Kotlin(tree_sitter_kotlin_sg::LANGUAGE.into())
    }
}

impl Stack for Kotlin {
    fn identifier_query(&self) -> String {
        "(simple_identifier) @identifier\n(identifier) @identifier".to_string()
    }

    fn q(&self, q: &str, _nt: &NodeType) -> Query {
        Query::new(&self.0, q).unwrap()
    }

    fn parse(&self, code: &str, _nt: &NodeType) -> Result<Tree> {
        let mut parser = Parser::new();
        parser.set_language(&self.0)?;

        Ok(parser.parse(code, None).context("failed to parse")?)
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
        find_class: &dyn Fn(&str) -> Option<NodeData>,
        parent_type: Option<&str>,
    ) -> Result<Option<Operand>> {
        if parent_type.is_none() {
            return Ok(None);
        }
        let parent_type = parent_type.unwrap();
        let nodedata = find_class(parent_type);
        Ok(match nodedata {
            Some(class) => Some(Operand {
                source: NodeKeys::new(&class.name, &class.file, class.start),
                target: NodeKeys::new(func_name, file, node.start_position().row),
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
        "#
        ))
    }

    fn add_endpoint_verb(&self, inst: &mut NodeData, call: &Option<String>) -> Option<String> {
        if let Some(c) = call {
            let verb = match c.as_str() {
                "get" => "GET",
                "post" => "POST",
                "put" => "PUT",
                "delete" => "DELETE",
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
        Some("app/models".to_string())
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

    fn is_test(&self, func_name: &str, _func_file: &str, _func_body: &str) -> bool {
        func_name.starts_with("test")
    }

    fn resolve_import_name(&self, import_name: &str) -> String {
        let import_name = import_name.to_string();
        let name = import_name
            .split('.')
            .last()
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
