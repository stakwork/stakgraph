use super::super::*;
use super::consts::*;
use shared::error::{Context, Result};
use tree_sitter::{Language, Parser, Query, Tree};

pub struct Java(Language);

impl Default for Java {
    fn default() -> Self {
        Self::new()
    }
}

impl Java {
    pub fn new() -> Self {
        Java(tree_sitter_java::LANGUAGE.into())
    }
}

impl Stack for Java {
    fn should_skip_function_call(&self, called: &str, operand: &Option<String>) -> bool {
        super::skips::java::should_skip(called, operand)
    }
    fn q(&self, q: &str, _nt: &NodeType) -> Query {
        Query::new(&self.0, q).unwrap()
    }

    fn parse(&self, code: &str, _nt: &NodeType) -> Result<Tree> {
        let mut parser = Parser::new();
        parser.set_language(&self.0)?;
        parser.parse(code, None).context("failed to parse")
    }

    fn lib_query(&self) -> Option<String> {
        Some(format!(
            r#"
            (package_declaration) @{LIBRARY}"#
        ))
    }

    fn imports_query(&self) -> Option<String> {
        Some(format!(
            r#"
            (package_declaration) @{IMPORTS}
            (import_declaration
                (scoped_identifier) @{IMPORTS_NAME} @{IMPORTS_FROM}
            ) @{IMPORTS}
            "#
        ))
    }
    fn variables_query(&self) -> Option<String> {
        Some(format!(
            r#"
            [
                (field_declaration
                    type: (_) @{VARIABLE_TYPE}
                    declarator: (variable_declarator
                        name: (identifier) @{VARIABLE_NAME}
                        value: (_)? @{VARIABLE_VALUE}
                    )
                ) @{VARIABLE_DECLARATION}

                (local_variable_declaration
                    (modifiers)? @{ATTRIBUTES}
                    type: (_) @{VARIABLE_TYPE}
                    declarator: (variable_declarator
                        name: (identifier) @{VARIABLE_NAME}
                        value: (_)? @{VARIABLE_VALUE}
                    )
                ) @{VARIABLE_DECLARATION}
            ]
            "#
        ))
    }

    fn trait_query(&self) -> Option<String> {
        Some(format!(
            r#"
            (interface_declaration
                name: (identifier) @{TRAIT_NAME}
            ) @{TRAIT}
            "#
        ))
    }

    fn implements_query(&self) -> Option<String> {
        Some(format!(
            r#"
            (class_declaration
                name: (identifier) @{CLASS_NAME}
                (super_interfaces
                    (type_list
                        [
                            (type_identifier) @{TRAIT_NAME}
                            (generic_type
                                (type_identifier) @{TRAIT_NAME}
                            )
                        ]
                    )
                )
            ) @{IMPLEMENTS}
            "#
        ))
    }

    fn class_definition_query(&self) -> String {
        format!(
            r#"
                 (class_declaration
                    (identifier)@{CLASS_NAME}
                    (superclass 
                        (type_identifier)@{CLASS_PARENT}
                    )?
                    (super_interfaces
                        (type_list
                            (type_identifier)@{INCLUDED_MODULES}
                        )
                    )?
                )@{CLASS_DEFINITION}
                "#
        )
    }

    fn instance_definition_query(&self) -> Option<String> {
        Some(format!(
            r#"
            [
            (field_declaration
                type: [
                    (type_identifier) @{CLASS_NAME}
                    (scoped_type_identifier) @{CLASS_NAME}
                    (generic_type
                        (type_identifier) @{CLASS_NAME}
                    )
                ]
                declarator: (variable_declarator
                    name: (identifier) @{INSTANCE_NAME}
                )
            )

            (local_variable_declaration
                type: [
                    (type_identifier) @{CLASS_NAME}
                    (scoped_type_identifier) @{CLASS_NAME}
                    (generic_type
                        (type_identifier) @{CLASS_NAME}
                    )
                ]
                declarator: (variable_declarator
                    name: (identifier) @{INSTANCE_NAME}
                )
            )
            ]@{INSTANCE}
            "#
        ))
    }
    fn function_definition_query(&self) -> String {
        format!(
            r#"
            [
                (method_declaration
                    (modifiers (annotation)*)? @{ATTRIBUTES}
                    type: (_) @{RETURN_TYPES}
                    name: (identifier) @{FUNCTION_NAME}
                    (formal_parameters
                        (formal_parameter)@{ARGUMENTS}
                    )?
                )@{FUNCTION_DEFINITION}

                (constructor_declaration
                    (modifiers (annotation)*)? @{ATTRIBUTES}
                    name: (identifier) @{FUNCTION_NAME}
                    (formal_parameters
                        (formal_parameter) @{ARGUMENTS}
                    )?
                )@{FUNCTION_DEFINITION}
            ]
            "#
        )
    }

    fn comment_query(&self) -> Option<String> {
        Some(format!(
            r#" [
                    (line_comment)+
                    (block_comment)+
                ] @{FUNCTION_COMMENT}
                "#
        ))
    }

    fn function_call_query(&self) -> String {
        format!(
            r#"
                [
                (method_invocation
                    object: (_)? @{OPERAND}
                    name: (identifier) @{FUNCTION_NAME}
                    arguments: (argument_list 
                    (_)* 
                    )@{ARGUMENTS}
                ) @{FUNCTION_CALL}

                (object_creation_expression
                    type: [
                        (type_identifier) @{FUNCTION_NAME}
                        (scoped_type_identifier
                            (type_identifier) @{FUNCTION_NAME}
                        )
                    ]
                    arguments: (argument_list
                        (_)*
                    ) @{ARGUMENTS}
                ) @{FUNCTION_CALL}
                ]
                
                "#
        )
    }

    fn test_query(&self) -> Option<String> {
        Some(format!(
            r#"
            (method_declaration
                (modifiers
                    [
                        (marker_annotation
                            name: (identifier) @test_anno (#eq? @test_anno "Test")
                        )
                        (annotation
                            name: (identifier) @test_anno2 (#eq? @test_anno2 "Test")
                        )
                    ]
                )
                name: (identifier) @{FUNCTION_NAME}
            ) @{FUNCTION_DEFINITION}
            "#
        ))
    }
    fn endpoint_finders(&self) -> Vec<String> {
        vec![
            format!(
                r#"
                (method_declaration
                    (modifiers
                    (annotation
                        name: (identifier) @{ENDPOINT_VERB} (#match? @{ENDPOINT_VERB} "GetMapping|PostMapping|PutMapping|DeleteMapping|RequestMapping|PatchMapping")
                        arguments: (annotation_argument_list 
                        [
                            (string_literal) @{ENDPOINT}
                            (element_value_pair
                                key: (identifier) @path_key (#match? @path_key "^path$|^value$")
                                value: (string_literal) @{ENDPOINT}
                            )
                            (element_value_pair
                                key: (identifier) @method_key (#eq? @method_key "method")
                                value: (field_access
                                    field: (identifier) @{ENDPOINT_VERB}
                                )
                            )
                        ]*
                        )?
                    )
                    )
                    name: (identifier) @{HANDLER}
                ) @{ROUTE}
                "#
            ),
            format!(
                r#"
                (method_invocation
                    name: (identifier) @{ENDPOINT_VERB} (#match? @{ENDPOINT_VERB} "^GET$|^POST$|^PUT$|^DELETE$|^PATCH$")
                    arguments: (argument_list
                        (string_literal) @{ENDPOINT}
                        [
                            (lambda_expression
                                parameters: (_) @{ARGUMENTS}
                                body: (_) @lambda.body
                            ) @{ANONYMOUS_FUNCTION}
                            (method_reference
                                (identifier) @{HANDLER}
                            )
                        ]
                    )
                ) @{ROUTE}
                "#
            ),
        ]
    }

    fn generate_anonymous_handler_name(
        &self,
        method: &str,
        path: &str,
        line: usize,
    ) -> Option<String> {
        let clean_method = method.to_uppercase();
        let clean_path = path
            .replace("/", "_")
            .replace(":", "param_")
            .replace("-", "_")
            .trim_start_matches('_')
            .to_string();

        Some(format!("{}_{}_lambda_L{}", clean_method, clean_path, line))
    }

    fn endpoint_group_find(&self) -> Option<String> {
        Some(format!(
            r#"
            (class_declaration
                (modifiers
                    (annotation
                        name: (identifier) @{ENDPOINT_VERB} (#eq? @{ENDPOINT_VERB} "RequestMapping")
                        arguments: (annotation_argument_list 
                            [
                                (string_literal) @{ENDPOINT}
                                (element_value_pair
                                    key: (identifier) @path_key (#match? @path_key "^path$|^value$")
                                    value: (string_literal) @{ENDPOINT}
                                )
                            ]*
                        )?
                    )
                )
                name: (identifier) @{ENDPOINT_GROUP}
            )@{ROUTE}
            "#
        ))
    }

    fn update_endpoint(&self, nd: &mut NodeData, _call: &Option<String>) {
        if let Some(verb_annotation) = nd.meta.get("verb").cloned() {
            let normalized = verb_annotation.to_uppercase();
            let http_verb = match normalized.as_str() {
                "GETMAPPING" => "GET",
                "POSTMAPPING" => "POST",
                "PUTMAPPING" => "PUT",
                "DELETEMAPPING" => "DELETE",
                "PATCHMAPPING" => "PATCH",
                "GET" => "GET",
                "POST" => "POST",
                "PUT" => "PUT",
                "DELETE" => "DELETE",
                "PATCH" => "PATCH",
                "REQUESTMAPPING" => {
                    if nd.body.contains("RequestMethod.GET") {
                        "GET"
                    } else if nd.body.contains("RequestMethod.POST") {
                        "POST"
                    } else if nd.body.contains("RequestMethod.PUT") {
                        "PUT"
                    } else if nd.body.contains("RequestMethod.DELETE") {
                        "DELETE"
                    } else if nd.body.contains("RequestMethod.PATCH") {
                        "PATCH"
                    } else {
                        "ANY"
                    }
                }
                _ => "GET",
            };

            nd.add_verb(http_verb);
            return;
        }
        //TODO: check for the presence of the verb in the function call
        // if all else fails, default to GET
        nd.add_verb("GET");
    }

    fn is_test_file(&self, file_name: &str) -> bool {
        let normalized = file_name.replace('\\', "/").to_lowercase();
        normalized.contains("/src/test/")
            || normalized.contains("/tests/")
            || normalized.ends_with("test.java")
    }

    fn is_test(&self, func_name: &str, func_file: &str, _func_body: &str) -> bool {
        self.is_test_file(func_file)
            || func_name.starts_with("test")
            || func_name.ends_with("_test")
    }

    fn classify_test(&self, _name: &str, file: &str, _body: &str) -> NodeType {
        let f = file.replace('\\', "/").to_lowercase();
        if f.contains("/integration/") {
            return NodeType::IntegrationTest;
        }
        if f.contains("/e2e/") {
            return NodeType::E2eTest;
        }
        NodeType::UnitTest
    }
    fn data_model_query(&self) -> Option<String> {
        Some(format!(
            r#"
                (class_declaration
                    (modifiers
                        (marker_annotation) @marker (#match? @marker "Entity")
                    )
                name: (identifier) @{STRUCT_NAME}
                ) @{STRUCT}
                (record_declaration
                name: (identifier) @{STRUCT_NAME}
                ) @{STRUCT}
                (class_declaration
                    (modifiers) @modifier (#match? @modifier ".final$")
                name: (identifier) @{STRUCT_NAME}
                ) @{STRUCT}
            "#
        ))
    }
    fn data_model_within_query(&self) -> Option<String> {
        Some(format!(
            r#"
                
            (method_declaration
            
                type: [
                
                (type_identifier) @{STRUCT_NAME}
                
                (generic_type
                    (type_identifier)?
                    (type_arguments
                    (type_identifier) @{STRUCT_NAME}
                    )
                )
                ]
            ) 
            "#
        ))
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
        if parts.len() > 1 {
            parts[..parts.len() - 1].join("/")
        } else {
            import_path
        }
    }
}
