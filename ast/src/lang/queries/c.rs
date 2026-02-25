use super::super::*;
use super::consts::*;
use crate::lang::parse::trim_quotes;
use shared::error::{Context, Result};
use tree_sitter::{Language, Parser, Query, Tree};

pub struct C(Language);

impl Default for C {
    fn default() -> Self {
        Self::new()
    }
}

impl C {
    pub fn new() -> Self {
        C(tree_sitter_c::LANGUAGE.into())
    }
}

impl Stack for C {
    fn q(&self, q: &str, nt: &NodeType) -> Query {
        if matches!(nt, NodeType::Library) {
            match Query::new(&tree_sitter_bash::LANGUAGE.into(), q) {
                Ok(query) => query,
                Err(err) => panic!("Failed to compile C library query '{}': {}", q, err),
            }
        } else {
            match Query::new(&self.0, q) {
                Ok(query) => query,
                Err(err) => panic!("Failed to compile C query '{}': {}", q, err),
            }
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

    fn is_test_file(&self, filename: &str) -> bool {
        let f = filename.replace('\\', "/").to_lowercase();
        let name = f.rsplit('/').next().unwrap_or(&f);

        f.contains("/test/")
            || f.contains("/tests/")
            || f.contains("/integration/")
            || f.contains("/e2e/")
            || name.ends_with("_test.c")
            || name.ends_with(".test.c")
            || name.ends_with(".spec.c")
            || name.starts_with("test_")
    }

    fn is_test(&self, func_name: &str, func_file: &str, _func_body: &str) -> bool {
        let n = func_name.to_lowercase();
        self.is_test_file(func_file)
            || n.starts_with("test_")
            || n.ends_with("_test")
            || n.starts_with("it_")
    }

    fn classify_test(&self, name: &str, file: &str, body: &str) -> NodeType {
        let f = file.replace('\\', "/").to_lowercase();
        let n = name.to_lowercase();
        let b = body.to_lowercase();

        if f.contains("/e2e/")
            || f.contains(".e2e.")
            || n.contains("e2e")
            || b.contains("selenium")
            || b.contains("playwright")
        {
            return NodeType::E2eTest;
        }

        if f.contains("/integration/")
            || f.contains(".integration.")
            || f.contains(".int.")
            || n.contains("integration")
        {
            return NodeType::IntegrationTest;
        }

        NodeType::UnitTest
    }

    fn lib_query(&self) -> Option<String> {
        Some(format!(
            r#"
            (command
                name : (command_name) @command_name (#match? @command_name "^find_package$|^add_library$")
                (subshell
                    (command
                        name: (command_name)@{LIBRARY_NAME}
                        argument: (word)? @{LIBRARY_VERSION}
                    )@{LIBRARY}
                )
            )
            "#
        ))
    }

    fn imports_query(&self) -> Option<String> {
        Some(format!(
            r#"
                (translation_unit
                    (preproc_include
                        path: (_)@{IMPORTS_FROM} @{IMPORTS_NAME}
                    )?@{IMPORTS}
                    (preproc_ifdef
                        name: (identifier) @condition
                        (preproc_include
                            path: (_)@{IMPORTS_FROM} @{IMPORTS_NAME}
                        )@{IMPORTS}
                    )?
                    (declaration
                        type: (type_identifier)
                        declarator : (identifier)@{IMPORTS_FROM} @{IMPORTS_NAME}
                    )?@{IMPORTS}
                )
                "#
        ))
    }

    fn variables_query(&self) -> Option<String> {
        Some(format!(
            r#"
            (translation_unit
                (declaration
                    type: (_) @{VARIABLE_TYPE}
                    declarator : (identifier)? @{VARIABLE_NAME}
                    declarator : (init_declarator
                        declarator : (_)@{VARIABLE_NAME}
                        value: (_)?@{VARIABLE_VALUE}
                    )?
                )@{VARIABLE_DECLARATION}
            )
            "#
        ))
    }

    fn class_definition_query(&self) -> String {
        format!(
            r#"
            [
                (struct_specifier
                    name: (type_identifier) @{CLASS_NAME}
                ) @{CLASS_DEFINITION}

                (union_specifier
                    name: (type_identifier) @{CLASS_NAME}
                ) @{CLASS_DEFINITION}

                (enum_specifier
                    name: (type_identifier) @{CLASS_NAME}
                ) @{CLASS_DEFINITION}

                (type_definition
                    type: [
                        (struct_specifier)
                        (union_specifier)
                        (enum_specifier)
                    ]
                    declarator: (type_identifier) @{CLASS_NAME}
                ) @{CLASS_DEFINITION}

                (enum_specifier
                    name: (type_identifier) @{CLASS_NAME}
                ) @{CLASS_DEFINITION}
            ]
            "#
        )
    }

    fn function_definition_query(&self) -> String {
        format!(
            r#"
            (
                [
                (function_definition
                            type : (_) @{RETURN_TYPES}
                            declarator: (function_declarator
                                declarator : (identifier) @{FUNCTION_NAME}
                                parameters: (parameter_list
                                    (parameter_declaration)@{ARGUMENTS}
                                )?
                            )
                )@{FUNCTION_DEFINITION}
                (function_definition
                            type : (_) @{RETURN_TYPES}
                            declarator: (pointer_declarator
                                declarator: (function_declarator
                                    declarator : (identifier) @{FUNCTION_NAME}
                                    parameters: (parameter_list
                                        (parameter_declaration)@{ARGUMENTS}
                                    )?
                                )
                            )
                )@{FUNCTION_DEFINITION}
                ]
            )
            "#
        )
    }

    fn comment_query(&self) -> Option<String> {
        Some(format!(r#"(comment) @comment"#))
    }

    fn class_comment_query(&self) -> Option<String> {
        Some(format!(r#"(comment) @comment"#))
    }

    fn data_model_comment_query(&self) -> Option<String> {
        Some(format!(r#"(comment) @comment"#))
    }

    fn function_call_query(&self) -> String {
        format!(
            r#"
            (
            [
                (call_expression
                    function : (identifier)@{FUNCTION_NAME}
                    arguments: (argument_list)?@{ARGUMENTS}
                )@{FUNCTION_CALL}
                (expression_statement
                    (call_expression
                        function: (field_expression
                                argument: (identifier)@{OPERAND}
                                field : (field_identifier)@{FUNCTION_NAME}
                            )?
                        arguments: (argument_list)?@{ARGUMENTS}
                    )
                )@{FUNCTION_CALL}
                (call_expression
                    function: (pointer_expression
                        argument: (identifier)@{OPERAND}
                    )
                    arguments: (argument_list)?@{ARGUMENTS}
                )@{FUNCTION_CALL}
            ]
            )
            "#
        )
    }

    fn data_model_query(&self) -> Option<String> {
        Some(format!(
            r#"
            [
                (struct_specifier
                    name: (type_identifier)@{STRUCT_NAME}
                    body: (_)
                )@{STRUCT}
                (enum_specifier
                    name: (type_identifier)@{STRUCT_NAME}
                    body: (_)
                )@{STRUCT}
                (union_specifier
                    name: (type_identifier)@{STRUCT_NAME}
                    body: (_)
                )@{STRUCT}
            ]
            "#
        ))
    }

    fn endpoint_finders(&self) -> Vec<String> {
        vec![
            // libonion pattern
            format!(
                r#"[
                    (expression_statement
                        (call_expression
                            function: (identifier) @route_fn (#match? @route_fn "^onion_url_add(_with_data)?$")
                            arguments: (argument_list
                                (identifier) @{OPERAND}
                                (string_literal) @{ENDPOINT}
                                (identifier) @{HANDLER}
                            )
                        )@{ROUTE}
                    )
                ]"#
            ),
            // libmicrohttpd pattern
            format!(
                r#"(expression_statement
                    (call_expression
                        function: (identifier) @route_fn (#match? @route_fn "^MHD_(add_response_entry|create_response_from_callback)$")
                        arguments: (argument_list
                            (string_literal) @{ENDPOINT}
                            (identifier) @{HANDLER}
                        )
                    )@{ROUTE}
                )"#
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
            .replace("<", "")
            .replace(">", "")
            .trim_start_matches('_')
            .to_string();

        Some(format!(
            "{}_{}_callback_L{}",
            clean_method, clean_path, line
        ))
    }

    fn add_endpoint_verb(&self, nd: &mut NodeData, _call: &Option<String>) -> Option<String> {
        if let Some(verb) = nd.meta.get("verb") {
            nd.meta
                .insert("verb".to_string(), verb.to_uppercase().to_string());
        }
        None
    }

    fn update_endpoint(&self, nd: &mut NodeData, _call: &Option<String>) {
        if let Some(verb_annotation) = nd.meta.get("verb").cloned() {
            let c = verb_annotation.trim();
            let verb = if let Some(stripped) = c.strip_suffix("_METHOD") {
                trim_quotes(stripped).to_uppercase()
            } else {
                trim_quotes(c).to_uppercase()
            };
            if !verb.is_empty() {
                nd.add_verb(&verb);
            }
        } else {
            nd.add_verb("GET");
        }
    }

    fn instance_definition_query(&self) -> Option<String> {
        Some(format!(
            r#"
            (declaration
                type: (type_identifier)? @{CLASS_NAME}
                declarator: (init_declarator
                    declarator: (identifier) @{INSTANCE_NAME}
                )  
            )@{INSTANCE}
            "#
        ))
    }

    fn should_skip_function_call(&self, called: &str, operand: &Option<String>) -> bool {
        super::skips::c::should_skip(called, operand)
    }
}
