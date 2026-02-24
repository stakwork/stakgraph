use super::super::*;
use super::consts::*;
use shared::error::{Context, Result};
// use lsp::{Cmd as LspCmd, CmdSender, Position, Res as LspRes};
use crate::lang::parse::trim_quotes;
use tree_sitter::{Language, Parser, Query, Tree};

pub struct Cpp(Language);

impl Default for Cpp {
    fn default() -> Self {
        Self::new()
    }
}

impl Cpp {
    pub fn new() -> Self {
        Cpp(tree_sitter_cpp::LANGUAGE.into())
    }

    fn is_cuda_keyword(text: &str) -> bool {
        matches!(
            text,
            "__global__"
                | "__device__"
                | "__host__"
                | "__shared__"
                | "__constant__"
                | "__managed__"
                | "__restrict__"
        )
    }
}
impl Stack for Cpp {
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
                    (storage_class_specifier)? @{ATTRIBUTES}
                    (type_identifier)? @{ATTRIBUTES}
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
            (translation_unit
                (class_specifier
                    name: (type_identifier)@{CLASS_NAME}
                    (base_class_clause
                        (type_identifier)@{CLASS_PARENT}
                    )?
                )@{CLASS_DEFINITION}
            )
            "#
        )
    }
    fn instance_definition_query(&self) -> Option<String> {
        Some(format!(
            r#"
            [
                (declaration
                    type: (type_identifier)? @{CLASS_NAME}
                    declarator: (init_declarator
                        declarator: (identifier) @{INSTANCE_NAME}
                    )  
                )@{INSTANCE}

                (declaration
                        type: (qualified_identifier
                            scope: (namespace_identifier) @scope
                            name : (type_identifier) @included_module
                        ) @{CLASS_NAME}
                        
                        declarator: (identifier) @{INSTANCE_NAME}
                )@{INSTANCE}
            ]
            "#
        ))
    }

    fn function_definition_query(&self) -> String {
        format!(
            r#"
            (
                [
                (class_specifier
                    name:(type_identifier)@{PARENT_TYPE}
                    body: (field_declaration_list
                        (function_definition
                                (storage_class_specifier)? @{ATTRIBUTES}
                                (type_identifier)? @{ATTRIBUTES}
                                type : (_) @{RETURN_TYPES}
                                declarator: (function_declarator
                                    declarator : (field_identifier) @{FUNCTION_NAME}
                                    parameters: (parameter_list
                                        (parameter_declaration)@{ARGUMENTS}
                                    )?
                                )
                        )?@{FUNCTION_DEFINITION}
                    )
                )
                (struct_specifier
                    name: (type_identifier) @{PARENT_TYPE}
                    body: (field_declaration_list
                        (function_definition
                                (storage_class_specifier)? @{ATTRIBUTES}
                                (type_identifier)? @{ATTRIBUTES}
                                type : (_) @{RETURN_TYPES}
                                declarator: (function_declarator
                                    declarator : (field_identifier) @{FUNCTION_NAME}
                                    parameters: (parameter_list
                                        (parameter_declaration)@{ARGUMENTS}
                                    )?
                                )
                        )?@{FUNCTION_DEFINITION}
                    )
                )?
                (function_definition
                                (storage_class_specifier)? @{ATTRIBUTES}
                                (type_identifier)? @{ATTRIBUTES}
                                type : (_) @{RETURN_TYPES}
                                declarator: (function_declarator
                                    declarator : (identifier) @{FUNCTION_NAME}
                                    parameters: (parameter_list
                                        (parameter_declaration)@{ARGUMENTS}
                                    )?
                                )
                )@{FUNCTION_DEFINITION}
                (function_definition
                                (storage_class_specifier)? @{ATTRIBUTES}
                                (type_identifier)? @{ATTRIBUTES}
                                type : (_) @{RETURN_TYPES}
                                declarator: (function_declarator
                                    declarator : (qualified_identifier
                                        name: (identifier) @{FUNCTION_NAME}
                                    )
                                    parameters: (parameter_list
                                        (parameter_declaration)@{ARGUMENTS}
                                    )?
                                )
                )@{FUNCTION_DEFINITION}
                ]
            )
            "#
        )
    }

    fn comment_query(&self) -> Option<String> {
        Some(format!(r#"(comment) @{FUNCTION_COMMENT}"#))
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
                        function: (qualified_identifier
                                scope: (namespace_identifier) @namespace
                                name: (identifier)@{FUNCTION_NAME}
                        )?
                        arguments: (argument_list)?@{ARGUMENTS}
                    )
                )@{FUNCTION_CALL}
            ]
            )
            "#
        )
    }

    fn data_model_query(&self) -> Option<String> {
        Some(format!(
            r#"
            (struct_specifier
                name: (type_identifier)@{STRUCT_NAME}
                body: (_)
            )@{STRUCT}
                
            "#
        ))
    }

    fn is_test_file(&self, filename: &str) -> bool {
        let f = filename.replace('\\', "/").to_lowercase();
        let name = f.rsplit('/').next().unwrap_or(&f);

        f.contains("/test/")
            || f.contains("/tests/")
            || f.contains("/integration/")
            || f.contains("/e2e/")
            || name.ends_with("_test.cpp")
            || name.ends_with("_test.cc")
            || name.ends_with("_test.cxx")
            || name.ends_with(".test.cpp")
            || name.ends_with(".spec.cpp")
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

    fn endpoint_finders(&self) -> Vec<String> {
        vec![format!(
            r#"[
                    (expression_statement
                        (call_expression
                        function: (call_expression
                            function: (identifier) @route_macro (#match? @route_macro "^CROW_(ROUTE|WEBSOCKET_ROUTE|BP_ROUTE)$")
                            arguments: (argument_list
                                (identifier) @{OPERAND}
                            (string_literal) @{ENDPOINT}
                            )
                        ) 
                         arguments: (argument_list
                            (lambda_expression
                                body: (compound_statement
                                    (return_statement
                                        (call_expression
                                            function: (identifier) @{HANDLER}
                                        )
                                    )
                                )
                            )
                        )
                    )@{ROUTE}
                    )

                    (call_expression
                    function: (call_expression
                        (field_expression
                            argument: (call_expression
                                function: (identifier) @route_macro (#match? @route_macro "^CROW_(ROUTE|WEBSOCKET_ROUTE|BP_ROUTE)$")
                                arguments: (argument_list
                                    (identifier) @{OPERAND}
                                    (string_literal) @{ENDPOINT}
                                )
                            ) 
                        )
                        arguments: (argument_list
                            (user_defined_literal)@{ENDPOINT_VERB}
                        )

                    )
                    arguments: (argument_list
                        (lambda_expression
                            body: (compound_statement
                                (return_statement
                                    (call_expression
                                        function: (identifier) @{HANDLER}
                                    )
                                )
                            )
                        )
                    )
                    )@{ROUTE}

                    (expression_statement
                        (call_expression
                            function: (call_expression
                                function: (identifier) @route_macro (#match? @route_macro "^CROW_(ROUTE|WEBSOCKET_ROUTE|BP_ROUTE)$")
                                arguments: (argument_list
                                    (identifier) @{OPERAND}
                                    (string_literal) @{ENDPOINT}
                                )
                            )
                            arguments: (argument_list
                                (lambda_expression) @{ANONYMOUS_FUNCTION}
                            )
                        ) @{ROUTE}
                    )

                    (call_expression
                        function: (call_expression
                            function: (field_expression
                                argument: (call_expression
                                    function: (identifier) @route_macro (#match? @route_macro "^CROW_(ROUTE|WEBSOCKET_ROUTE|BP_ROUTE)$")
                                    arguments: (argument_list
                                        (identifier) @{OPERAND}
                                        (string_literal) @{ENDPOINT}
                                    )
                                )
                                field: (field_identifier) @method_name (#match? @method_name "methods")
                            )
                            arguments: (argument_list
                                (user_defined_literal) @{ENDPOINT_VERB}
                            )
                        )
                        arguments: (argument_list
                            (lambda_expression) @{ANONYMOUS_FUNCTION}
                        )
                    ) @{ROUTE}
                    ]
                "#
        )]
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
            .trim_start_matches('_')
            .to_string();

        Some(format!("{}_{}_lambda_L{}", clean_method, clean_path, line))
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
            nd.add_verb("ANY");
        }
    }

    fn filter_attribute(&self, attr: &str, _capture_name: &str) -> Option<String> {
        if Self::is_cuda_keyword(attr) {
            return Some(attr.to_string());
        }

        if matches!(
            attr,
            "static" | "extern" | "const" | "volatile" | "mutable" | "thread_local"
        ) {
            return Some(attr.to_string());
        }

        None
    }
}
