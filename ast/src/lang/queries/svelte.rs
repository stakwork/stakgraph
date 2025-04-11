use super::super::*;
use super::consts::*;
use anyhow::{Context, Result};
use tree_sitter::{Language, Parser, Query, Tree};

pub struct Svelte(Language);

impl Svelte {
    pub fn new() -> Self {
        Svelte(tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into())
    }
}

impl Stack for Svelte {
    fn q(&self, q: &str, _nt: &NodeType) -> Query {
        Query::new(&self.0, q).unwrap()
    }
    fn parse(&self, code: &str, _nt: &NodeType) -> Result<Tree> {
        let mut parser = Parser::new();
        parser.set_language(&self.0)?;
        Ok(parser.parse(code, None).context("failed to parse")?)
    }
    
    fn imports_query(&self) -> Option<String> {
        Some(format!(
            r#"(program
                (import_statement)+ @{IMPORTS}
            )"#,
        ))
    }

    fn class_definition_query(&self) -> String {
        format!(
            r#"
            (class_declaration
                name: (type_identifier) @{CLASS_NAME}
                (class_heritage
                    (implements_clause
                    (type_identifier) @{PARENT_NAME}
                    )?
                )?
            ) @{CLASS_DEFINITION}
            "#
        )
    }

    fn function_definition_query(&self) -> String {
        format!(
            r#"
            (function_declaration
                (identifier) @{FUNCTION_NAME}
                parameters: (formal_parameters)
            )@{FUNCTION_DEFINITION}
            "#
        )
    }

    fn function_call_query(&self) -> String {
        format!(
            r#"
            (call_expression
                function: (identifier) @{FUNCTION_NAME}
                arguments: (arguments) @{ARGUMENTS}
            )@{FUNCTION_CALL}

            (call_expression
            function: (member_expression
                object: (identifier) @{CLASS_NAME}
                property: (property_identifier) @{FUNCTION_NAME}
            )
                arguments: (arguments) @{ARGUMENTS}
            )@{FUNCTION_CALL}
            "#
        )
    }

    fn request_finder(&self) -> Option<String> {
        Some(format!(
            r#"
            (call_expression
                (_) @{ENDPOINT}
                (#match? @{ENDPOINT} "fetch|get|post|put|delete")
            )@{REQUEST_CALL}
            "#
        ))
    }
    fn data_model_query(&self) -> Option<String> {
        Some(format!(
            r#"
                (class_declaration
                    name: (type_identifier) @{STRUCT_NAME}
                 
                )@{STRUCT}

             "#
        ))
    }
    fn data_model_within_query(&self) -> Option<String> {
        Some(format!(
            r#"(
                (type_identifier) @{STRUCT_NAME} (#match? @{STRUCT_NAME} "^[A-Z].*")
            )"#
        ))
    }
}
