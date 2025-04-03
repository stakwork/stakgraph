use super::super::*;
use super::consts::*;
use anyhow::{Context, Result};
use tree_sitter::{Language, Parser, Query, Tree};

pub struct Angular(Language);

impl Angular {
    pub fn new() -> Self {
        Angular(tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into())
    }
}

impl Stack for Angular {
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
            r#"(pair
                key: (string (_) @dependency_type) (#match? @dependency_type "^(dependencies|devDependencies)$")
                value: (object
                    (pair
                    key: (string (_) @{LIBRARY_NAME}) (#match? @{LIBRARY_NAME} "^[@a-zA-Z]")
                    value: (string (_) @{LIBRARY_VERSION}) (#match? @{LIBRARY_VERSION} "^[\\^~]?\\d|\\*")
                    ) @{LIBRARY}
                )
                )"#
        ))
    }

    fn is_lib_file(&self, file_name: &str) -> bool {
        file_name.contains("node_modules/")
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


            (method_definition
                name: (property_identifier) @{FUNCTION_NAME}
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
                (_) @{ARGUMENTS}
            )@{REQUEST_CALL}
            "#
        ))
    }
    fn data_model_query(&self) -> Option<String> {
        Some(format!(
            r#"
                (interface_declaration
                    name: (type_identifier) @{STRUCT_NAME}
                    body: (interface_body) @{STRUCT}
                )

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
