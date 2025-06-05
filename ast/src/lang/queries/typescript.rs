use super::super::*;
use super::consts::*;
use anyhow::{Context, Result};
use tree_sitter::{Language, Parser, Query, Tree};

pub struct TypeScript(Language);

impl TypeScript {
    pub fn new() -> Self {
        TypeScript(tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into())
    }
}

impl Stack for TypeScript {
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
            || file_name.contains("/lib/")
            || file_name.ends_with(".d.ts")
            || file_name.starts_with("/usr")
            || file_name.contains(".nvm/")
    }
    //copied from react
    fn imports_query(&self) -> Option<String> {
        Some(format!(
            r#"
            (import_statement
                (import_clause
                    (identifier)? @{IMPORTS_NAME}
                    (named_imports
                        (import_specifier
                            name:(identifier) @{IMPORTS_NAME}
                        )
                    )?

                )?
                source: (string) @{IMPORTS_FROM}
            )@{IMPORTS}
            (export_statement
                (export_clause
                    (export_specifier
                        name: (identifier)@{IMPORTS_NAME}
                    )
                )
                source: (string) @{IMPORTS_FROM}
            )@{IMPORTS}
            "#,
        ))
    }
    fn variables_query(&self) -> Option<String> {
        let types = "(string)(template_string)(number)(object)(array)(true)(false)(new_expression)";
        Some(format!(
            r#"(program
                    (export_statement
                        (variable_declaration
                            (variable_declarator
                                name: (identifier) @{VARIABLE_NAME}
                                type: (_)? @{VARIABLE_TYPE}
                                value: [{types}]+ @{VARIABLE_VALUE}

                            )
                        )
                    )?@{VARIABLE_DECLARATION}
                )
                (program
                    (export_statement
                        (lexical_declaration
                            (variable_declarator
                                name: (identifier) @{VARIABLE_NAME}
                                type: (_)? @{VARIABLE_TYPE}
                                value: [{types}]+ @{VARIABLE_VALUE}

                            )
                        )
                    )?@{VARIABLE_DECLARATION}
                )
                (program
                        (lexical_declaration
                            (variable_declarator
                                name: (identifier) @{VARIABLE_NAME}
                                type: (_)? @{VARIABLE_TYPE}
                                value: [{types}]+ @{VARIABLE_VALUE}
                            )
                        )@{VARIABLE_DECLARATION}
                    
                )
                (program
                        (variable_declaration
                            (variable_declarator
                                name: (identifier) @{VARIABLE_NAME}
                                type: (_)? @{VARIABLE_TYPE}
                                value: [{types}]+ @{VARIABLE_VALUE}
                            )
                        ) @{VARIABLE_DECLARATION}
                    
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
                name: (identifier) @{FUNCTION_NAME}
            ) @{FUNCTION_DEFINITION}
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

    fn endpoint_finders(&self) -> Vec<String> {
        vec![format!(
            r#"(call_expression
                function: (member_expression
                    object: (identifier)
                    property: (property_identifier) @{ENDPOINT_VERB} (#match? @{ENDPOINT_VERB} "^get$|^post$|^put$|^delete$")
                )
                arguments: (arguments
                    (string) @{ENDPOINT}
                    (identifier) @{HANDLER}
                )
                ) @{ROUTE}
            "#
        )]
    }
    fn add_endpoint_verb(&self, inst: &mut NodeData, call: &Option<String>) {
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
    }

    fn data_model_query(&self) -> Option<String> {
        Some(format!(
            r#"
                ;; Find TypeScript interfaces related to the model
                (interface_declaration
                    name: (type_identifier) @{STRUCT_NAME} 
                    body: (interface_body) @{STRUCT}
                )
                ;; sequelize
                (class_declaration
                    name: (type_identifier) @{STRUCT_NAME}
                    (class_heritage
                        (extends_clause
                            value: (identifier) @model (#eq? @model "Model")
                        )
                    )
                ) @{STRUCT}
                ;; typeorm
                (
                    (decorator
                        (call_expression
                            function: (identifier) @entity (#eq? @entity "Entity")
                        )
                    )
                    (class_declaration
                        name: (type_identifier) @{STRUCT_NAME}
                    ) @{STRUCT}
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
    fn resolve_import_path(&self, import_path: &str, _current_file: &str) -> String {
        let mut path = import_path.trim().to_string();
        if path.starts_with("./") {
            path = path[2..].to_string();
        } else if path.starts_with(".\\") {
            path = path[2..].to_string();
        } else if path.starts_with('/') {
            path = path[1..].to_string();
        }

        if (path.starts_with('"') && path.ends_with('"'))
            || (path.starts_with('\'') && path.ends_with('\''))
            || (path.starts_with('`') && path.ends_with('`'))
        {
            path = path[1..path.len() - 1].to_string();
        }

        if path.ends_with(".js") {
            path = path.replace(".js", ".ts");
        }

        path
    }
}
