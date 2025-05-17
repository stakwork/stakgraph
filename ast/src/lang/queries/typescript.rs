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

    fn trait_query(&self) -> Option<String> {
        Some(format!(
            r#"
            (interface_declaration
                name: (type_identifier) @{TRAIT_NAME}
                body: (interface_body) @{TRAIT}
            )
            "#
        ))
    }

    fn is_data_model(&self, body: tree_sitter::Node, _code: &str) -> bool {
        let mut only_properties = true;
        for i in 0..body.named_child_count() {
            if let Some(child) = body.named_child(i) {
                println!("child: {:?}", child);
                if child.kind() != "property_signature" {
                    only_properties = false;
                    break;
                }
            }
        }
        only_properties && body.named_child_count() > 0
    }

    fn is_trait(&self, body: tree_sitter::Node, _code: &str) -> bool {
        for i in 0..body.named_child_count() {
            if let Some(child) = body.named_child(i) {
                if child.kind() == "method_signature" {
                    return true;
                }
            }
        }
        false
    }
}
