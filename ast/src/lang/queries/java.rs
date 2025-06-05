use super::super::*;
use super::consts::*;
use anyhow::{Context, Result};
use tree_sitter::{Language, Parser, Query, Tree};

pub struct Java(Language);

impl Java {
    pub fn new() -> Self {
        Java(tree_sitter_java::LANGUAGE.into())
    }
}

impl Stack for Java {
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
            (program
                (local_variable_declaration
                    (modifiers)
                    type: (_)@{VARIABLE_TYPE}
                    declarator: (variable_declarator
                        name: (identifier) @{VARIABLE_NAME}
                        value: (_) @{VARIABLE_VALUE}
                    )
                )@{VARIABLE_DECLARATION}
            )
            "#
        ))
    }

    fn class_definition_query(&self) -> String {
        format!(
            r#"
                 (class_declaration
                    (identifier)@{CLASS_NAME}
                    (class_body)@{CLASS_DEFINITION}
                    (superclass 
                        (type_identifier)@{CLASS_PARENT}
                    )?
                    (super_interfaces
                        (type_list
                            (type_identifier)@{INCLUDED_MODULES}
                        )
                    )?
                )
                "#
        )
    }

    fn instance_definition_query(&self) -> Option<String> {
        Some(format!(
            r#"
            (field_declaration
                (type_identifier) @{CLASS_NAME}
                (variable_declarator
                    (identifier) @{INSTANCE_NAME}
                )
            )
            "#
        ))
    }
    fn function_definition_query(&self) -> String {
        format!(
            r#"
            (method_declaration
                type: (_) @{RETURN_TYPES}
                name: (identifier) @{FUNCTION_NAME}                
                (formal_parameters
                    (formal_parameter)@{ARGUMENTS}
                )?  
            )@{FUNCTION_DEFINITION}
            "#
        )
    }

    fn function_call_query(&self) -> String {
        format!(
            r#"
                (method_invocation
                    object: (_)? @{OPERAND}
                    name: (identifier) @{FUNCTION_NAME}
                    arguments: (argument_list 
                    (_)* 
                    )@{ARGUMENTS}
                ) @function-call
                
                "#
        )
    }
    fn endpoint_finders(&self) -> Vec<String> {
        vec![format!(
            r#"
            (method_declaration
                (modifiers
                (annotation
                    name: (identifier) @{ENDPOINT_VERB} (#match? @{ENDPOINT_VERB} "GetMapping|PostMapping|PutMapping|DeleteMapping|RequestMapping|PatchMapping")
                    arguments: (annotation_argument_list 
                    (string_literal) @{ENDPOINT}
                    )?
                )
                )
                name: (identifier) @{HANDLER}
            ) @{ROUTE}
            "#
        )]
    }

    fn endpoint_group_find(&self) -> Option<String> {
        Some(format!(
            r#"
            (class_declaration
                (modifiers
                    (annotation
                        name: (identifier) @{ENDPOINT_VERB} (#eq? @{ENDPOINT_VERB} "RequestMapping")
                        arguments: (annotation_argument_list 
                            (string_literal) @{ENDPOINT}
                        )?
                    )
                )
                name: (identifier) @{ENDPOINT_GROUP}
            )@{ROUTE}
            "#
        ))
    }

    fn update_endpoint_verb(&self, nd: &mut NodeData, _call: &Option<String>) {
        if let Some(verb_annotation) = nd.meta.get("verb").cloned() {
            let http_verb = match verb_annotation.as_str() {
                "GETMAPPING" => "GET",
                "POSTMAPPING" => "POST",
                "PUTMAPPING" => "PUT",
                "DELETEMAPPING" => "DELETE",
                "PATCHMAPPING" => "PATCH",
                _ => "GET",
            };

            nd.add_verb(http_verb);
            return;
        }
        //TODO: check for the presence of the verb in the function call
        // if all else fails, default to GET
        nd.add_verb("GET");
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
            .last()
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
