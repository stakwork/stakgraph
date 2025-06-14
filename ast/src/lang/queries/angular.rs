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
    fn component_template_query(&self) -> Option<String> {
        Some(format!(
            r#"
            [
                (decorator
                    (call_expression
                        function: (identifier) @{DECORATOR_NAME} (#eq? @{DECORATOR_NAME} "Component")
                        arguments: (arguments
                            (object
                                (pair
                                    key: (property_identifier) @{TEMPLATE_KEY} (#match? @{TEMPLATE_KEY} "^(templateUrl|styleUrls)$")
                                    value: (_) @{TEMPLATE_VALUE}
                                )
                            )
                        )
                    )
                )
                (decorator
                    (call_expression
                        function: (identifier) @{DECORATOR_NAME} (#eq? @{DECORATOR_NAME} "Component")
                        arguments: (arguments
                            (object
                                (pair
                                    key: (property_identifier) @selector (#eq? @selector "selector")
                                    value: (string) @selector_value
                                )
                            )
                        )
                    )
                )
            ]
            "#
        ))
    }
    fn nested_component_query(&self) -> Option<String> {
        Some(format!(
            r#"
            [
                (element
                    (start_tag
                        name: (_) @component_name (#match? @component_name "^app-.*$")
                    )
                )
                (element
                    (start_tag
                        name: (_) @router_outlet (#eq? @router_outlet "router-outlet")
                    )
                )
            ]
            "#
        ))
    }
    fn template_ext(&self) -> Option<&str> {
        Some(".component.ts")
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

        path
    }
    fn is_extra_page(&self, file_name: &str) -> bool {
        file_name.ends_with(".html") || 
        file_name.ends_with(".css") || 
        file_name.ends_with(".scss") || 
        file_name.ends_with(".sass")
    }
    
    fn use_extra_page_finder(&self) -> bool {
        true
    }
    
    fn extra_page_finder(
        &self,
        file_path: &str,
        find_fn: &dyn Fn(&str, &str) -> Option<NodeData>,
    ) -> Option<Edge> {
        let path = std::path::Path::new(file_path);
        let extension = path.extension()?.to_str()?;
        let file_stem = path.file_stem()?.to_str()?;
        
        if extension == "html" {
            let component_name = format!("{}Component", file_stem.replace("-", " ")
                .split_whitespace()
                .map(|s| {
                    let mut c = s.chars();
                    match c.next() {
                        None => String::new(),
                        Some(f) => f.to_uppercase().collect::<String>() + c.as_str(),
                    }
                })
                .collect::<String>());
            
            let component_file = format!("{}.component.ts", file_stem);
            
            if let Some(component) = find_fn(&component_name, &component_file) {
                let page = NodeData::name_file(file_stem, file_path);
                return Some(Edge::renders(&component, &page));
            }
        }
        else if extension == "css" || extension == "scss" || extension == "sass" {
            let component_name = format!("{}Component", file_stem.replace("-", " ")
                .split_whitespace()
                .map(|s| {
                    let mut c = s.chars();
                    match c.next() {
                        None => String::new(),
                        Some(f) => f.to_uppercase().collect::<String>() + c.as_str(),
                    }
                })
                .collect::<String>());
            
            let component_file = format!("{}.component.ts", file_stem);
            
            if let Some(component) = find_fn(&component_name, &component_file) {
                let page = NodeData::name_file(file_stem, file_path);
                return Some(Edge::renders(&component, &page));
            }
        }
        
        None
    }
}
