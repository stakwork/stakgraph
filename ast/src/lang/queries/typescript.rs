use super::super::*;
use super::consts::*;
use shared::error::{Context, Result};
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
                            alias: (identifier)? @{IMPORTS_ALIAS}
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
            (
              (decorator)* @{ATTRIBUTES}
              .
              (function_declaration
                name: (identifier) @{FUNCTION_NAME}
                parameters : (formal_parameters)? @{ARGUMENTS}
                return_type: (type_annotation)? @{RETURN_TYPES}
              ) @{FUNCTION_DEFINITION}
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
                    object: (identifier) @{ENDPOINT_OBJECT}
                    property: (property_identifier) @{ENDPOINT_VERB} (#match? @{ENDPOINT_VERB} "^get$|^post$|^put$|^delete$|^use$")
                )
                arguments: (arguments
                    (string) @{ENDPOINT}
                    (identifier) @{HANDLER}
                )
                ) @{ROUTE}
            "#
        )]
    }

    fn endpoint_group_find(&self) -> Option<String> {
        Some(format!(
            r#"(call_expression
                function: (member_expression
                    object: (identifier)
                    property: (property_identifier) @{ENDPOINT_VERB} (#match? @{ENDPOINT_VERB} "^use$")
                )
                arguments: (arguments
                    (string) @{ENDPOINT}
                    (identifier) @{ENDPOINT_GROUP}
                )
            ) @{ROUTE}"#
        ))
    }

    fn handler_method_query(&self) -> Option<String> {
        Some(format!(
            r#"
            ;; Matches: router.get(...), app.post(...), etc.
            (call_expression
                function: (member_expression
                    object: (identifier)
                    property: (property_identifier) @method (#match? @method "^(get|post|put|delete|patch)$")
                )
            ) @route
            "#
        ))
    }

    fn add_endpoint_verb(&self, inst: &mut NodeData, call: &Option<String>) -> Option<String> {
        if let Some(c) = call {
            let (verb, result) = match c.as_str() {
                "get" => ("GET", Some("GET".to_string())),
                "post" => ("POST", Some("POST".to_string())),
                "put" => ("PUT", Some("PUT".to_string())),
                "delete" => ("DELETE", Some("DELETE".to_string())),
                "patch" => ("PATCH", Some("PATCH".to_string())),
                "use" => {
                    return Some("USE".to_string());
                }
                _ => ("", None),
            };

            if !verb.is_empty() {
                inst.meta.insert("verb".to_string(), verb.to_string());
                return result;
            }
        }
        None
    }

    /*
    POSSIBLE QUERY FOR DATA MODEL that picks up interfaces without methods -- needs work
    (interface_declaration
        name: (type_identifier) @struct-name
        body: (interface_body

            (method_signature) @method
            (#is_not? @method _)

        )
    ) @struct
     */
    fn data_model_query(&self) -> Option<String> {
        Some(format!(
            r#"
                [
                    (interface_declaration
                        name: (type_identifier) @{STRUCT_NAME} 
                    )
                    (type_alias_declaration
                        name: (type_identifier) @{STRUCT_NAME}
                    )
                    (enum_declaration
                        name: (identifier) @{STRUCT_NAME}
                    )
                    (class_declaration
                        name: (type_identifier) @{STRUCT_NAME}
                        (class_heritage
                            (extends_clause
                                value: (identifier) @model (#eq? @model "Model")
                            )
                        )
                    ) 
                    (
                        (decorator
                            (call_expression
                                function: (identifier) @entity (#eq? @entity "Entity")
                            )
                        )
                        (class_declaration
                            name: (type_identifier) @{STRUCT_NAME}
                        ) 
                    )
                ] @{STRUCT}
             "#
        ))
    }

    fn trait_query(&self) -> Option<String> {
        Some(format!(
            r#"
            [
                (interface_declaration
                    name: (type_identifier) @{TRAIT_NAME}
                    body: (interface_body
                        (method_signature)+
                    )
                )
                (type_alias_declaration
                    name: (type_identifier)@{TRAIT_NAME}
                    value: (object_type
                            (method_signature)+
                        )
                )
            ]@{TRAIT}
            "#
        ))
    }

    fn implements_query(&self) -> Option<String> {
        Some(format!(
            r#"
            (class_declaration
                name: (type_identifier) @{CLASS_NAME}
                (class_heritage
                    (implements_clause
                        (type_identifier) @{TRAIT_NAME}
                    )
                )
            )@{IMPLEMENTS}
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
    fn is_test_file(&self, file_name: &str) -> bool {
        file_name.contains("__tests__")
            || file_name.contains(".test.")
            || file_name.contains(".spec.")
            || file_name.ends_with(".test.ts")
            || file_name.ends_with(".test.tsx")
            || file_name.ends_with(".test.js")
            || file_name.ends_with(".test.jsx")
            || file_name.ends_with(".spec.ts")
            || file_name.ends_with(".spec.tsx")
            || file_name.ends_with(".spec.js")
            || file_name.ends_with(".spec.jsx")
    }
    fn tests_are_functions(&self) -> bool {
        false
    }

    fn is_e2e_test_file(&self, file: &str, code: &str) -> bool {
        let f = file.replace('\\', "/");
        let lower_code = code.to_lowercase();
        let fname = f.rsplit('/').next().unwrap_or(&f).to_lowercase();

        let is_e2e_dir = f.contains("/tests/e2e/")
            || f.contains("/test/e2e")
            || f.contains("/e2e/")
            || f.contains("/__e2e__/")
            || f.contains("e2e.");
        let has_e2e_in_name = fname.contains("e2e");
        let has_playwright = lower_code.contains("@playwright/test");
        let has_cypress = lower_code.contains("cy.");
        let has_puppeteer =
            lower_code.contains("puppeteer") || lower_code.contains("browser.newpage");

        is_e2e_dir || has_e2e_in_name || has_playwright || has_cypress || has_puppeteer
    }

    fn is_test(&self, _func_name: &str, func_file: &str) -> bool {
        if self.is_test_file(func_file) {
            true
        } else {
            false
        }
    }

    fn should_skip_function_call(&self, called: &str, operand: &Option<String>) -> bool {
        consts::should_skip_js_function_call(called, operand)
    }
}
