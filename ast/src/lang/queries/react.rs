use super::super::*;
use super::consts::*;
use anyhow::{Context, Result};
use tree_sitter::{Language, Parser, Query, Tree};

pub struct ReactTs(Language);

impl ReactTs {
    pub fn new() -> Self {
        ReactTs(tree_sitter_typescript::LANGUAGE_TSX.into())
    }
}

impl Stack for ReactTs {
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
    fn is_component(&self, func_name: &str) -> bool {
        if func_name.len() < 1 {
            return false;
        }
        func_name.chars().next().unwrap().is_uppercase()
    }
    fn class_definition_query(&self) -> String {
        format!(
            "(class_declaration
                name: (type_identifier) @{CLASS_NAME}
            ) @{CLASS_DEFINITION}"
        )
    }
    // FIXME "render" is always discluded to avoid jsx classes
    fn function_definition_query(&self) -> String {
        format!(
            r#"[
                (function_declaration
                    name: (identifier) @{FUNCTION_NAME}
                    parameters: (formal_parameters) @{ARGUMENTS}
                )
                (method_definition
                    name: (property_identifier) @{FUNCTION_NAME} (#not-eq? @{FUNCTION_NAME} "render")
                    parameters: (formal_parameters) @{ARGUMENTS}
                )
                (lexical_declaration
                    (variable_declarator
                        name: (identifier) @{FUNCTION_NAME}
                        value: (arrow_function
                            parameters: (formal_parameters) @{ARGUMENTS}
                        )
                    )
                )
                (export_statement
                    (lexical_declaration
                        (variable_declarator
                            name: (identifier) @{FUNCTION_NAME}
                            value: (arrow_function
                                parameters: (formal_parameters) @{ARGUMENTS}
                            )
                        )
                    )
                )
                (export_statement
                    (function_declaration
                        name: (identifier) @{FUNCTION_NAME}
                        parameters: (formal_parameters) @{ARGUMENTS}
                    )
                )
                (variable_declarator
                    name: (identifier) @{FUNCTION_NAME}
                    value: (arrow_function
                        parameters: (formal_parameters) @{ARGUMENTS}
                    )
                )
                (expression_statement
                    (assignment_expression
                        left: (identifier) @{FUNCTION_NAME}
                        right: (arrow_function
                            parameters: (formal_parameters) @{ARGUMENTS}
                        )
                    )
                )
                (public_field_definition
                    name: (property_identifier) @{FUNCTION_NAME}
                    value: [
                        (function_expression
                            parameters: (formal_parameters) @{ARGUMENTS}
                        )
                        (arrow_function
                            parameters: (formal_parameters) @{ARGUMENTS}
                        )
                    ]
                )
                (pair
                    key: (property_identifier) @function-name
                    value: [
                        (function_expression
                                parameters: (formal_parameters) @{ARGUMENTS}
                        )
                        (arrow_function
                                parameters: (formal_parameters) @{ARGUMENTS}
                        )
                    ]
                )
                (variable_declarator
                    name: (identifier) @{FUNCTION_NAME}
                    value: (call_expression
                        function: (_)
                        arguments: (arguments
                            (arrow_function
                                parameters: (formal_parameters)
                                body: (statement_block
                                    (return_statement
                                        [
                                            (jsx_element)
                                            (parenthesized_expression
                                                (jsx_element)
                                            )
                                        ]
                                    )
                                )
                            )
                        )
                    )
                )
                (class_declaration
                    name: (type_identifier) @{FUNCTION_NAME}
                    (class_heritage
                        (extends_clause
                            value: (member_expression
                                object: (identifier) @react (#eq @react "React")
                                property: (property_identifier) @component (#eq @component "Component")
                            )
                        )
                    )
                    body: (class_body
                        (method_definition
                            name: (property_identifier) @render (#eq @render "render")
                            body: (statement_block
                                (return_statement
                                    [
                                        (jsx_element)
                                        (parenthesized_expression
                                            (jsx_element)
                                        )
                                    ]
                                )
                            )
                        )
                    )
                )
                (lexical_declaration
                    (variable_declarator
                        name: (identifier) @{FUNCTION_NAME}
                        value: (call_expression
                            function: (member_expression
                                object: (identifier) @styled-object (#eq @styled-object "styled")
                                property: (property_identifier) @styled-method
                            )
                        )
                    )
                )
            ] @{FUNCTION_DEFINITION}"#
        )
    }
    fn data_model_query(&self) -> Option<String> {
        Some(format!(
            r#"[
                (type_alias_declaration
                    name: (type_identifier) @{STRUCT_NAME}
                ) @{STRUCT}
                (interface_declaration
                    name: (type_identifier) @{STRUCT_NAME}
                ) @{STRUCT}
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
            ]"#
        ))
    }
    fn data_model_within_query(&self) -> Option<String> {
        Some(format!(
            r#"(
                (type_identifier) @{STRUCT_NAME} (#match? @{STRUCT_NAME} "^[A-Z].*")
            )"#
        ))
    }
    fn test_query(&self) -> Option<String> {
        Some(format!(
            r#"[
                    (call_expression
                        function: (identifier) @it (#eq? @it "it")
                        arguments: (arguments
                            (string) @{FUNCTION_NAME}
                        )
                    )
                    (call_expression
                        function: (member_expression
                            object: (member_expression
                                object: (identifier) @cypress (#eq? @cypress "Cypress")
                                property: (property_identifier) @commands (#eq? @commands "Commands")
                            )
                            property: (property_identifier) @add (#eq? @add "add")
                        )
                        arguments: (arguments
                            (string) @{FUNCTION_NAME}
                        )
                    )
                ] @{FUNCTION_DEFINITION}"#
        ))
    }
    fn request_finder(&self) -> Option<String> {
        Some(format!(
            r#"(call_expression
                function: [
                    (identifier) @{REQUEST_CALL}
                    (member_expression
                        property: (property_identifier) @{REQUEST_CALL}
                    )
                ] (#match? @{REQUEST_CALL} "^fetch$|^get$|^post$|^put$|^delete$")
                arguments: (arguments
                    [(string) (template_string)] @{ENDPOINT}
                )
            ) @{ROUTE}"#
        ))
    }
    fn function_call_query(&self) -> String {
        format!(
            "[
                (call_expression
                    function: [
                        (identifier) @{FUNCTION_NAME}
                        (member_expression
                            object: (identifier) @{OPERAND}
                            property: (property_identifier) @{FUNCTION_NAME}
                        )
                    ]
                )
                [
                    (jsx_element
                        open_tag: (jsx_opening_element
                            name: (identifier) @{FUNCTION_NAME}
                        )
                    )
                    (jsx_self_closing_element
                        name: (identifier) @{FUNCTION_NAME}
                    )
                ]
            ] @{FUNCTION_CALL}"
        )
    }
    fn add_endpoint_verb(&self, inst: &mut NodeData, call: &Option<String>) {
        if inst.meta.get("verb").is_none() {
            if let Some(call) = call {
                match call.as_str() {
                    "get" => inst.add_verb("GET"),
                    "post" => inst.add_verb("POST"),
                    "put" => inst.add_verb("PUT"),
                    "delete" => inst.add_verb("DELETE"),
                    "fetch" => {
                        inst.body.find("GET").map(|_| inst.add_verb("GET"));
                        inst.body.find("POST").map(|_| inst.add_verb("POST"));
                        inst.body.find("PUT").map(|_| inst.add_verb("PUT"));
                        inst.body.find("DELETE").map(|_| inst.add_verb("DELETE"));
                    }
                    _ => (),
                }
            }
        }
        if inst.meta.get("verb").is_none() {
            inst.add_verb("GET");
        }
    }
    fn is_router_file(&self, file_name: &str, code: &str) -> bool {
        // next.js or react-router-dom
        // file_name.contains("src/pages/") || code.contains("react-router-dom")
        !file_name.contains("__tests__") && !file_name.contains("test")
    }
    fn page_query(&self) -> Option<String> {
        let component_attribute = format!(
            r#"(jsx_attribute
                (property_identifier) @header-attr (#eq? @header-attr "header")
                (jsx_expression
                    (jsx_self_closing_element
                        name: (identifier) @{PAGE_HEADER}
                    )
                )
            )?"#
        );
        Some(format!(
            r#"[
                (jsx_self_closing_element
                    name: (
                        (identifier) @tag (#match? @tag "Route")
                    )
                    attribute: (jsx_attribute
                        (property_identifier) @path-attr (#eq? @path-attr "path")
                        (_) @{PAGE_PATHS}
                    )
                    attribute: (jsx_attribute
                        (property_identifier) @component-attr (#match? @component-attr "^component$|^element$")
                        (jsx_expression [
                            (identifier) @page-component
                            (jsx_self_closing_element
                                (identifier) @page-component
                            )
                        ])
                    )?
                )
                (jsx_element
                    open_tag: (jsx_opening_element
                        name: (
                            (identifier) @tag (#match? @tag "Route")
                        )
                        (_)*   ; allow any children before
                        (jsx_attribute
                            (property_identifier) @path-attr (#eq? @path-attr "path")
                            (_) @{PAGE_PATHS}
                        )
                        (_)*   ; allow any children after
                    )
                    [
                        (jsx_element(jsx_opening_element
                            name: (identifier) @{PAGE_COMPONENT}
                            {component_attribute}
                        ) (jsx_self_closing_element
                            name: (identifier) @{PAGE_CHILD}
                        ))
                        (jsx_self_closing_element
                            name: (identifier) @{PAGE_COMPONENT}
                            {component_attribute}
                        )
                    ]
                )
            ] @{PAGE}"#
        ))
    }
    fn find_function_parent(
        &self,
        node: TreeNode,
        code: &str,
        file: &str,
        func_name: &str,
        _callback: &dyn Fn(&str) -> Option<NodeData>,
        _parent_type: Option<&str>,
    ) -> Result<Option<Operand>> {
        let mut parent = node.parent();
        while parent.is_some() {
            if parent.unwrap().kind().to_string() == "method_definition" {
                // this is not a method, but a function defined within a method!!! skip it
                return Ok(None);
            }
            if parent.unwrap().kind().to_string() == "class_declaration" {
                // found it!
                break;
            }
            parent = parent.unwrap().parent();
        }
        let parent_of = match parent {
            Some(p) => {
                let query = self.q("(type_identifier) @class_name", &NodeType::Class);
                match query_to_ident(query, p, code)? {
                    Some(parent_name) => Some(Operand {
                        source: NodeKeys::new(&parent_name, file, p.start_position().row),
                        target: NodeKeys::new(func_name, file, node.start_position().row),
                    }),
                    None => None,
                }
            }
            None => None,
        };
        Ok(parent_of)
    }
}
