use std::fs;

use super::super::*;
use super::consts::*;
use lsp::strip_tmp;
use shared::error::{Context, Result};
use tree_sitter::{Language, Parser, Query, QueryCursor, Tree};

/// Unified TypeScript/React parser using TSX grammar
/// Handles both .ts and .tsx files as TSX is a superset of TypeScript
pub struct TypeScriptReact {
    tsx: Language,
}

impl Default for TypeScriptReact {
    fn default() -> Self {
        Self::new()
    }
}

impl TypeScriptReact {
    pub fn new() -> Self {
        TypeScriptReact {
            tsx: tree_sitter_typescript::LANGUAGE_TSX.into(),
        }
    }
}

impl Stack for TypeScriptReact {
    fn q(&self, q: &str, _nt: &NodeType) -> Query {
        match Query::new(&self.tsx, q) {
            Ok(query) => query,
            Err(err) => panic!("Failed to compile TypeScriptReact query '{}': {}", q, err),
        }
    }

    fn parse(&self, code: &str, _nt: &NodeType) -> Result<Tree> {
        let mut parser = Parser::new();
        parser.set_language(&self.tsx)?;
        parser.parse(code, None).context("failed to parse")
    }

    // FROM REACT: lib_query
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

    // FROM REACT (identical in both): classify_test
    fn classify_test(&self, name: &str, file: &str, body: &str) -> NodeType {
        // 1. Path based (strongest signal)
        let f = file.replace('\\', "/");
        let fname = f.rsplit('/').next().unwrap_or(&f).to_lowercase();
        let is_e2e_dir =
            f.contains("/tests/e2e/") || f.contains("/test/e2e") || f.contains("/e2e/");
        if is_e2e_dir
            || f.contains("/__e2e__/")
            || f.contains(".e2e.")
            || fname.starts_with("e2e.")
            || fname.starts_with("e2e_")
            || fname.starts_with("e2e-")
            || fname.contains(".e2e.test")
            || fname.contains(".e2e.spec")
        {
            return NodeType::E2eTest;
        }
        if f.contains("/integration/") || f.contains(".int.") || f.contains(".integration.") {
            return NodeType::IntegrationTest;
        }
        if f.contains("/unit/") || f.contains(".unit.") {
            return NodeType::UnitTest;
        }

        let lower_name = name.to_lowercase();
        // 2. Explicit tokens in test name
        if lower_name.contains("e2e") {
            return NodeType::E2eTest;
        }
        if lower_name.contains("integration") {
            return NodeType::IntegrationTest;
        }

        // 3. Body heuristics (tighter): network => integration; real browser automation => e2e
        let body_l = body.to_lowercase();
        let has_playwright_import = body_l.contains("@playwright/test")
            || body_l.contains("from '@playwright/test'")
            || body_l.contains("from \"@playwright/test\"");
        let has_browser_actions = body_l.contains("page.goto(")
            || body_l.contains("page.click(")
            || body_l.contains("page.evaluate(");
        let has_cypress = body_l.contains("from 'cypress'")
            || body_l.contains("from \"cypress\"")
            || body_l.contains("require('cypress')")
            || body_l.contains("require(\"cypress\")");
        let has_puppeteer = body_l.contains("from 'puppeteer'")
            || body_l.contains("from \"puppeteer\"")
            || body_l.contains("require('puppeteer')")
            || body_l.contains("require(\"puppeteer\")");
        if (has_playwright_import && has_browser_actions) || has_cypress || has_puppeteer {
            return NodeType::E2eTest;
        }

        const NETWORK_MARKERS: [&str; 11] = [
            "fetch(",
            "axios.",
            "axios(",
            "supertest(",
            "request(",
            "new request(",
            "/api/",
            "http://",
            "https://",
            "globalthis.fetch",
            "cy.request(",
        ];
        if NETWORK_MARKERS.iter().any(|m| body_l.contains(m)) {
            return NodeType::IntegrationTest;
        }
        NodeType::UnitTest
    }

    // MERGED: is_lib_file (TypeScript has more patterns)
    fn is_lib_file(&self, file_name: &str) -> bool {
        file_name.contains("node_modules/")
            || file_name.contains("/lib/")
            || file_name.ends_with(".d.ts")
            || file_name.starts_with("/usr")
            || file_name.contains(".nvm/")
    }

    // FROM TYPESCRIPT: imports_query (has IMPORTS_ALIAS which React doesn't have)
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

    // FROM REACT (identical in both): variables_query
    fn variables_query(&self) -> Option<String> {
        let types = "(string)(template_string)(number)(object)(array)(true)(false)(new_expression)(member_expression)(identifier)";
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

    // FROM REACT: is_component
    fn is_component(&self, func_name: &str) -> bool {
        if func_name.is_empty() {
            return false;
        }
        func_name
            .chars()
            .next()
            .map(|ch| ch.is_uppercase())
            .unwrap_or(false)
    }

    // MERGED: class_definition_query (TypeScript version has PARENT_NAME for implements)
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

    // MERGED: function_definition_query (React has more patterns including JSX components)
    fn function_definition_query(&self) -> String {
        format!(
            r#"[
            ;; TypeScript: decorated function declarations
            (
              (decorator)* @{ATTRIBUTES}
              .
              (function_declaration
                name: (identifier) @{FUNCTION_NAME}
                parameters : (formal_parameters)? @{ARGUMENTS}
                return_type: (type_annotation)? @{RETURN_TYPES}
              ) @{FUNCTION_DEFINITION}
            )

            ;; export function hello()
            (export_statement
                (function_declaration
                    name: (identifier) @{FUNCTION_NAME}
                    parameters: (formal_parameters)? @{ARGUMENTS}
                    return_type: (type_annotation)? @{RETURN_TYPES}
                )
            ) @{FUNCTION_DEFINITION}
             
            ;; export const hello = () => 
            (export_statement
                (lexical_declaration
                    (variable_declarator
                        name: (identifier) @{FUNCTION_NAME}
                        value: (arrow_function
                            parameters: (formal_parameters)? @{ARGUMENTS}
                            return_type: (type_annotation)? @{RETURN_TYPES}
                        )
                    )
                )
            ) @{FUNCTION_DEFINITION}

            ;; export const hello = create()
            (export_statement
                (lexical_declaration
                    (variable_declarator
                        name: (identifier) @function-name
                        value: (call_expression)
                    )
                )
            ) @function-definition

            (program
                (function_declaration
                    name: (identifier) @{FUNCTION_NAME}
                    parameters: (formal_parameters)? @{ARGUMENTS}
                    return_type: (type_annotation)? @{RETURN_TYPES}
                ) @{FUNCTION_DEFINITION}
            )

            (statement_block
                (function_declaration
                    name: (identifier) @{FUNCTION_NAME}
                    parameters: (formal_parameters)? @{ARGUMENTS}
                    return_type: (type_annotation)? @{RETURN_TYPES}
                ) @{FUNCTION_DEFINITION}
            )

            (program
                (lexical_declaration
                    (variable_declarator
                        name: (identifier) @{FUNCTION_NAME}
                        value: [
                            (arrow_function
                                parameters: (formal_parameters)? @{ARGUMENTS}
                                return_type: (type_annotation)? @{RETURN_TYPES}
                            )
                            (function_expression
                                parameters: (formal_parameters)? @{ARGUMENTS}
                                return_type: (type_annotation)? @{RETURN_TYPES}
                            )
                        ]
                    )
                ) @{FUNCTION_DEFINITION}
            )

            (method_definition
                name: (property_identifier) @{FUNCTION_NAME} (#not-eq? @{FUNCTION_NAME} "render")
                parameters: (formal_parameters)? @{ARGUMENTS}
                return_type: (type_annotation)? @{RETURN_TYPES}
            ) @{FUNCTION_DEFINITION}

            (variable_declarator
                name: (identifier) @{FUNCTION_NAME}
                value: (arrow_function
                    parameters: (formal_parameters)? @{ARGUMENTS}
                    return_type: (type_annotation)? @{RETURN_TYPES}
                )
            ) @{FUNCTION_DEFINITION}
             
            (expression_statement
                (assignment_expression
                    left: (identifier) @{FUNCTION_NAME}
                    right: (arrow_function
                        parameters: (formal_parameters)? @{ARGUMENTS}
                        return_type: (type_annotation)? @{RETURN_TYPES}
                    )
                )
            ) @{FUNCTION_DEFINITION}

            (public_field_definition
                name: (property_identifier) @{FUNCTION_NAME}
                value: [
                    (function_expression
                        parameters: (formal_parameters)? @{ARGUMENTS}
                        return_type: (type_annotation)? @{RETURN_TYPES}
                    )
                    (arrow_function
                        parameters: (formal_parameters)? @{ARGUMENTS}
                        return_type: (type_annotation)? @{RETURN_TYPES}
                    )
                ]
            ) @{FUNCTION_DEFINITION}

            (pair
                key: (property_identifier) @{FUNCTION_NAME}
                value: [
                    (function_expression
                            parameters: (formal_parameters)? @{ARGUMENTS}
                            return_type: (type_annotation)? @{RETURN_TYPES}
                    )
                    (arrow_function
                            parameters: (formal_parameters)? @{ARGUMENTS}
                            return_type: (type_annotation)? @{RETURN_TYPES}
                    )
                ]
            ) @{FUNCTION_DEFINITION}

            (variable_declarator
                name: (identifier) @{FUNCTION_NAME}
                value: (call_expression
                    function: (_)
                    arguments: (arguments
                        (arrow_function
                            parameters: (formal_parameters)? @{ARGUMENTS}
                            return_type: (type_annotation)? @{RETURN_TYPES}
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
            ) @{FUNCTION_DEFINITION}

            (class_declaration
                name: (type_identifier) @{FUNCTION_NAME}
                (class_heritage
                    (extends_clause
                        value: (member_expression
                            object: (identifier) @react (#eq? @react "React")
                            property: (property_identifier) @component (#eq? @component "Component")
                        )
                    )
                )
                body: (class_body
                    (method_definition
                        name: (property_identifier) @render (#eq? @render "render")
                        return_type: (type_annotation)? @{RETURN_TYPES}
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
            ) @{FUNCTION_DEFINITION}

            ;; export const hello = styled.div
            (export_statement
                (lexical_declaration
                    (variable_declarator
                        name: (identifier) @{FUNCTION_NAME}
                        value: (call_expression
                            function: (member_expression
                                object: (identifier) @styled-object (#eq? @styled-object "styled")
                                property: (property_identifier) @styled-method
                            )
                        )
                    )
                )
            ) @{FUNCTION_DEFINITION}

            ;; const hello = styled.div
            (program
                (lexical_declaration
                    (variable_declarator
                        name: (identifier) @{FUNCTION_NAME}
                        value: (call_expression
                            function: (member_expression
                                object: (identifier) @styled-object (#eq? @styled-object "styled")
                                property: (property_identifier) @styled-method
                            )
                        )
                    )
                ) @{FUNCTION_DEFINITION}
            )

            ]"#
        )
    }

    // FROM REACT (identical in both): comment_query
    fn comment_query(&self) -> Option<String> {
        Some(format!(r#"(comment) @{FUNCTION_COMMENT}"#))
    }
    fn class_comment_query(&self) -> Option<String> {
        Some(format!(r#"(comment) @{CLASS_COMMENT}"#))
    }
    fn data_model_comment_query(&self) -> Option<String> {
        Some(format!(r#"(comment) @{STRUCT_COMMENT}"#))
    }
    fn endpoint_comment_query(&self) -> Option<String> {
        Some(format!(r#"(comment) @{ENDPOINT_COMMENT}"#))
    }
    fn var_comment_query(&self) -> Option<String> {
        Some(format!(r#"(comment) @{VAR_COMMENT}"#))
    }

    // MERGED: data_model_query (React has slightly different ordering but same patterns)
    fn data_model_query(&self) -> Option<String> {
        Some(format!(
            r#"[
                (type_alias_declaration
                    name: (type_identifier) @{STRUCT_NAME}
                ) 
                (interface_declaration
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

    // FROM REACT (identical in both): data_model_within_query
    fn data_model_within_query(&self) -> Option<String> {
        Some(format!(
            r#"(
                (type_identifier) @{STRUCT_NAME} (#match? @{STRUCT_NAME} "^[A-Z].*")
            )"#
        ))
    }

    // FROM REACT: test_query (React version has @{FUNCTION_DEFINITION} captures)
    fn test_query(&self) -> Option<String> {
        Some(format!(
            r#"[
                    (call_expression
                        function: (identifier) @desc (#eq? @desc "describe")
                        arguments: (arguments [ (string) (template_string) ] @{FUNCTION_NAME})
                    )@{FUNCTION_DEFINITION}
                    (call_expression
                        function: (member_expression
                            object: (identifier) @desc2 (#eq? @desc2 "describe")
                            property: (property_identifier) @mod (#match? @mod "^(only|skip|todo)$")
                        )
                        arguments: (arguments [ (string) (template_string) ] @{FUNCTION_NAME})
                    )@{FUNCTION_DEFINITION}
                     (program
                        (expression_statement
                            (call_expression
                                function: (identifier) @test (#match? @test "^(describe|test|it)$")
                                arguments: (arguments [ (string) (template_string) ] @{FUNCTION_NAME})
                            ) @{FUNCTION_DEFINITION}
                        )
                    )
                     (program
                        (expression_statement
                            (call_expression
                            function: (member_expression
                                object: (identifier) @obj (#eq? @obj "test")
                                property: (property_identifier) @prop (#match? @prop "^(describe|skip|only|todo)$")
                            )
                            arguments: (arguments [ (string) (template_string) ] @{FUNCTION_NAME})
                            )
                        )@{FUNCTION_DEFINITION}
                    )
                ] "#
        ))
    }

    // FROM REACT (identical in both): e2e_test_query
    fn e2e_test_query(&self) -> Option<String> {
        Some(format!(
            r#"
            (program
                (expression_statement
                    (call_expression
                        function: [
                            (identifier) @func
                            (member_expression
                                property: (property_identifier) @func
                            )
                        ]
                        (#match? @func "^(describe|test|it)$")
                        arguments: (arguments 
                            [ (string) (template_string) ] @{E2E_TEST_NAME} 
                            (arrow_function)
                        )
                    ) @{E2E_TEST}
                )
            )
        "#
        ))
    }

    // MERGED: endpoint_finders (union of React and TypeScript patterns)
    fn endpoint_finders(&self) -> Vec<String> {
        vec![
            // FROM REACT: Next.js style exports + basic router pattern
            format!(
                r#"
            (export_statement
                (function_declaration
                    name: (identifier) @{ENDPOINT} @{ENDPOINT_VERB} @{HANDLER} (#match? @{ENDPOINT_VERB} "^(GET|POST|PUT|PATCH|DELETE)$")
                ) @{ROUTE}
            )
            (export_statement
                (lexical_declaration
                        (variable_declarator
                            name: (identifier) @{ENDPOINT} @{ENDPOINT_VERB} @{HANDLER} (#match? @{ENDPOINT_VERB} "^(GET|POST|PUT|PATCH|DELETE)$")
                        )
                )@{ROUTE}
            )
            (call_expression
                function: (member_expression
                    object: (identifier) @{ENDPOINT_OBJECT}
                    property: (property_identifier) @{ENDPOINT_VERB} (#match? @{ENDPOINT_VERB} "^get$|^post$|^put$|^delete$|^use$|^patch$")
                )
                arguments: (arguments
                    (string) @{ENDPOINT}
                    (identifier) @{HANDLER}
                )
            ) @{ROUTE}
        "#
            ),
            // FROM TYPESCRIPT: router.method(path, identifier) - middleware or named handler
            format!(
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
            ),
            // FROM TYPESCRIPT: arrow function handlers
            format!(
                r#"(call_expression
                    function: (member_expression
                        object: (identifier) @{ENDPOINT_OBJECT}
                        property: (property_identifier) @{ENDPOINT_VERB} (#match? @{ENDPOINT_VERB} "^get$|^post$|^put$|^delete$|^use$")
                    )
                    arguments: (arguments
                        (string) @{ENDPOINT}
                        (arrow_function) @{ANONYMOUS_FUNCTION}
                    )
                    ) @{ROUTE}
                "#
            ),
            // FROM TYPESCRIPT: generic get handler
            format!(
                r#"(call_expression
                    function: (member_expression
                        object: (identifier) @{ENDPOINT_OBJECT}
                        property: (property_identifier) @{ENDPOINT_VERB} (#match? @{ENDPOINT_VERB} "^get$")
                    )
                    arguments: (arguments
                        (string) @{ENDPOINT}
                        (_) @{HANDLER}
                    )
                    ) @{ROUTE}
                "#
            ),
            // FROM TYPESCRIPT: chained route pattern router.route('/path').get()
            format!(
                r#"(call_expression
                    function: (member_expression
                        object: (call_expression
                            function: (member_expression
                                object: (identifier)
                                property: (property_identifier) @base_method (#match? @base_method "route")
                            )
                            arguments: (arguments (string) @{ENDPOINT})
                        )
                        property: (property_identifier) @{ENDPOINT_VERB} (#match? @{ENDPOINT_VERB} "^get$|^post$|^put$|^delete$|^patch$")
                    )
                    arguments: (arguments (arrow_function) @{ANONYMOUS_FUNCTION})
                    ) @{ROUTE}
                "#
            ),
            // FROM TYPESCRIPT: deeply chained route pattern
            format!(
                r#"(call_expression
                    function: (member_expression
                        object: (call_expression
                            function: (member_expression
                                object: (call_expression
                                    function: (member_expression
                                        object: (identifier)
                                        property: (property_identifier) @base_method (#match? @base_method "route")
                                    )
                                    arguments: (arguments (string) @{ENDPOINT})
                                )
                                property: (property_identifier)
                            )
                            arguments: (arguments (arrow_function))
                        )
                        property: (property_identifier) @{ENDPOINT_VERB} (#match? @{ENDPOINT_VERB} "^get$|^post$|^put$|^delete$|^patch$")
                    )
                    arguments: (arguments (arrow_function) @{ANONYMOUS_FUNCTION})
                    ) @{ROUTE}
                "#
            ),
        ]
    }

    // FROM REACT: request_finder (TypeScript doesn't have this)
    fn request_finder(&self) -> Option<String> {
        Some(format!(
            r#"
                ;; Matches: fetch('/api/...')
                (call_expression
                    function: (identifier) @{REQUEST_CALL} (#eq? @{REQUEST_CALL} "fetch")
                    arguments: (arguments [ (string) (template_string) ] @{ENDPOINT})
                ) @{ROUTE}

                ;; Matches: axios.get('/api/...'), ky.post('/api/...'), api.get('/api/...') etc.
                ;; to make it more specific: (#match? @lib "^(axios|ky|superagent|api)$")
                (call_expression
                    function: (member_expression
                        object: (identifier) @lib
                        property: (property_identifier) @{REQUEST_CALL} (#match? @{REQUEST_CALL} "^(get|post|put|delete|patch)$")
                    )
                    arguments: (arguments [ (string) (template_string) ] @{ENDPOINT})
                ) @{ROUTE}

                ;; Matches: axios({{ url: '/api/...' }})
                (call_expression
                    function: (identifier) @lib (#match? @lib "^(axios|ky|superagent)$")
                    arguments: (arguments
                        (object
                            (pair
                                key: (property_identifier) @url_key (#eq? @url_key "url")
                                value: [ (string) (template_string) ] @{ENDPOINT}
                            )
                        )
                    )
                ) @{ROUTE}

                 ;; Matches: new Request('/api/...', {{ method: 'POST', ... }})
                (new_expression
                    constructor: (identifier) @constructor (#eq? @constructor "Request")
                    arguments: (arguments
                        [ (string) (template_string) ] @{ENDPOINT}
                        (object)?
                    )
                ) @{ROUTE}

                 ;; Matches: new NextRequest('/api/...')
                (new_expression
                    constructor: (identifier) @constructor (#eq? @constructor "NextRequest")
                    arguments: (arguments [ (string) (template_string) ] @{ENDPOINT})
                ) @{ROUTE}
            "#
        ))
    }

    // FROM REACT (identical in both): endpoint_group_find
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

    // FROM REACT (identical in both): handler_method_query
    fn handler_method_query(&self) -> Option<String> {
        Some(
            r#"
            ;; Matches: router.get(...), app.post(...), etc.
            (call_expression
                function: (member_expression
                    object: (identifier)
                    property: (property_identifier) @method (#match? @method "^(get|post|put|delete|patch)$")
                )
            ) @route
            "#
            .to_string(),
        )
    }

    // MERGED: function_call_query (React version has JSX patterns)
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
                        (member_expression
                            object: (member_expression
                                object: (identifier) @{OPERAND}
                                property: (property_identifier)
                            )
                            property: (property_identifier) @{FUNCTION_NAME}
                        )
                        (member_expression
                            object: (member_expression
                                object: (member_expression
                                    object: (identifier) @{OPERAND}
                                    property: (property_identifier)
                                )
                                property: (property_identifier)
                            )
                            property: (property_identifier) @{FUNCTION_NAME}
                        )
                    ]
                )
                (
                    new_expression (identifier) @{CLASS_NAME}
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

    // MERGED: add_endpoint_verb (React version has fetch handling)
    fn add_endpoint_verb(&self, inst: &mut NodeData, call: &Option<String>) -> Option<String> {
        if !inst.meta.contains_key("verb") {
            if let Some(call) = call {
                match call.as_str() {
                    "get" => {
                        inst.add_verb("GET");
                        return Some("GET".to_string());
                    }
                    "post" => {
                        inst.add_verb("POST");
                        return Some("POST".to_string());
                    }
                    "put" => {
                        inst.add_verb("PUT");
                        return Some("PUT".to_string());
                    }
                    "delete" => {
                        inst.add_verb("DELETE");
                        return Some("DELETE".to_string());
                    }
                    "patch" => {
                        inst.add_verb("PATCH");
                        return Some("PATCH".to_string());
                    }
                    "use" => {
                        return Some("USE".to_string());
                    }
                    "fetch" => {
                        if inst.body.contains("GET") {
                            inst.add_verb("GET")
                        }
                        if inst.body.contains("POST") {
                            inst.add_verb("POST")
                        }
                        if inst.body.contains("PUT") {
                            inst.add_verb("PUT")
                        }
                        if inst.body.contains("DELETE") {
                            inst.add_verb("DELETE")
                        }
                        if let Some(v) = inst.meta.get("verb") {
                            return Some(v.clone());
                        }
                    }
                    _ => (),
                }
            }
        } else if let Some(v) = inst.meta.get("verb") {
            return Some(v.clone());
        }

        if !inst.meta.contains_key("verb") {
            inst.add_verb("GET");
        }
        Some("GET".to_string())
    }

    fn generate_anonymous_handler_name(
        &self,
        method: &str,
        path: &str,
        line: usize,
    ) -> Option<String> {
        let clean_method = method.to_lowercase();
        let clean_path = path
            .replace("/", "_")
            .replace(":", "param_")
            .replace("-", "_")
            .replace(" ", "_")
            .trim_start_matches('_')
            .trim_end_matches('_')
            .to_string();

        let handler_name = if clean_path.is_empty() || clean_path == "_" {
            format!("{}_handler_L{}", clean_method, line)
        } else {
            format!("{}_{}_handler_L{}", clean_method, clean_path, line)
        };

        Some(handler_name)
    }

    // FROM REACT: update_endpoint (for Next.js)
    fn update_endpoint(&self, nd: &mut NodeData, _call: &Option<String>) {
        // for next.js - update endpoint name from file path
        if matches!(
            nd.name.as_str(),
            "GET" | "POST" | "PUT" | "DELETE" | "PATCH"
        ) {
            nd.name = endpoint_name_from_file(&nd.file);
        }
        // Only set handler from verb if handler is not already set (TypeScript Express style sets it from capture)
        if nd.meta.get("handler").is_none() {
            if let Some(verb) = nd.meta.get("verb") {
                nd.meta.insert("handler".to_string(), verb.to_string());
            } else {
                nd.meta.insert("handler".to_string(), "GET".to_string());
            }
        }
    }

    // Use false to enable the fallback path that uses node_data_finder
    // This supports imported handlers for Express-style routes
    fn use_handler_finder(&self) -> bool {
        false
    }

    // FROM REACT: handler_finder
    // Handles both Next.js style (GET/POST as handler) and Express style (named handlers)
    fn handler_finder(
        &self,
        endpoint: NodeData,
        find_fn: &dyn Fn(&str, &str) -> Option<NodeData>,
        find_fns_in: &dyn Fn(&str) -> Vec<NodeData>,
        _handler_params: HandlerParams,
    ) -> Vec<(NodeData, Option<Edge>)> {
        let http_verbs = ["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"];

        // Helper to find HTTP verb handler directly in the same file
        // This avoids the start-position guard in node_data_finder
        let find_http_verb_handler = |verb: &str| -> Option<NodeData> {
            let verb_upper = verb.to_uppercase();
            let verb_lower = verb.to_lowercase();

            // Get all functions in the endpoint's file
            let functions_in_file = find_fns_in(&endpoint.file);

            // Find function matching the verb (case-insensitive)
            functions_in_file.into_iter().find(|f| {
                f.name.to_uppercase() == verb_upper || f.name.to_lowercase() == verb_lower
            })
        };

        // First try: look for explicit "handler" meta
        if let Some(handler) = endpoint.meta.get("handler") {
            let is_http_verb = http_verbs.contains(&handler.to_uppercase().as_str());

            if is_http_verb {
                // Next.js style: handler IS the HTTP verb, use direct lookup
                if let Some(nd) = find_http_verb_handler(handler) {
                    let edge = Edge::handler(&endpoint, &nd);
                    return vec![(endpoint, Some(edge))];
                }
            } else {
                // Express style: named handler, use node_data_finder for import resolution
                if let Some(nd) = find_fn(handler, &endpoint.file) {
                    let edge = Edge::handler(&endpoint, &nd);
                    return vec![(endpoint, Some(edge))];
                }
            }
        }

        // Second try: for Next.js style where verb IS the handler name
        if let Some(verb) = endpoint.meta.get("verb") {
            if http_verbs.contains(&verb.to_uppercase().as_str()) {
                if let Some(handler_node) = find_http_verb_handler(verb) {
                    let edge = Edge::handler(&endpoint, &handler_node);
                    return vec![(endpoint, Some(edge))];
                }
            }
        }

        vec![(endpoint, None)]
    }

    // FROM REACT: is_router_file
    fn is_router_file(&self, file_name: &str, _code: &str) -> bool {
        !file_name.contains("__tests__")
    }

    // FROM REACT: page_query
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

    // FROM TYPESCRIPT: trait_query
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

    fn trait_comment_query(&self) -> Option<String> {
        Some(format!(r#"(comment) @{TRAIT_COMMENT}"#))
    }

    // FROM TYPESCRIPT: implements_query
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

    // FROM REACT: find_function_parent
    fn find_function_parent(
        &self,
        node: TreeNode,
        code: &str,
        file: &str,
        func_name: &str,
        _callback: &dyn Fn(&str) -> Option<(NodeData, NodeType)>,
        _parent_type: Option<&str>,
    ) -> Result<Option<Operand>> {
        let mut parent = node.parent();
        while let Some(current) = parent {
            if current.kind() == "method_definition" {
                // this is not a method, but a function defined within a method!!! skip it
                return Ok(None);
            }
            if current.kind() == "class_declaration" {
                // found it!
                break;
            }
            parent = current.parent();
        }
        let parent_of = match parent {
            Some(p) => {
                let query = self.q("(type_identifier) @class_name", &NodeType::Class);
                query_to_ident(query, p, code)?.map(|parent_name| Operand {
                    source: NodeKeys::new(&parent_name, file, p.start_position().row),
                    target: NodeKeys::new(func_name, file, node.start_position().row),
                    source_type: NodeType::Class,
                })
            }
            None => None,
        };
        Ok(parent_of)
    }

    // MERGED: resolve_import_path (TypeScript version has .js -> .ts replacement)
    fn resolve_import_path(&self, import_path: &str, _current_file: &str) -> String {
        let mut path = import_path.trim().to_string();
        if path.starts_with("./") || path.starts_with(".\\") {
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

        // FROM TYPESCRIPT: .js -> .ts replacement
        if path.ends_with(".js") {
            path = path.replace(".js", ".ts");
        }

        path
    }

    // FROM REACT: extra_calls_queries
    fn extra_calls_queries(&self) -> Vec<String> {
        let mut extra_regex = "^use.*tore".to_string();
        if let Ok(env_regex) = std::env::var("EXTRA_REGEX_REACT") {
            extra_regex = env_regex;
        }
        vec![format!(
            r#"
            (lexical_declaration
                (variable_declarator
                    name: (object_pattern
                        ;; first only
                        . (shorthand_property_identifier_pattern) @{EXTRA_PROP}
                    )?
                    value: (call_expression
                        function: (identifier) @{EXTRA_NAME} (#match? @{EXTRA_NAME} "{extra_regex}")
                    )
                )
            ) @{EXTRA}?
            "#,
        )]
    }

    // FROM REACT: use_extra_page_finder
    fn use_extra_page_finder(&self) -> bool {
        true
    }

    // FROM REACT: is_extra_page
    fn is_extra_page(&self, file_name: &str) -> bool {
        // Ignore false positives
        let ignore_patterns = [
            "/node_modules/",
            "/dist/",
            "/.next/",
            "/build/",
            "/out/",
            "/vendor/",
            "/__tests__/",
            "/test/",
            "/coverage/",
        ];
        for pat in &ignore_patterns {
            if file_name.contains(pat) {
                return false;
            }
        }

        // App Router
        if file_name.contains("/app/")
            && (file_name.ends_with("/page.tsx")
                || file_name.ends_with("/page.jsx")
                || file_name.ends_with("page.mdx")
                || file_name.ends_with("page.md"))
        {
            return true;
        }
        // Pages Router: must be under /pages/ and not _app, _document, _error, or api
        if let Some(idx) = file_name.find("/pages/") {
            let after = &file_name[idx + 7..];
            if after.starts_with("api/")
                || after.starts_with("_app")
                || after.starts_with("_document")
                || after.starts_with("_error")
            {
                return false;
            }

            if !(after.ends_with(".tsx")
                || after.ends_with(".jsx")
                || after.ends_with(".js")
                || after.ends_with(".ts"))
                || after.ends_with(".md")
                || after.ends_with(".mdx")
            {
                return false;
            }

            // Only allow all-lowercase or dynamic ([...]) segments
            for segment in after.split('/') {
                if segment.is_empty() {
                    continue;
                }
                // skip dynamic routes like [id]
                if segment.starts_with('[') && segment.ends_with(']') {
                    continue;
                }
                // skip extension for file segment
                let segment = segment.split('.').next().unwrap_or(segment);
                if segment
                    .chars()
                    .next()
                    .map(|c| c.is_uppercase())
                    .unwrap_or(false)
                {
                    return false;
                }
            }
            return true;
        }
        false
    }

    // FROM REACT: extra_page_finder
    fn extra_page_finder(
        &self,
        file_path: &str,
        _find_fn: &dyn Fn(&str, &str) -> Option<NodeData>,
        find_fns_in: &dyn Fn(&str) -> Vec<NodeData>,
    ) -> Option<(NodeData, Option<Edge>)> {
        let path = std::path::Path::new(file_path);

        let filename = strip_tmp(path).display().to_string();

        let name = page_name(&filename);

        let mut page = NodeData::name_file(&name, &filename);
        page.body = route_from_path(&filename);

        let code = fs::read_to_string(file_path).ok()?;

        let default_export = find_default_export_name(&code, self.tsx.clone());

        let all_functions = find_fns_in(&filename);

        let target = if let Some(default_name) = default_export {
            all_functions.into_iter().find(|f| f.name == default_name)
        } else {
            None
        };

        let edge = if let Some(target) = target {
            Edge::renders(&page, &target)
        } else {
            return Some((page, None));
        };
        Some((page, Some(edge)))
    }

    // MERGED: is_test_file (union of both patterns)
    fn is_test_file(&self, file_name: &str) -> bool {
        file_name.ends_with(".test.ts")
            || file_name.ends_with(".test.tsx")
            || file_name.ends_with(".test.jsx")
            || file_name.ends_with(".test.js")
            || file_name.ends_with(".e2e.ts")
            || file_name.ends_with(".e2e.tsx")
            || file_name.ends_with(".e2e.jsx")
            || file_name.ends_with(".e2e.js")
            || file_name.ends_with(".spec.ts")
            || file_name.ends_with(".spec.tsx")
            || file_name.ends_with(".spec.jsx")
            || file_name.ends_with(".spec.js")
            || file_name.contains("/__tests__/")
            || file_name.contains("/tests/")
            || file_name.contains("/test/")
            || file_name.contains("__tests__")
            || file_name.contains(".test.")
            || file_name.contains(".spec.")
    }

    // FROM REACT (identical in both): is_e2e_test_file
    fn is_e2e_test_file(&self, file: &str, code: &str) -> bool {
        let f = file.replace('\\', "/");
        let lower_code = code.to_lowercase();
        let fname = f.rsplit('/').next().unwrap_or(&f).to_lowercase();

        let is_e2e_dir = f.contains("/tests/e2e/")
            || f.contains("/test/e2e")
            || f.contains("/e2e/")
            || f.contains("/__e2e__/")
            || f.contains(".e2e.test")
            || f.contains(".e2e.spec");
        let has_e2e_in_name = fname.starts_with("e2e.")
            || fname.starts_with("e2e-")
            || fname.starts_with("e2e_")
            || fname.contains(".e2e.");
        let has_playwright = lower_code.contains("@playwright/test")
            || lower_code.contains("from '@playwright/test'")
            || lower_code.contains("from \"@playwright/test\"");
        let has_cypress = lower_code.contains("from 'cypress'")
            || lower_code.contains("from \"cypress\"")
            || lower_code.contains("require('cypress')")
            || lower_code.contains("require(\"cypress\")");
        let has_puppeteer = lower_code.contains("from 'puppeteer'")
            || lower_code.contains("from \"puppeteer\"")
            || lower_code.contains("require('puppeteer')")
            || lower_code.contains("require(\"puppeteer\")");

        is_e2e_dir || has_e2e_in_name || has_playwright || has_cypress || has_puppeteer
    }

    // FROM REACT (identical in both): is_test
    fn is_test(&self, _func_name: &str, func_file: &str, _func_body: &str) -> bool {
        self.is_test_file(func_file)
    }

    // FROM REACT (identical in both): tests_are_functions
    fn tests_are_functions(&self) -> bool {
        false
    }

    fn should_skip_function_call(&self, called: &str, operand: &Option<String>) -> bool {
        super::skips::react_ts::should_skip(called, operand)
    }

    // MERGED: parse_imports_from_file (using TypeScript version with IMPORTS_ALIAS)
    fn parse_imports_from_file(
        &self,
        file: &str,
        find_import_node: &dyn Fn(&str) -> Option<NodeData>,
    ) -> Option<Vec<(String, Vec<String>)>> {
        use super::consts::{IMPORTS_ALIAS, IMPORTS_FROM, IMPORTS_NAME};
        use crate::lang::parse::utils::trim_quotes;
        use tree_sitter::QueryCursor;

        let import_node = find_import_node(file)?;
        let code = import_node.body.as_str();

        let imports_query = self.imports_query()?;
        let q = match tree_sitter::Query::new(&self.tsx, &imports_query) {
            Ok(query) => query,
            Err(_) => return None,
        };

        let tree = match self.parse(code, &NodeType::Import) {
            Ok(t) => t,
            Err(_) => return None,
        };

        let mut cursor = QueryCursor::new();
        let mut matches = cursor.matches(&q, tree.root_node(), code.as_bytes());
        let mut results = Vec::new();

        while let Some(m) = matches.next() {
            let mut import_names = Vec::new();
            let mut import_source = None;
            let mut import_aliases = Vec::new();

            for capture in m.captures {
                let capture_name = q.capture_names()[capture.index as usize];
                let text = capture.node.utf8_text(code.as_bytes()).unwrap_or("");

                if capture_name == IMPORTS_NAME {
                    import_names.push(text.to_string());
                } else if capture_name == IMPORTS_ALIAS {
                    import_aliases.push(text.to_string());
                } else if capture_name == IMPORTS_FROM {
                    import_source = Some(trim_quotes(text).to_string());
                }
            }

            if !import_aliases.is_empty() {
                import_names = import_aliases;
            }

            if let Some(source_path) = import_source {
                let mut resolved_path = self.resolve_import_path(&source_path, file);

                if resolved_path.starts_with("@/") {
                    resolved_path = resolved_path[2..].to_string();
                }

                let exts = [".ts", ".tsx", ".js", ".jsx"];
                if let Some(ext) = exts.iter().find(|&&e| resolved_path.ends_with(e)) {
                    resolved_path = resolved_path.trim_end_matches(ext).to_string();
                }

                results.push((resolved_path, import_names));
            }
        }

        if results.is_empty() {
            None
        } else {
            Some(results)
        }
    }

    // FROM REACT (identical in both): match_endpoint_groups
    fn match_endpoint_groups(
        &self,
        groups: &[NodeData],
        endpoints: &[NodeData],
        find_import_node: &dyn Fn(&str) -> Option<NodeData>,
    ) -> Vec<(NodeData, String)> {
        let mut matches = Vec::new();

        for group in groups {
            if let Some(router_var_name) = group.meta.get("group") {
                let prefix = &group.name;
                let group_file = &group.file;

                for endpoint in endpoints {
                    let endpoint_name = &endpoint.name;
                    let endpoint_file = &endpoint.file;

                    if endpoint_name.starts_with(prefix) {
                        continue;
                    }

                    if endpoint_file == group_file
                        && !endpoint_name.contains("/:")
                        && endpoint.meta.get("object") == Some(router_var_name)
                    {
                        matches.push((endpoint.clone(), prefix.clone()));
                        continue;
                    }

                    if let Some(resolved_source) =
                        self.resolve_import_source(router_var_name, group_file, find_import_node)
                    {
                        if endpoint_file.contains(&resolved_source) {
                            matches.push((endpoint.clone(), prefix.clone()));
                        } else {
                            let source_basename = std::path::Path::new(&resolved_source)
                                .file_stem()
                                .map(|s| s.to_string_lossy().to_string())
                                .unwrap_or_default();

                            let endpoint_basename = std::path::Path::new(endpoint_file)
                                .file_stem()
                                .map(|s| s.to_string_lossy().to_string())
                                .unwrap_or_default();

                            if !source_basename.is_empty()
                                && source_basename == endpoint_basename
                                && (resolved_source.starts_with('@')
                                    || resolved_source.starts_with("@/"))
                            {
                                matches.push((endpoint.clone(), prefix.clone()));
                            }
                        }
                    }
                }
            }
        }
        matches
    }
}

// Helper functions from react.rs

pub fn endpoint_name_from_file(file: &str) -> String {
    let path = file.replace('\\', "/");
    let route_path = if let Some(idx) = path.find("/api/") {
        let after_api = &path[idx..];
        after_api
            .trim_end_matches("/route.ts")
            .trim_end_matches("/route.js")
            .to_string()
    } else {
        file.to_string()
    };

    route_path
}

fn find_default_export_name(code: &str, language: Language) -> Option<String> {
    let query_str = r#"
    [
        (export_statement
            "default"
            (identifier) @component_name
        ) @export
        (export_statement
            "default" 
            (arrow_function) @arrow_func
        )@export
        (export_statement
            "default"
            (function_declaration
            name: (identifier) @component_name
            )
        ) @export
        (export_statement
        (export_clause
            (export_specifier
            name: (identifier) @component_name
            alias: (identifier) @alias (#eq? @alias "default")
            )
        )
        ) @export
    ]
    "#;

    let query = Query::new(&language, query_str).ok()?;
    let mut parser = Parser::new();
    parser.set_language(&language).ok()?;
    let tree = parser.parse(code, None)?;
    let root = tree.root_node();

    let mut cursor = QueryCursor::new();
    let mut matches = cursor.matches(&query, root, code.as_bytes());

    while let Some(m) = matches.next() {
        for cap in m.captures.iter() {
            let name = cap.node.utf8_text(code.as_bytes()).ok()?.to_string();
            // to handle the case for Arrow Functions
            if query.capture_names()[cap.index as usize] == "component_name" {
                return Some(name);
            }
        }
    }

    None
}

fn route_from_path(path: &str) -> String {
    if let Some(app_idx) = path.find("/app/") {
        let after_app = &path[app_idx + 4..];

        let after_app = after_app.strip_prefix('/').unwrap_or(after_app);

        let page_suffixes = ["/page.tsx", "/page.jsx", "/page.mdx", "/page.md"];

        let mut route = after_app;
        for suffix in &page_suffixes {
            if route == suffix.strip_prefix('/').unwrap_or(suffix) {
                // If the route is exactly "page.tsx" or "page.jsx", it's root
                return "/".to_string();
            }
            if route.ends_with(suffix) {
                route = &route[..route.len() - suffix.len()];
                break;
            }
        }
        if route.is_empty() {
            return "/".to_string();
        } else {
            return format!("/{}", route);
        }
    }

    if let Some(pages_idx) = path.find("/pages/") {
        let after_pages = &path[pages_idx + 6..];

        let after_pages = after_pages.strip_prefix('/').unwrap_or(after_pages);

        let file = after_pages;

        let file = file
            .trim_end_matches(".tsx")
            .trim_end_matches(".jsx")
            .trim_end_matches(".js")
            .trim_end_matches(".ts");

        if file == "index" || file.is_empty() {
            return "/".to_string();
        }

        if let Some(stripped) = file.strip_suffix("/index") {
            return format!("/{}", stripped);
        }

        return format!("/{}", file);
    }

    "/".to_string()
}

fn page_name(filename: &str) -> String {
    // App Router: use directory name
    if filename.contains("/app/") {
        let path = std::path::Path::new(filename);
        return path
            .parent()
            .and_then(|p| p.file_name())
            .and_then(|n| n.to_str())
            .unwrap_or("app")
            .to_string();
    }

    // Pages Router: use last part of the path || dir name if it's "index" || index if it's root
    if let Some(pages_idx) = filename.find("/pages/") {
        let after = &filename[pages_idx + 7..];
        let after = after.strip_prefix('/').unwrap_or(after);
        let file = after
            .trim_end_matches(".tsx")
            .trim_end_matches(".jsx")
            .trim_end_matches(".js")
            .trim_end_matches(".ts");

        if file == "index" || file.is_empty() {
            //root page
            return "index".to_string();
        }
        if file.ends_with("/index") {
            // index inside dir
            return file.rsplit('/').nth(1).unwrap_or("index").to_string();
        }
        // normal page
        return file.rsplit('/').next().unwrap_or(file).to_string();
    }

    "page".to_string()
}
