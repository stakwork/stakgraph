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
            ] @{FUNCTION_CALL}"
        )
    }

    fn endpoint_finders(&self) -> Vec<String> {
        vec![
            // Pattern 1: router.method(path, identifier) - middleware or named handler
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
            format!(
                r#"(call_expression
                    function: (member_expression
                        object: (identifier) @{ENDPOINT_OBJECT}
                        property: (property_identifier) @{ENDPOINT_VERB} (#match? @{ENDPOINT_VERB} "^get$|^post$|^put$|^delete$|^use$")
                    )
                    arguments: (arguments
                        (string) @{ENDPOINT}
                        (arrow_function) @{ARROW_FUNCTION_HANDLER}
                    )
                    ) @{ROUTE}
                "#
            ),
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
                    arguments: (arguments (arrow_function) @{ARROW_FUNCTION_HANDLER})
                    ) @{ROUTE}
                "#
            ),
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
                    arguments: (arguments (arrow_function) @{ARROW_FUNCTION_HANDLER})
                    ) @{ROUTE}
                "#
            ),
        ]
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

    fn generate_arrow_handler_name(&self, method: &str, path: &str, line: usize) -> Option<String> {
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
            || file_name.contains("/tests/")
            || file_name.contains("/test/")
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

    fn is_test(&self, _func_name: &str, func_file: &str, _func_body: &str) -> bool {
        if self.is_test_file(func_file) {
            true
        } else {
            false
        }
    }

    fn test_query(&self) -> Option<String> {
        Some(format!(
            r#"[
                    (call_expression
                        function: (identifier) @desc (#eq? @desc "describe")
                        arguments: (arguments [ (string) (template_string) ] @{FUNCTION_NAME})
                    )
                    (call_expression
                        function: (member_expression
                            object: (identifier) @desc2 (#eq? @desc2 "describe")
                            property: (property_identifier) @mod (#match? @mod "^(only|skip|todo)$")
                        )
                        arguments: (arguments [ (string) (template_string) ] @{FUNCTION_NAME})
                    )
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
                                object: (identifier) @obj (#eq? @test "test")
                                property: (property_identifier) @prop (#match? @prop "^(describe|skip|only|todo)$")
                            )
                            arguments: (arguments) @{FUNCTION_NAME}
                            )
                        )@{FUNCTION_DEFINITION}
                        )
                ] @{FUNCTION_DEFINITION}"#
        ))
    }
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

    fn should_skip_function_call(&self, called: &str, operand: &Option<String>) -> bool {
        consts::should_skip_js_function_call(called, operand)
    }

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
        let q = tree_sitter::Query::new(&self.0, &imports_query).unwrap();

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

                    // Skip if already prefixed
                    if endpoint_name.starts_with(prefix) {
                        continue;
                    }

                    // Same-file matching: check if endpoint's object metadata matches router variable
                    if endpoint_file == group_file
                        && !endpoint_name.contains("/:")
                        && endpoint.meta.get("object") == Some(router_var_name)
                    {
                        matches.push((endpoint.clone(), prefix.clone()));
                        continue;
                    }

                    // Cross-file matching: resolve import source and check if endpoint is from that file
                    // Check if router is imported in group file
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
