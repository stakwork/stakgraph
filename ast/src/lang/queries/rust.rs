use super::super::*;
use super::consts::*;
use crate::lang::parse::utils::trim_quotes;
use shared::error::{Context, Result};
use std::collections::HashMap;
use tree_sitter::{Language, Parser, Query, Tree};
pub struct Rust(Language);

impl Default for Rust {
    fn default() -> Self {
        Self::new()
    }
}

impl Rust {
    pub fn new() -> Self {
        Rust(tree_sitter_rust::LANGUAGE.into())
    }
}

impl Rust {
    // Actix-specific helper: captures .service(handler) calls (on scope or chained)
    fn actix_service_registration_query() -> String {
        use super::consts::HANDLER_REF;
        format!(
            r#"
            (call_expression
                function: (field_expression
                    field: (field_identifier) @service_name (#eq? @service_name "service")
                )
                arguments: (arguments
                    (identifier) @{HANDLER_REF}
                )
            ) @service_call
            "#
        )
    }

    // Walk up AST from .service() call to find the parent web::scope("/prefix")
    fn find_scope_prefix(node: tree_sitter::Node, code: &str) -> Option<String> {
        // The node passed in is the call_expression for .service()
        // Pattern: call_expression { function: field_expression { value: ..., field: "service" } }

        if let Some(func_node) = node.child_by_field_name("function") {
            if func_node.kind() == "field_expression" {
                if let Some(value_node) = func_node.child_by_field_name("value") {
                    if value_node.kind() == "call_expression" {
                        // Check if this is web::scope()
                        if let Some(scope_func) = value_node.child_by_field_name("function") {
                            let func_text = scope_func.utf8_text(code.as_bytes()).ok()?;
                            if func_text == "web::scope" {
                                // Found it! Extract the prefix argument
                                if let Some(args) = value_node.child_by_field_name("arguments") {
                                    for i in 0..args.child_count() {
                                        let child = args.child(i)?;
                                        if child.kind() == "string_literal" {
                                            let prefix = child.utf8_text(code.as_bytes()).ok()?;
                                            return Some(trim_quotes(prefix).to_string());
                                        }
                                    }
                                }
                            } else {
                                // Not web::scope, but could be chained .service()
                                // Recurse on this call_expression
                                return Self::find_scope_prefix(value_node, code);
                            }
                        }
                    }
                }
            }
        }

        None
    }

    fn find_nest_prefix(node: tree_sitter::Node, code: &str) -> Option<String> {
        if let Some(func_node) = node.child_by_field_name("function") {
            if func_node.kind() == "field_expression" {
                if let Some(field_node) = func_node.child_by_field_name("field") {
                    let field_text = field_node.utf8_text(code.as_bytes()).ok()?;

                    if field_text == "nest" {
                        if let Some(args_node) = node.child_by_field_name("arguments") {
                            for i in 0..args_node.child_count() {
                                let child = args_node.child(i)?;
                                if child.kind() == "string_literal" {
                                    let prefix = child.utf8_text(code.as_bytes()).ok()?;
                                    return Some(trim_quotes(prefix).to_string());
                                }
                            }
                        }
                    }
                }

                if let Some(value_node) = func_node.child_by_field_name("value") {
                    if let Some(parent_prefix) = Self::find_nest_prefix(value_node, code) {
                        return Some(parent_prefix);
                    }
                }
            }
        }

        None
    }

    fn extract_rocket_handlers(token_tree_text: &str) -> Vec<String> {
        let trimmed = token_tree_text.trim();

        let inner = trimmed
            .strip_prefix('[')
            .and_then(|s| s.strip_suffix(']'))
            .unwrap_or(trimmed);

        inner
            .split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect()
    }

    fn extract_axum_handlers(
        node: tree_sitter::Node,
        code: &str,
        handler_map: &mut HashMap<(String, String), Vec<String>>,
        file: &str,
        prefix: &str,
    ) {
        if node.kind() == "call_expression" {
            if let Some(func) = node.child_by_field_name("function") {
                if func.kind() == "field_expression" {
                    if let Some(field) = func.child_by_field_name("field") {
                        let field_text = field.utf8_text(code.as_bytes()).unwrap_or("");

                        if field_text == "route" {
                            if let Some(args) = node.child_by_field_name("arguments") {
                                let mut handler_name = None;

                                for i in 0..args.child_count() {
                                    if let Some(arg) = args.child(i) {
                                        if arg.kind() == "call_expression" {
                                            if let Some(handler_args) =
                                                arg.child_by_field_name("arguments")
                                            {
                                                for j in 0..handler_args.child_count() {
                                                    if let Some(h_arg) = handler_args.child(j) {
                                                        if h_arg.kind() == "identifier" {
                                                            handler_name = Some(
                                                                h_arg
                                                                    .utf8_text(code.as_bytes())
                                                                    .unwrap_or("")
                                                                    .to_string(),
                                                            );
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }

                                if let Some(h) = handler_name {
                                    let key = (h, file.to_string());
                                    handler_map.entry(key).or_default().push(prefix.to_string());
                                }
                            }
                        }
                    }

                    if let Some(value) = func.child_by_field_name("value") {
                        Self::extract_axum_handlers(value, code, handler_map, file, prefix);
                    }
                }
            }
        }
    }
}

impl Stack for Rust {
    fn q(&self, q: &str, nt: &NodeType) -> Query {
        if matches!(nt, NodeType::Library) {
            Query::new(&tree_sitter_toml_ng::LANGUAGE.into(), q).unwrap()
        } else {
            Query::new(&self.0, q).unwrap()
        }
    }

    fn parse(&self, code: &str, nt: &NodeType) -> Result<Tree> {
        let mut parser = Parser::new();

        if matches!(nt, NodeType::Library) {
            parser.set_language(&tree_sitter_toml_ng::LANGUAGE.into())?;
        } else {
            parser.set_language(&self.0)?;
        }

        parser.parse(code, None).context("failed to parse")
    }

    fn lib_query(&self) -> Option<String> {
        Some(format!(
            r#"(document
          (table 
            (bare_key) @section (#eq? @section "dependencies")
            (pair 
              (bare_key) @{LIBRARY_NAME} 
              (#not-eq? @{LIBRARY_NAME} "version")
              [
                ; Simple version string: package = "1.0.0"
                (string) @{LIBRARY_VERSION}
                
                ; Inline table with version
                (inline_table
                  (pair
                    (bare_key) @version_key (#eq? @version_key "version")
                    (string) @{LIBRARY_VERSION}
                  )
                )
              ]
            ) @{LIBRARY}
          )
        )"#
        ))
    }

    fn imports_query(&self) -> Option<String> {
        Some(format!(
            r#"
           (use_declaration 
                    argument: (scoped_use_list
                        path:(scoped_identifier
                            path: (crate)
                            name: (identifier)@{IMPORTS_FROM}
                        )?
                        name: (identifier)? @{IMPORTS_NAME}
                        path: (crate)?
                        list: (use_list
                                (identifier)?@{IMPORTS_NAME}
                                (scoped_identifier
                                    path: (identifier) @{IMPORTS_FROM}
                                    name: (identifier) @{IMPORTS_NAME}
                                )?
                        )?
                    )?
                argument:(scoped_identifier
                        path: (super)?@{IMPORTS_FROM}
                        path: (scoped_identifier
                            path:(_)
                            name:(identifier)@{IMPORTS_FROM}
                        )?
                        name: (identifier) @{IMPORTS_NAME}
                        
                        
                    )?
                )@{IMPORTS}

        "#
        ))
    }
    fn variables_query(&self) -> Option<String> {
        Some(format!(
            r#"
                (source_file
                    (const_item
                        name: (identifier) @{VARIABLE_NAME}
                        type: (_)? @{VARIABLE_TYPE}
                        value: (_) @{VARIABLE_VALUE}
                    )?@{VARIABLE_DECLARATION}
                    (static_item
                        name: (identifier) @{VARIABLE_NAME}
                        type: (_)? @{VARIABLE_TYPE}
                        value: (_) @{VARIABLE_VALUE}
                    )?@{VARIABLE_DECLARATION}
                    (let_declaration
                        pattern: (identifier) @{VARIABLE_NAME}
                        type: (_)? @{VARIABLE_TYPE}
                        value: (_) @{VARIABLE_VALUE}
                    )?@{VARIABLE_DECLARATION}
                )
                "#
        ))
    }
    fn trait_query(&self) -> Option<String> {
        Some(format!(
            r#"
            (trait_item
                name: (type_identifier) @{TRAIT_NAME}
                body: (declaration_list)
            ) @{TRAIT}
            "#
        ))
    }

    fn trait_comment_query(&self) -> Option<String> {
        Some(format!(
            r#"
            [
              (line_comment)+
              (block_comment)+
            ] @{TRAIT_COMMENT}
        "#
        ))
    }

    fn class_definition_query(&self) -> String {
        format!(
            r#"
           [
                (struct_item
                    name: (type_identifier) @{CLASS_NAME}
                )
                (enum_item
                    name: (type_identifier) @{CLASS_NAME}
                )
            ]@{CLASS_DEFINITION}
            "#
        )
    }

    fn implements_query(&self) -> Option<String> {
        Some(format!(
            r#"
        (impl_item
            trait: (type_identifier)? @{TRAIT_NAME}
            type: (type_identifier) @{CLASS_NAME}
            body: (declaration_list)?
        ) @{IMPLEMENTS}
        "#
        ))
    }

    fn function_definition_query(&self) -> String {
        format!(
            r#"
            (
              (attribute_item)* @{ATTRIBUTES}
              .
              (function_item
                name: (identifier) @{FUNCTION_NAME}
                parameters: (parameters) @{ARGUMENTS}
                return_type: (_)? @{RETURN_TYPES}
                body: (block)? @function.body) @{FUNCTION_DEFINITION}
            )
              
            (
              (attribute_item)* @{ATTRIBUTES}
              .
              (function_signature_item
                name: (identifier) @{FUNCTION_NAME}
                parameters: (parameters) @{ARGUMENTS}
                return_type: (_)? @{RETURN_TYPES}
               )@{FUNCTION_DEFINITION}
            )
            
            (macro_definition
              name: (identifier) @{FUNCTION_NAME}
              (token_tree)?
            ) @{MACRO} @{FUNCTION_DEFINITION}
            
            (impl_item
              type: (_) @{PARENT_TYPE}
              body: (declaration_list
                (
                  (attribute_item)* @{ATTRIBUTES}
                  .
                  (function_item
                    name: (identifier) @{FUNCTION_NAME}
                    parameters: (parameters) @{ARGUMENTS}
                    return_type: (_)? @{RETURN_TYPES}
                    body: (block)? @method.body) @method @{FUNCTION_DEFINITION}
                )
              )) @impl
            "#
        )
    }
    fn comment_query(&self) -> Option<String> {
        Some(format!(
            r#"
            [
              (line_comment)+
              (block_comment)+
            ] @{FUNCTION_COMMENT}
        "#
        ))
    }
    fn class_comment_query(&self) -> Option<String> {
        Some(format!(
            r#"
            [
              (line_comment)+
              (block_comment)+
            ] @{CLASS_COMMENT}
        "#
        ))
    }
    fn data_model_comment_query(&self) -> Option<String> {
        Some(format!(
            r#"
            [
              (line_comment)+
              (block_comment)+
            ] @{STRUCT_COMMENT}
        "#
        ))
    }
    fn endpoint_comment_query(&self) -> Option<String> {
        Some(format!(
            r#"
            [
              (line_comment)+
              (block_comment)+
            ] @{ENDPOINT_COMMENT}
        "#
        ))
    }
    fn var_comment_query(&self) -> Option<String> {
        Some(format!(
            r#"
            [
              (line_comment)+
              (block_comment)+
            ] @{VAR_COMMENT}
        "#
        ))
    }
    fn function_call_query(&self) -> String {
        format!(
            r#"
                (call_expression
                    function: [
                        (identifier) @{FUNCTION_NAME}
                        ;; module method
                        (scoped_identifier
                            path: (identifier) @{PARENT_NAME}
                            name: (identifier) @{FUNCTION_NAME}
                        )
                        ;; chained call
                        (field_expression
                            value: (identifier)? @{OPERAND}
                            field: (field_identifier) @{FUNCTION_NAME}
                        )
                    ]
                    arguments: (arguments) @{ARGUMENTS}
                ) @{FUNCTION_CALL}
                "#
        )
    }

    fn endpoint_finders(&self) -> Vec<String> {
        vec![
            format!(
                r#"
                (call_expression
                    (arguments
                        (string_literal) @{ENDPOINT}
                        (call_expression
                            function: (identifier) @{ENDPOINT_VERB} (#match? @{ENDPOINT_VERB} "^get$|^post$|^put$|^delete$")
                            arguments: (arguments
                                (identifier) @{HANDLER}
                            )
                        )
                    )
                ) @{ROUTE}
                "#
            ),
            // Method-specific routes (.get("/path", handler))
            format!(
                r#"
                (call_expression
                    function: (field_expression
                        field: (field_identifier) @http_method (#match? @http_method "^get$|^post$|^put$|^delete$")
                    )
                    arguments: (arguments
                        (string_literal) @{ENDPOINT}
                        (identifier) @{HANDLER}
                    )
                ) @{ROUTE}
        "#
            ),
            // Nested routes (.nest("/base", Router...))
            format!(
                r#"
                (call_expression
                    function: (field_expression
                        field: (field_identifier) @nest_method (#eq? @nest_method "nest")
                    )
                    arguments: (arguments
                        (string_literal) @base_path
                        (_) @nested_router
                    )
                ) @{ROUTE}
                "#
            ),
            // Actix/Rocket endpoint finder (#[get("/path")] or #[post("/path", data = "...")])
            format!(
                r#"
                 (
                    (attribute_item
                        (attribute
                        (identifier) @http_method (#match? @http_method "^get$|^post$|^put$|^delete$")
                        arguments: (token_tree
                            (string_literal) @{ENDPOINT} (#match? @{ENDPOINT} "^\"\/")
                        )
                        )
                    )
                    .
                    (function_item
                        name: (identifier) @{HANDLER}
                    )
                ) @{ROUTE}
            "#,
            ),
            // Anonymous closures in routes (.route("/path", get(|args| { ... })))
            format!(
                r#"
                (call_expression
                    function: (field_expression
                        field: (field_identifier) @route_method (#eq? @route_method "route")
                    )
                    arguments: (arguments
                        (string_literal) @{ENDPOINT}
                        (call_expression
                            function: (identifier) @endpoint-verb (#match? @endpoint-verb "^get$|^post$|^put$|^delete$|^patch$")
                            arguments: (arguments
                                (closure_expression) @{ANONYMOUS_FUNCTION}
                            )
                        )
                    )
                ) @{ROUTE}
                "#
            ),
        ]
    }

    fn endpoint_group_find(&self) -> Option<String> {
        Some(format!(
            r#"
            [
                (call_expression
                    function: (field_expression
                        value: (call_expression
                            function: (scoped_identifier
                                path: (identifier) @web (#eq? @web "web")
                                name: (identifier) @scope (#eq? @scope "scope")
                            )
                            arguments: (arguments
                                (string_literal) @{ENDPOINT}
                            )
                        )
                        field: (field_identifier) @configure (#eq? @configure "configure")
                    )
                    arguments: (arguments
                        (identifier) @{ENDPOINT_GROUP}
                    )
                ) @{ROUTE}
                
                (call_expression
                    function: (scoped_identifier
                        path: (identifier) @web2 (#eq? @web2 "web")
                        name: (identifier) @scope2 (#eq? @scope2 "scope")
                    )
                    arguments: (arguments
                        (string_literal) @{ENDPOINT}
                    )
                ) @{ROUTE}
                
                (call_expression
                    function: (field_expression
                        field: (field_identifier) @nest_method (#eq? @nest_method "nest")
                    )
                    arguments: (arguments
                        (string_literal) @{ENDPOINT}
                        (_) @router_arg
                    )
                ) @{ROUTE}
                
                (call_expression
                    function: (field_expression
                        field: (field_identifier) @mount_method (#eq? @mount_method "mount")
                    )
                    arguments: (arguments
                        (string_literal) @{ENDPOINT}
                        (macro_invocation
                            macro: (identifier) @routes_macro (#eq? @routes_macro "routes")
                            (token_tree) @{ENDPOINT_GROUP}
                        )
                    )
                ) @{ROUTE}
            ]
            "#
        ))
    }

    fn data_model_query(&self) -> Option<String> {
        Some(format!(
            r#"
                [
                    (
                        (attribute_item)* @{ATTRIBUTES}
                        .
                        (struct_item
                            name: (type_identifier) @{STRUCT_NAME}
                        ) @{STRUCT}
                    )
                    (
                        (struct_item
                            name: (type_identifier) @{STRUCT_NAME}
                        ) @{STRUCT}
                    )
                    (
                        (attribute_item)* @{ATTRIBUTES}
                        .
                        (enum_item
                            name: (type_identifier) @{STRUCT_NAME}
                        ) @{STRUCT}
                    )
                    (
                        (enum_item
                            name: (type_identifier) @{STRUCT_NAME}
                        ) @{STRUCT}
                    )
                    (type_item
                        name: (type_identifier)@{STRUCT_NAME}
                        type :(_)
                    )@{STRUCT}
                ]
            "#
        ))
    }
    fn data_model_within_query(&self) -> Option<String> {
        Some(format!(
            r#"
                (type_identifier) @{STRUCT_NAME}
            "#,
        ))
    }

    fn test_query(&self) -> Option<String> {
        None
    }

    fn add_endpoint_verb(&self, endpoint: &mut NodeData, call: &Option<String>) -> Option<String> {
        if let Some(verb) = endpoint.meta.remove("http_method") {
            let verb_upper = verb.to_uppercase();
            endpoint.add_verb(&verb_upper);
            return None;
        }

        if let Some(call_text) = call {
            if call_text.contains(".get(") || call_text.contains("get(") {
                endpoint.add_verb("GET");
                return None;
            } else if call_text.contains(".post(") || call_text.contains("post(") {
                endpoint.add_verb("POST");
                return None;
            } else if call_text.contains(".put(") || call_text.contains("put(") {
                endpoint.add_verb("PUT");
                return None;
            } else if call_text.contains(".delete(") || call_text.contains("delete(") {
                endpoint.add_verb("DELETE");
                return None;
            }
        }

        if let Some(handler) = endpoint.meta.get("handler") {
            let handler_lower = handler.to_lowercase();
            if handler_lower.starts_with("get_") {
                endpoint.add_verb("GET");
            } else if handler_lower.starts_with("post_") || handler_lower.starts_with("create_") {
                endpoint.add_verb("POST");
            } else if handler_lower.starts_with("put_") || handler_lower.starts_with("update_") {
                endpoint.add_verb("PUT");
            } else if handler_lower.starts_with("delete_") || handler_lower.starts_with("remove_") {
                endpoint.add_verb("DELETE");
            }
        }

        // Default to GET if no verb is found
        if !endpoint.meta.contains_key("verb") {
            endpoint.add_verb("GET");
        }
        None
    }

    fn resolve_import_path(&self, import_path: &str, _current_file: &str) -> String {
        let mut path = import_path.to_string();
        path = path.replace("::", "/");
        path
    }

    fn generate_anonymous_handler_name(
        &self,
        method: &str,
        path: &str,
        line: usize,
    ) -> Option<String> {
        let clean_method = method.to_uppercase();
        let clean_path = path
            .replace("/", "_")
            .replace(":", "param_")
            .trim_start_matches('_')
            .to_string();

        Some(format!("{}_{}_closure_L{}", clean_method, clean_path, line))
    }

    fn filter_by_implements(&self) -> bool {
        true
    }

    fn is_test_file(&self, filename: &str) -> bool {
        // Simplified: only used for test classification, not identification
        let normalized = filename.replace('\\', "/");
        normalized.contains("/tests/") || normalized.contains("/benches/")
    }

    fn is_test(&self, _func_name: &str, _func_file: &str, func_body: &str) -> bool {
        let test_patterns = [
            "#[test",
            "#[tokio::test",
            "#[actix_rt::test",
            "#[actix_web::test",
            "#[rstest",
            "#[rstest(",
            "#[proptest",
            "#[quickcheck",
            "#[wasm_bindgen_test",
            "#[bench",
        ];

        for pattern in &test_patterns {
            if func_body.contains(pattern) {
                return true;
            }
        }

        false
    }

    fn classify_test(&self, name: &str, file: &str, body: &str) -> NodeType {
        let f = file.replace('\\', "/");
        let fname = f.rsplit('/').next().unwrap_or(&f).to_lowercase();
        let name_lower = name.to_lowercase();

        if f.contains("/tests/e2e/")
            || f.contains("/e2e/")
            || fname.starts_with("e2e_")
            || fname.contains("e2e.rs")
            || name_lower.starts_with("e2e_")
            || name_lower.contains("_e2e_")
            || name_lower.contains("end_to_end")
        {
            return NodeType::E2eTest;
        }

        if f.contains("/tests/integration/")
            || fname.starts_with("integration_")
            || fname.contains("integration.rs")
            || name_lower.starts_with("integration_")
            || name_lower.contains("_integration_")
        {
            return NodeType::IntegrationTest;
        }

        if f.contains("/tests/") && !f.contains("/src/") {
            return NodeType::IntegrationTest;
        }

        let body_l = body.to_lowercase();
        let http_markers = [
            "reqwest::",
            "hyper::client",
            "actix_web::test",
            "rocket::local",
            ".get(",
            ".post(",
            "http://",
            "https://",
        ];

        if http_markers.iter().any(|m| body_l.contains(m)) {
            return NodeType::IntegrationTest;
        }

        NodeType::UnitTest
    }

    fn parse_imports_from_file(
        &self,
        file: &str,
        find_import_node: &dyn Fn(&str) -> Option<NodeData>,
    ) -> Option<Vec<(String, Vec<String>)>> {
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

            for capture in m.captures {
                let capture_name = q.capture_names()[capture.index as usize];
                let text = capture.node.utf8_text(code.as_bytes()).unwrap_or("");

                if capture_name == IMPORTS_NAME {
                    import_names.push(text.to_string());
                } else if capture_name == IMPORTS_FROM {
                    import_source = Some(trim_quotes(text).to_string());
                }
            }

            if let Some(source_path) = import_source {
                let resolved_path = self.resolve_import_path(&source_path, file);
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

        // Build handler-to-prefix mapping using Actix service registration query
        let reg_query_str = Rust::actix_service_registration_query();

        let reg_query = match tree_sitter::Query::new(&self.0, &reg_query_str) {
            Ok(q) => q,
            Err(_e) => {
                return matches;
            }
        };

        // Collect all (handler, file) -> prefix mappings from all files
        // Key is (handler_name, file_path) to avoid cross-framework collisions
        let mut handler_to_prefix: HashMap<(String, String), Vec<String>> = HashMap::new();

        // Get unique files from groups
        let mut files_to_scan: std::collections::HashSet<String> = std::collections::HashSet::new();
        for group in groups {
            files_to_scan.insert(group.file.clone());
        }

        for file in &files_to_scan {
            // Read the full file
            let code = match std::fs::read_to_string(file) {
                Ok(c) => c,
                Err(_e) => {
                    continue;
                }
            };

            let tree = match self.parse(&code, &NodeType::Endpoint) {
                Ok(t) => t,
                Err(_e) => {
                    continue;
                }
            };
            let mut cursor = QueryCursor::new();
            let mut query_matches = cursor.matches(&reg_query, tree.root_node(), code.as_bytes());

            while let Some(m) = query_matches.next() {
                let mut handler = None;
                let mut service_node = None;

                for capture in m.captures {
                    let capture_name = &reg_query.capture_names()[capture.index as usize];
                    let text = capture.node.utf8_text(code.as_bytes()).unwrap_or("");

                    if capture_name == &HANDLER_REF {
                        handler = Some(text.to_string());
                    } else if capture_name == &"service_call" {
                        service_node = Some(capture.node);
                    }
                }

                // Walk up AST to find web::scope() call
                if let (Some(h), Some(node)) = (handler, service_node) {
                    if let Some(prefix) = Self::find_scope_prefix(node, &code) {
                        // Store with file path to avoid matching across frameworks
                        let key = (h, file.clone());
                        handler_to_prefix.entry(key).or_default().push(prefix);
                    }
                }
            }
        }

        // Match endpoints by handler name AND file path (Actix)
        for endpoint in endpoints {
            if let Some(handler_name) = endpoint.meta.get("handler") {
                let key = (handler_name.clone(), endpoint.file.clone());
                if let Some(prefixes) = handler_to_prefix.get(&key) {
                    for prefix in prefixes {
                        matches.push((endpoint.clone(), prefix.clone()));
                    }
                }
            }
        }

        // Axum: scan for .nest() patterns
        let nest_query_str = r#"
            (call_expression
                function: (field_expression
                    field: (field_identifier) @nest_method
                )
                arguments: (arguments) @nest_args
            ) @nest_call
        "#;

        let nest_query = match tree_sitter::Query::new(&self.0, nest_query_str) {
            Ok(q) => q,
            Err(_) => {
                return matches;
            }
        };

        let mut axum_handler_to_prefix: HashMap<(String, String), Vec<String>> = HashMap::new();

        for file in &files_to_scan {
            let code = match std::fs::read_to_string(file) {
                Ok(c) => c,
                Err(_) => continue,
            };

            let tree = match self.parse(&code, &NodeType::Endpoint) {
                Ok(t) => t,
                Err(_) => continue,
            };

            let mut cursor = QueryCursor::new();
            let mut query_matches = cursor.matches(&nest_query, tree.root_node(), code.as_bytes());

            while let Some(m) = query_matches.next() {
                let mut nest_node = None;
                let mut is_nest = false;

                for capture in m.captures {
                    let capture_name = nest_query.capture_names()[capture.index as usize];
                    let text = capture.node.utf8_text(code.as_bytes()).unwrap_or("");

                    if capture_name == "nest_method" && text == "nest" {
                        is_nest = true;
                    } else if capture_name == "nest_call" {
                        nest_node = Some(capture.node);
                    }
                }

                if !is_nest {
                    continue;
                }

                if let Some(node) = nest_node {
                    if let Some(prefix) = Self::find_nest_prefix(node, &code) {
                        if let Some(args) = node.child_by_field_name("arguments") {
                            for i in 0..args.child_count() {
                                if let Some(child) = args.child(i) {
                                    if child.kind() == "call_expression" {
                                        let router_text =
                                            child.utf8_text(code.as_bytes()).unwrap_or("");

                                        if router_text.contains("Router::new()") {
                                            Self::extract_axum_handlers(
                                                child,
                                                &code,
                                                &mut axum_handler_to_prefix,
                                                file,
                                                &prefix,
                                            );
                                        } else if let Some(func) =
                                            child.child_by_field_name("function")
                                        {
                                            let func_text =
                                                func.utf8_text(code.as_bytes()).unwrap_or("");

                                            if let Some(resolved_source) = self
                                                .resolve_import_source(
                                                    func_text,
                                                    file,
                                                    find_import_node,
                                                )
                                            {
                                                for endpoint in endpoints {
                                                    if endpoint.file.contains(&resolved_source) {
                                                        if let Some(handler) =
                                                            endpoint.meta.get("handler")
                                                        {
                                                            let key = (
                                                                handler.clone(),
                                                                endpoint.file.clone(),
                                                            );
                                                            axum_handler_to_prefix
                                                                .entry(key)
                                                                .or_default()
                                                                .push(prefix.clone());
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        for endpoint in endpoints {
            if let Some(handler_name) = endpoint.meta.get("handler") {
                let key = (handler_name.clone(), endpoint.file.clone());
                if let Some(prefixes) = axum_handler_to_prefix.get(&key) {
                    for prefix in prefixes {
                        matches.push((endpoint.clone(), prefix.clone()));
                    }
                }
            }
        }

        let mount_query_str = r#"
            (call_expression
                function: (field_expression
                    field: (field_identifier) @mount_method
                )
                arguments: (arguments
                    (string_literal) @prefix
                    (macro_invocation
                        macro: (identifier) @macro_name
                        (token_tree) @handlers_token
                    )
                )
            ) @mount_call
        "#;

        let mount_query = match tree_sitter::Query::new(&self.0, mount_query_str) {
            Ok(q) => q,
            Err(_) => {
                return matches;
            }
        };

        let mut rocket_handler_to_prefix: HashMap<(String, String), Vec<String>> = HashMap::new();

        for file in &files_to_scan {
            let code = match std::fs::read_to_string(file) {
                Ok(c) => c,
                Err(_) => continue,
            };

            let tree = match self.parse(&code, &NodeType::Endpoint) {
                Ok(t) => t,
                Err(_) => continue,
            };

            let mut cursor = QueryCursor::new();
            let mut query_matches = cursor.matches(&mount_query, tree.root_node(), code.as_bytes());

            while let Some(m) = query_matches.next() {
                let mut prefix = None;
                let mut handlers_text = None;
                let mut is_mount = false;
                let mut is_routes_macro = false;

                for capture in m.captures {
                    let capture_name = mount_query.capture_names()[capture.index as usize];
                    let text = capture.node.utf8_text(code.as_bytes()).unwrap_or("");

                    if capture_name == "mount_method" && text == "mount" {
                        is_mount = true;
                    } else if capture_name == "macro_name" && text == "routes" {
                        is_routes_macro = true;
                    } else if capture_name == "prefix" {
                        prefix = Some(text.trim_matches('"').to_string());
                    } else if capture_name == "handlers_token" {
                        handlers_text = Some(text.to_string());
                    }
                }

                if !is_mount || !is_routes_macro {
                    continue;
                }

                if let (Some(p), Some(token_text)) = (prefix, handlers_text) {
                    if p == "/" {
                        continue;
                    }

                    let handlers = Self::extract_rocket_handlers(&token_text);

                    for handler in &handlers {
                        for endpoint in endpoints {
                            if let Some(ep_handler) = endpoint.meta.get("handler") {
                                if ep_handler == handler && endpoint.file.contains("rocket") {
                                    let key = (ep_handler.clone(), endpoint.file.clone());
                                    rocket_handler_to_prefix
                                        .entry(key)
                                        .or_default()
                                        .push(p.clone());
                                }
                            }
                        }
                    }
                }
            }
        }

        for endpoint in endpoints {
            if let Some(handler_name) = endpoint.meta.get("handler") {
                let key = (handler_name.clone(), endpoint.file.clone());
                if let Some(prefixes) = rocket_handler_to_prefix.get(&key) {
                    for prefix in prefixes {
                        matches.push((endpoint.clone(), prefix.clone()));
                    }
                }
            }
        }

        // Also handle cross-file grouping via .configure() as before (Actix)
        for group in groups {
            let prefix = &group.name;
            let group_file = &group.file;

            if let Some(router_var) = group.meta.get("group") {
                for endpoint in endpoints {
                    let endpoint_name = &endpoint.name;
                    let endpoint_file = &endpoint.file;

                    if endpoint_name.starts_with(prefix) {
                        continue;
                    }

                    if let Some(resolved_source) =
                        self.resolve_import_source(router_var, group_file, find_import_node)
                    {
                        if endpoint_file.contains(&resolved_source) {
                            matches.push((endpoint.clone(), prefix.clone()));
                        }
                    }
                }
            }
        }

        matches
    }
}
