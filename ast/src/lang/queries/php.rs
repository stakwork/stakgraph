use super::super::*;
use super::consts::*;
use super::HandlerParams;
use crate::lang::parse::trim_quotes;
use inflection_rs::inflection;
use shared::error::{Context, Result};
use tree_sitter::{Language, Parser, Query, Tree};

pub struct Php(Language);

impl Default for Php {
    fn default() -> Self {
        Self::new()
    }
}

impl Php {
    pub fn new() -> Self {
        Php(tree_sitter_php::LANGUAGE_PHP.into())
    }
}

impl Stack for Php {
    fn q(&self, q: &str, nt: &NodeType) -> Query {
        if matches!(nt, NodeType::Library) {
            Query::new(&tree_sitter_json::LANGUAGE.into(), q).unwrap()
        } else {
            Query::new(&self.0, q).unwrap()
        }
    }

    fn parse(&self, code: &str, nt: &NodeType) -> Result<Tree> {
        let mut parser = Parser::new();
        if matches!(nt, NodeType::Library) {
            parser.set_language(&tree_sitter_json::LANGUAGE.into())?;
        } else {
            parser.set_language(&self.0)?;
        }
        parser.parse(code, None).context("failed to parse")
    }

    fn lib_query(&self) -> Option<String> {
        Some(format!(
            r#"(pair
                key: (string (string_content) @req (#eq? @req "require"))
                value: (object
                    (pair
                        key: (string (string_content) @{LIBRARY_NAME})
                        value: (string (string_content) @{LIBRARY_VERSION})
                    ) @{LIBRARY}
                )
            )"#
        ))
    }

    fn imports_query(&self) -> Option<String> {
        Some(format!(
            r#"(namespace_use_declaration
                (namespace_use_clause
                    (qualified_name) @{IMPORTS_NAME}
                )
            ) @{IMPORTS}
            
            (function_call_expression
                function: (name) @fn_name (#match? @fn_name "^(require|require_once|include|include_once)$")
                arguments: (arguments
                    (argument
                        (string) @{IMPORTS_NAME}
                    )
                )
            ) @{IMPORTS}
            "#
        ))
    }

    fn class_definition_query(&self) -> String {
        format!(
            r#"[
                (class_declaration
                    name: (name) @{CLASS_NAME}
                    (base_clause
                        (name) @{CLASS_PARENT}
                    )?
                    (class_interface_clause
                        (name) @{INCLUDED_MODULES}
                    )?
                    (declaration_list
                        (use_declaration
                            (name) @{INCLUDED_MODULES}
                        )?
                    )?
                ) @{CLASS_DEFINITION}
                
                (class_declaration
                    name: (name) @{CLASS_NAME}
                    (base_clause
                        (name) @{CLASS_PARENT}
                    )?
                    (class_interface_clause
                        (name) @{INCLUDED_MODULES}
                    )?
                ) @{CLASS_DEFINITION}
                
                (interface_declaration
                    name: (name) @{CLASS_NAME}
                ) @{CLASS_DEFINITION}
                
                (trait_declaration
                    name: (name) @{CLASS_NAME}
                ) @{CLASS_DEFINITION}
            ]"#
        )
    }

    fn trait_query(&self) -> Option<String> {
        Some(format!(
            r#"(trait_declaration
                name: (name) @{TRAIT_NAME}
            ) @{CLASS_DEFINITION}"#
        ))
    }

    fn function_definition_query(&self) -> String {
        format!(
            r#"[
                (function_definition
                    name: (name) @{FUNCTION_NAME}
                    parameters: (formal_parameters) @{ARGUMENTS}
                ) @{FUNCTION_DEFINITION}
                
                (method_declaration
                    name: (name) @{FUNCTION_NAME}
                    parameters: (formal_parameters) @{ARGUMENTS}
                ) @{FUNCTION_DEFINITION}
            ]"#
        )
    }

    fn comment_query(&self) -> Option<String> {
        Some(format!(
            r#"[
                (comment)
            ]+ @{FUNCTION_COMMENT}"#
        ))
    }

    fn function_call_query(&self) -> String {
        format!(
            r#"[
                (function_call_expression
                    function: (name) @{FUNCTION_NAME}
                    arguments: (arguments) @{ARGUMENTS}
                ) @{FUNCTION_CALL}
                
                (function_call_expression
                    function: (qualified_name) @{FUNCTION_NAME}
                    arguments: (arguments) @{ARGUMENTS}
                ) @{FUNCTION_CALL}
                
                (member_call_expression
                    object: (_) @{OPERAND}
                    name: (name) @{FUNCTION_NAME}
                    arguments: (arguments) @{ARGUMENTS}
                ) @{FUNCTION_CALL}
                
                (scoped_call_expression
                    scope: (_) @{OPERAND}
                    name: (name) @{FUNCTION_NAME}
                    arguments: (arguments) @{ARGUMENTS}
                ) @{FUNCTION_CALL}
            ]"#
        )
    }

    fn variables_query(&self) -> Option<String> {
        Some(format!(
            r#"(expression_statement
                (assignment_expression
                    left: (variable_name) @{VARIABLE_NAME}
                    right: (_) @{VARIABLE_VALUE}
                )
            ) @{VARIABLE_DECLARATION}"#
        ))
    }

    fn identifier_query(&self) -> String {
        "(name) @identifier".to_string()
    }

    fn is_test(&self, func_name: &str, func_file: &str, func_body: &str) -> bool {
        if self.is_test_file(func_file) {
            return true;
        }
        let lower_name = func_name.to_lowercase();
        lower_name.starts_with("test")
            || func_body.contains("@test")
            || func_body.contains("PHPUnit")
    }

    fn is_test_file(&self, filename: &str) -> bool {
        let f = filename.to_lowercase();
        f.ends_with("test.php")
            || f.ends_with("_test.php")
            || f.contains("/tests/")
            || f.contains("/test/")
            || f.contains("/spec/")
    }

    fn is_e2e_test_file(&self, file: &str, code: &str) -> bool {
        let f = file.to_lowercase();
        let c = code.to_lowercase();

        if f.contains("/e2e/")
            || f.contains("/browser/")
            || f.contains("/acceptance/")
            || f.contains("/feature/")
        {
            return true;
        }

        c.contains("dusk") || c.contains("panther") || c.contains("mink") || c.contains("selenium")
    }

    fn classify_test(&self, name: &str, file: &str, body: &str) -> NodeType {
        let f = file.to_lowercase();
        let b = body.to_lowercase();

        if f.contains("/e2e/")
            || f.contains("/browser/")
            || f.contains("/acceptance/")
            || b.contains("dusk")
            || b.contains("panther")
        {
            return NodeType::E2eTest;
        }

        if f.contains("/feature/")
            || f.contains("/integration/")
            || f.contains("/api/")
            || b.contains("$this->get(")
            || b.contains("$this->post(")
            || b.contains("$this->json(")
            || b.contains("actingas")
        {
            return NodeType::IntegrationTest;
        }

        if f.contains("/unit/") || f.contains("/models/") || f.contains("/services/") {
            return NodeType::UnitTest;
        }

        let lname = name.to_lowercase();
        if lname.contains("e2e") || lname.contains("browser") {
            return NodeType::E2eTest;
        }
        if lname.contains("integration") || lname.contains("feature") || lname.contains("api") {
            return NodeType::IntegrationTest;
        }

        NodeType::UnitTest
    }

    fn test_query(&self) -> Option<String> {
        Some(format!(
            r#"[
                (method_declaration
                    (attribute_list
                        (attribute_group
                            (attribute
                                (name) @attr_name (#eq? @attr_name "Test")
                            )
                        )
                    )+
                    name: (name) @test_name @{FUNCTION_NAME}
                    (#match? @test_name "^test")
                ) @{FUNCTION_DEFINITION}
                
                (method_declaration
                    (attribute_list
                        (attribute_group
                            (attribute
                                (name) @attr_name (#eq? @attr_name "Test")
                            )
                        )
                    )+
                    name: (name) @{FUNCTION_NAME}
                ) @{FUNCTION_DEFINITION}

                (function_call_expression
                    function: (name) @test_func (#match? @test_func "^test$|^it$")
                    arguments: (arguments)
                ) @{FUNCTION_DEFINITION}
            ]"#
        ))
    }

    fn endpoint_finders(&self) -> Vec<String> {
        vec![
            format!(
                r#"(scoped_call_expression
                    scope: (name) @scope (#eq? @scope "Route")
                    name: (name) @{ENDPOINT_VERB} (#match? @{ENDPOINT_VERB} "^resource$|^apiResource$")
                    arguments: (arguments
                        (argument !name (string) @{ENDPOINT})
                        (argument) @{HANDLER}
                    )
                ) @{ROUTE}"#
            ),
            format!(
                r#"(scoped_call_expression
                    scope: (name) @scope (#eq? @scope "Route")
                    name: (name) @{ENDPOINT_VERB} (#match? @{ENDPOINT_VERB} "^get$|^post$|^put$|^delete$|^patch$|^options$")
                    arguments: (arguments
                        (argument) @{ENDPOINT}
                        (argument) @{HANDLER}
                        (#not-match? @{HANDLER} "^function|^fn|^static")
                    )
                ) @{ROUTE}"#
            ),
            format!(
                r#"(method_declaration
                    (attribute_list
                        (attribute_group
                            (attribute
                                (name) @attr_name (#eq? @attr_name "Route")
                                parameters: (arguments
                                    (argument !name (string) @{ENDPOINT})
                                    (_)*
                                )
                            )
                        )
                    )
                    name: (name) @{HANDLER}
                ) @{ROUTE}"#
            ),
            format!(
                r#"(scoped_call_expression
                    scope: (name) @scope (#eq? @scope "Route")
                    name: (name) @{ENDPOINT_VERB} (#match? @{ENDPOINT_VERB} "^get$|^post$|^put$|^delete$|^patch$|^options$")
                    arguments: (arguments
                        (argument !name (string) @{ENDPOINT})
                        (argument
                            [
                                (anonymous_function)
                                (arrow_function)
                            ] @{ANONYMOUS_FUNCTION}
                        )
                    )
                ) @{ROUTE}"#
            ),
            // Chained calls (e.g. Route::middleware(...)->get(...)) - Closures
            format!(
                r#"(member_call_expression
                    name: (name) @{ENDPOINT_VERB} (#match? @{ENDPOINT_VERB} "^get$|^post$|^put$|^delete$|^patch$|^options$")
                    arguments: (arguments
                        (argument !name (string) @{ENDPOINT})
                        (argument
                            [
                                (anonymous_function)
                                (arrow_function)
                            ] @{ANONYMOUS_FUNCTION}
                        )
                    )
                ) @{ROUTE}"#
            ),
            // Chained calls - Named Handlers
            format!(
                r#"(member_call_expression
                    name: (name) @{ENDPOINT_VERB} (#match? @{ENDPOINT_VERB} "^get$|^post$|^put$|^delete$|^patch$|^options$")
                    arguments: (arguments
                        (argument !name (string) @{ENDPOINT})
                        (argument) @{HANDLER}
                    )
                ) @{ROUTE}"#
            ),
            // Route::controller(UserController::class)->group(...)
            format!(
                r#"(expression_statement
                    (member_call_expression
                        (scoped_call_expression
                            scope: (name) @scope (#eq? @scope "Route")
                            name: (name) @c_verb (#eq? @c_verb "controller")
                            arguments: (arguments (argument) @{CONTROLLER_CONTEXT})
                        )
                        name: (name) @g_verb (#eq? @g_verb "group")
                        arguments: (arguments 
                            (argument 
                                (anonymous_function 
                                    body: (compound_statement
                                        (expression_statement
                                            (scoped_call_expression
                                                name: (name) @{ENDPOINT_VERB} (#match? @{ENDPOINT_VERB} "^get$|^post$|^put$|^delete$|^patch$|^options$")
                                                arguments: (arguments
                                                    (argument) @{ENDPOINT}
                                                    (argument) @{HANDLER}
                                                )
                                            ) @{ROUTE}
                                        )
                                    )
                                )
                            )
                        )
                    )
                )"#
            ),
        ]
    }

    fn e2e_test_id_finder_string(&self) -> Option<String> {
        Some(
            r#"
            (method_declaration
               name: (name) @test_name
               (#match? @test_name "^test_")
            ) @{FUNCTION_DEFINITION}
            "#
            .to_string(),
        )
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
        while parent.is_some() && parent.unwrap().kind() != "class_declaration" {
            parent = parent.unwrap().parent();
        }
        let parent_of = match parent {
            Some(p) => {
                let query = self.q(&self.identifier_query(), &NodeType::Class);
                query_to_ident(query, p, code)?.map(|parent_name| Operand {
                    source: NodeKeys::new(&parent_name, file, p.start_position().row),
                    target: NodeKeys::new(func_name, file, node.start_position().row),
                })
            }
            None => None,
        };
        Ok(parent_of)
    }

    fn find_endpoint_parents(
        &self,
        node: TreeNode,
        code: &str,
        _file: &str,
        _callback: &dyn Fn(&str) -> Option<NodeData>,
    ) -> Result<Vec<HandlerItem>> {
        let mut parents = Vec::new();
        let mut current = node.parent();

        while let Some(parent_node) = current {
            // Walk the chain of method calls for this parent
            let mut chain_node = Some(parent_node.clone());
            while let Some(node) = chain_node {
                if node.kind() == "member_call_expression" {
                    if let Some(method) = node.child_by_field_name("name") {
                        let method_name = method.utf8_text(code.as_bytes()).unwrap_or("");
                        if method_name == "prefix" || method_name == "name" {
                            if let Some(args) = node.child_by_field_name("arguments") {
                                if let Some(first_arg) = args.named_child(0) {
                                    let prefix = first_arg.utf8_text(code.as_bytes()).unwrap_or("");
                                    let cleaned =
                                        trim_quotes(prefix).trim_start_matches('/').to_string();
                                    if !cleaned.is_empty() {
                                        parents.push(HandlerItem {
                                            name: cleaned,
                                            item_type: HandlerItemType::Namespace,
                                        });
                                    }
                                }
                            }
                        } else if method_name == "middleware" {
                            if let Some(args) = node.child_by_field_name("arguments") {
                                if let Some(first_arg) = args.named_child(0) {
                                    let content =
                                        first_arg.utf8_text(code.as_bytes()).unwrap_or("");
                                    let cleaned = trim_quotes(content).to_string();
                                    if !cleaned.is_empty() {
                                        parents.push(HandlerItem {
                                            name: cleaned,
                                            item_type: HandlerItemType::Middleware,
                                        });
                                    }
                                }
                            }
                        }
                    }
                    chain_node = node.child_by_field_name("object");
                } else if node.kind() == "scoped_call_expression" {
                    // Route::prefix(...) or Route::middleware(...)
                    if let Some(scope) = node.child_by_field_name("scope") {
                        let scope_name = scope.utf8_text(code.as_bytes()).unwrap_or("");
                        if scope_name == "Route" {
                            if let Some(method) = node.child_by_field_name("name") {
                                let method_name = method.utf8_text(code.as_bytes()).unwrap_or("");
                                if method_name == "prefix" {
                                    if let Some(args) = node.child_by_field_name("arguments") {
                                        if let Some(first_arg) = args.named_child(0) {
                                            let prefix =
                                                first_arg.utf8_text(code.as_bytes()).unwrap_or("");
                                            let cleaned = trim_quotes(prefix)
                                                .trim_start_matches('/')
                                                .to_string();
                                            if !cleaned.is_empty() {
                                                parents.push(HandlerItem {
                                                    name: cleaned,
                                                    item_type: HandlerItemType::Namespace,
                                                });
                                            }
                                        }
                                    }
                                } else if method_name == "middleware" {
                                    if let Some(args) = node.child_by_field_name("arguments") {
                                        if let Some(first_arg) = args.named_child(0) {
                                            let mw =
                                                first_arg.utf8_text(code.as_bytes()).unwrap_or("");
                                            let cleaned = trim_quotes(mw).to_string();
                                            if !cleaned.is_empty() {
                                                parents.push(HandlerItem {
                                                    name: cleaned,
                                                    item_type: HandlerItemType::Middleware,
                                                });
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    chain_node = None;
                } else {
                    chain_node = None;
                }
            }
            // Check for class_declaration with #[Route('/prefix')] attribute
            if parent_node.kind() == "class_declaration" {
                // Look for attribute_list child
                for child in parent_node.children(&mut parent_node.walk()) {
                    if child.kind() == "attribute_list" {
                        // Look for attribute_group containing Route
                        for attr_group in child.children(&mut child.walk()) {
                            if attr_group.kind() == "attribute_group" {
                                for attr in attr_group.children(&mut attr_group.walk()) {
                                    if attr.kind() == "attribute" {
                                        // Check if this is a Route attribute
                                        if let Some(name_node) = attr.child_by_field_name("name") {
                                            let attr_name =
                                                name_node.utf8_text(code.as_bytes()).unwrap_or("");
                                            if attr_name == "Route" {
                                                // Extract the first argument (the path)
                                                if let Some(params) =
                                                    attr.child_by_field_name("parameters")
                                                {
                                                    if let Some(first_arg) = params.named_child(0) {
                                                        // Get the argument value (could be nested)
                                                        let path = first_arg
                                                            .utf8_text(code.as_bytes())
                                                            .unwrap_or("");
                                                        let cleaned = trim_quotes(path)
                                                            .trim_start_matches('/')
                                                            .to_string();
                                                        if !cleaned.is_empty() {
                                                            parents.push(HandlerItem {
                                                                name: cleaned,
                                                                item_type:
                                                                    HandlerItemType::Namespace,
                                                            });
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
            current = parent_node.parent();
        }

        // Reverse so outermost parents come first
        parents.reverse();
        Ok(parents)
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

    fn handler_finder(
        &self,
        mut endpoint: NodeData,
        find_fn: &dyn Fn(&str, &str) -> Option<NodeData>,
        find_fns_in: &dyn Fn(&str) -> Vec<NodeData>,
        _handler_params: HandlerParams,
    ) -> Vec<(NodeData, Option<Edge>)> {
        let mut edges = Vec::new();

        // Handle controller_context (from Route::controller group)
        if let Some(handler) = endpoint.meta.get("handler").cloned() {
            if let Some(ctx) = endpoint.meta.get("controller_context") {
                // If handler is just a method (e.g. 'index') and ctx is a class (e.g. UserController::class)
                // Construct full handler: [UserController::class, 'index']
                if !handler.contains('[') && !handler.contains("::") {
                    let new_handler = format!("[{}, '{}']", ctx, handler);
                    endpoint.meta.insert("handler".to_string(), new_handler);
                }
            }
        }

        // Normalize verb to lowercase to handle potential uppercasing by system
        let verb_raw = endpoint.meta.get("verb").map(|s| s.as_str()).unwrap_or("");
        let verb = verb_raw.to_lowercase();

        if verb == "resource" || verb == "apiresource" {
            let resource_name = trim_quotes(&endpoint.name);
            let singular_name = inflection::singularize(&resource_name);
            let handler_str = endpoint
                .meta
                .get("handler")
                .unwrap_or(&String::new())
                .to_string();

            // Clean controller name: "UserController::class" -> "UserController"
            // Also handle fully qualified names: "App\Http\Controllers\UserController" -> "UserController"
            let msg = handler_str.replace("::class", "");
            let parts: Vec<&str> = msg.split('\\').collect();
            let controller_name = parts
                .last()
                .unwrap_or(&msg.as_str())
                .trim_matches('\'')
                .trim_matches('"');

            // Find controller file (heuristic: ends with ControllerName.php)
            let methods = find_fns_in(&format!("{}.php", controller_name));

            // Define standard REST actions
            let mut actions = vec![
                ("index", "GET", format!("/{}", resource_name)),
                ("store", "POST", format!("/{}", resource_name)),
                (
                    "show",
                    "GET",
                    format!("/{}/{{{}}}", resource_name, singular_name),
                ),
                (
                    "update",
                    "PUT",
                    format!("/{}/{{{}}}", resource_name, singular_name),
                ), // PUT/PATCH
                (
                    "destroy",
                    "DELETE",
                    format!("/{}/{{{}}}", resource_name, singular_name),
                ),
            ];

            if verb == "resource" {
                actions.push(("create", "GET", format!("/{}/create", resource_name)));
                actions.push((
                    "edit",
                    "GET",
                    format!("/{}/{{{}}}/edit", resource_name, singular_name),
                ));
            }

            for (method, method_verb, path) in actions {
                if let Some(target_method) = methods.iter().find(|m| m.name == method) {
                    let mut new_endpoint = endpoint.clone();
                    new_endpoint.name = path;
                    new_endpoint
                        .meta
                        .insert("verb".to_string(), method_verb.to_string());

                    edges.push((
                        new_endpoint.clone(),
                        Some(Edge::handler(&new_endpoint, target_method)),
                    ));
                } else {
                    let mut new_endpoint = endpoint.clone();
                    new_endpoint.name = path;
                    new_endpoint
                        .meta
                        .insert("verb".to_string(), method_verb.to_string());
                    edges.push((new_endpoint, None));
                }
            }

            return edges;
        }

        let mut edge = None;

        if let Some(handler) = endpoint.meta.get("handler") {
            // Basic support for [Class::class, 'method']
            if handler.starts_with('[') && handler.ends_with(']') {
                let content = &handler[1..handler.len() - 1];
                let parts: Vec<&str> = content.split(',').map(|s| s.trim()).collect();
                if parts.len() == 2 {
                    // Try to match the method name for now (part 1)
                    let method_part = parts[1].trim_matches('\'').trim_matches('"');

                    // Extract controller name for file lookup?
                    let class_part = parts[0];
                    // Class::class or "Class"
                    let cls = class_part.replace("::class", "");
                    let cls_parts: Vec<&str> = cls.split('\\').collect();
                    let cls_name = cls_parts
                        .last()
                        .unwrap_or(&cls.as_str())
                        .trim_matches('\'')
                        .trim_matches('"');

                    if let Some(nd) = find_fn(method_part, &format!("{}.php", cls_name)) {
                        edge = Some(Edge::handler(&endpoint, &nd));
                    } else if let Some(nd) = find_fn(method_part, &endpoint.file) {
                        // Fallback to same file
                        edge = Some(Edge::handler(&endpoint, &nd));
                    }
                }
            } else {
                if let Some(nd) = find_fn(handler, &endpoint.file) {
                    edge = Some(Edge::handler(&endpoint, &nd));
                }
            }
        }

        vec![(endpoint, edge)]
    }

    fn convert_association_to_name(&self, name: &str) -> String {
        // Handle PHP class references like "Post" from class_constant_access
        name.to_string()
    }
}

fn query_to_ident(query: Query, node: TreeNode, code: &str) -> Result<Option<String>> {
    let mut cursor = tree_sitter::QueryCursor::new();
    let mut matches = cursor.matches(&query, node, code.as_bytes());
    if let Some(m) = matches.next() {
        if let Some(cap) = m.captures.first() {
            let text = cap.node.utf8_text(code.as_bytes())?;
            return Ok(Some(text.to_string()));
        }
    }
    Ok(None)
}
