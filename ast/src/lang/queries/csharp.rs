use super::super::*;
use super::consts::*;
use super::HandlerParams;
use crate::lang::parse::trim_quotes;
use shared::error::{Context, Result};
use tree_sitter::{Language, Parser, Query, Tree};

pub struct CSharp(Language);

impl Default for CSharp {
    fn default() -> Self {
        Self::new()
    }
}

impl CSharp {
    pub fn new() -> Self {
        CSharp(tree_sitter_c_sharp::LANGUAGE.into())
    }
}

impl Stack for CSharp {
    fn q(&self, q: &str, nt: &NodeType) -> Query {
        if matches!(nt, NodeType::Library) {
            // .csproj files are XML
            Query::new(&tree_sitter_html::LANGUAGE.into(), q).unwrap()
        } else {
            Query::new(&self.0, q).unwrap()
        }
    }

    fn parse(&self, code: &str, nt: &NodeType) -> Result<Tree> {
        let mut parser = Parser::new();
        if matches!(nt, NodeType::Library) {
            parser.set_language(&tree_sitter_html::LANGUAGE.into())?;
        } else {
            parser.set_language(&self.0)?;
        }
        parser.parse(code, None).context("failed to parse")
    }

    fn lib_query(&self) -> Option<String> {
        // Parse NuGet PackageReference from .csproj files (XML-like)
        Some(format!(
            r#"[
                (element
                    (start_tag
                        (tag_name) @tag (#eq? @tag "PackageReference")
                        (_)*
                        (attribute
                            (attribute_name) @attr (#eq? @attr "Include")
                            (quoted_attribute_value (attribute_value) @{LIBRARY_NAME})
                        )
                        (_)*
                        (attribute
                            (attribute_name) @ver_attr (#eq? @ver_attr "Version")
                            (quoted_attribute_value (attribute_value) @{LIBRARY_VERSION})
                        )?
                    )
                ) @{LIBRARY}

                (self_closing_tag
                    (tag_name) @tag (#eq? @tag "PackageReference")
                    (_)*
                    (attribute
                        (attribute_name) @attr (#eq? @attr "Include")
                        (quoted_attribute_value (attribute_value) @{LIBRARY_NAME})
                    )
                    (_)*
                    (attribute
                        (attribute_name) @ver_attr (#eq? @ver_attr "Version")
                        (quoted_attribute_value (attribute_value) @{LIBRARY_VERSION})
                    )?
                ) @{LIBRARY}
            ]"#
        ))
    }

    fn imports_query(&self) -> Option<String> {
        Some(format!(
            r#"[
                (using_directive
                    (identifier) @{IMPORTS_NAME}
                ) @{IMPORTS}
                
                (using_directive
                    (qualified_name) @{IMPORTS_NAME}
                ) @{IMPORTS}
            ]"#
        ))
    }

    fn class_definition_query(&self) -> String {
        format!(
            r#"[
                (class_declaration
                    name: (identifier) @{CLASS_NAME}
                    (base_list
                        (identifier) @{CLASS_PARENT}
                    )?
                ) @{CLASS_DEFINITION}
                
                (interface_declaration
                    name: (identifier) @{CLASS_NAME}
                ) @{CLASS_DEFINITION}
                
                (struct_declaration
                    name: (identifier) @{CLASS_NAME}
                ) @{CLASS_DEFINITION}
                
                (record_declaration
                    name: (identifier) @{CLASS_NAME}
                ) @{CLASS_DEFINITION}
                
                (enum_declaration
                    name: (identifier) @{CLASS_NAME}
                ) @{CLASS_DEFINITION}
            ]"#
        )
    }

    fn trait_query(&self) -> Option<String> {
        Some(format!(
            r#"(interface_declaration
                name: (identifier) @{TRAIT_NAME}
            ) @{CLASS_DEFINITION}"#
        ))
    }

    fn function_definition_query(&self) -> String {
        format!(
            r#"[
                (method_declaration
                    name: (identifier) @{FUNCTION_NAME}
                    parameters: (parameter_list) @{ARGUMENTS}
                ) @{FUNCTION_DEFINITION}
                
                (constructor_declaration
                    name: (identifier) @{FUNCTION_NAME}
                    parameters: (parameter_list) @{ARGUMENTS}
                ) @{FUNCTION_DEFINITION}
                
                (local_function_statement
                    name: (identifier) @{FUNCTION_NAME}
                    parameters: (parameter_list) @{ARGUMENTS}
                ) @{FUNCTION_DEFINITION}
            ]"#
        )
    }

    fn comment_query(&self) -> Option<String> {
        Some(format!(
            r#"[
                (comment)+
            ] @{FUNCTION_COMMENT}"#
        ))
    }

    fn function_call_query(&self) -> String {
        format!(
            r#"[
                (invocation_expression
                    function: (identifier) @{FUNCTION_NAME}
                    arguments: (argument_list) @{ARGUMENTS}
                ) @{FUNCTION_CALL}
                
                (invocation_expression
                    function: (member_access_expression
                        expression: (_) @{OPERAND}
                        name: (identifier) @{FUNCTION_NAME}
                    )
                    arguments: (argument_list) @{ARGUMENTS}
                ) @{FUNCTION_CALL}
                
                (invocation_expression
                    function: (generic_name
                        (identifier) @{FUNCTION_NAME}
                    )
                    arguments: (argument_list) @{ARGUMENTS}
                ) @{FUNCTION_CALL}
                
                (object_creation_expression
                    type: (identifier) @{FUNCTION_NAME}
                    arguments: (argument_list) @{ARGUMENTS}
                ) @{FUNCTION_CALL}
                
                (object_creation_expression
                    type: (generic_name
                        (identifier) @{FUNCTION_NAME}
                    )
                    arguments: (argument_list) @{ARGUMENTS}
                ) @{FUNCTION_CALL}
            ]"#
        )
    }

    fn variables_query(&self) -> Option<String> {
        Some(format!(
            r#"(field_declaration
                (variable_declaration
                    type: (_)
                    (variable_declarator
                        (identifier) @{VARIABLE_NAME}
                    )
                )
            ) @{VARIABLE_DECLARATION}"#
        ))
    }

    fn identifier_query(&self) -> String {
        "(identifier) @identifier".to_string()
    }

    fn is_test(&self, func_name: &str, func_file: &str, func_body: &str) -> bool {
        if self.is_test_file(func_file) {
            return true;
        }
        func_body.contains("[Fact]")
            || func_body.contains("[Theory]")
            || func_body.contains("[Test]")
            || func_body.contains("[TestMethod]")
            || func_name.starts_with("Test")
            || func_name.ends_with("Test")
    }

    fn is_test_file(&self, filename: &str) -> bool {
        let f = filename.to_lowercase();
        f.ends_with("tests.cs")
            || f.ends_with("test.cs")
            || f.ends_with("_tests.cs")
            || f.ends_with("_test.cs")
            || f.contains("/tests/")
            || f.contains("/test/")
    }

    fn is_e2e_test_file(&self, file: &str, code: &str) -> bool {
        let f = file.to_lowercase();
        let c = code.to_lowercase();

        if f.contains("/e2e/")
            || f.contains("/acceptance/")
            || f.contains("/integration/")
            || f.contains("/functionaltest/")
        {
            return true;
        }

        c.contains("webapplicationfactory") || c.contains("selenium") || c.contains("playwright")
    }

    fn classify_test(&self, name: &str, file: &str, body: &str) -> NodeType {
        let f = file.to_lowercase();
        let b = body.to_lowercase();

        if f.contains("/e2e/")
            || f.contains("/acceptance/")
            || b.contains("selenium")
            || b.contains("playwright")
        {
            return NodeType::E2eTest;
        }

        if f.contains("/integration/")
            || f.contains("/functionaltest/")
            || b.contains("webapplicationfactory")
            || b.contains("httpclient")
            || b.contains("testserver")
        {
            return NodeType::IntegrationTest;
        }

        if f.contains("/unit/") || f.contains("/unittests/") {
            return NodeType::UnitTest;
        }

        let lname = name.to_lowercase();
        if lname.contains("e2e") || lname.contains("acceptance") {
            return NodeType::E2eTest;
        }
        if lname.contains("integration") || lname.contains("functional") {
            return NodeType::IntegrationTest;
        }

        NodeType::UnitTest
    }

    fn test_query(&self) -> Option<String> {
        Some(format!(
            r#"[
                (method_declaration
                    (attribute_list
                        (attribute
                            name: (identifier) @attr_name (#match? @attr_name "^(Fact|Theory|Test|TestMethod)$")
                        )
                    )
                    name: (identifier) @{FUNCTION_NAME}
                ) @{FUNCTION_DEFINITION}
            ]"#
        ))
    }

    fn generate_anonymous_handler_name(
        &self,
        method: &str,
        path: &str,
        line: usize,
    ) -> Option<String> {
        let method_str = method.to_uppercase();
        let path_str = path.replace('/', "_").replace(['{', '}', ':'], "");
        Some(format!("{}_{}_closure_L{}", method_str, path_str, line))
    }

    fn endpoint_finders(&self) -> Vec<String> {
        vec![
            // Controller action with [HttpGet], [HttpPost], etc.
            format!(
                r#"(method_declaration
                    (attribute_list
                        (attribute
                            name: (identifier) @{ENDPOINT_VERB} (#match? @{ENDPOINT_VERB} "^Http(Get|Post|Put|Delete|Patch|Options)$")
                            (attribute_argument_list
                                (attribute_argument
                                    (string_literal) @{ENDPOINT}
                                )?
                            )?
                        )
                    )
                    name: (identifier) @{HANDLER}
                ) @{ROUTE}"#
            ),
            // [Route("path")] attribute on class or method
            format!(
                r#"(attribute
                    name: (identifier) @attr (#eq? @attr "Route")
                    (attribute_argument_list
                        (attribute_argument
                            (string_literal) @{ENDPOINT}
                        )
                    )
                ) @{ROUTE}"#
            ),
            // Minimal API: app.MapGet("/path", handler)
            format!(
                r#"(invocation_expression
                    function: (member_access_expression
                        name: (identifier) @{ENDPOINT_VERB} (#match? @{ENDPOINT_VERB} "^Map(Get|Post|Put|Delete|Patch)$")
                    )
                    arguments: (argument_list
                        (argument
                            (string_literal) @{ENDPOINT}
                        )
                        (argument
                            [
                                (identifier)
                                (member_access_expression)
                            ] @{HANDLER}
                        )
                    )
                ) @{ROUTE}"#
            ),
            // Minimal API with lambda: app.MapGet("/path", () => ...)
            format!(
                r#"(invocation_expression
                    function: (member_access_expression
                        name: (identifier) @{ENDPOINT_VERB} (#match? @{ENDPOINT_VERB} "^Map(Get|Post|Put|Delete|Patch)$")
                    )
                    arguments: (argument_list
                        (argument
                            (string_literal) @{ENDPOINT}
                        )
                        (argument
                            (lambda_expression) @{ANONYMOUS_FUNCTION}
                        )
                    )
                ) @{ROUTE}"#
            ),
        ]
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

    fn use_handler_finder(&self) -> bool {
        true
    }

    fn handler_finder(
        &self,
        endpoint: NodeData,
        find_fn: &dyn Fn(&str, &str) -> Option<NodeData>,
        _find_fns_in: &dyn Fn(&str) -> Vec<NodeData>,
        _handler_params: HandlerParams,
    ) -> Vec<(NodeData, Option<Edge>)> {
        let mut edge = None;

        if let Some(handler) = endpoint.meta.get("handler") {
            let handler_name = trim_quotes(handler);
            if let Some(nd) = find_fn(&handler_name, &endpoint.file) {
                edge = Some(Edge::handler(&endpoint, &nd));
            }
        }

        vec![(endpoint, edge)]
    }

    fn add_endpoint_verb(&self, nd: &mut NodeData, call: &Option<String>) -> Option<String> {
        if let Some(verb_attr) = call {
            let verb = verb_attr
                .trim_start_matches("Http")
                .trim_start_matches("Map")
                .to_uppercase();
            nd.meta.insert("verb".to_string(), verb.clone());
            return Some(verb);
        }
        None
    }

    fn data_model_query(&self) -> Option<String> {
        // Entity Framework DbSet properties and classes with data annotations
        Some(format!(
            r#"[
                (property_declaration
                    type: (generic_name
                        (identifier) @db_set (#eq? @db_set "DbSet")
                        (type_argument_list
                            (identifier) @{STRUCT_NAME}
                        )
                    )
                ) @{STRUCT}
                
                (class_declaration
                    (attribute_list
                        (attribute
                            name: (identifier) @table_attr (#eq? @table_attr "Table")
                        )
                    )
                    name: (identifier) @{STRUCT_NAME}
                ) @{STRUCT}
            ]"#
        ))
    }

    fn program_node_name(&self) -> String {
        "compilation_unit".to_string()
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
