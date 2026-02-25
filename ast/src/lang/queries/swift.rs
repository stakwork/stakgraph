use super::super::*;
use super::consts::*;
use shared::error::{Context, Result};
use tree_sitter::{Language, Node as TreeNode, Parser, Query, Tree};

pub struct Swift(Language);

impl Default for Swift {
    fn default() -> Self {
        Self::new()
    }
}

impl Swift {
    pub fn new() -> Self {
        Swift(tree_sitter_swift::LANGUAGE.into())
    }
}

impl Stack for Swift {
    fn should_skip_function_call(&self, called: &str, operand: &Option<String>) -> bool {
        super::skips::java::should_skip(called, operand)
    }
    fn q(&self, q: &str, _nt: &NodeType) -> Query {
        match Query::new(&self.0, q) {
            Ok(query) => query,
            Err(err) => panic!("Failed to compile Swift query '{}': {}", q, err),
        }
    }
    fn parse(&self, code: &str, _nt: &NodeType) -> Result<Tree> {
        let mut parser = Parser::new();

        parser.set_language(&self.0)?;
        parser.parse(code, None).context("failed to parse")
    }

    fn imports_query(&self) -> Option<String> {
        Some(format!(
            r#"
            (import_declaration
                (identifier) 
            ) @{IMPORTS}
            "#
        ))
    }

    fn variables_query(&self) -> Option<String> {
        Some(format!(
            r#"
            (source_file
                (property_declaration 
                    (pattern
                        (simple_identifier) @{VARIABLE_NAME}
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
                name: (type_identifier) @{CLASS_NAME}
            ) @{CLASS_DEFINITION}
            
            (class_declaration
                name: (user_type) @{CLASS_NAME}
            ) @{CLASS_DEFINITION}
            "#
        )
    }

    fn function_definition_query(&self) -> String {
        format!(
            r#"
        (function_declaration
            name: (simple_identifier) @{FUNCTION_NAME}
        ) @{FUNCTION_DEFINITION}
        (init_declaration
            name: (_) @{FUNCTION_NAME}
        ) @{FUNCTION_DEFINITION}
        "#
        )
    }
    fn comment_query(&self) -> Option<String> {
        Some(format!(
            r#"
             [
                (comment)+
                (multiline_comment)+
             ] @{FUNCTION_COMMENT}
        "#
        ))
    }

    fn function_call_query(&self) -> String {
        format!(
            r#"
            (call_expression
                 (simple_identifier) @{ARGUMENTS}
            ) @{FUNCTION_CALL}
            "#
        )
    }

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
            if current.kind() == "class_declaration" {
                break;
            }
            parent = current.parent();
        }
        let parent_of = match parent {
            Some(p) => {
                let query = self.q("name: (type_identifier) @class-name", &NodeType::Class);
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

    fn request_finder(&self) -> Option<String> {
        Some(format!(
            r#"
            (call_expression
                (simple_identifier) @{REQUEST_CALL} (#match? @{REQUEST_CALL} "^createRequest$")
            ) @{ROUTE}
        "#
        ))
    }
    fn add_endpoint_verb(&self, inst: &mut NodeData, _call: &Option<String>) -> Option<String> {
        if !inst.meta.contains_key("verb") {
            if inst.body.contains("method: \"GET\"") || inst.body.contains("bodyParams: nil") {
                inst.add_verb("GET");
            } else if inst.body.contains("method: \"POST\"") {
                inst.add_verb("POST");
            } else if inst.body.contains("method: \"PUT\"") {
                inst.add_verb("PUT");
            } else if inst.body.contains("method: \"DELETE\"") {
                inst.add_verb("DELETE");
            }

            if !inst.meta.contains_key("verb") {
                inst.add_verb("GET"); // Default
            }
        }
        if inst.name.is_empty() {
            let url_start = inst.body.find("url:");
            if let Some(start_pos) = url_start {
                if let Some(quote_start) = inst.body[start_pos..].find("\"") {
                    let start_idx = start_pos + quote_start + 1;
                    if let Some(quote_end) = inst.body[start_idx..].find("\"") {
                        let url_section = &inst.body[start_idx..start_idx + quote_end];
                        if let Some(path_start) = url_section.rfind("/") {
                            let path = &url_section[path_start..];
                            if !path.is_empty() {
                                inst.name = path.to_string();
                            }
                        }
                    }
                }
            }
        }
        None
    }

    fn data_model_query(&self) -> Option<String> {
        Some(format!(
            r#"
            (class_declaration
                (type_identifier) @{STRUCT_NAME}
                (_)*
            ) @{STRUCT}
        "#
        ))
    }

    fn data_model_path_filter(&self) -> Option<String> {
        Some("CoreData".to_string())
    }

    fn data_model_within_query(&self) -> Option<String> {
        Some(format!(
            r#"[
                (identifier) @{STRUCT_NAME} (#match? @{STRUCT_NAME} "^[A-Z].*")

                (call_expression
                    (simple_identifier) @{STRUCT_NAME} (#match? @{STRUCT_NAME} "^[A-Z].*")
                )

                ]@{STRUCT}
            "#
        ))
    }

    fn test_query(&self) -> Option<String> {
        Some(format!(
            r#"
            (function_declaration
                name: (simple_identifier) @{FUNCTION_NAME}
                (#match? @{FUNCTION_NAME} "^test")
            ) @{FUNCTION_DEFINITION}
            "#
        ))
    }

    fn integration_test_query(&self) -> Option<String> {
        Some(format!(
            r#"
            (function_declaration
                name: (simple_identifier) @{FUNCTION_NAME}
                (#match? @{FUNCTION_NAME} "^test")
            ) @{FUNCTION_DEFINITION}
            "#
        ))
    }

    fn e2e_test_query(&self) -> Option<String> {
        Some(format!(
            r#"
            (function_declaration
                name: (simple_identifier) @{FUNCTION_NAME}
                (#match? @{FUNCTION_NAME} "^test")
            ) @{FUNCTION_DEFINITION}
            "#
        ))
    }

    fn is_test_file(&self, path: &str) -> bool {
        let normalized = path.replace("\\", "/");
        normalized.contains("/Tests/")
            || normalized.contains("/UITests/")
            || normalized.ends_with("Tests.swift")
            || normalized.ends_with("Test.swift")
    }

    fn is_e2e_test_file(&self, path: &str, code: &str) -> bool {
        let normalized = path.replace("\\", "/");
        normalized.contains("/UITests/") 
            || code.contains("import XCUITest")
            || code.contains("XCUIApplication")
    }

    fn is_test(&self, func_name: &str, _func_file: &str, _func_body: &str) -> bool {
        func_name.starts_with("test")
    }

    fn tests_are_functions(&self) -> bool {
        true
    }

    fn classify_test(&self, _name: &str, file: &str, body: &str) -> NodeType {
        let normalized = file.replace("\\", "/");
        
        if normalized.contains("/UITests/") 
            || body.contains("import XCUITest")
            || body.contains("XCUIApplication") {
            return NodeType::E2eTest;
        }
        
        if normalized.contains("/IntegrationTests/") {
            return NodeType::IntegrationTest;
        }
        
        NodeType::UnitTest    }
}