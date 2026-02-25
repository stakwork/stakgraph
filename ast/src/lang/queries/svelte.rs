use super::super::*;
use super::consts::*;
use shared::error::{Context, Result};
use tree_sitter::{Language, Node as TreeNode, Parser, Query, Tree};

pub struct Svelte(Language);

impl Default for Svelte {
    fn default() -> Self {
        Self::new()
    }
}

impl Svelte {
    pub fn new() -> Self {
        Svelte(tree_sitter_svelte_ng::LANGUAGE.into())
    }

    fn extract_script_content(code: &str) -> Option<(String, usize)> {
        let start_tag = "<script";
        let end_tag = "</script>";
        
        let start_idx = code.find(start_tag)?;
        let tag_end = code[start_idx..].find('>')? + start_idx + 1;
        let end_idx = code.find(end_tag)?;
        
        if end_idx <= tag_end {
            return None;
        }
        
        let script_content = code[tag_end..end_idx].to_string();
        let line_offset = code[..tag_end].matches('\n').count();
        
        Some((script_content, line_offset))
    }
}

impl Stack for Svelte {
    fn q(&self, q: &str, nt: &NodeType) -> Query {
        let grammar = match nt {
            NodeType::Function
            | NodeType::UnitTest
            | NodeType::IntegrationTest
            | NodeType::E2eTest => tree_sitter_typescript::LANGUAGE_TSX.into(),
            _ => self.0.clone(),
        };
        Query::new(&grammar, q).unwrap()
    }

    fn parse(&self, code: &str, nt: &NodeType) -> Result<Tree> {
        let mut parser = Parser::new();
        
        let (parse_code, grammar): (String, Language) = match nt {
            NodeType::Function
            | NodeType::UnitTest
            | NodeType::IntegrationTest
            | NodeType::E2eTest => {
                if let Some((script, _offset)) = Self::extract_script_content(code) {
                    (script, tree_sitter_typescript::LANGUAGE_TSX.into())
                } else {
                    (code.to_string(), self.0.clone())
                }
            }
            _ => (code.to_string(), self.0.clone()),
        };
        
        parser.set_language(&grammar)?;
        parser.parse(&parse_code, None).context("failed to parse")
    }

    fn imports_query(&self) -> Option<String> {
        Some(format!(
            r#"
            (document
                (_) @{IMPORTS}
            )
        "#
        ))
    }

    fn class_definition_query(&self) -> String {
        format!(
            r#"(script_element) @{CLASS_DEFINITION}"#
        )
    }

    fn function_definition_query(&self) -> String {
        format!(
            r#"[
            (function_declaration
                name: (identifier) @{FUNCTION_NAME}
            ) @{FUNCTION_DEFINITION}
            
            (lexical_declaration
                (variable_declarator
                    name: (identifier) @{FUNCTION_NAME}
                    value: (arrow_function)
                )
            ) @{FUNCTION_DEFINITION}
            ]"#
        )
    }
    fn comment_query(&self) -> Option<String> {
        Some(format!(r#"(comment) @{FUNCTION_COMMENT}"#))
    }

    fn function_call_query(&self) -> String {
        r#"
            (expression
                (_) @args
            ) @FUNCTION_CALL
            "#.to_string()
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

    fn request_finder(&self) -> Option<String> {
        Some(format!(
            r#"
            (_
                (_) @{ENDPOINT}
                (#match? @{ENDPOINT} "fetch|get|post|put|delete")
            ) @{REQUEST_CALL}
            "#
        ))
    }

    fn data_model_query(&self) -> Option<String> {
        Some(format!(
            r#"
            (document
                (_
                    (_) + @{STRUCT_NAME}
                )
            ) @{STRUCT}
            "#
        ))
    }
    fn identifier_query(&self) -> String {
        r#"(tag_name) @identifier"#.to_string()
    }

    fn data_model_within_query(&self) -> Option<String> {
        Some(format!(
            r#"
            [
                    (_) @{STRUCT_NAME} (#match? @{STRUCT_NAME} "^[A-Z].*")
                (expression
                     (_) @{STRUCT_NAME} (#match? @{STRUCT_NAME} "^[A-Z].*")
                )
            ]
            "#
        ))
    }

    fn test_query(&self) -> Option<String> {
        Some(format!(
            r#"(function_declaration
                name: (identifier) @{FUNCTION_NAME}
                (#match? @{FUNCTION_NAME} "^(test|it)")
            ) @{FUNCTION_DEFINITION}"#
        ))
    }

    fn integration_test_query(&self) -> Option<String> {
        Some(format!(
            r#"(function_declaration
                name: (identifier) @{FUNCTION_NAME}
                (#match? @{FUNCTION_NAME} "^(test|it)")
            ) @{FUNCTION_DEFINITION}"#
        ))
    }

    fn e2e_test_query(&self) -> Option<String> {
        Some(format!(
            r#"(function_declaration
                name: (identifier) @{FUNCTION_NAME}
                (#match? @{FUNCTION_NAME} "^(test|it)")
            ) @{FUNCTION_DEFINITION}"#
        ))
    }

    fn is_test_file(&self, path: &str) -> bool {
        let normalized = path.replace("\\", "/");
        normalized.contains("/test/")
            || normalized.contains("/tests/")
            || normalized.contains("/__tests__/")
            || normalized.ends_with(".spec.svelte")
            || normalized.ends_with(".test.svelte")
            || normalized.ends_with(".spec.ts")
            || normalized.ends_with(".test.ts")
            || normalized.ends_with(".spec.js")
            || normalized.ends_with(".test.js")
            || normalized.ends_with(".spec.js")
            || normalized.ends_with(".test.js")
    }

    fn is_e2e_test_file(&self, path: &str, _code: &str) -> bool {
        let normalized = path.replace("\\", "/");
        normalized.contains("/e2e/")
            || normalized.contains("/integration/")
            || normalized.contains(".e2e.spec.")
            || normalized.contains(".e2e.test.")
    }

    fn is_test(&self, func_name: &str, _func_file: &str, func_body: &str) -> bool {
        func_name.starts_with("test")
            || func_body.contains("test(")
            || func_body.contains("it(")
            || func_body.contains("describe(")
    }

    fn tests_are_functions(&self) -> bool {
        true
    }

    fn classify_test(&self, _name: &str, file: &str, _body: &str) -> NodeType {
        let normalized = file.replace("\\", "/");
        
        // E2E tests
        if normalized.contains("/e2e/") 
            || normalized.contains(".e2e.spec.") 
            || normalized.contains(".e2e.test.") {
            return NodeType::E2eTest;
        }
        
        // Integration tests
        if normalized.contains("/integration/") {
            return NodeType::IntegrationTest;
        }
        
        // Unit tests (default)
        NodeType::UnitTest    }
}