use super::super::*;
use super::consts::*;
use lsp::{Cmd as LspCmd, CmdSender, Position, Res as LspRes};
use shared::error::{Context, Result};
use tree_sitter::{Language, Parser, Query, Tree};

pub struct Go(Language);

impl Default for Go {
    fn default() -> Self {
        Self::new()
    }
}

impl Go {
    pub fn new() -> Self {
        Go(tree_sitter_go::LANGUAGE.into())
    }
}

impl Stack for Go {
    fn q(&self, q: &str, nt: &NodeType) -> Query {
        if matches!(nt, NodeType::Library) {
            Query::new(&tree_sitter_bash::LANGUAGE.into(), q).unwrap()
        } else {
            Query::new(&self.0, q).unwrap()
        }
    }
    fn parse(&self, code: &str, nt: &NodeType) -> Result<Tree> {
        let mut parser = Parser::new();
        if matches!(nt, NodeType::Library) {
            parser.set_language(&tree_sitter_bash::LANGUAGE.into())?;
        } else {
            parser.set_language(&self.0)?;
        }
        parser.parse(code, None).context("failed to parse")
    }
    // fn is_lib_file(&self, file_name: &str) -> bool {
    //     file_name.contains("/go/pkg/mod/")
    // }
    fn lib_query(&self) -> Option<String> {
        Some(format!(
            r#"(command
                name: (command_name) @require (#eq? @require "require")
                (subshell
                    (command
                    name: (command_name) @{LIBRARY_NAME}
                    argument: (word) @{LIBRARY_VERSION}
                    )
                )
            ) @{LIBRARY}"#
        ))
    }
    fn module_query(&self) -> Option<String> {
        Some(format!("(package_identifier) @{MODULE_NAME}"))
    }
    fn imports_query(&self) -> Option<String> {
        Some(format!(
            "(source_file
                (import_declaration)+ @{IMPORTS}
            )"
        ))
    }
    fn variables_query(&self) -> Option<String> {
        Some(format!(
            r#"
            (source_file
                (var_declaration
                    (var_spec
                        name: (identifier) @{VARIABLE_NAME}
                        type: (type_identifier)? @{VARIABLE_TYPE}
                        value: (expression_list)? @{VARIABLE_VALUE}
                    )
                )? @{VARIABLE_DECLARATION}
                (const_declaration
                    (const_spec
                        name: (identifier) @{VARIABLE_NAME}
                        type: (type_identifier)? @{VARIABLE_TYPE}
                        value: (expression_list)? @{VARIABLE_VALUE}
                    )
                )? @{VARIABLE_DECLARATION}
            )
            "#
        ))
    }
    fn trait_query(&self) -> Option<String> {
        Some(format!(
            r#"(type_declaration
                (type_spec
                    name: (type_identifier) @{TRAIT_NAME}
                    type: (interface_type)
                )
            ) @{TRAIT}"#
        ))
    }
    // FIXME for go this just gets every struct. Filter them out later. If class has no methods, delete it.
    fn class_definition_query(&self) -> String {
        format!(
            "(type_spec
                name: (type_identifier) @{CLASS_NAME}
                type_parameters: (type_parameter_list)?
            ) @{CLASS_DEFINITION}"
        )
    }
    //capture as variables instead
    fn instance_definition_query(&self) -> Option<String> {
        Some(format!(
            "(source_file
                (var_declaration
                    (var_spec
                        name: (identifier) @{INSTANCE_NAME}
                        type: (type_identifier) @{CLASS_NAME}
                    )
                ) @{INSTANCE}
            )"
        ))
    }
    fn function_definition_query(&self) -> String {
        let return_type = format!(r#"result: (_)? @{RETURN_TYPES}"#);
        format!(
            "[
                (function_declaration
                    name: (identifier) @{FUNCTION_NAME}
                    type_parameters: (type_parameter_list)?
                    parameters: (parameter_list) @{ARGUMENTS}
                    {return_type}
                )
                (method_declaration
                    receiver: (parameter_list
                        (parameter_declaration
                            name: (identifier) @{PARENT_NAME}
                            type: [
                                (type_identifier)
                                (pointer_type) (type_identifier)
                                (generic_type)
                                (pointer_type) (generic_type)
                            ] @{PARENT_TYPE}
                        )
                    )
                    name: (field_identifier) @{FUNCTION_NAME}
                    parameters: (parameter_list) @{ARGUMENTS}
                    {return_type}
                )
            ] @{FUNCTION_DEFINITION}"
        )
    }

    fn comment_query(&self) -> Option<String> {
        Some(format!(r#"(comment) @{FUNCTION_COMMENT}"#))
    }
    fn class_comment_query(&self) -> Option<String> {
        Some(format!(r#"(comment) @{CLASS_COMMENT}"#))
    }
    fn data_model_comment_query(&self) -> Option<String> {
        Some(format!(r#"(comment) @{STRUCT_COMMENT}"#))
    }
    fn trait_comment_query(&self) -> Option<String> {
        Some(format!(r#"(comment) @{TRAIT_COMMENT}"#))
    }
    fn function_call_query(&self) -> String {
        format!(
            "(call_expression
                function: [
                    (identifier) @{FUNCTION_NAME}
                    (selector_expression
                        operand: [
                            (identifier) @{OPERAND}
                            (selector_expression) @{OPERAND}
                            (call_expression)
                        ]
                        field: (field_identifier) @{FUNCTION_NAME}
                    )
                ]
                arguments: (argument_list) @{ARGUMENTS}
            ) @{FUNCTION_CALL}"
        )
    }
    //     fn endpoint_handler_queries(&self) -> Vec<String> {
    //         let q1 = r#"("func"
    //     parameters: (parameter_list
    //         (parameter_declaration
    //             type: (qualified_type) @res (#eq? @res "http.ResponseWriter")
    //         )
    //         (parameter_declaration
    //             type: (pointer_type) @req (#eq? @req "*http.Request")
    //         )
    //     )
    // )"#;
    //         vec![q1.to_string()]
    //     }
    fn endpoint_finders(&self) -> Vec<String> {
        vec![format!(
            r#"(call_expression
                function: (selector_expression
                    operand: (identifier)
                    field: (field_identifier) @{ENDPOINT_VERB} (#match? @{ENDPOINT_VERB} "^(GET|POST|PUT|DELETE|PATCH|Get|Post|Put|Delete|Patch)$")
                )
                arguments: (argument_list
                    (interpreted_string_literal) @{ENDPOINT}
                    [
                        (selector_expression
                            field: (field_identifier) @{HANDLER}
                        )
                        (identifier) @{HANDLER}
                    ]
                )
            ) @{ROUTE}"#
        )]
    }
    fn endpoint_group_find(&self) -> Option<String> {
        Some(format!(
            r#"(call_expression
                function: (selector_expression
                    operand: (identifier)
                    field: (field_identifier) @{ENDPOINT_VERB} (#match? @{ENDPOINT_VERB} "^(Mount|Group)$")
                )
                arguments: (argument_list
                    (interpreted_string_literal) @{ENDPOINT}
                    (call_expression
                        function: (identifier) @{ENDPOINT_GROUP}
                    )
                )
            ) @{ROUTE}"#
        ))
    }
    fn find_function_parent(
        &self,
        node: TreeNode,
        _code: &str,
        file: &str,
        func_name: &str,
        find_class: &dyn Fn(&str) -> Option<NodeData>,
        parent_type: Option<&str>,
    ) -> Result<Option<Operand>> {
        if parent_type.is_none() {
            return Ok(None);
        }
        let parent_str = parent_type.unwrap();
        // Clean parent type: remove pointer * and generic [T]
        let cleaned_type = parent_str.trim_start_matches('*');
        let cleaned_type = if let Some(idx) = cleaned_type.find('[') {
            &cleaned_type[..idx]
        } else {
            cleaned_type
        };

        let nodedata = find_class(cleaned_type);
        Ok(match nodedata {
            Some(class) => Some(Operand {
                source: NodeKeys::new(&class.name, &class.file, class.start),
                target: NodeKeys::new(func_name, file, node.start_position().row),
            }),
            None => None,
        })
    }
    fn find_trait_operand(
        &self,
        pos: Position,
        nd: &NodeData,
        find_trait: &dyn Fn(u32, &str) -> Option<NodeData>,
        lsp_tx: &Option<CmdSender>,
    ) -> Result<Option<Edge>> {
        if let Some(lsp) = lsp_tx {
            let res = LspCmd::GotoImplementations(pos.clone()).send(lsp)?;
            if let LspRes::GotoImplementations(Some(imp)) = res {
                let tr = find_trait(imp.line, &imp.file.display().to_string());
                if let Some(tr) = tr {
                    let edge = Edge::trait_operand(&tr, nd);
                    return Ok(Some(edge));
                }
            }
        }
        Ok(None)
    }
    //     fn data_model_query(&self) -> Option<String> {
    //         Some(format!(
    //             "(type_declaration
    //     (type_spec
    //     	name: (type_identifier) @{STRUCT_NAME}
    //         type: (struct_type
    //         	(field_declaration_list
    //             	(field_declaration
    //                 	tag: (_)
    //                 )
    //             )
    //         )
    //     )
    // ) @{STRUCT}"
    //         ))
    //     }
    // It duplicates with classes...
    fn data_model_query(&self) -> Option<String> {
        Some(format!(
            "(type_declaration
                (type_spec
                    name: (type_identifier) @{STRUCT_NAME}
                    type: (_)
                )
            ) @{STRUCT}"
        ))
    }
    fn data_model_within_query(&self) -> Option<String> {
        // the surrounding () is required to match the match work
        let type_finder = format!(
            r#"(
                (type_identifier) @{STRUCT_NAME} (#match? @{STRUCT_NAME} "^[A-Z].*")
            )"#
        );
        Some(type_finder)
    }
    fn classify_test(&self, _name: &str, file: &str, body: &str) -> NodeType {
        let f = file.replace('\\', "/").to_lowercase();
        let fname = f.rsplit('/').next().unwrap_or(&f);

        if f.contains("/tests/e2e/")
            || f.contains("/test/e2e/")
            || f.contains("/e2e/")
            || fname.contains("e2e")
        {
            return NodeType::E2eTest;
        }

        if f.contains("/tests/integration/")
            || f.contains("/test/integration/")
            || f.contains("/integration/")
            || fname.contains("integration_test")
        {
            return NodeType::IntegrationTest;
        }

        let has_browser_driver = body.contains("chromedp")
            || body.contains("selenium")
            || body.contains("playwright")
            || body.contains("rod");

        if has_browser_driver {
            return NodeType::E2eTest;
        }

        let has_httptest =
            body.contains("httptest.NewRecorder") || body.contains("net/http/httptest");

        if has_httptest {
            return NodeType::IntegrationTest;
        }

        NodeType::UnitTest
    }

    fn is_test(&self, func_name: &str, _func_file: &str, _func_body: &str) -> bool {
        func_name.starts_with("Test")
            || func_name.starts_with("Benchmark")
            || func_name.starts_with("Example")
    }
    fn is_test_file(&self, filename: &str) -> bool {
        filename.ends_with("_test.go")
    }

    fn test_query(&self) -> Option<String> {
        Some(format!(
            r#"(function_declaration
                name: (identifier) @{FUNCTION_NAME} (#match? @{FUNCTION_NAME} "^(Test|Benchmark|Example)")
            ) @{FUNCTION_DEFINITION}"#
        ))
    }

    fn is_e2e_test_file(&self, file: &str, code: &str) -> bool {
        let f = file.replace('\\', "/").to_lowercase();
        let lower_code = code.to_lowercase();

        if f.contains("/e2e/") || f.contains("/test/e2e/") || f.contains("/tests/e2e/") {
            return true;
        }

        let fname = f.rsplit('/').next().unwrap_or(&f);
        if fname.contains("e2e") || fname.contains("_e2e_test.go") {
            return true;
        }

        let has_selenium = lower_code.contains("selenium")
            || lower_code.contains("webdriver")
            || lower_code.contains("github.com/tebeka/selenium");

        let has_chromedp =
            lower_code.contains("chromedp") || lower_code.contains("github.com/chromedp/chromedp");

        let has_playwright = lower_code.contains("playwright")
            || lower_code.contains("github.com/playwright-community/playwright-go");

        let has_rod = lower_code.contains("github.com/go-rod/rod");

        has_selenium || has_chromedp || has_playwright || has_rod
    }

    fn integration_test_query(&self) -> Option<String> {
        Some(format!(
            r#"(call_expression
                function: (selector_expression) @hff (#eq? @hff "http.HandlerFunc")
                arguments: (argument_list
                    (selector_expression
                        operand: (identifier)
                        field: (field_identifier) @{HANDLER}
                    )
                )
            )"#
        ))
    }
    fn clean_graph(&self, callback: &mut dyn FnMut(NodeType, NodeType, &str)) {
        callback(NodeType::Class, NodeType::Function, "operand");
    }
}
