use super::super::*;
use super::consts::*;
use shared::{Context, Result};
use tree_sitter::{Language, Parser, Query, Tree};

pub struct Angular(Language);

impl Default for Angular {
    fn default() -> Self {
        Self::new()
    }
}

impl Angular {
    pub fn new() -> Self {
        Angular(tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into())
    }
}

impl Stack for Angular {
    fn q(&self, q: &str, _nt: &NodeType) -> Query {
        Query::new(&self.0, q).unwrap()
    }
    fn parse(&self, code: &str, _nt: &NodeType) -> Result<Tree> {
        let mut parser = Parser::new();
        parser.set_language(&self.0)?;
        parser.parse(code, None).context("failed to parse")
    }
    fn component_template_query(&self) -> Option<String> {
        Some(format!(
            r#"
            (decorator
                (call_expression
                    function: (identifier) @{DECORATOR_NAME} (#eq? @{DECORATOR_NAME} "Component")
                    arguments: (arguments
                        (object
                            (pair
                                key: (property_identifier) @{TEMPLATE_KEY} (#match? @{TEMPLATE_KEY} "^(templateUrl|styleUrls|selector)$")
                                value: (_) @{TEMPLATE_VALUE}
                            )
                        )
                    )
                )
            )
            "#
        ))
    }
    fn template_ext(&self) -> Option<&str> {
        Some(".component.ts")
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
            r#"
            (import_statement
                (import_clause
                    (identifier)? @{IMPORTS_NAME}
                    (named_imports
                        (import_specifier
                            name:(identifier) @{IMPORTS_NAME}
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
                parameters: (formal_parameters)? @{ARGUMENTS}
                return_type: (type_annotation)? @{RETURN_TYPES}
              ) @{FUNCTION_DEFINITION}
            )

            (
              (decorator)* @{ATTRIBUTES}
              .
              (method_definition
                name: (property_identifier) @{FUNCTION_NAME}
                parameters: (formal_parameters)? @{ARGUMENTS}
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

    fn request_finder(&self) -> Option<String> {
        Some(format!(
            r#"
            (call_expression
                (_) @{ENDPOINT}
                (_) @{ARGUMENTS}
            )@{REQUEST_CALL}
            "#
        ))
    }
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
                ] @{STRUCT}

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

        path
    }
    fn is_extra_page(&self, file_name: &str) -> bool {
        file_name.ends_with(".html")
            || file_name.ends_with(".css")
            || file_name.ends_with(".scss")
            || file_name.ends_with(".sass")
    }

    fn use_extra_page_finder(&self) -> bool {
        true
    }

    fn extra_page_finder(
        &self,
        file_path: &str,
        find_fn: &dyn Fn(&str, &str) -> Option<NodeData>,
        _find_all_in_file: &dyn Fn(&str) -> Vec<NodeData>,
    ) -> Option<(NodeData, Option<Edge>)> {
        let path = std::path::Path::new(file_path);

        let file_name = path.file_name()?.to_str()?;
        let base = file_name
            .strip_suffix(".component.html")
            .or_else(|| file_name.strip_suffix(".component.css"))
            .or_else(|| file_name.strip_suffix(".component.scss"))
            .or_else(|| file_name.strip_suffix(".component.sass"))
            .unwrap_or(file_name);

        let page = NodeData::name_file(file_name, file_path);
        let mut edge = None;

        let component_class = base
            .split('-')
            .map(|s| {
                let mut c = s.chars();
                match c.next() {
                    None => String::new(),
                    Some(f) => f.to_uppercase().collect::<String>() + c.as_str(),
                }
            })
            .collect::<String>()
            + "Component";
        let component_file = format!("{}.component.ts", base);

        if let Some(component) = find_fn(&component_class, &component_file) {
            edge = Some(Edge::new(
                EdgeType::Renders,
                NodeRef::from((&component).into(), NodeType::Class),
                NodeRef::from((&page).into(), NodeType::Page),
            ));
        }

        Some((page, edge))
    }

    fn component_selector_to_template_map(
        &self,
        files: &[(String, String)],
    ) -> std::collections::HashMap<String, String> {
        let mut map = std::collections::HashMap::new();

        for (filename, code) in files {
            if !filename.ends_with(".component.ts") {
                continue;
            }

            if let Some(query_str) = self.component_template_query() {
                let tree = match self.parse(code, &NodeType::Class) {
                    Ok(tree) => tree,
                    Err(_) => continue,
                };

                let query = self.q(&query_str, &NodeType::Class);
                let mut cursor = tree_sitter::QueryCursor::new();
                let mut matches = cursor.matches(&query, tree.root_node(), code.as_bytes());

                let mut selector = None;
                let mut template_url = None;

                while let Some(m) = matches.next() {
                    let mut key = String::new();
                    let mut value = String::new();

                    for o in query.capture_names().iter() {
                        if let Some(ci) = query.capture_index_for_name(o) {
                            let mut nodes = m.nodes_for_capture_index(ci);
                            if let Some(node) = nodes.next() {
                                if let Ok(text) = node.utf8_text(code.as_bytes()) {
                                    if o == &TEMPLATE_KEY {
                                        key = text.to_string();
                                    } else if o == &TEMPLATE_VALUE {
                                        value = text.to_string();
                                    }
                                }
                            }
                        }
                    }

                    if !key.is_empty() && !value.is_empty() {
                        match key.as_str() {
                            "selector" => {
                                selector = Some(parse::trim_quotes(&value).to_string());
                            }
                            "templateUrl" => {
                                template_url = Some(parse::trim_quotes(&value).to_string());
                            }
                            _ => {}
                        }
                    }
                }

                if let (Some(sel), Some(tmpl)) = (selector, template_url) {
                    let resolved_template = self.resolve_import_path(&tmpl, filename);
                    let base = std::path::Path::new(filename).parent().unwrap();
                    let full_template_path =
                        base.join(&resolved_template).to_string_lossy().to_string();
                    map.insert(sel, full_template_path);
                }
            }
        }
        map
    }

    fn page_component_renders_finder(
        &self,
        file_path: &str,
        code: &str,
        selector_map: &std::collections::HashMap<String, String>,
        find_page_fn: &dyn Fn(&str) -> Option<NodeData>,
    ) -> Vec<Edge> {
        let mut edges = Vec::new();

        if !file_path.ends_with(".html") {
            return edges;
        }

        if let Some(current_page) = find_page_fn(file_path) {
            for (selector, target_html_path) in selector_map {
                if code.contains(&format!("<{}", selector)) {
                    if let Some(target_page) = find_page_fn(target_html_path) {
                        edges.push(Edge::new(
                            EdgeType::Renders,
                            NodeRef::from((&current_page).into(), NodeType::Page),
                            NodeRef::from((&target_page).into(), NodeType::Page),
                        ));
                    }
                }
            }
        }
        edges
    }
}
