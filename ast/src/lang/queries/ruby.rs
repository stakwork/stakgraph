use super::super::*;
use super::consts::*;
use crate::builder::get_page_name;
use crate::lang::parse::trim_quotes;
use crate::lang::queries::rails_routes;
use anyhow::{Context, Result};
use convert_case::{Case, Casing};
use inflection_rs::inflection;
use std::collections::BTreeMap;
use std::path::Path;
use tracing::debug;
use tree_sitter::{Language, Parser, Query, Tree};

pub struct Ruby(Language);

const CONTROLLER_FILE_SUFFIX: &str = "_controller.rb";
const MAILER_FILE_SUFFIX: &str = "_mailer.rb";

impl Ruby {
    pub fn new() -> Self {
        Ruby(tree_sitter_ruby::LANGUAGE.into())
    }
}

impl Stack for Ruby {
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
            r#"(call
                method: (identifier) @gem (#eq? @gem "gem")
                arguments: (argument_list
                    . (string) @{LIBRARY_NAME}
                    (string)? @{LIBRARY_VERSION}
                )
            ) @{LIBRARY}"#
        ))
    }

    fn imports_query(&self) -> Option<String> {
        Some(format!(
            r#"
            (call
                method: (identifier) @method (#any-of? @method "require" "require_relative" "load" "include" "extend")
                arguments: (argument_list) @{IMPORTS_NAME} @{IMPORTS_FROM}
            )@{IMPORTS}
            "#
        ))
    }

    fn variables_query(&self) -> Option<String> {
        Some(format!(
            r#"
            (program
                (assignment
                    left: (_) @{VARIABLE_NAME}
                    right: (_) @{VARIABLE_VALUE}
                )@{VARIABLE_DECLARATION}
            )
            "#
        ))
    }

    fn class_definition_query(&self) -> String {
        format!(
            r#"
            (
              (class
                name: (_) @{CLASS_NAME}
                superclass: (superclass (_) @{CLASS_PARENT})?
                body: (body_statement)?
              ) @{CLASS_DEFINITION}
            )
            (
                (class
                    name: (_) @{CLASS_NAME}
                    superclass: (superclass (_) @{CLASS_PARENT})?
                    body:  (body_statement
                                (call
                                    method: (_) @{ASSOCIATION_TYPE} (#match? @{ASSOCIATION_TYPE} "^has_one$|^has_many$|^belongs_to$|^has_and_belongs_to_many$")
                                    arguments: (argument_list
                                        (simple_symbol)@{ASSOCIATION_TARGET}
                                        (pair
                                            key: (hash_key_symbol) @association.option.key
                                            value: (_) @association.option.value
                                        )? @{ASSOCIATION_OPTION}
                                    )
                                )?
                            )?
                ) @{CLASS_DEFINITION}
                )
                (module
                    name: (constant) @{MODULE_NAME}
                    (body_statement
                    (class
                        name: (constant)@{CLASS_NAME}
                        superclass: (superclass
                            (constant)@{CLASS_PARENT}
                        )?
                        (body_statement
                        (call
                            (argument_list (_)* @{INCLUDED_MODULES} )
                        )?
                        )
                    )@{CLASS_DEFINITION}
                    )
                )
            "#
        )
    }

    fn function_definition_query(&self) -> String {
        format!(
            "[
                (method
                    name: (identifier) @{FUNCTION_NAME}
                    parameters: (method_parameters)? @{ARGUMENTS}
                )
                (singleton_method
                    name: (identifier) @{FUNCTION_NAME}
                    parameters: (method_parameters)? @{ARGUMENTS}
                )
            ] @{FUNCTION_DEFINITION}"
        )
    }
    fn function_call_query(&self) -> String {
        format!(
            "(call
                receiver: [
                    (identifier)
                    (constant)
                    (call)
                ] @{OPERAND}
                method: (identifier) @{FUNCTION_NAME}
                arguments: (argument_list) @{ARGUMENTS}
            ) @{FUNCTION_CALL}"
        )
    }
    fn endpoint_finders(&self) -> Vec<String> {
        super::rails_routes::ruby_endpoint_finders_func()
    }
    fn endpoint_path_filter(&self) -> Option<String> {
        Some("routes.rb".to_string())
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
        while parent.is_some() && parent.unwrap().kind().to_string() != "class" {
            parent = parent.unwrap().parent();
        }
        let parent_of = match parent {
            Some(p) => {
                let query = self.q(&self.identifier_query(), &NodeType::Class);
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
    fn identifier_query(&self) -> String {
        format!("name: [(constant) (scope_resolution)] @identifier")
    }
    fn data_model_name(&self, dm_name: &str) -> String {
        inflection::pluralize(dm_name).to_lowercase()
    }
    fn data_model_query(&self) -> Option<String> {
        Some(format!(
            r#"(call
            receiver: [
                (element_reference
                    object: (scope_resolution
                        scope: (constant) @scope (#eq? @scope "ActiveRecord")
                        name: (constant) @name (#eq? @name "Schema")
                    )
                )
                (scope_resolution
                    scope: (constant) @scope (#eq? @scope "ActiveRecord")
                    name: (constant) @name (#eq? @name "Schema")
                )
            ]
            block: (do_block
                body: (body_statement
                    (call
                        method: (identifier) @create (#eq? @create "create_table")
                        arguments: (argument_list
                            (string) @{STRUCT_NAME}
                        )
                    ) @{STRUCT}
                )
            )
            )"#
        ))
    }
    fn data_model_path_filter(&self) -> Option<String> {
        Some("db/schema.rb".to_string())
    }
    fn use_data_model_within_finder(&self) -> bool {
        true
    }
    fn data_model_within_finder(
        &self,
        data_model: &NodeData,
        find_fns_in: &dyn Fn(&str) -> Vec<NodeData>,
    ) -> Vec<Edge> {
        // file: app/controllers/api/advisor_groups_controller.rb
        let mut models = Vec::new();

        // println!("{}{}", &data_model.name, CONTROLLER_FILE_SUFFIX);

        let funcs = find_fns_in(format!("{}{}", &data_model.name, CONTROLLER_FILE_SUFFIX).as_str());

        for func in funcs {
            models.push(Edge::contains(
                NodeType::Function,
                &func,
                NodeType::DataModel,
                data_model,
            ));
        }

        // without: Returning Graph with 12726 nodes and 13283 edges
        // if edge:Handler with source.node_data.name == name, then the target -> Contains this data model
        // "advisor_groups"
        models
    }
    fn is_test(&self, _func_name: &str, func_file: &str) -> bool {
        self.is_test_file(func_file)
    }
    fn is_test_file(&self, filename: &str) -> bool {
        filename.ends_with("_spec.rb")
    }
    fn e2e_test_id_finder_string(&self) -> Option<String> {
        Some("get_by_test_id".to_string())
    }
    fn use_handler_finder(&self) -> bool {
        true
    }
    fn handler_finder(
        &self,
        endpoint: NodeData,
        find_fn: &dyn Fn(&str, &str) -> Option<NodeData>,
        find_fns_in: &dyn Fn(&str) -> Vec<NodeData>,
        params: HandlerParams,
    ) -> Vec<(NodeData, Option<Edge>)> {
        if endpoint.meta.get("handler").is_none() {
            return Vec::new();
        }

        let handler_string = endpoint.meta.get("handler").unwrap();
        // tracing::info!("handler_finder: {} {:?}", handler_string, params);
        let mut explicit_path = false;
        // intermediate nodes (src/target)
        let mut inter = Vec::new();
        // let mut targets = Vec::new();
        if let Some(item) = &params.item {
            debug!("===> found item: {}", item.name);
            if let Some(nd) = find_fn(
                &item.name,
                format!("{}{}", &handler_string, &CONTROLLER_FILE_SUFFIX).as_str(),
            ) {
                inter.push((endpoint, nd));
            }
        } else if handler_string.contains("#") {
            // put 'request_center/:id', to: 'request_center#update'
            let arr = handler_string.split("#").collect::<Vec<&str>>();
            if arr.len() != 2 {
                return Vec::new();
            }
            let controller = arr[0];
            let name = arr[1];
            // debug!("controller: {}, name: {}", controller, name);
            if let Some(nd) = find_fn(
                name,
                format!("{}{}", &controller, &CONTROLLER_FILE_SUFFIX).as_str(),
            ) {
                inter.push((endpoint, nd));
                explicit_path = true;
            }
        } else {
            // https://guides.rubyonrails.org/routing.html  section 2.2 CRUD, Verbs, and Actions
            let ror_actions = vec![
                "index", "show", "new", "create", "edit", "update", "destroy",
            ]
            .iter()
            .map(|s| s.to_string())
            .collect();
            let verb_mapping = vec![
                ("GET", "index"),
                ("GET", "show"),
                ("GET", "new"),
                ("POST", "create"),
                ("GET", "edit"),
                ("PUT", "update"),
                ("DELETE", "destroy"),
            ];
            let mut verb_map = BTreeMap::new();
            for (verb, action) in verb_mapping {
                verb_map.insert(action.to_string(), verb.to_string());
            }
            let actions = match &params.actions_array {
                Some(aa) => {
                    let aaa = trim_array_string(aa);
                    // split on commas, and trim_quotes
                    aaa.split(",")
                        .map(|s| trim_quotes(s).to_string())
                        .collect::<Vec<String>>()
                }
                None => ror_actions,
            };
            // resources :request_center
            let controllers =
                find_fns_in(format!("{}{}", &handler_string, CONTROLLER_FILE_SUFFIX).as_str());
            debug!(
                "ror endpoint controllers for {}: {:?}",
                handler_string,
                controllers.len()
            );
            for nd in controllers {
                debug!("checking controller: {}", nd.name);
                if actions.contains(&nd.name) {
                    debug!("===> found action: {}", nd.name);
                    let mut endp_ = endpoint.clone();
                    endp_.add_action(&nd.name);
                    if let Some(verb) = verb_map.get(&nd.name) {
                        endp_.add_verb(verb);
                    }
                    inter.push((endp_, nd));
                }
            }
        }

        let ret = inter
            .iter()
            .map(|(src, target)| {
                let mut src = src.clone();
                if !explicit_path {
                    if let Some(pathy) = rails_routes::generate_endpoint_path(&src, &params) {
                        src.name = pathy;
                    }
                }
                let edge = Edge::handler(&src, &target);
                (src.clone(), Some(edge))
            })
            .collect::<Vec<(NodeData, Option<Edge>)>>();

        ret
    }
    fn find_endpoint_parents(
        &self,
        node: TreeNode,
        code: &str,
        _file: &str,
        _callback: &dyn Fn(&str) -> Option<NodeData>,
    ) -> Result<Vec<HandlerItem>> {
        let mut parents = Vec::new();
        let mut parent = node.parent();

        while parent.is_some() {
            let parent_node = parent.unwrap();
            if parent_node.kind().to_string() == "call" {
                // Check if this is a namespace or resources call
                if let Some(method_node) = parent_node.child_by_field_name("method") {
                    if method_node.kind().to_string() == "identifier" {
                        let method_name = method_node.utf8_text(code.as_bytes()).unwrap_or("");
                        if method_name == "namespace" || method_name == "resources" {
                            // Get the first argument which should be the route name
                            if let Some(args_node) = parent_node.child_by_field_name("arguments") {
                                if let Some(first_arg) = args_node.named_child(0) {
                                    let route_name =
                                        first_arg.utf8_text(code.as_bytes()).unwrap_or("");
                                    let item_type = if method_name == "namespace" {
                                        HandlerItemType::Namespace
                                    } else {
                                        HandlerItemType::ResourceMember
                                    };
                                    // Create HandlerItem for this parent route
                                    parents.push(HandlerItem {
                                        name: trim_quotes(route_name).to_string(),
                                        item_type,
                                    });
                                }
                            }
                        }
                    }
                }
            }
            parent = parent_node.parent();
        }

        // Reverse the order so that outermost parents come first
        parents.reverse();
        Ok(parents)
    }
    fn integration_test_query(&self) -> Option<String> {
        Some(format!(
            r#"(call
                method: (identifier) @describe (#eq? @describe "describe")
                arguments: [
                    (argument_list
                        [
                            (constant)
                            (scope_resolution)
                        ] @{HANDLER}
                    )
                    (argument_list
                        (string) @{E2E_TEST_NAME}
                        (pair) @js-true (#eq? @js-true "js: true")
                    )
                ]
            ) @{INTEGRATION_TEST}"#
        ))
    }
    fn use_integration_test_finder(&self) -> bool {
        true
    }
    fn integration_test_edge_finder(
        &self,
        nd: &NodeData,
        find_class: &dyn Fn(&str) -> Option<NodeData>,
        tt: NodeType,
    ) -> Option<Edge> {
        let cla = find_class(&nd.name);
        if let Some(cl) = cla {
            Some(Edge::calls(tt, nd, NodeType::Class, &cl))
        } else {
            None
        }
    }
    fn use_extra_page_finder(&self) -> bool {
        true
    }
    fn is_extra_page(&self, file_name: &str) -> bool {
        let is_good_ext = file_name.ends_with(".erb")
            || file_name.ends_with(".haml")
            || file_name.ends_with(".slim")
            || file_name.ends_with(".html");
        let pagename = get_page_name(file_name);
        if pagename.is_none() {
            return false;
        }
        // let is_underscore = pagename.as_ref().unwrap().starts_with("_");
        let is_view = file_name.contains("/views/");
        is_view && is_good_ext
    }
    fn extra_page_finder(
        &self,
        file_path: &str,
        find_fn: &dyn Fn(&str, &str) -> Option<NodeData>,
    ) -> Option<Edge> {
        let pagename = get_page_name(file_path);
        if pagename.is_none() {
            return None;
        }
        let pagename = pagename.unwrap();
        let page = NodeData::name_file(&pagename, file_path);
        // get the handler name
        let p = std::path::Path::new(file_path);
        let func_name = remove_all_extensions(p);
        let parent_name = p.parent()?.file_name()?.to_str()?;
        println!("func_name: {}, parent_name: {}", func_name, parent_name);
        let controller_handler = find_fn(
            &func_name,
            &format!("{}{}", parent_name, CONTROLLER_FILE_SUFFIX),
        );
        if let Some(h) = controller_handler {
            return Some(Edge::renders(&page, &h));
        }
        let parent_name_no_mailer = parent_name.strip_suffix("_mailer").unwrap_or(parent_name);
        let mailer_handler = find_fn(
            &func_name,
            &format!("{}{}", parent_name_no_mailer, MAILER_FILE_SUFFIX),
        );
        if let Some(h) = mailer_handler {
            return Some(Edge::renders(&page, &h));
        }
        println!("no handler found for {} {}", func_name, file_path);
        None
    }
    fn direct_class_calls(&self) -> bool {
        true
    }
    fn convert_association_to_name(&self, name: &str) -> String {
        let target_class = inflection_rs::inflection::singularize(name);
        target_class.to_case(Case::Pascal)
    }

    fn resolve_import_path(&self, import_path: &str, _current_file: &str) -> String {
        let mut path = import_path.to_string();

        if path.starts_with("(") {
            path = path[1..path.len() - 1].to_string();
        }

        if path.contains(":") {
            path = path.replace(":", "");
        }

        path
    }
    fn resolve_import_name(&self, import_name: &str) -> String {
        let mut name = import_name.to_string();

        if name.starts_with("(") {
            name = name[1..name.len() - 1].to_string();
        }

        if name.contains(":") {
            name = name.replace(":", "");
        }

        if name.starts_with("\"") && name.ends_with("\"") {
            name = name[1..name.len() - 1].to_string();
        }

        inflection::camelize(name)
    }
}

fn remove_all_extensions(path: &Path) -> String {
    let mut stem = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_string();

    while let Some(s) = Path::new(&stem).file_stem() {
        if let Some(s_str) = s.to_str() {
            if s_str == stem {
                break;
            }
            stem = s_str.to_string();
        } else {
            break;
        }
    }

    stem
}

fn trim_array_string(s: &str) -> String {
    s.trim_start_matches("%i")
        .trim_start_matches("[")
        .trim_end_matches("]")
        .to_string()
}

// fn is_controller(nd: &NodeData, controller: &str) -> bool {
//     nd.file.ends_with(&format!("{}_controller.rb", controller))
// }
