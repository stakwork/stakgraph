use anyhow::{Ok, Result};
use tracing::info;

use crate::lang::graph::{EdgeType, Graph, Node};
use crate::lang::queries::HandlerParams;
use crate::lang::{linker::normalize_backend_path, Lang};
use crate::repo::Repo;
use crate::utils::logger;

pub struct BackendTester {
    graph: Graph,
    lang: Lang,
}

struct BackendArtefacts {
    endpoints: Vec<Node>,
    data_models: Vec<Node>,
    handler_functions: Vec<Node>,
}
impl BackendTester {
    pub async fn new(lang: Lang) -> Result<Self> {
        logger();

        let language_kind = lang.kind.clone();
        let language_in_repository = Lang::from_language(language_kind);

        let repo = Repo::new(
            &format!("src/testing/{}", lang.kind.to_string()),
            language_in_repository,
            false,
            Vec::new(),
            Vec::new(),
        )
        .unwrap();

        Ok(Self {
            graph: repo.build_graph().await?,
            lang,
        })
    }

    pub fn test_language_node(&self) -> Result<()> {
        let language_nodes = self
            .graph
            .nodes
            .iter()
            .filter(|n| matches!(n, Node::Language(_)))
            .collect::<Vec<_>>();

        assert!(
            !language_nodes.is_empty(),
            "Language node not found for {}",
            self.lang
        );

        let language_node = language_nodes[0].into_data();
        assert_eq!(
            language_node.name,
            self.lang.kind.to_string(),
            "Language node name mismatch"
        );

        info!("✓ Found Language node for {}", self.lang);

        Ok(())
    }

    pub fn test_package_file(&self) -> Result<()> {
        let package_file_name = self.lang.kind.pkg_file();

        let file_nodes = self
            .graph
            .nodes
            .iter()
            .filter(|n| matches!(n, Node::File(_)))
            .collect::<Vec<_>>();

        let package_files: Vec<_> = file_nodes
            .iter()
            .filter(|n| {
                let file_data = n.into_data();
                file_data.name.contains(&package_file_name)
            })
            .collect();

        assert!(
            !package_files.is_empty(),
            "No package file found matching {}",
            package_file_name
        );

        info!("✓ Found package file {}", package_file_name);

        Ok(())
    }

    fn get_data_models(&self, name: &str) -> Vec<Node> {
        self.graph
            .nodes
            .iter()
            .filter(|n| matches!(n, Node::DataModel(_)))
            .filter(|n| {
                let data_model = n.into_data();
                data_model.name == name
            })
            .cloned()
            .collect()
    }

    pub fn test_data_model(&self) -> Result<()> {
        let data_models: Vec<_> = self.get_data_models("Person");

        assert!(!data_models.is_empty(), "Person data model not found");

        info!("✓ Found Person data model");

        Ok(())
    }

    fn normalize_function_name(name: &str) -> String {
        name.replace('_', "").to_lowercase()
    }

    fn function_name_matches(function_name: &str, base_pattern: &str) -> bool {
        let normalized_function = Self::normalize_function_name(function_name);
        let normalized_pattern = Self::normalize_function_name(base_pattern);
        normalized_function.contains(&normalized_pattern)
    }

    fn find_endpoints(&self) -> Vec<Node> {
        self.graph
            .nodes
            .iter()
            .filter(|n| matches!(n, Node::Endpoint(_)))
            .cloned()
            .collect()
    }

    fn get_endpoint(&self, verb: &str, path: Option<&str>) -> Option<Node> {
        self.graph
            .nodes
            .iter()
            .filter(|n| matches!(n, Node::Endpoint(_)))
            .find(|n| {
                let endpoint_data = n.into_data();

                let verb_matches = match (
                    endpoint_data.meta.get("verb"),
                    endpoint_data.meta.get("methods"),
                ) {
                    (Some(verb_value), _) => verb_value.as_str() == verb,
                    (_, Some(methods)) => methods
                        .split(|c| c == ',' || c == ' ' || c == '[' || c == ']')
                        .map(|m| m.trim())
                        .filter(|m| !m.is_empty())
                        .any(|m| m.to_uppercase() == verb.to_uppercase()),

                    // Default route with no method specified often defaults to GET
                    (None, None) => verb.to_uppercase() == "GET",
                };

                if !verb_matches {
                    return false;
                }

                let endpoint_path = normalize_backend_path(&endpoint_data.name).unwrap_or_default();

                path.map_or(true, |p| {
                    let normalized_path = normalize_backend_path(p).unwrap_or_default();
                    endpoint_path == normalized_path || endpoint_path.contains(&normalized_path)
                })
            })
            .cloned()
    }

    pub fn test_endpoints(&self) -> Result<()> {
        let endpoints = self.find_endpoints();

        assert!(
            endpoints.len() >= 2,
            "Expected at least 2 endpoints, found {}",
            endpoints.len()
        );

        let get_endpoint = self.get_endpoint("GET", Some("person"));
        let post_endpoint = self.get_endpoint("POST", Some("person"));

        assert!(get_endpoint.is_some(), "GET person endpoint not found");
        assert!(post_endpoint.is_some(), "POST person endpoint not found");

        info!("✓ Found required endpoints: GET person/:param and POST person");

        Ok(())
    }

    pub fn test_handler_functions(&self) -> Result<()> {
        let endpoints = self.find_endpoints();

        let person_model = self.get_data_models("Person")[0].clone();

        let mut found_get = false;
        let mut found_post = false;

        let mut get_handler_verified = false;
        let mut post_handler_verified = false;

        for endpoint in &endpoints {
            let endpoint_data = endpoint.into_data();
            let is_get = match endpoint_data.meta.get("verb") {
                Some(verb) if verb == "GET" => {
                    found_get = true;
                    true
                }
                _ => found_get,
            };

            let is_post = match endpoint_data.meta.get("verb") {
                Some(verb) if verb == "POST" => {
                    found_post = true;
                    true
                }
                _ => found_post,
            };

            if !is_get && !is_post {
                continue;
            }

            let handler_params = HandlerParams::default();

            let handler_results =
                self.lang
                    .lang()
                    .handler_finder(endpoint_data.clone(), &self.graph, handler_params);

            println!("handler_results: {:?}", handler_results);

            for (_, edge_opt) in handler_results {
                if let Some(edge) = edge_opt {
                    let function_nodes = self
                        .graph
                        .nodes
                        .iter()
                        .filter(|n| matches!(n, Node::Function(_)))
                        .filter(|n| {
                            let function_data = n.into_data();

                            function_data.name == edge.target.node_data.name
                                && function_data.file == edge.target.node_data.file
                        })
                        .collect::<Vec<_>>();

                    if let Some(function_node) = function_nodes.first() {
                        let function_data = function_node.into_data();
                        let function_name = function_data.name.clone();

                        let contains_person = self.graph.edges.iter().any(|e| {
                            e.edge == EdgeType::Contains
                                && e.source.node_data.name == function_data.name.clone()
                                && e.source.node_data.file == function_data.file
                                && e.target.node_data.name == person_model.into_data().name
                        });

                        if contains_person {
                            println!("Functin name : {}", &function_name);
                            if is_get && Self::function_name_matches(&function_name, "GetPerson") {
                                info!(
                                    "✓ GET handlder '{}' references Person datamodel",
                                    function_name
                                );
                                get_handler_verified = true;
                            } else if is_post
                                && Self::function_name_matches(&function_name, "CreatePerson")
                            {
                                info!(
                                    "✓ POST handler '{}' references Person datamodel",
                                    function_name
                                );
                                post_handler_verified = true;
                            }
                        }
                    }
                }
            }
        }

        assert!(
            get_handler_verified,
            "GET endpoint handler for Person not found or doesn't reference Person datamodel",
        );

        assert!(
            post_handler_verified,
            "POST endpoint handler for Person not found or doesn't reference Person datamodel",
        );

        Ok(())
    }

    pub fn test_backend(&self) -> Result<()> {
        self.test_language_node()?;
        self.test_package_file()?;
        self.test_data_model()?;
        self.test_endpoints()?;
        self.test_handler_functions()?;

        Ok(())
    }
}
