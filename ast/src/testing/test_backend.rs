use crate::lang::graphs::NodeType;
use crate::lang::Graph;
use crate::lang::{linker::normalize_backend_path, Lang};
use crate::repo::Repo;
use anyhow::Context;
use tracing::{error, info};

pub struct BackendTester<G: Graph> {
    graph: G,
    lang: Lang,
    repo: Option<String>,
}

impl<G: Graph> BackendTester<G> {
    pub async fn from_repo(lang: Lang, repo: Option<String>) -> Result<Self, anyhow::Error>
    where
        G: Default,
    {
        let language_name = lang.kind.clone();
        let language_in_repository = Lang::from_language(language_name.clone());
        let repo_path = repo.clone().unwrap_or_else(|| language_name.to_string());

        let repository = Repo::new(
            &format!("src/testing/{}", repo_path),
            language_in_repository,
            false,
            Vec::new(),
            Vec::new(),
        )?;

        let graph = repository
            .build_graph_inner()
            .await
            .with_context(|| format!("Failed to build graph for {}", repo_path))?;

        Ok(Self {
            graph,
            lang,
            repo: Some(repo_path),
        })
    }

    pub fn test_backend(&self) -> Result<(), anyhow::Error> {
        info!(
            "\n\nTesting backend for {} at src/testing/{}\n\n",
            self.lang.kind.to_string().to_uppercase(),
            self.repo.as_ref().unwrap()
        );

        self.test_language()?;
        self.test_package_file()?;

        let data_model = self.lang.lang().data_model_name("Person");

        let expected_endpoints = vec![("GET", "person/:param"), ("POST", "person")];

        self.test_data_model(data_model.as_str())?;

        self.test_endpoints(expected_endpoints.clone())?;

        self.test_handler_functions(expected_endpoints, data_model.as_str())?;

        Ok(())
    }

    fn test_language(&self) -> Result<(), anyhow::Error> {
        let language_nodes = self.graph.find_nodes_by_type(NodeType::Language);

        assert!(!language_nodes.is_empty(), "Language node not found");

        let language_node = &language_nodes[0];
        assert_eq!(
            language_node.name,
            self.lang.kind.to_string(),
            "Language node name mismatch"
        );

        Ok(())
    }

    fn test_package_file(&self) -> Result<(), anyhow::Error> {
        let package_file_names = self.lang.kind.pkg_files();
        let package_file_name = package_file_names.first().unwrap();

        let file_nodes = self
            .graph
            .find_nodes_by_name(NodeType::File, &package_file_name);

        assert!(
            !file_nodes.is_empty(),
            "No package file found matching {}",
            package_file_name
        );

        info!("✓ Found package file {}", package_file_name);

        Ok(())
    }

    fn test_data_model(&self, name: &str) -> Result<(), anyhow::Error> {
        let data_model_nodes = self
            .graph
            .find_nodes_by_name_contains(NodeType::DataModel, name);

        if !data_model_nodes.is_empty() {
            info!("✓ Found data model {}", name);
            Ok(())
        } else {
            anyhow::bail!("Data model {} not found", name)
        }
    }

    fn test_endpoints(&self, endpoints: Vec<(&str, &str)>) -> Result<(), anyhow::Error> {
        for (method, path) in endpoints {
            let normalized_expected_path = normalize_backend_path(path).unwrap();

            // Using find_resource_nodes which we added to the Graph trait
            let matching_endpoints = self.graph.find_resource_nodes(
                NodeType::Endpoint,
                method,
                &normalized_expected_path,
            );

            if !matching_endpoints.is_empty() {
                info!("✓ Found endpoint {} {}", method, path);
            } else {
                anyhow::bail!("Endpoint {} {} not found", method, path);
            }
        }

        Ok(())
    }

    fn test_handler_functions(
        &self,
        expected_enpoints: Vec<(&str, &str)>,
        data_model: &str,
    ) -> Result<(), anyhow::Error> {
        for (verb, path) in expected_enpoints {
            let normalized_path = normalize_path(path);

            let matching_endpoints =
                self.graph
                    .find_resource_nodes(NodeType::Endpoint, verb, &normalized_path);
            if matching_endpoints.is_empty() {
                anyhow::bail!("Endpoint {} {} not found", verb, path);
            }

            let mut found_handler = false;
            let mut last_endpoint_name = String::new();
            for endpoint in &matching_endpoints {
                let handlers = self.graph.find_handlers_for_endpoint(endpoint);

                if handlers.is_empty() {
                    info!("No handler found for endpoint {}", endpoint.name);
                    last_endpoint_name = endpoint.name.clone();
                    continue;
                }

                found_handler = true;

                let handler = &handlers[0];
                let handler_name = &handler.name;
                let formatted_handler = normalize_function_name(handler_name);

                info!("✓ Found handler {}", formatted_handler);

                let direct_connection = self
                    .graph
                    .check_direct_data_model_usage(&handler_name, data_model);

                if direct_connection {
                    info!(
                        "✓ Handler {} directly uses data model {}",
                        formatted_handler, data_model
                    );
                    continue;
                }

                let triggered_functions = self.graph.find_functions_called_by(handler);

                if triggered_functions.is_empty() {
                    error!("No functions triggered by handler {}", formatted_handler);
                }

                let mut data_model_found = false;
                let mut functions_to_check = triggered_functions.clone();
                functions_to_check.push(handler.clone());

                for func in &functions_to_check {
                    // Check if this function directly uses the data model
                    if self
                        .graph
                        .check_direct_data_model_usage(&func.name, data_model)
                    {
                        data_model_found = true;
                        break;
                    }

                    let mut visited = Vec::new();

                    if self.check_indirect_data_model_usage(&func.name, data_model, &mut visited) {
                        data_model_found = true;
                        info!(
                            "✓ Found function {} that indirectly triggers data model {}",
                            func.name, data_model
                        );
                        break;
                    }
                }

                if data_model_found {
                    info!(
                        "✓ Data model {} used by handler {}",
                        data_model, formatted_handler
                    );
                } else {
                    error!(
                        "Data model {} not used by handler {}",
                        data_model, formatted_handler
                    );
                }

                assert!(
                    data_model_found,
                    "No function triggers data model {}",
                    data_model
                );
            }
            if !found_handler {
                error!("No handler found for endpoint {}", last_endpoint_name);
            }
        }

        Ok(())
    }

    fn check_indirect_data_model_usage(
        &self,
        function_name: &str,
        data_model: &str,
        visited: &mut Vec<String>,
    ) -> bool {
        if visited.contains(&function_name.to_string()) {
            return false;
        }
        visited.push(function_name.to_string());

        if self
            .graph
            .check_direct_data_model_usage(function_name, data_model)
        {
            return true;
        }

        let function_nodes = self
            .graph
            .find_nodes_by_name(NodeType::Function, function_name);

        if function_nodes.is_empty() {
            return false;
        }

        let function_node = &function_nodes[0];

        let called_functions = self.graph.find_functions_called_by(&function_node);

        for called_function in called_functions {
            if self.check_indirect_data_model_usage(&called_function.name, data_model, visited) {
                return true;
            }
        }
        false
    }
}

pub fn normalize_path(path: &str) -> String {
    let path_with_slash = if path.starts_with("/") {
        path.to_string()
    } else {
        format!("/{}", path)
    };

    normalize_backend_path(&path_with_slash).unwrap_or(path_with_slash)
}
fn normalize_function_name(name: &str) -> String {
    name.replace('_', "").to_lowercase()
}
