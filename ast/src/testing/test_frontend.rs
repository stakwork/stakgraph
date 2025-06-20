use crate::lang::graphs::NodeType;
use crate::lang::{Graph, Lang};
use crate::repo::Repo;
use anyhow::Context;
use lsp::Language as LspLanguage;
use std::collections::HashMap;
use std::result::Result;
use tracing::info;

pub struct FrontendTester<G: Graph> {
    graph: G,
    lang: Lang,
    repo: Option<String>,
}

pub struct FrontendArtefact<'a> {
    pub components: Vec<&'a str>,
    pub request: Vec<(&'a str, &'a str)>,
    pub pages: Vec<&'a str>,
    pub data_model: &'a str,
    pub contains_pages_and_components: Vec<&'a LspLanguage>,
}

impl FrontendArtefact<'_> {
    pub fn default() -> FrontendArtefact<'static> {
        FrontendArtefact {
            components: vec!["NewPerson", "People"],
            request: vec![("GET", "/people"), ("POST", "/person")],
            pages: vec!["/new-person", "/people"],
            data_model: "Person",
            contains_pages_and_components: vec![&LspLanguage::React],
        }
    }
}
impl<G: Graph> FrontendTester<G> {
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

    pub fn test_frontend(&self) -> Result<(), anyhow::Error> {
        let artefact = FrontendArtefact::default();

        info!(
            "\n\nTesting frontend for {} at src/testing/{}\n\n",
            self.lang.kind.to_string().to_uppercase(),
            self.repo.as_ref().unwrap()
        );

        self.test_language()?;
        self.test_package_file()?;
        self.test_data_model(artefact.data_model)?;
        if artefact
            .contains_pages_and_components
            .contains(&&self.lang.kind)
        {
            self.test_components(artefact.components)?;
            self.test_pages(artefact.pages)?;
        }
        self.test_requests(artefact.request)?;

        Ok(())
    }

    fn test_language(&self) -> Result<(), anyhow::Error> {
        let language_nodes = self
            .graph
            .find_nodes_by_type(NodeType::Language)
            .first()
            .cloned()
            .unwrap();

        assert_eq!(
            language_nodes.name,
            self.lang.kind.to_string(),
            "Language node name mismatch"
        );

        Ok(())
    }

    fn test_package_file(&self) -> Result<(), anyhow::Error> {
        let package_file_names = self.lang.kind.pkg_files();
        let package_file_name = package_file_names.first().unwrap();

        let pkg_file_nodes = self
            .graph
            .find_nodes_by_name_contains(NodeType::File, &package_file_name);

        assert!(
            pkg_file_nodes.len() >= 1,
            "No package file found matching {}",
            package_file_name
        );

        info!("✓ Found package file {}", package_file_name);

        Ok(())
    }

    fn test_data_model(&self, data_model: &str) -> Result<(), anyhow::Error> {
        let data_model_nodes = self
            .graph
            .find_nodes_by_name_contains(NodeType::DataModel, data_model);

        info!("✓ Found data model {}", data_model);

        assert!(
            data_model_nodes.len() >= 1,
            "No data model found matching {}",
            data_model
        );

        Ok(())
    }

    fn test_components(&self, expected_components: Vec<&str>) -> Result<(), anyhow::Error> {
        let mut found_components: HashMap<&str, bool> = expected_components
            .clone()
            .into_iter()
            .map(|item| (item, false))
            .collect();

        for component in expected_components {
            if let Some(node) = self
                .graph
                .find_nodes_by_name(NodeType::Function, component)
                .first()
            {
                let component_name = node.name.as_str();
                if let Some(entry) = found_components.get_mut(component_name) {
                    *entry = true;
                    info!("✓ Found component {}", component_name);
                }
            }
        }

        for (component_name, found) in found_components.iter() {
            assert!(*found, "Component {} not found in graph", component_name);
        }
        Ok(())
    }

    fn test_pages(&self, expected_pages: Vec<&str>) -> Result<(), anyhow::Error> {
        let mut found_pages: HashMap<&str, bool> = expected_pages
            .clone()
            .into_iter()
            .map(|item| (item, false))
            .collect();

        for page in expected_pages {
            if let Some(node) = self.graph.find_nodes_by_name(NodeType::Page, page).last() {
                let page_name = node.name.as_str();
                if let Some(entry) = found_pages.get_mut(page_name) {
                    *entry = true;
                    info!("✓ Found page {}", page_name);
                }
            }
        }
        for (page_name, found) in found_pages.iter() {
            assert!(*found, "Page {} not found in graph", page_name);
        }
        Ok(())
    }

    fn test_requests(&self, expected_requests: Vec<(&str, &str)>) -> Result<(), anyhow::Error> {
        let mut found_requests = HashMap::new();
        for (verb, path) in expected_requests.iter() {
            found_requests.insert((verb.to_string(), path.to_string()), false);
        }
        for ((verb, path), found) in found_requests.iter_mut() {
            let matching_nodes = self
                .graph
                .find_resource_nodes(NodeType::Request, verb, path);
            if !matching_nodes.is_empty() {
                *found = true;
                info!("✓ Found request {} {}", verb, path);
            }
        }

        for ((verb, path), found) in found_requests.iter() {
            assert!(*found, "Request {} {} not found in graph", verb, path);
        }

        Ok(())
    }
}
