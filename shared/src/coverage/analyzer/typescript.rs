use crate::coverage::types::{Language, PackageManager, ProjectConfig, TestFramework};
use crate::Result;
use std::path::Path;

pub struct TypeScriptAnalyzer;

impl TypeScriptAnalyzer {
    pub fn new() -> Self {
        Self
    }

    pub fn analyze(&self, repo_path: &Path) -> Result<ProjectConfig> {
        let mut config = ProjectConfig {
            language: Language::TypeScript,
            package_manager: self.detect_package_manager(repo_path),
            test_frameworks: Vec::new(),
            config_files: Vec::new(),
        };

        let package_json = repo_path.join("package.json");
        if package_json.exists() {
            config.config_files.push(package_json.clone());
            
            if let Ok(content) = std::fs::read_to_string(&package_json) {
                if let Ok(pkg_json) = serde_json::from_str::<serde_json::Value>(&content) {
                    config.test_frameworks = self.detect_test_frameworks(&pkg_json);
                }
            }
        }

        if repo_path.join("tsconfig.json").exists() {
            config.config_files.push(repo_path.join("tsconfig.json"));
        }

        Ok(config)
    }

    fn detect_package_manager(&self, repo_path: &Path) -> Option<PackageManager> {
        if repo_path.join("pnpm-lock.yaml").exists() {
            Some(PackageManager::Pnpm)
        } else if repo_path.join("yarn.lock").exists() {
            Some(PackageManager::Yarn)
        } else if repo_path.join("package-lock.json").exists() {
            Some(PackageManager::Npm)
        } else {
            None
        }
    }

    fn detect_test_frameworks(&self, package_json: &serde_json::Value) -> Vec<TestFramework> {
        let mut frameworks = Vec::new();
        let deps = ["dependencies", "devDependencies"];

        for dep_type in deps {
            if let Some(deps_obj) = package_json.get(dep_type).and_then(|v| v.as_object()) {
                for dep_name in deps_obj.keys() {
                    let framework = match dep_name.as_str() {
                        "vitest" | "@vitest/ui" => Some(TestFramework::Vitest),
                        "jest" | "@jest/core" | "@types/jest" => Some(TestFramework::Jest),
                        "playwright" | "@playwright/test" => Some(TestFramework::Playwright),
                        "cypress" => Some(TestFramework::Cypress),
                        "mocha" | "@types/mocha" => Some(TestFramework::Mocha),
                        "jasmine" | "@types/jasmine" | "karma-jasmine" | "protractor" => Some(TestFramework::Jasmine),
                        _ => None,
                    };

                    if let Some(fw) = framework {
                        if !frameworks.contains(&fw) {
                            frameworks.push(fw);
                        }
                    }
                }
            }
        }

   
        if let Some(scripts) = package_json.get("scripts").and_then(|v| v.as_object()) {
            for script_value in scripts.values() {
                if let Some(script_str) = script_value.as_str() {
                    if script_str.contains("karma") && !frameworks.contains(&TestFramework::Jasmine) {
                        frameworks.push(TestFramework::Jasmine);
                    }
                    if script_str.contains("protractor") && !frameworks.contains(&TestFramework::Jasmine) {
                        frameworks.push(TestFramework::Jasmine);
                    }
                }
            }
        }

        frameworks
    }
}