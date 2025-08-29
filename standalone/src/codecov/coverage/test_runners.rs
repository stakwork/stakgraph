use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq)]
pub enum TestRunner {
    Vitest,
    Jest,
    Playwright,
    Cypress,
    Pytest,
    Unittest,
    Tox,
    CargoTest,
    Maven,
}

#[derive(Debug, Clone)]
pub struct TestScript {
    pub name: String,
    pub runner: TestRunner,
    pub has_coverage: bool,
}

impl TestRunner {
    pub fn detect_from_package_json(scripts: &HashMap<String, String>) -> Vec<TestScript> {
        let mut test_scripts = Vec::new();
        
        let test_script_patterns = [
            "test:coverage", "test", "test:unit", "coverage", "test:integration"
        ];
        
        for script_name in test_script_patterns {
            if let Some(script_content) = scripts.get(script_name) {
                let (runner, has_coverage) = Self::detect_runner_from_script(script_content);
                test_scripts.push(TestScript {
                    name: script_name.to_string(),
                    runner,
                    has_coverage: has_coverage || script_name.contains("coverage"),
                });
            }
        }
        
        test_scripts
    }
    
    pub fn detect_from_python_project(repo_path: &std::path::Path) -> Vec<TestScript> {
        let mut test_scripts = Vec::new();
        
        if repo_path.join("tox.ini").exists() {
            test_scripts.push(TestScript {
                name: "tox".to_string(),
                runner: TestRunner::Tox,
                has_coverage: Self::tox_has_coverage(repo_path),
            });
        }
        
        if Self::has_pytest_config(repo_path) {
            test_scripts.push(TestScript {
                name: "pytest".to_string(),
                runner: TestRunner::Pytest,
                has_coverage: Self::pytest_has_coverage(repo_path),
            });
        }
        
        if test_scripts.is_empty() && Self::has_python_test_files(repo_path) {
            test_scripts.push(TestScript {
                name: "unittest".to_string(),
                runner: TestRunner::Unittest,
                has_coverage: false,
            });
        }
        
        test_scripts
    }
    
    fn has_pytest_config(repo_path: &std::path::Path) -> bool {
        repo_path.join("pytest.ini").exists() || 
        repo_path.join("pyproject.toml").exists() ||
        repo_path.join("setup.cfg").exists()
    }
    
    fn has_python_test_files(repo_path: &std::path::Path) -> bool {
        use std::fs;

        if repo_path.join("test").exists() || repo_path.join("tests").exists() {
            return true;
        }

        if let Ok(entries) = fs::read_dir(repo_path) {
            for entry in entries.flatten() {
                if let Some(name) = entry.file_name().to_str() {
                    if name.starts_with("test_") && name.ends_with(".py") {
                        return true;
                    }
                }
            }
        }
        
        false
    }
    
    fn tox_has_coverage(repo_path: &std::path::Path) -> bool {
        use std::fs;
        
        if let Ok(content) = fs::read_to_string(repo_path.join("tox.ini")) {
            content.contains("coverage") || content.contains("pytest-cov")
        } else {
            false
        }
    }
    
    fn pytest_has_coverage(repo_path: &std::path::Path) -> bool {
        use std::fs;

        if repo_path.join(".coveragerc").exists() {
            return true;
        }

        if let Ok(content) = fs::read_to_string(repo_path.join("pyproject.toml")) {
            if content.contains("[tool.coverage") || content.contains("pytest-cov") {
                return true;
            }
        }
        
        false
    }
    
    fn detect_runner_from_script(script: &str) -> (TestRunner, bool) {
        if script.contains("vitest") {
            (TestRunner::Vitest, script.contains("coverage"))
        } else if script.contains("jest") {
            (TestRunner::Jest, script.contains("coverage"))
        } else if script.contains("playwright") {
            (TestRunner::Playwright, false)
        } else if script.contains("cypress") {
            (TestRunner::Cypress, false)
        } else {
            (TestRunner::Jest, false) 
        }
    }
    
    pub fn coverage_dependency(&self) -> Option<&'static str> {
        match self {
            TestRunner::Vitest => Some("@vitest/coverage-v8"),
            TestRunner::Jest => Some("@jest/globals"),
            TestRunner::Pytest => Some("pytest-cov"),
            TestRunner::Unittest => Some("coverage"),
            TestRunner::Tox => Some("coverage"),
            _ => None,
        }
    }
    
    pub fn coverage_reporters(&self) -> Vec<&'static str> {
        match self {
            TestRunner::Vitest => vec!["json-summary", "json"],
            TestRunner::Jest => vec!["json-summary", "json"],
            TestRunner::Pytest => vec!["json", "xml"],
            TestRunner::Unittest => vec!["json", "xml"], 
            TestRunner::Tox => vec!["json", "xml"],
            _ => vec![],
        }
    }
}
