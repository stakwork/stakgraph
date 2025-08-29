use super::{TestCoverage, package_managers::PackageManager, test_runners::{TestRunner, TestScript}, coverage_tools::CoverageTool, execution::{CommandRunner, find_test_files}};
use crate::codecov::utils::{has_any_files_with_ext, parse_summary_or_final};
use crate::types::{LanguageReport, Metric};
use lsp::Language;
use shared::{Error, Result};
use std::fs;
use std::path::{Path, PathBuf};

pub struct PythonCoverage;

impl PythonCoverage {
    fn find_python_executable(&self) -> &'static str {
        for python_cmd in &["python3", "python", "python3.12", "python3.11", "python3.10", "python3.9"] {
            if std::process::Command::new(python_cmd)
                .arg("--version")
                .output()
                .map(|out| out.status.success())
                .unwrap_or(false)
            {
                return python_cmd;
            }
        }
        "python3"
    }
    
    fn install_coverage_dependencies(&self, repo_path: &Path, test_scripts: &[TestScript]) -> Result<()> {
        for script in test_scripts {
            if let Some(dep) = script.runner.coverage_dependency() {
                if !self.check_python_package_available(repo_path, dep) {
                    let package_manager = PackageManager::primary_for_repo(repo_path)
                        .unwrap_or(PackageManager::Pip);
                    
                    let (cmd, mut args) = package_manager.install_cmd();
                    args.push(dep.to_string());
                    CommandRunner::run_with_string_args(repo_path, cmd, &args)?;
                }
            }
        }
        Ok(())
    }
    
    fn check_python_package_available(&self, repo_path: &Path, package: &str) -> bool {
        let python_cmd = self.find_python_executable();
        std::process::Command::new(python_cmd)
            .args(&["-c", &format!("import {}", package.replace("-", "_"))])
            .current_dir(repo_path)
            .output()
            .map(|out| out.status.success())
            .unwrap_or(false)
    }
    
    fn execute_test_script(&self, repo_path: &Path, script: &TestScript) -> Result<()> {
        let python_cmd = self.find_python_executable();
        
        match script.runner {
            TestRunner::Tox => {
                CommandRunner::run(repo_path, "tox", &[])
            }
            TestRunner::Pytest if script.has_coverage => {
                CommandRunner::run(repo_path, python_cmd, &[
                    "-m", "pytest", 
                    "--cov=.", 
                    "--cov-report=json",
                    "--cov-report=xml",
                    "--cov-report=term",
                ])
            }
            TestRunner::Pytest => {
                CommandRunner::run(repo_path, python_cmd, &[
                    "-m", "coverage", "run", 
                    "--source=.", 
                    "-m", "pytest"
                ])?;

                CommandRunner::run(repo_path, python_cmd, &[
                    "-m", "coverage", "json"
                ])?;
                CommandRunner::run(repo_path, python_cmd, &[
                    "-m", "coverage", "xml"
                ])
            }
            TestRunner::Unittest if script.has_coverage => {
                CommandRunner::run(repo_path, python_cmd, &[
                    "-m", "coverage", "run", 
                    "--source=.", 
                    "-m", "unittest", "discover"
                ])?;

                CommandRunner::run(repo_path, python_cmd, &[
                    "-m", "coverage", "json"
                ])?;
                CommandRunner::run(repo_path, python_cmd, &[
                    "-m", "coverage", "xml"
                ])
            }
            TestRunner::Unittest => {

                CommandRunner::run(repo_path, python_cmd, &[
                    "-m", "unittest", "discover"
                ])?;
                CommandRunner::run(repo_path, python_cmd, &[
                    "-m", "coverage", "run", 
                    "--source=.", 
                    "-m", "unittest", "discover"
                ])?;
                CommandRunner::run(repo_path, python_cmd, &[
                    "-m", "coverage", "json"
                ])?;
                CommandRunner::run(repo_path, python_cmd, &[
                    "-m", "coverage", "xml"
                ])
            }
            _ => {

               return Err(shared::Error::Custom(format!("Unsupported test runner: {:?}", script.runner)));
            }
        }
    }

    
    fn parse_coverage_json(&self, repo_path: &Path) -> Result<Option<LanguageReport>> {
        let coverage_file = repo_path.join("coverage.json");
        if !coverage_file.exists() {
            return Ok(None);
        }
        
        let content = fs::read_to_string(coverage_file)?;
        let coverage_data: serde_json::Value = serde_json::from_str(&content)
            .map_err(|e| shared::Error::Custom(format!("Failed to parse coverage.json: {}", e)))?;
        
        if let Some(totals) = coverage_data.get("totals") {
            let lines_metric = if let (Some(covered), Some(total), Some(pct)) = (
                totals.get("covered_lines").and_then(|v| v.as_u64()),
                totals.get("num_statements").and_then(|v| v.as_u64()),
                totals.get("percent_covered").and_then(|v| v.as_f64())
            ) {
                Some(Metric { total, covered, pct })
            } else {
                None
            };
            
            let branches_metric = if let (Some(covered), Some(total), Some(pct)) = (
                totals.get("covered_branches").and_then(|v| v.as_u64()),
                totals.get("num_branches").and_then(|v| v.as_u64()),
                totals.get("percent_covered").and_then(|v| v.as_f64())
            ) {
                Some(Metric { total, covered, pct })
            } else {
                None
            };
            
            return Ok(Some(LanguageReport {
                language: "python".into(),
                lines: lines_metric,
                branches: branches_metric,
                functions: None,
                statements: None,
            }));
        }
        
        Ok(None)
    }
}

impl TestCoverage for PythonCoverage {
    fn name(&self) -> &'static str {
        "python"
    }
    
    fn detect(&self, repo_path: &Path) -> bool {
        if !has_any_files_with_ext(repo_path, &Language::Python.exts()).unwrap_or(false) {
            return false;
        }
        
        repo_path.join("setup.py").exists() ||
        repo_path.join("pyproject.toml").exists() ||
        repo_path.join("requirements.txt").exists() ||
        repo_path.join("Pipfile").exists() ||
        repo_path.join("tox.ini").exists()
    }
    
    fn needs_install(&self, repo_path: &Path) -> bool {
        if std::env::var("CODECOV_SKIP_INSTALL").is_ok() {
            return false;
        }
        
        repo_path.join("requirements.txt").exists() ||
        repo_path.join("setup.py").exists() ||
        repo_path.join("pyproject.toml").exists()
    }
    
    fn install(&self, repo_path: &Path) -> Result<()> {
        let package_manager = PackageManager::primary_for_repo(repo_path)
            .unwrap_or(PackageManager::Pip);
        
        if repo_path.join("requirements.txt").exists() {
            let (cmd, args) = package_manager.install_cmd();
            CommandRunner::run_with_string_args(repo_path, cmd, &args)?;
        }

        if repo_path.join("setup.py").exists() {
            CommandRunner::run(repo_path, "python", &["-m", "pip", "install", "-e", "."])?;
        }
        
        Ok(())
    }
    
    fn prepare(&self, repo_path: &Path) -> Result<()> {
        if !CoverageTool::CoveragePy.check_availability(repo_path) {
            CommandRunner::run(repo_path, "python", &["-m", "pip", "install", "coverage"])?;
        }

        let test_scripts = TestRunner::detect_from_python_project(repo_path);
        if !test_scripts.is_empty() {
            self.install_coverage_dependencies(repo_path, &test_scripts)?;
            return Ok(());
        }

        let extensions = &["py"];
        let test_files = find_test_files(repo_path, extensions);
        if !test_files.is_empty() {
           return Err(Error::Custom("Test files not found".to_string()));
        }
        
        Ok(())
    }
    
    fn has_existing_coverage(&self, repo_path: &Path) -> bool {
        repo_path.join("coverage.json").exists() || 
        repo_path.join("coverage.xml").exists() ||
        repo_path.join(".coverage").exists()
    }
    
    fn execute(&self, repo_path: &Path) -> Result<()> {
        let test_scripts = TestRunner::detect_from_python_project(repo_path);
        
        for script in &test_scripts {
            if let Ok(()) = self.execute_test_script(repo_path, script) {
                return Ok(());
            }
        }

        Ok(())
    }
    
    fn parse(&self, repo_path: &Path) -> Result<Option<LanguageReport>> {
        if let Ok(Some(report)) = self.parse_coverage_json(repo_path) {
            return Ok(Some(report));
        }
        
        let (lines, branches, functions, statements) = parse_summary_or_final(repo_path)?;
        
        let empty = lines.is_none() && branches.is_none() && functions.is_none() && statements.is_none();
        if empty {
            return Ok(None);
        }
        
        Ok(Some(LanguageReport {
            language: "python".into(),
            lines,
            branches,
            functions,
            statements,
        }))
    }
    
    fn artifact_paths(&self, repo_path: &Path) -> Vec<PathBuf> {
        let mut paths = Vec::new();

        let coverage_files = [
            "coverage.json", "coverage.xml", ".coverage", 
            "htmlcov/index.html", "coverage_test_runner.py"
        ];
        
        for file in &coverage_files {
            let path = repo_path.join(file);
            if path.exists() {
                paths.push(path);
            }
        }
        
        let cov_dir = repo_path.join("coverage");
        if cov_dir.exists() {
            if let Ok(entries) = fs::read_dir(&cov_dir) {
                for entry in entries.flatten() {
                    paths.push(entry.path());
                }
            }
        }
        
        paths
    }
}
