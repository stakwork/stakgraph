use super::TestCoverage;
use crate::codecov::LanguageReport;
use crate::codecov::coverage::{
    coverage_tools::CoverageTool,
    execution::{CommandRunner, find_test_files, read_package_json_scripts},
    package_managers::PackageManager,
    test_runners::{TestRunner, TestScript},
};
use crate::codecov::utils::{has_any_files_with_ext, parse_summary_or_final};
use crate::{Error, Result};
use std::path::{Path, PathBuf};

pub struct TypeScriptCoverage;

impl TypeScriptCoverage {
    fn install_coverage_dependencies(
        &self,
        repo_path: &Path,
        test_scripts: &[TestScript],
    ) -> Result<()> {
        for script in test_scripts {
            if let Some(dep) = script.runner.coverage_dependency() {
                if !self.check_dependency_available(repo_path, dep) {
                    let package_manager =
                        PackageManager::primary_for_repo(repo_path).unwrap_or(PackageManager::Npm);
                    let (cmd, mut args) = package_manager.install_cmd();
                    args.push("--save-dev".to_string());
                    args.push(dep.to_string());
                    CommandRunner::run_with_string_args(repo_path, cmd, &args)?;
                }
            }
        }
        Ok(())
    }

    fn check_dependency_available(&self, repo_path: &Path, dep: &str) -> bool {
        std::process::Command::new("npx")
            .args(&[dep.split('/').last().unwrap_or(dep), "--version"])
            .current_dir(repo_path)
            .output()
            .map(|out| out.status.success())
            .unwrap_or(false)
    }

    fn execute_test_script(&self, repo_path: &Path, script: &TestScript) -> Result<()> {
        let package_manager =
            PackageManager::primary_for_repo(repo_path).unwrap_or(PackageManager::Npm);
        match script.runner {
            TestRunner::Vitest if script.has_coverage => {
                let args = vec![
                    "vitest",
                    "--coverage",
                    "--coverage.reporter=json-summary",
                    "--coverage.reporter=json",
                    "--coverage.reportsDirectory=./coverage",
                ];
                CommandRunner::run(repo_path, "npx", &args)
            }
            TestRunner::Jest if script.has_coverage => {
                let args = vec![
                    "jest",
                    "--coverage",
                    "--coverageReporters=json-summary",
                    "--coverageReporters=json",
                    "--coverageDirectory=./coverage",
                ];
                CommandRunner::run(repo_path, "npx", &args)
            }
            _ => {
                if script.has_coverage {
                    let (cmd, args) = package_manager.run_script_cmd(&script.name);
                    CommandRunner::run_with_string_args(repo_path, cmd, &args)
                } else {
                    let (script_cmd, script_args) = package_manager.run_script_cmd(&script.name);
                    let mut args = vec![
                        "c8".to_string(),
                        "--reporter=json-summary".to_string(),
                        "--reporter=json".to_string(),
                        "--reports-dir=./coverage".to_string(),
                        script_cmd.to_string(),
                    ];
                    args.extend(script_args);
                    let str_args: Vec<&str> = args.iter().map(|s| s.as_str()).collect();
                    CommandRunner::run(repo_path, "npx", &str_args)
                }
            }
        }
    }
}

impl TestCoverage for TypeScriptCoverage {
    fn name(&self) -> &'static str {
        "typescript"
    }

    fn detect(&self, repo_path: &Path) -> bool {
        let pkg = repo_path.join("package.json");
        if !pkg.exists() {
            return false;
        }
        // Basic extension detection (avoid pulling in lsp crate)
        const EXTS: &[&str] = &["ts", "tsx", "js", "jsx"];
        has_any_files_with_ext(repo_path, EXTS).unwrap_or(false)
    }

    fn needs_install(&self, repo_path: &Path) -> bool {
        if std::env::var("CODECOV_SKIP_INSTALL").is_ok() {
            return false;
        }
        PackageManager::detect(repo_path)
            .iter()
            .any(|pm| pm.needs_install(repo_path))
    }

    fn install(&self, repo_path: &Path) -> Result<()> {
        if let Some(package_manager) = PackageManager::primary_for_repo(repo_path) {
            let (cmd, args) = package_manager.install_cmd();
            CommandRunner::run_with_string_args(repo_path, cmd, &args)?;
        }
        Ok(())
    }

    fn prepare(&self, repo_path: &Path) -> Result<()> {
        if !CoverageTool::C8.check_availability(repo_path) {
            if let Some((cmd, args_vec)) = CoverageTool::C8.install_command() {
                let args: Vec<&str> = args_vec.iter().copied().collect();
                CommandRunner::run(repo_path, cmd, &args)?;
            }
        }
        if let Ok(Some(scripts)) = read_package_json_scripts(repo_path) {
            let test_scripts = TestRunner::detect_from_package_json(&scripts);
            if !test_scripts.is_empty() {
                self.install_coverage_dependencies(repo_path, &test_scripts)?;
                return Ok(());
            }
        }

        let extensions = &["js", "ts", "jsx", "tsx"];
        let test_files = find_test_files(repo_path, extensions);
        if test_files.is_empty() {
            return Err(Error::Custom("Test files not found".to_string()));
        }
        Ok(())
    }

    fn has_existing_coverage(&self, repo_path: &Path) -> bool {
        repo_path.join("coverage/coverage-summary.json").exists()
    }

    fn execute(&self, repo_path: &Path) -> Result<()> {
        if let Ok(Some(scripts)) = read_package_json_scripts(repo_path) {
            let test_scripts = TestRunner::detect_from_package_json(&scripts);
            for script in &test_scripts {
                if let Ok(()) = self.execute_test_script(repo_path, script) {
                    return Ok(());
                }
            }
        }
        if !repo_path.join("coverage/coverage-summary.json").exists() {
            let pm = PackageManager::primary_for_repo(repo_path).unwrap_or(PackageManager::Npm);
            let mut args: Vec<String> = vec![
                "c8".into(),
                "--reporter=json-summary".into(),
                "--reporter=json".into(),
                "--reports-dir=./coverage".into(),
            ];
            match pm {
                PackageManager::Npm => {
                    args.push("npm".into());
                    args.push("test".into());
                    args.push("--".into());
                    args.push("--coverage".into());
                }
                PackageManager::Yarn => {
                    args.push("yarn".into());
                    args.push("test".into());
                    args.push("--coverage".into());
                }
                PackageManager::Pnpm => {
                    args.push("pnpm".into());
                    args.push("test".into());
                    args.push("--".into());
                    args.push("--coverage".into());
                }
                _ => {}
            }
            let str_args: Vec<&str> = args.iter().map(|s| s.as_str()).collect();
            let _ = CommandRunner::run(repo_path, "npx", &str_args);
        }
        if !repo_path.join("coverage/coverage-summary.json").exists() {
            return Err(Error::Custom("coverage not produced".into()));
        }
        Ok(())
    }

    fn parse(&self, repo_path: &Path) -> Result<Option<LanguageReport>> {
        let (lines, branches, functions, statements) = parse_summary_or_final(repo_path)?;
        if lines.is_none() && branches.is_none() && functions.is_none() && statements.is_none() {
            return Ok(None);
        }
        Ok(Some(LanguageReport {
            language: "typescript".into(),
            lines,
            branches,
            functions,
            statements,
        }))
    }

    fn artifact_paths(&self, repo_path: &Path) -> Vec<PathBuf> {
        let cov_dir = repo_path.join("coverage");
        [
            "coverage-summary.json",
            "coverage-final.json",
            "coverage-run.log",
        ]
        .into_iter()
        .map(|n| cov_dir.join(n))
        .filter(|p| p.exists())
        .collect()
    }
}
