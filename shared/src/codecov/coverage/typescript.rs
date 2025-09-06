use super::TestCoverage;
use crate::codecov::coverage::{
	coverage_tools::CoverageTool,
	execution::{find_test_files, CommandRunner},
	package_managers::PackageManager,
	test_runners::{
		coverage_dependency, detect_from_package_json, execute_with_fallback, TestScript, detect_runners,
	},
};
use crate::codecov::utils::{has_any_files_with_ext, parse_summary_or_final};
use crate::codecov::LanguageReport;
use crate::{Error, Result};
use std::{collections::HashMap, fs, path::{Path, PathBuf}};

pub struct TypeScriptCoverage;

impl TypeScriptCoverage {
	fn install_coverage_dependencies(
		&self,
		repo_path: &Path,
		test_scripts: &[TestScript],
	) -> Result<()> {
		for script in test_scripts {
			if let Some(dep) = coverage_dependency(&script.runner_id) {
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
		// Use new generic executor with fallback attempts.
		execute_with_fallback(repo_path, script, 3)
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
		if let Some(scripts) = self.read_test_scripts(repo_path)? {
			let test_scripts = detect_from_package_json(&scripts);
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
		if let Some(scripts) = self.read_test_scripts(repo_path)? {
			let test_scripts = detect_from_package_json(&scripts);
			for script in &test_scripts {
				if let Ok(()) = self.execute_test_script(repo_path, script) {
					if repo_path.join("coverage/coverage-summary.json").exists() { return Ok(()); }
				}
			}
			let deps = self.read_test_dependencies(repo_path)?;
			let runner_ids = detect_runners(repo_path, &scripts, &deps);
			for rid in runner_ids {
				let pseudo = TestScript { name: "test".into(), runner_id: rid.clone(), has_coverage: true };
				let _ = self.execute_test_script(repo_path, &pseudo);
				if repo_path.join("coverage/coverage-summary.json").exists() { return Ok(()); }
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

	fn read_test_scripts(&self, repo_path: &Path) -> Result<Option<std::collections::HashMap<String,String>>> {
		    let pkg_path = repo_path.join("package.json");
    if !pkg_path.exists() {
        return Ok(None);
    }
    let pkg_content = fs::read_to_string(pkg_path)?;
    let pkg_json: serde_json::Value = serde_json::from_str(&pkg_content)
        .map_err(|e| Error::Custom(format!("Failed to parse package.json: {}", e)))?;
    if let Some(scripts) = pkg_json.get("scripts").and_then(|s| s.as_object()) {
        let mut result = HashMap::new();
        for (key, value) in scripts {
            if let Some(script) = value.as_str() {
                result.insert(key.clone(), script.to_string());
            }
        }
        Ok(Some(result))
    } else {
        Ok(None)
    }
	}

	fn read_test_dependencies(&self, repo_path: &Path) -> Result<Vec<String>> {
		 let pkg_path = repo_path.join("package.json");
    if !pkg_path.exists() {
        return Ok(vec![]);
    }
    let pkg_content = fs::read_to_string(pkg_path)?;
    let pkg_json: serde_json::Value = serde_json::from_str(&pkg_content)
        .map_err(|e| Error::Custom(format!("Failed to parse package.json: {}", e)))?;
    let mut deps = Vec::new();
    if let Some(obj) = pkg_json.get("dependencies").and_then(|v| v.as_object()) {
        for k in obj.keys() { deps.push(k.clone()); }
    }
    if let Some(obj) = pkg_json.get("devDependencies").and_then(|v| v.as_object()) {
        for k in obj.keys() { if !deps.contains(k) { deps.push(k.clone()); } }
    }
    Ok(deps)
	}

	fn discover_test_files(&self, repo_path: &Path) -> Result<Vec<PathBuf>> {
		let extensions = &["js", "ts", "jsx", "tsx"]; // minimal list here
		Ok(find_test_files(repo_path, extensions))
	}
}
