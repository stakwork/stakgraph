use std::collections::HashMap;
use crate::codecov::coverage::execution::CommandRunner;
use crate::{Error, Result};
use crate::codecov::coverage::package_managers::PackageManager;


#[derive(Debug, Clone)]
pub struct RunnerSpec {
    pub id: &'static str,                 // e.g. "jest"
    pub detect_tokens: &'static [&'static str], // script substrings to detect runner
    pub coverage_dependency: Option<&'static str>, // optional extra dep for coverage
    pub coverage_style: CoverageStyle,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CoverageStyle {
    // Native flag patterns
    Vitest,   // uses --coverage --coverage.reporter
    Jest,     // uses --coverage --coverageReporters
    Jasmine,  // needs c8 wrapper unless script already has coverage
    Mocha,    // can be wrapped with c8
    Generic,  // just append --coverage if supported
}

#[derive(Debug, Clone)]
pub struct TestScript {
    pub name: String,
    pub runner_id: String,
    pub has_coverage: bool,
}

// Static registry of known runners. Extend here to support more.
fn registry() -> &'static [RunnerSpec] {
    &[
        RunnerSpec {
            id: "vitest",
            detect_tokens: &["vitest"],
            coverage_dependency: Some("@vitest/coverage-v8"),
            coverage_style: CoverageStyle::Vitest,
        },
        RunnerSpec {
            id: "jest",
            detect_tokens: &["jest"],
            coverage_dependency: None,
            coverage_style: CoverageStyle::Jest,
        },
        RunnerSpec {
            id: "jasmine",
            detect_tokens: &["jasmine"],
            coverage_dependency: None,
            coverage_style: CoverageStyle::Jasmine,
        },
        RunnerSpec {
            id: "mocha",
            detect_tokens: &["mocha"],
            coverage_dependency: None,
            coverage_style: CoverageStyle::Mocha,
        },
    ]
}

pub fn get_runner(id: &str) -> Option<&'static RunnerSpec> {
    registry().iter().find(|r| r.id == id)
}

pub fn detect_from_package_json(scripts: &HashMap<String, String>) -> Vec<TestScript> {
    let patterns = [
        "test:coverage",
        "test",
        "test:unit",
        "coverage",
        "test:integration",
    ];
    let mut out = Vec::new();
    for script_name in patterns {
        if let Some(script_content) = scripts.get(script_name) {
            let script_lower = script_content.to_lowercase();
            if let Some(spec) = registry().iter().find(|spec| spec
                .detect_tokens
                .iter()
                .any(|tok| script_lower.contains(&tok.to_lowercase())))
            {
                let has_cov = script_lower.contains("coverage") || script_name.contains("coverage");
                out.push(TestScript {
                    name: script_name.to_string(),
                    runner_id: spec.id.to_string(),
                    has_coverage: has_cov,
                });
                continue;
            }
            out.push(TestScript {
                name: script_name.to_string(),
                runner_id: "jest".to_string(), // sane default
                has_coverage: script_lower.contains("coverage") || script_name.contains("coverage"),
            });
        }
    }
    out
}
pub fn detect_runners(_repo_path: &std::path::Path, scripts: &HashMap<String,String>, deps: &[String]) -> Vec<String> {
    let mut runners: Vec<String> = Vec::new();
    // From scripts
    for ts in detect_from_package_json(scripts) { if !runners.contains(&ts.runner_id) { runners.push(ts.runner_id); } }
    // From dependencies
    let deps_lower: Vec<String> = deps.iter().map(|d| d.to_lowercase()).collect();
    for spec in registry() {
        if deps_lower.iter().any(|d| spec.detect_tokens.iter().any(|tok| d.contains(&tok.to_lowercase()))) {
            if !runners.iter().any(|r| r == spec.id) { runners.push(spec.id.to_string()); }
        }
    }
    runners
}

pub fn execute_with_fallback(
    repo_path: &std::path::Path,
    script: &TestScript,
    max_attempts: usize,
) -> Result<()> {
    let package_manager = PackageManager::primary_for_repo(repo_path).unwrap_or(PackageManager::Npm);
    let mut attempt = 0usize;
    let runner_spec = get_runner(&script.runner_id);
    let reports_dir = "./coverage";
    loop {
        attempt += 1;
        if let Some(spec) = runner_spec {
            if script.has_coverage {
                if let Some(mut args) = coverage_reporter_args(&script.runner_id, reports_dir) {
                    args.insert(0, spec.id.to_string());
                    let str_args: Vec<&str> = args.iter().map(|s| s.as_str()).collect();
                    if CommandRunner::run(repo_path, "npx", &str_args).is_ok() { return Ok(()); }
                }
            }
        }
        let (script_cmd, script_args) = package_manager.run_script_cmd(&script.name);
        let mut args = vec![
            "c8".to_string(),
            "--reporter=json-summary".to_string(),
            "--reporter=json".to_string(),
            "--reports-dir=./coverage".to_string(),
            script_cmd.to_string(),
        ];
        args.extend(script_args);
        if CommandRunner::run(repo_path, "npx", &args.iter().map(|s| s.as_str()).collect::<Vec<&str>>()).is_ok() {
            return Ok(());
        }
        if let Some(spec) = runner_spec {
            let args = vec![spec.id.to_string(), "--coverage".to_string()];
            if CommandRunner::run(repo_path, "npx", &args.iter().map(|s| s.as_str()).collect::<Vec<&str>>()).is_ok() { return Ok(()); }
        }

        if attempt >= max_attempts { return Err(Error::Custom("all coverage attempts failed".into())); }
    }
}

pub fn coverage_dependency(runner_id: &str) -> Option<&'static str> {
    get_runner(runner_id).and_then(|r| r.coverage_dependency)
}

pub fn coverage_reporter_args(runner_id: &str, reports_dir: &str) -> Option<Vec<String>> {
    let spec = get_runner(runner_id)?;
    let args = match spec.coverage_style {
        CoverageStyle::Vitest => vec![
            "--coverage".into(),
            "--coverage.reporter=json-summary".into(),
            "--coverage.reporter=json".into(),
            format!("--coverage.reportsDirectory={}", reports_dir),
        ],
        CoverageStyle::Jest => vec![
            "--coverage".into(),
            "--coverageReporters=json-summary".into(),
            "--coverageReporters=json".into(),
            format!("--coverageDirectory={}", reports_dir),
        ],
        CoverageStyle::Jasmine | CoverageStyle::Mocha => {
            // These typically rely on c8 wrapper; return None so caller can wrap.
            return None;
        }
        CoverageStyle::Generic => vec!["--coverage".into()],
    };
    Some(args)
}
