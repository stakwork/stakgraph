use crate::coverage::types::{CoverageStrategy, TestFramework, PackageManager, ProjectConfig};
use crate::Result;
use std::path::Path;


pub struct TypeScriptRunner;

impl TypeScriptRunner {
    pub fn new() -> Self {
        Self
    }

    pub fn build_strategy(&self, repo_path: &Path, config: &ProjectConfig) -> Result<CoverageStrategy> {
        // Pick the first available framework (could be smarter about this) //TODO: optimize
        let framework = config.test_frameworks.first()
            .ok_or_else(|| crate::Error::Custom("No test frameworks found".to_string()))?;

        let (command, args) = match framework {
            TestFramework::Vitest => self.vitest_coverage_command(&config.package_manager),
            TestFramework::Jest => self.jest_coverage_command(&config.package_manager),
            TestFramework::Playwright => self.playwright_coverage_command(&config.package_manager),
            TestFramework::Cypress => self.cypress_coverage_command(&config.package_manager),
            TestFramework::Mocha => self.c8_coverage_command(&config.package_manager),
            TestFramework::Jasmine => self.c8_coverage_command(&config.package_manager),
            _ => return Err(crate::Error::Custom(format!(
                "Unsupported framework for TypeScript: {:?}", framework
            ))),
        };

        Ok(CoverageStrategy {
            framework: framework.clone(),
            command,
            args,
            working_dir: repo_path.to_path_buf(),
            output_dir: repo_path.join("coverage"),
        })
    }

     fn vitest_coverage_command(&self, pm: &Option<PackageManager>) -> (String, Vec<String>) {
        let (cmd, base_args) = self.get_package_manager_command(pm);
        let mut args = base_args;
        args.extend_from_slice(&[
            "vitest".to_string(),
            "--coverage".to_string(),
            "--coverage.reporter=json".to_string(),
            "--coverage.reporter=json-summary".to_string(),
        ]);
        (cmd, args)
    }

        fn jest_coverage_command(&self, pm: &Option<PackageManager>) -> (String, Vec<String>) {
        let (cmd, base_args) = self.get_package_manager_command(pm);
        let mut args = base_args;
        args.extend_from_slice(&[
            "jest".to_string(),
            "--coverage".to_string(),
            "--coverageReporters=json".to_string(),
            "--coverageReporters=json-summary".to_string(),
        ]);
        (cmd, args)
    }

    fn playwright_coverage_command(&self, pm: &Option<PackageManager>) -> (String, Vec<String>) {
        let (cmd, base_args) = self.get_package_manager_command(pm);
        let mut args = base_args;
        args.extend_from_slice(&[
            "playwright".to_string(),
            "test".to_string(),
            "--reporter=html".to_string(),
        ]);
        (cmd, args)
    }

    fn cypress_coverage_command(&self, pm: &Option<PackageManager>) -> (String, Vec<String>) {
        let (cmd, base_args) = self.get_package_manager_command(pm);
        let mut args = base_args;
        args.extend_from_slice(&[
            "cypress".to_string(),
            "run".to_string(),
            // NOTE: Cypress coverage usually requires @cypress/code-coverage plugin
        ]);
        (cmd, args)
    }

    fn c8_coverage_command(&self, pm: &Option<PackageManager>) -> (String, Vec<String>) {
        let (cmd, base_args) = self.get_package_manager_command(pm);
        let mut args = base_args;
        args.extend_from_slice(&[
            "c8".to_string(),
            "--reporter=json".to_string(),
            "--reporter=json-summary".to_string(),
            "--reports-dir=./coverage".to_string(),
            "mocha".to_string(),
        ]);
        (cmd, args)
    }

    // ... other framework command builders ...

        fn get_package_manager_command(&self, pm: &Option<PackageManager>) -> (String, Vec<String>) {
        match pm {
            Some(PackageManager::Yarn) => ("yarn".to_string(), vec![]),
            Some(PackageManager::Pnpm) => ("pnpm".to_string(), vec![]),
            Some(PackageManager::Npm) | None => ("npx".to_string(), vec![]),
            _ => ("npx".to_string(), vec![]),
        }
    }

}