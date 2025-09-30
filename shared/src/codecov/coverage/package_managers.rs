use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PackageManager {
    Npm,
    Yarn,
    Pnpm,
    Pip,
    Cargo,
    Maven,
    Gradle,
}

impl PackageManager {
    pub fn detect(repo_path: &Path) -> Vec<PackageManager> {
        let mut managers = Vec::new();
        if repo_path.join("package.json").exists() {
            if repo_path.join("yarn.lock").exists() {
                managers.push(PackageManager::Yarn);
            }
            if repo_path.join("pnpm-lock.yaml").exists() {
                managers.push(PackageManager::Pnpm);
            }
            if repo_path.join("package-lock.json").exists() || managers.is_empty() {
                managers.push(PackageManager::Npm);
            }
        }
        if repo_path.join("Cargo.toml").exists() {
            managers.push(PackageManager::Cargo);
        }
        if repo_path.join("requirements.txt").exists() || repo_path.join("pyproject.toml").exists()
        {
            managers.push(PackageManager::Pip);
        }
        if repo_path.join("pom.xml").exists() {
            managers.push(PackageManager::Maven);
        }
        if repo_path.join("build.gradle").exists() || repo_path.join("build.gradle.kts").exists() {
            managers.push(PackageManager::Gradle);
        }
        managers
    }

    pub fn primary_for_repo(repo_path: &Path) -> Option<PackageManager> {
        Self::detect(repo_path).into_iter().next()
    }

    pub fn needs_install(&self, repo_path: &Path) -> bool {
        match self {
            PackageManager::Npm | PackageManager::Yarn | PackageManager::Pnpm => {
                !repo_path.join("node_modules").exists()
            }
            PackageManager::Cargo => !repo_path.join("target").exists(),
            _ => false,
        }
    }

    pub fn install_cmd(&self) -> (&str, Vec<String>) {
        match self {
            PackageManager::Npm => ("npm", vec!["install".into()]),
            PackageManager::Yarn => ("yarn", vec!["install".into()]),
            PackageManager::Pnpm => ("pnpm", vec!["install".into()]),
            PackageManager::Pip => (
                "pip",
                vec!["install".into(), "-r".into(), "requirements.txt".into()],
            ),
            PackageManager::Cargo => ("cargo", vec!["build".into()]),
            PackageManager::Maven => ("mvn", vec!["install".into()]),
            PackageManager::Gradle => ("gradle", vec!["build".into()]),
        }
    }

    pub fn run_script_cmd(&self, script_name: &str) -> (&str, Vec<String>) {
        match self {
            PackageManager::Npm => ("npm", vec!["run".into(), script_name.into()]),
            PackageManager::Yarn => ("yarn", vec!["run".into(), script_name.into()]),
            PackageManager::Pnpm => ("pnpm", vec!["run".into(), script_name.into()]),
            PackageManager::Pip => ("python", vec!["-m".into(), script_name.into()]),
            PackageManager::Cargo => ("cargo", vec![script_name.into()]),
            PackageManager::Maven => ("mvn", vec![script_name.into()]),
            PackageManager::Gradle => ("gradle", vec![script_name.into()]),
        }
    }
}
