pub const ENTRY_POINT_NAMES: &[&str] = &[
    "main.rs",
    "lib.rs",
    "mod.rs",
    "index.ts",
    "index.js",
    "index.tsx",
    "index.jsx",
    "index.rs",
    "app.ts",
    "app.js",
    "app.tsx",
    "server.ts",
    "server.js",
    "server.rs",
    "routes.ts",
    "routes.js",
    "routes.rs",
    "main.py",
    "__init__.py",
    "main.go",
    "app.rb",
    "application.rb",
    "main.kt",
    "Application.kt",
    "AppDelegate.swift",
    "ContentView.swift",
    "Program.cs",
    "Startup.cs",
    "main.c",
    "main.cpp",
    "index.php",
];

pub const JUNK_DIRS: &[&str] = &[
    "dist",
    "build",
    "out",
    ".next",
    "__pycache__",
    ".turbo",
    ".cache",
    "coverage",
    ".nyc_output",
    "node_modules",
    "vendor",
    ".venv",
    "venv",
    "env",
    ".git",
    ".svn",
    ".hg",
    ".vscode",
    ".idea",
    "target",
    "obj",
    ".gradle",
    "migrations",
    "seeds",
    "fixtures",
    "__snapshots__",
    ".parcel-cache",
    "storybook-static",
];

pub const JUNK_FILE_ENDS: &[&str] = &[
    ".lock",
    ".sum",
    ".generated.ts",
    ".generated.js",
    ".pb.go",
    ".pb.ts",
    ".min.js",
    ".min.css",
    ".d.ts",
    ".snap",
    ".map",
    ".log",
];

pub const JUNK_EXACT_FILES: &[&str] = &[
    "Cargo.lock",
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    "poetry.lock",
    "go.sum",
    "Gemfile.lock",
    "composer.lock",
];

pub const TEST_DIRS: &[&str] = &[
    "tests",
    "test",
    "spec",
    "__tests__",
    "e2e",
    "integration",
];

pub const TEST_FILE_PATTERNS: &[&str] = &[
    "_test.go",
    "_spec.rb",
    ".test.ts",
    ".test.js",
    ".test.tsx",
    ".test.jsx",
    ".spec.ts",
    ".spec.js",
    ".spec.tsx",
    ".spec.jsx",
    ".e2e.ts",
    ".e2e.tsx",
    ".e2e.jsx",
    ".e2e.js",
    "test_.py",
];

pub const PRIORITY_SOURCE_DIRS: &[&str] = &[
    "src", "app", "lib", "api", "routes", "routers", "controllers", "services",
    "components", "pages", "pkg", "cmd", "internal", "crates", "server", "client",
    "frontend", "backend", "mcp", "ast", "lsp", "shared", "cli", "web",
];

pub const ALWAYS_EXPAND_DIRS: &[&str] = &[
    "src", "app", "lib", "api", "server", "client", "backend", "frontend",
];

pub const COLLAPSE_DIRS: &[&str] = &[
    "migrations", "seeds", "fixtures", "tests", "test", "spec", "__tests__", "e2e",
    "integration", "docs", "doc", "public", "assets", "static", ".github", ".ai",
    ".cursorrules", ".husky",
    "dist", "build", "out", ".next", "coverage", "node_modules", "vendor", "target",
    "__pycache__", "obj", ".turbo", ".cache", "storybook-static",
];

pub const PRIORITY_ROOT_FILES: &[&str] = &[
    "package.json", "Cargo.toml", "README.md", "AGENTS.md", "CLAUDE.md", "Dockerfile",
    "docker-compose.yml", "docker-compose.yaml", "compose.yml", "compose.yaml",
    "next.config.ts", "next.config.js", "tsconfig.json", "schema.prisma", "go.mod",
    "pyproject.toml", "requirements.txt", "Gemfile", "composer.json", "pom.xml",
    "build.gradle", "build.gradle.kts",
];

pub fn should_skip_dir(name: &str) -> bool {
    name.starts_with('.') || JUNK_DIRS.contains(&name) || TEST_DIRS.contains(&name)
}

pub fn is_junk_file(name: &str) -> bool {
    if JUNK_EXACT_FILES.contains(&name) {
        return true;
    }
    JUNK_FILE_ENDS.iter().any(|end| name.ends_with(end))
}

pub fn is_test_file(name: &str) -> bool {
    TEST_FILE_PATTERNS.iter().any(|p| name.contains(p))
}

pub fn test_file_patterns_regex() -> String {
    TEST_FILE_PATTERNS
        .iter()
        .map(|p| p.replace(".", "\\."))
        .collect::<Vec<_>>()
        .join("|")
}
