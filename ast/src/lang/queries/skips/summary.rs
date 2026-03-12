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
    "test_.py",
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
