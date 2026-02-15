use serde::{Deserialize, Serialize};
use shared::error::{Error, Result};
use std::{fmt::Display, str::FromStr};

#[derive(Serialize, Deserialize, Clone, Debug, Eq, PartialEq, Hash)]
pub enum Language {
    Bash,
    Toml,
    Rust,
    Go,
    Typescript,
    Python,
    Ruby,
    Kotlin,
    Swift,
    Java,
    Svelte,
    Angular,
    C,
    Cpp,
    Php,
    CSharp,
}

pub const PROGRAMMING_LANGUAGES: [Language; 14] = [
    Language::Rust,
    Language::Go,
    Language::Typescript,
    Language::Python,
    Language::Ruby,
    Language::Kotlin,
    Language::Swift,
    Language::Java,
    Language::Svelte,
    Language::Angular,
    Language::C,
    Language::Cpp,
    Language::Php,
    Language::CSharp,
];

impl Language {
    pub fn is_frontend(&self) -> bool {
        matches!(self, Self::Typescript | Self::Kotlin | Self::Swift)
    }
    pub fn pkg_files(&self) -> Vec<&'static str> {
        match self {
            Self::Rust => vec!["Cargo.toml"],
            Self::Go => vec!["go.mod", "go.work"],
            Self::Typescript => vec!["package.json"],
            Self::Python => vec!["requirements.txt", "pyproject.toml"],
            Self::Ruby => vec!["Gemfile"],
            Self::Kotlin => vec![
                "build.gradle.kts",
                "build.gradle",
                "settings.gradle.kts",
                ".properties",
            ],
            Self::Swift => vec!["Package.swift", "Podfile", "Cartfile"],
            Self::Java => vec!["pom.xml", "build.gradle", "build.gradle.kts"],
            Self::Bash => vec![],
            Self::Toml => vec![],
            Self::Svelte => vec!["package.json"],
            Self::Angular => vec!["package.json"],
            Self::C => vec!["CMakeLists.txt", "Makefile", "meson.build"],
            Self::Cpp => vec!["CMakeLists.txt", "Makefile", "meson.build"],
            Self::Php => vec!["composer.json"],
            Self::CSharp => vec![".csproj", ".sln"],
        }
    }

    pub fn exts(&self) -> Vec<&'static str> {
        match self {
            Self::Rust => vec!["rs"],
            Self::Go => vec!["go"],
            Self::Python => vec!["py", "ipynb"],
            Self::Ruby => vec!["rb"],
            Self::Kotlin => vec!["kt", "kts", "java"],
            Self::Swift => vec!["swift", "plist"],
            Self::Java => vec!["java", "gradle", "gradlew"],
            Self::Bash => vec!["sh"],
            Self::Toml => vec!["toml"],
            Self::Typescript => vec!["ts", "js", "jsx", "tsx", "mdx", "html", "css"],
            Self::Svelte => vec!["svelte", "ts", "js", "html", "css"],
            Self::Angular => vec!["ts", "js", "html", "css"],
            Self::C => vec!["c", "h"],
            Self::Cpp => vec!["cpp", "hpp", "cc", "cxx", "hxx", "h"],
            Self::Php => vec!["php"],
            Self::CSharp => vec!["cs"],
        }
    }

    pub fn overrides(&self) -> Vec<Language> {
        match self {
            Self::Svelte => vec![Self::Typescript],
            Self::Angular => vec![Self::Typescript],
            _ => Vec::new(),
        }
    }

    // Used to distinguish between similar languages (e.g., Angular vs plain TypeScript).
    pub fn required_indicator_files(&self) -> Vec<&'static str> {
        match self {
            Self::Svelte => vec!["svelte.config.js", "svelte.config.ts"],
            Self::Angular => vec!["angular.json"],
            _ => vec![],
        }
    }

    pub fn skip_dirs(&self) -> Vec<&'static str> {
        match self {
            Self::Rust => vec!["target", ".git"],
            Self::Go => vec!["vendor", ".git"],
            Self::Typescript => vec!["node_modules", ".git"],
            Self::Python => vec!["__pycache__", ".git", ".venv", "venv"],
            Self::Ruby => vec!["migrate", "tmp", ".git"],
            Self::Kotlin => vec!["build", ".git"],
            Self::Swift => vec![".git", "Pods"],
            Self::Java => vec![".idea", "build", ".git"],
            Self::Bash => vec![".git"],
            Self::Toml => vec![".git"],
            Self::Svelte => vec![".git", "node_modules"],
            Self::Angular => vec![".git", "node_modules"],
            Self::C => vec![".git", "build", "out", "CMakeFiles", ".cmake"],
            Self::Cpp => vec![".git", "build", "out", "CMakeFiles"],
            Self::Php => vec![".git", "vendor"],
            Self::CSharp => vec![".git", "bin", "obj", "packages", ".vs"],
        }
    }

    pub fn skip_file_ends(&self) -> Vec<&'static str> {
        match self {
            Self::Typescript => vec![".min.js"],
            Self::Svelte => vec![".config.ts", ".config.ts"],
            Self::Angular => vec!["spec.ts"],
            Self::Kotlin => vec!["gradlew"],
            _ => Vec::new(),
        }
    }

    pub fn only_include_files(&self) -> Vec<&'static str> {
        match self {
            Self::Rust => Vec::new(),
            Self::Go => Vec::new(),
            Self::Typescript => Vec::new(),
            Self::Python => Vec::new(),
            Self::Ruby => Vec::new(),
            Self::Kotlin => Vec::new(),
            Self::Swift => Vec::new(),
            Self::Java => Vec::new(),
            Self::Bash => Vec::new(),
            Self::Toml => Vec::new(),
            Self::Svelte => Vec::new(),
            Self::Angular => Vec::new(),
            Self::C => Vec::new(),
            Self::Cpp => Vec::new(),
            Self::Php => Vec::new(),
            Self::CSharp => Vec::new(),
        }
    }

    pub fn default_do_lsp(&self) -> bool {
        if let Ok(use_lsp) = std::env::var("USE_LSP") {
            if use_lsp == "true" || use_lsp == "1" {
                matches!(self, Self::Rust | Self::Go | Self::Typescript | Self::Java);
            }
        }
        false
    }

    pub fn has_lsp_support(&self) -> bool {
        !self.lsp_exec().is_empty()
    }

    pub fn lsp_exec(&self) -> String {
        match self {
            Self::Rust => "rust-analyzer",
            Self::Go => "gopls",
            Self::Typescript => "typescript-language-server",
            Self::Python => "pylsp",
            Self::Ruby => "ruby-lsp",
            Self::Kotlin => "kotlin-language-server",
            Self::Swift => "",
            Self::Java => "",
            Self::Bash => "",
            Self::Toml => "",
            Self::Svelte => "",
            Self::Angular => "",
            Self::C => "",
            Self::Cpp => "",
            Self::Php => "",
            Self::CSharp => "",
        }
        .to_string()
    }

    pub fn version_arg(&self) -> String {
        match self {
            Self::Rust => "--version",
            Self::Go => "version",
            Self::Typescript => "--version",
            Self::Python => "--version",
            Self::Ruby => "--version",
            Self::Kotlin => "--version",
            Self::Swift => "--version",
            Self::Java => "--version",
            Self::Bash => "",
            Self::Toml => "",
            Self::Svelte => "--version",
            Self::Angular => "--version",
            Self::C => "--version",
            Self::Cpp => "--version",
            Self::Php => "--version",
            Self::CSharp => "--version",
        }
        .to_string()
    }

    pub fn lsp_args(&self) -> Vec<String> {
        match self {
            Self::Rust => Vec::new(),
            Self::Go => Vec::new(),
            Self::Typescript => vec!["--stdio".to_string()],
            Self::Python => Vec::new(),
            Self::Ruby => Vec::new(),
            Self::Kotlin => Vec::new(),
            Self::Swift => Vec::new(),
            Self::Java => Vec::new(),
            Self::Bash => Vec::new(),
            Self::Toml => Vec::new(),
            Self::Svelte => Vec::new(),
            Self::Angular => Vec::new(),
            Self::C => Vec::new(),
            Self::Cpp => Vec::new(),
            Self::Php => Vec::new(),
            Self::CSharp => Vec::new(),
        }
    }

    pub fn post_clone_cmd(&self, use_lsp: bool) -> Vec<&'static str> {
        if let Ok(lsp_skip) = std::env::var("LSP_SKIP_POST_CLONE") {
            if lsp_skip == "true" || lsp_skip == "1" {
                return Vec::new();
            }
        }
        if let Ok(repo_path) = std::env::var("REPO_PATH") {
            if !repo_path.is_empty() {
                tracing::info!("skipping post clone cmd for local repo. If its a js/ts repo, run npm install first!");
                return Vec::new();
            }
        }
        if !use_lsp {
            return Vec::new();
        }
        if let Ok(use_lsp_env) = std::env::var("USE_LSP") {
            if use_lsp_env == "false" || use_lsp_env == "0" {
                return Vec::new();
            }
        }
        match self {
            Self::Rust => Vec::new(),
            Self::Go => Vec::new(),
            Self::Typescript => vec!["npm install --force"],
            Self::Python => Vec::new(),
            Self::Ruby => Vec::new(),
            Self::Kotlin => Vec::new(),
            Self::Swift => Vec::new(),
            Self::Java => Vec::new(),
            Self::Bash => Vec::new(),
            Self::Toml => Vec::new(),
            Self::Svelte => Vec::new(),
            Self::Angular => Vec::new(),
            Self::Cpp => Vec::new(),
            Self::Php => Vec::new(),
            Self::C => Vec::new(),
            Self::CSharp => Vec::new(),
        }
    }

    pub fn test_id_regex(&self) -> Option<&'static str> {
        match self {
            Self::Typescript => Some(r#"data-testid=(?:["']([^"']+)["']|\{['"`]([^'"`]+)['"`]\})"#),
            Self::Python => Some("get_by_test_id"),
            Self::Ruby => Some(r#"get_by_test_id\(['"]([^'"]+)['"]\)"#),
            _ => None,
        }
    }
    pub fn is_package_file(&self, file_name: &str) -> bool {
        self.pkg_files()
            .iter()
            .any(|pkg_file| file_name.ends_with(pkg_file))
    }
    pub fn is_from_language(&self, file: &str) -> bool {
        if !file.contains('.') {
            return true;
        }
        if let Some(ext) = file.split('.').next_back() {
            self.exts().contains(&ext) || self.is_package_file(file)
        } else {
            true //dirs, lang, repo
        }
    }
    pub fn is_source_file(&self, file_name: &str) -> bool {
        if self.is_package_file(file_name) {
            return true;
        }

        if let Some(ext) = file_name.split('.').next_back() {
            self.exts().contains(&ext)
        } else {
            false
        }
    }
    pub fn from_path(path: &str) -> Option<Self> {
        let ext = std::path::Path::new(path)
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_lowercase();
        for lang in PROGRAMMING_LANGUAGES.iter() {
            if lang.exts().iter().any(|e| e.eq_ignore_ascii_case(&ext)) {
                return Some(lang.clone());
            }
        }
        None
    }
}

impl Display for Language {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Self::Rust => "rust",
            Self::Go => "go",
            Self::Typescript => "typescript",
            Self::Python => "python",
            Self::Ruby => "ruby",
            Self::Kotlin => "kotlin",
            Self::Swift => "swift",
            Self::Java => "java",
            Self::Bash => "bash",
            Self::Toml => "toml",
            Self::Svelte => "svelte",
            Self::Angular => "angular",
            Self::C => "c",
            Self::Cpp => "cpp",
            Self::Php => "php",
            Self::CSharp => "csharp",
        };
        write!(f, "{}", s)
    }
}

impl FromStr for Language {
    type Err = shared::Error;
    fn from_str(s: &str) -> Result<Self> {
        match s {
            "python" => Ok(Language::Python),
            "Python" => Ok(Language::Python),
            "go" => Ok(Language::Go),
            "Go" => Ok(Language::Go),
            "golang" => Ok(Language::Go),
            "Golang" => Ok(Language::Go),
            "react" => Ok(Language::Typescript),
            "React" => Ok(Language::Typescript),
            "tsx" => Ok(Language::Typescript),
            "jsx" => Ok(Language::Typescript),
            "ts" => Ok(Language::Typescript),
            "js" => Ok(Language::Typescript),
            "typescript" => Ok(Language::Typescript),
            "TypeScript" => Ok(Language::Typescript),
            "javascript" => Ok(Language::Typescript),
            "JavaScript" => Ok(Language::Typescript),
            "ruby" => Ok(Language::Ruby),
            "Ruby" => Ok(Language::Ruby),
            "RubyOnRails" => Ok(Language::Ruby),
            "rust" => Ok(Language::Rust),
            "Rust" => Ok(Language::Rust),
            "bash" => Ok(Language::Bash),
            "Bash" => Ok(Language::Bash),
            "toml" => Ok(Language::Toml),
            "Toml" => Ok(Language::Toml),
            "kotlin" => Ok(Language::Kotlin),
            "Kotlin" => Ok(Language::Kotlin),
            "swift" => Ok(Language::Swift),
            "Swift" => Ok(Language::Swift),
            "java" => Ok(Language::Java),
            "Java" => Ok(Language::Java),
            "svelte" => Ok(Language::Svelte),
            "Svelte" => Ok(Language::Svelte),
            "angular" => Ok(Language::Angular),
            "Angular" => Ok(Language::Angular),
            "c" => Ok(Language::C),
            "C" => Ok(Language::C),
            "cpp" => Ok(Language::Cpp),
            "Cpp" => Ok(Language::Cpp),
            "c++" => Ok(Language::Cpp),
            "C++" => Ok(Language::Cpp),
            "php" => Ok(Language::Php),
            "Php" => Ok(Language::Php),
            "PHP" => Ok(Language::Php),
            "csharp" => Ok(Language::CSharp),
            "CSharp" => Ok(Language::CSharp),
            "c#" => Ok(Language::CSharp),
            "C#" => Ok(Language::CSharp),

            _ => Err(Error::Custom("unsupported language".to_string())),
        }
    }
}

pub fn common_binary_exts() -> Vec<&'static str> {
    vec![
        "png", "jpg", "jpeg", "gif", "bmp", "svg", "ico", "tif", "tiff", "webp", "mp4", "mov",
        "avi", "mkv", "webm", "mp3", "wav", "ogg", "flac", "ttf", "otf", "woff", "woff2", "zip",
        "rar", "7z", "tar", "gz", "deb", "pkg", "dmg", "pdf", "doc", "docx", "xls", "xlsx", "ppt",
        "pptx", "exe", "dll", "so", "a", "o", "jar", "class", "pyc",
    ]
}
pub fn junk_directories() -> Vec<&'static str> {
    vec![
        ".git",
        ".vscode",
        "target",
        "build",
        "dist",
        "venv",
        "node_modules",
    ]
}
