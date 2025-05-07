use serde::{Deserialize, Serialize};
use std::str::FromStr;

#[derive(Serialize, Deserialize, Clone, Debug, Eq, PartialEq)]
pub enum Language {
    Bash,
    Toml,
    Rust,
    React,
    Go,
    Typescript,
    Python,
    Ruby,
    Kotlin,
    Swift,
    Java,
    Svelte,
    Angular,
}

pub const PROGRAMMING_LANGUAGES: [Language; 11] = [
    Language::Rust,
    Language::Go,
    Language::Typescript,
    Language::React,
    Language::Python,
    Language::Ruby,
    Language::Kotlin,
    Language::Swift,
    Language::Java,
    Language::Svelte,
    Language::Angular,
];

impl Language {
    pub fn is_frontend(&self) -> bool {
        matches!(
            self,
            Self::Typescript | Self::React | Self::Kotlin | Self::Swift
        )
    }
    pub fn pkg_file(&self) -> &'static str {
        match self {
            Self::Rust => "Cargo.toml",
            Self::Go => "go.mod",
            Self::Typescript | Self::React => "package.json",
            Self::Python => "requirements.txt",
            Self::Ruby => "Gemfile",
            Self::Kotlin => "build.gradle.kts",
            Self::Swift => "Podfile",
            Self::Java => "pom.xml",
            Self::Bash => "",
            Self::Toml => "",
            Self::Svelte => "package.json",
            Self::Angular => "package.json",
        }
    }

    pub fn exts(&self) -> Vec<&'static str> {
        match self {
            Self::Rust => vec!["rs"],
            Self::Go => vec!["go"],
            Self::Python => vec!["py", "ipynb"],
            Self::Ruby => vec!["rb"],
            Self::Kotlin => vec!["kt", "kts"],
            Self::Swift => vec!["swift", "xcodeproj", "xcworkspace"],
            Self::Java => vec!["java", "gradle", "gradlew"],
            Self::Bash => vec!["sh"],
            Self::Toml => vec!["toml"],
            // how to separate ts and js?
            Self::Typescript => vec!["ts", "js"],
            Self::React => vec!["jsx", "tsx", "ts", "js"],
            Self::Svelte => vec!["svelte", "ts", "js"],
            Self::Angular => vec!["ts", "js"],
        }
    }

    // React overrides Typescript if detected
    pub fn overrides(&self) -> Vec<Language> {
        match self {
            Self::React => vec![Self::Typescript, Self::Svelte, Self::Angular],
            Self::Svelte => vec![Self::Typescript],
            Self::Angular => vec![Self::Typescript],
            _ => Vec::new(),
        }
    }

    pub fn skip_dirs(&self) -> Vec<&'static str> {
        match self {
            Self::Rust => vec!["target", ".git"],
            Self::Go => vec!["vendor", ".git"],
            Self::Typescript | Self::React => vec!["node_modules", ".git"],
            Self::Python => vec!["__pycache__", ".git", ".venv", "venv"],
            Self::Ruby => vec!["migrate", "tmp", ".git"],
            Self::Kotlin => vec![".gradle", ".idea", "build", ".git"],
            Self::Swift => vec![".git", "Pods"],
            Self::Java => vec![".gradle", ".idea", "build", ".git"],
            Self::Bash => vec![".git"],
            Self::Toml => vec![".git"],
            Self::Svelte => vec![".git", " node_modules"],
            Self::Angular => vec![".git", " node_modules"],
        }
    }

    pub fn skip_file_ends(&self) -> Vec<&'static str> {
        match self {
            Self::Typescript | Self::React => vec![".min.js"],
            Self::Svelte => vec![".config.ts", ".config.ts"],
            Self::Angular => vec!["spec.ts"],
            _ => Vec::new(),
        }
    }

    pub fn only_include_files(&self) -> Vec<&'static str> {
        match self {
            Self::Rust => Vec::new(),
            Self::Go => Vec::new(),
            Self::Typescript | Self::React => Vec::new(),
            Self::Python => Vec::new(),
            Self::Ruby => Vec::new(),
            Self::Kotlin => Vec::new(),
            Self::Swift => Vec::new(),
            Self::Java => Vec::new(),
            Self::Bash => Vec::new(),
            Self::Toml => Vec::new(),
            Self::Svelte => Vec::new(),
            Self::Angular => Vec::new(),
        }
    }

    pub fn default_do_lsp(&self) -> bool {
        if let Ok(use_lsp) = std::env::var("USE_LSP") {
            if use_lsp == "false" || use_lsp == "0" {
                return false;
            }
        }
        match self {
            Self::Rust => true,
            Self::Go => true,
            Self::Typescript | Self::React => true,
            Self::Python => false,
            Self::Ruby => false,
            Self::Kotlin => true,
            Self::Swift => false,
            Self::Java => true,
            Self::Bash => false,
            Self::Toml => false,
            Self::Svelte => false,
            Self::Angular => false,
        }
    }

    pub fn lsp_exec(&self) -> String {
        match self {
            Self::Rust => "rust-analyzer",
            Self::Go => "gopls",
            Self::Typescript | Self::React => "typescript-language-server",
            Self::Python => "pylsp",
            Self::Ruby => "ruby-lsp",
            Self::Kotlin => "kotlin-language-server",
            Self::Swift => "sourcekit-lsp",
            Self::Java => "jdtls",
            Self::Bash => "",
            Self::Toml => "",
            Self::Svelte => "svelte-language-server",
            Self::Angular => "angular-language-server",
        }
        .to_string()
    }

    pub fn version_arg(&self) -> String {
        match self {
            Self::Rust => "--version",
            Self::Go => "version",
            Self::Typescript | Self::React => "--version",
            Self::Python => "--version",
            Self::Ruby => "--version",
            Self::Kotlin => "--version",
            Self::Swift => "--version",
            Self::Java => "--version",
            Self::Bash => "",
            Self::Toml => "",
            Self::Svelte => "--version",
            Self::Angular => "--version",
        }
        .to_string()
    }

    pub fn lsp_args(&self) -> Vec<String> {
        match self {
            Self::Rust => Vec::new(),
            Self::Go => Vec::new(),
            Self::Typescript | Self::React => vec!["--stdio".to_string()],
            Self::Python => Vec::new(),
            Self::Ruby => Vec::new(),
            Self::Kotlin => Vec::new(),
            Self::Swift => Vec::new(),
            Self::Java => Vec::new(),
            Self::Bash => Vec::new(),
            Self::Toml => Vec::new(),
            Self::Svelte => Vec::new(),
            Self::Angular => Vec::new(),
        }
    }

    pub fn to_string(&self) -> String {
        match self {
            Self::Rust => "rust",
            Self::Go => "go",
            Self::Typescript => "typescript",
            Self::React => "react",
            Self::Python => "python",
            Self::Ruby => "ruby",
            Self::Kotlin => "kotlin",
            Self::Swift => "swift",
            Self::Java => "java",
            Self::Bash => "bash",
            Self::Toml => "toml",
            Self::Svelte => "svelte",
            Self::Angular => "angular",
        }
        .to_string()
    }

    pub fn post_clone_cmd(&self) -> Vec<&'static str> {
        if let Ok(lsp_skip) = std::env::var("LSP_SKIP_POST_CLONE") {
            if lsp_skip == "true" || lsp_skip == "1" {
                return Vec::new();
            }
        }
        match self {
            Self::Rust => Vec::new(),
            Self::Go => Vec::new(),
            Self::Typescript | Self::React => vec!["npm install --force"],
            Self::Python => Vec::new(),
            Self::Ruby => Vec::new(),
            Self::Kotlin => Vec::new(),
            Self::Swift => Vec::new(),
            Self::Java => Vec::new(),
            Self::Bash => Vec::new(),
            Self::Toml => Vec::new(),
            Self::Svelte => Vec::new(),
            Self::Angular => Vec::new(),
        }
    }

    pub fn test_id_regex(&self) -> Option<&'static str> {
        match self {
            Self::Typescript | Self::React => {
                Some(r#"data-testid=(?:["']([^"']+)["']|\{['"`]([^'"`]+)['"`]\})"#)
            }
            Self::Python => Some("get_by_test_id"),
            Self::Ruby => Some(r#"get_by_test_id\(['"]([^'"]+)['"]\)"#),
            _ => None,
        }
    }
}

impl FromStr for Language {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "python" => Ok(Language::Python),
            "Python" => Ok(Language::Python),
            "go" => Ok(Language::Go),
            "Go" => Ok(Language::Go),
            "golang" => Ok(Language::Go),
            "Golang" => Ok(Language::Go),
            "react" => Ok(Language::React),
            "React" => Ok(Language::React),
            "tsx" => Ok(Language::React),
            "jsx" => Ok(Language::React),
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
            _ => Err(anyhow::anyhow!("unsupported language")),
        }
    }
}
