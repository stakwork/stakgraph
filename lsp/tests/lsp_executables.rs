use lsp::utils::get_lsp_version;
use lsp::Language;

#[tokio::test]
async fn verify_kotlin_and_ruby_lsp() {
    // Only run when USE_LSP env var is true to match CI behavior
    let use_lsp = std::env::var("USE_LSP").ok().map(|v| v == "true").unwrap_or(false);
    if !use_lsp {
        eprintln!("USE_LSP not set; skipping kotlin and ruby lsp executable checks");
        return;
    }

    // Verify kotlin-language-server
    match get_lsp_version(Language::Kotlin).await {
        Ok(v) => {
            assert!(!v.trim().is_empty(), "kotlin-language-server returned empty version output");
            eprintln!("kotlin-language-server: {}", v.trim());
        }
        Err(e) => panic!("Failed to run kotlin-language-server: {:?}", e),
    }

    // Verify ruby-lsp
    match get_lsp_version(Language::Ruby).await {
        Ok(v) => {
            assert!(!v.trim().is_empty(), "ruby-lsp returned empty version output");
            eprintln!("ruby-lsp: {}", v.trim());
        }
        Err(e) => panic!("Failed to run ruby-lsp: {:?}", e),
    }
}
