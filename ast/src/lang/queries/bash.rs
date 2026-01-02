use tree_sitter::Language;

#[allow(dead_code)]
pub struct Bash(Language);

impl Default for Bash {
    fn default() -> Self {
        Self::new()
    }
}

impl Bash {
    pub fn new() -> Self {
        Bash(tree_sitter_bash::LANGUAGE.into())
    }
}
