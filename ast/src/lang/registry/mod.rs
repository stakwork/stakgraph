pub mod typescript;

use crate::lang::graphs::Graph;
use crate::lang::Lang;
use lsp::Language;

pub trait Registry: Send + Sync {
    fn resolve_type(&self, file: &str, var_name: &str) -> Option<&str>;
    fn resolve_method(&self, type_name: &str, method_name: &str) -> Option<&str>;
}

pub fn build(lang: &Lang, graph: &impl Graph) -> Option<Box<dyn Registry>> {
    match lang.kind {
        Language::Typescript => Some(Box::new(typescript::TypeScriptRegistry::new(graph))),
        _ => None,
    }
}
