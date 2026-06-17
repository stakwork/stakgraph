pub mod ts_resolver;
pub mod typescript;

use crate::lang::asg::NodeKeys;
use crate::lang::graphs::Graph;
use crate::lang::Lang;
use lsp::Language;

pub trait Registry: Send + Sync {
    fn resolve_type(&self, file: &str, var_name: &str) -> Option<&str>;
    fn resolve_method(&self, type_name: &str, method_name: &str) -> Option<&str>;
    fn resolve_field(&self, _type_name: &str, _field_name: &str) -> Option<&str> {
        None
    }
    fn resolve_call_at(&self, _file: &str, _row: usize, _col: usize) -> Option<NodeKeys> {
        None
    }
}

pub fn build(
    lang: &Lang,
    graph: &impl Graph,
    filez: &[(String, String)],
) -> Option<Box<dyn Registry>> {
    match lang.kind {
        Language::Typescript => Some(Box::new(typescript::TypeScriptRegistry::new(graph, filez))),
        _ => None,
    }
}
