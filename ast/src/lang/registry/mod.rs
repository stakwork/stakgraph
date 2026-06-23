pub mod cs_resolver;
pub mod csharp_registry;
pub mod scope;
pub mod go_resolver;
pub mod golang;
pub mod java_registry;
pub mod java_resolver;
pub mod kotlin_registry;
pub mod kotlin_resolver;
pub mod php_registry;
pub mod php_resolver;
pub mod py_resolver;
pub mod python;
pub mod rust_registry;
pub mod rust_resolver;
pub mod swift_registry;
pub mod swift_resolver;
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
        Language::Python => Some(Box::new(python::PythonRegistry::new(graph, filez))),
        Language::Go => Some(Box::new(golang::GoRegistry::new(graph, filez))),
        Language::Rust => Some(Box::new(rust_registry::RustRegistry::new(graph, filez))),
        Language::Java => Some(Box::new(java_registry::JavaRegistry::new(graph, filez))),
        Language::CSharp => Some(Box::new(csharp_registry::CSharpRegistry::new(graph, filez))),
        Language::Kotlin => Some(Box::new(kotlin_registry::KotlinRegistry::new(graph, filez))),
        Language::Swift => Some(Box::new(swift_registry::SwiftRegistry::new(graph, filez))),
        Language::Php => Some(Box::new(php_registry::PhpRegistry::new(graph, filez))),
        _ => None,
    }
}
