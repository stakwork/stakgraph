use crate::lang::parse::trim_quotes;
use crate::lang::queries::react_ts::TypeScriptReact;
use crate::lang::queries::ruby::Ruby;
use crate::lang::queries::Stack;

#[test]
fn trim_quotes_strips_matching_quotes() {
    assert_eq!(trim_quotes("\"abc\""), "abc");
    assert_eq!(trim_quotes("'abc'"), "abc");
    assert_eq!(trim_quotes("`abc`"), "abc");
    assert_eq!(trim_quotes("\"\""), "");
    assert_eq!(trim_quotes(":sym"), "sym");
}

#[test]
fn trim_quotes_leaves_unbalanced_quotes_alone() {
    assert_eq!(trim_quotes("\"abc"), "\"abc");
    assert_eq!(trim_quotes("\"abc'"), "\"abc'");
}

#[test]
fn trim_quotes_lone_quote_does_not_panic() {
    assert_eq!(trim_quotes("\""), "\"");
    assert_eq!(trim_quotes("'"), "'");
    assert_eq!(trim_quotes("`"), "`");
}

#[test]
fn ruby_import_path_strips_matching_parens_only() {
    let ruby = Ruby::default();
    assert_eq!(ruby.resolve_import_path("(foo)", ""), "foo");
    assert_eq!(ruby.resolve_import_path("(", ""), "(");
    assert_eq!(ruby.resolve_import_path("(abc", ""), "(abc");
    assert_eq!(ruby.resolve_import_path("(café", ""), "(café");
}

#[test]
fn ruby_import_name_strips_matching_quotes_only() {
    let ruby = Ruby::default();
    assert_eq!(ruby.resolve_import_name("\"foo\""), "Foo");
    assert_eq!(ruby.resolve_import_name("'foo'"), "Foo");
    assert_eq!(ruby.resolve_import_name("\""), "\"");
    assert_eq!(ruby.resolve_import_name("'"), "'");
}

#[test]
fn react_ts_import_path_lone_quote_does_not_panic() {
    let ts = TypeScriptReact::default();
    assert_eq!(ts.resolve_import_path("\"x\"", ""), "x");
    assert_eq!(ts.resolve_import_path("\"", ""), "\"");
}
