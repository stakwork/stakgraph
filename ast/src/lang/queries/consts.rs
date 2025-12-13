pub const IDENTIFIER: &str = "identifier";
pub const LIBRARY: &str = "LIBRARY";
pub const LIBRARY_NAME: &str = "LIBRARY_NAME";
pub const LIBRARY_VERSION: &str = "LIBRARY_VERSION";
pub const IMPORTS: &str = "imports";
pub const IMPORTS_FROM: &str = "imports-from";
pub const IMPORTS_NAME: &str = "imports-name";
pub const IMPORTS_ALIAS: &str = "imports-alias";
pub const TRAIT: &str = "trait";
pub const TRAIT_NAME: &str = "trait-name";
pub const TRAIT_METHOD_NAME: &str = "trait-method-name";
pub const MODULE_NAME: &str = "module-name";
pub const CLASS_NAME: &str = "class-name";
pub const CLASS_PARENT: &str = "class-parent";
pub const INCLUDED_MODULES: &str = "included-modules";
pub const INSTANCE: &str = "instance";
pub const INSTANCE_NAME: &str = "instance-name";
pub const CLASS_DEFINITION: &str = "class-definition";
pub const FUNCTION_NAME: &str = "function-name";
pub const FUNCTION_DEFINITION: &str = "function-definition";
pub const ARGUMENTS: &str = "arguments";
pub const RETURN_TYPES: &str = "return-types";
pub const PARENT_NAME: &str = "parent-name";
pub const PARENT_TYPE: &str = "parent-type";
pub const FUNCTION_CALL: &str = "function-call";
pub const FUNCTION_COMMENT: &str = "function-comment";
pub const OPERAND: &str = "operand";
pub const MEMBER_EXPRESSION: &str = "member-expression";
pub const ASSOCIATION_TYPE: &str = "association-type";
pub const ASSOCIATION_TARGET: &str = "association-target";
pub const ASSOCIATION_OPTION: &str = "association-option";

pub const HANDLER: &str = "handler";
pub const HANDLER_ACTIONS_ARRAY: &str = "handler-actions-array";
pub const COLLECTION_ITEM: &str = "collection-item";
pub const MEMBER_ITEM: &str = "member-item";
pub const RESOURCE_ITEM: &str = "resource-item";
pub const ENDPOINT_ACTION: &str = "endpoint-action";
pub const ENDPOINT: &str = "endpoint";
pub const ENDPOINT_ALIAS: &str = "endpoint-alias";
pub const ENDPOINT_VERB: &str = "endpoint-verb";
pub const ENDPOINT_OBJECT: &str = "endpoint-object";
pub const ROUTE: &str = "route";
pub const REQUEST_CALL: &str = "call";
pub const ENDPOINT_GROUP: &str = "endpoint-group";
pub const SERVICE_REGISTRATION: &str = "service-registration";
pub const SCOPE_PREFIX: &str = "scope-prefix";
pub const HANDLER_REF: &str = "handler-ref";
pub const NAMESPACE_NAME: &str = "namespace-name";
pub const SCOPE_PATH: &str = "scope-path";
pub const SINGULAR_RESOURCE: &str = "singular-resource";

pub const INTEGRATION_TEST: &str = "integration-test";
pub const E2E_TEST: &str = "e2e-test";
pub const E2E_TEST_NAME: &str = "e2e-test-name";
pub const TEST_NAME: &str = "test-name";

pub const STRUCT: &str = "struct";
pub const STRUCT_NAME: &str = "struct-name";
pub const PAGE: &str = "page";
pub const PAGE_COMPONENT: &str = "page-component";
pub const PAGE_PATHS: &str = "page-paths";
pub const PAGE_HEADER: &str = "page-header";
pub const PAGE_CHILD: &str = "page-child";
pub const VARIABLE_DECLARATION: &str = "variable-declaration";
pub const VARIABLE_NAME: &str = "variable-name";
pub const VARIABLE_VALUE: &str = "variable-value";
pub const VARIABLE_TYPE: &str = "variable-type";

pub const EXTRA: &str = "extra";
pub const EXTRA_NAME: &str = "extra-name";
pub const EXTRA_PROP: &str = "extra-prop";

pub const DECORATOR_NAME: &str = "decorator_name";
pub const TEMPLATE_KEY: &str = "key";
pub const TEMPLATE_VALUE: &str = "value";

pub const IMPLEMENTS: &str = "implements";
pub const ATTRIBUTES: &str = "attributes";
pub const MACRO: &str = "macro";

pub const ARRAY_METHODS: [&str; 30] = [
    "push",
    "pop",
    "shift",
    "unshift",
    "slice",
    "splice",
    "concat",
    "join",
    "reverse",
    "sort",
    "indexOf",
    "lastIndexOf",
    "forEach",
    "map",
    "filter",
    "reduce",
    "reduceRight",
    "every",
    "some",
    "find",
    "findIndex",
    "includes",
    "flat",
    "flatMap",
    "fill",
    "copyWithin",
    "entries",
    "keys",
    "values",
    "at",
];

pub const STRING_METHODS: [&str; 30] = [
    "charAt",
    "charCodeAt",
    "concat",
    "includes",
    "indexOf",
    "lastIndexOf",
    "match",
    "matchAll",
    "replace",
    "replaceAll",
    "search",
    "slice",
    "split",
    "substring",
    "toLowerCase",
    "toUpperCase",
    "trim",
    "trimStart",
    "trimEnd",
    "padStart",
    "padEnd",
    "repeat",
    "startsWith",
    "endsWith",
    "localeCompare",
    "normalize",
    "at",
    "codePointAt",
    "fromCharCode",
    "fromCodePoint",
];

pub const OBJECT_METHODS: [&str; 15] = [
    "hasOwnProperty",
    "isPrototypeOf",
    "propertyIsEnumerable",
    "toLocaleString",
    "toString",
    "valueOf",
    "keys",
    "values",
    "entries",
    "assign",
    "create",
    "defineProperty",
    "freeze",
    "seal",
    "preventExtensions",
];

pub const ASYNC_METHODS: [&str; 6] = ["then", "catch", "finally", "all", "race", "allSettled"];

pub const DOM_METHODS: [&str; 20] = [
    "addEventListener",
    "removeEventListener",
    "querySelector",
    "querySelectorAll",
    "getElementById",
    "getElementsByClassName",
    "getElementsByTagName",
    "appendChild",
    "removeChild",
    "replaceChild",
    "insertBefore",
    "cloneNode",
    "setAttribute",
    "getAttribute",
    "removeAttribute",
    "classList",
    "focus",
    "blur",
    "click",
    "submit",
];

pub const JSX_HTML_ELEMENTS: [&str; 134] = [
    "div",
    "span",
    "p",
    "section",
    "article",
    "nav",
    "header",
    "footer",
    "main",
    "aside",
    "address",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "form",
    "input",
    "button",
    "label",
    "select",
    "textarea",
    "fieldset",
    "legend",
    "datalist",
    "option",
    "optgroup",
    "ul",
    "ol",
    "li",
    "dl",
    "dt",
    "dd",
    "table",
    "tbody",
    "thead",
    "tfoot",
    "tr",
    "td",
    "th",
    "caption",
    "colgroup",
    "col",
    "img",
    "video",
    "audio",
    "source",
    "track",
    "canvas",
    "iframe",
    "embed",
    "object",
    "param",
    "map",
    "area",
    "strong",
    "em",
    "code",
    "pre",
    "kbd",
    "var",
    "samp",
    "mark",
    "ins",
    "del",
    "sub",
    "sup",
    "small",
    "b",
    "i",
    "u",
    "s",
    "q",
    "bdi",
    "bdo",
    "details",
    "summary",
    "dialog",
    "meta",
    "link",
    "style",
    "svg",
    "g",
    "path",
    "rect",
    "circle",
    "ellipse",
    "line",
    "polyline",
    "polygon",
    "text",
    "tspan",
    "defs",
    "use",
    "symbol",
    "marker",
    "clipPath",
    "mask",
    "pattern",
    "image",
    "foreignObject",
    "switch",
    "a",
    "view",
    "animate",
    "animateMotion",
    "animateTransform",
    "set",
    "desc",
    "title",
    "metadata",
    "stop",
    "linearGradient",
    "radialGradient",
    "feBlend",
    "feColorMatrix",
    "feComponentTransfer",
    "feComposite",
    "feConvolveMatrix",
    "feDiffuseLighting",
    "feDisplacementMap",
    "feFlood",
    "feGaussianBlur",
    "feImage",
    "feMerge",
    "feMorphology",
    "feOffset",
    "feSpecularLighting",
    "feTile",
    "feTurbulence",
    "feDistantLight",
    "fePointLight",
    "feSpotLight",
];

pub fn should_skip_js_function_call(called: &str, operand: &Option<String>) -> bool {
    if let Some(op) = operand {
        if let Some(first_char) = op.chars().next() {
            if first_char.is_lowercase() {
                if ARRAY_METHODS.contains(&called)
                    || STRING_METHODS.contains(&called)
                    || OBJECT_METHODS.contains(&called)
                    || ASYNC_METHODS.contains(&called)
                    || DOM_METHODS.contains(&called)
                {
                    return true;
                }
            }
        }
    }

    if JSX_HTML_ELEMENTS.contains(&called) {
        return true;
    }

    false
}
