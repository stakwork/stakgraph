const SVELTE_LIFECYCLE: [&str; 8] = [
    "onMount",
    "onDestroy",
    "beforeUpdate",
    "afterUpdate",
    "tick",
    "createEventDispatcher",
    "bind",
    "animate",
];

const WEB_API: [&str; 20] = [
    "fetch",
    "then",
    "catch",
    "finally",
    "json",
    "text",
    "blob",
    "ok",
    "status",
    "headers",
    "body",
    "clone",
    "JSON",
    "parse",
    "stringify",
    "addEventListener",
    "removeEventListener",
    "preventDefault",
    "stopPropagation",
    "querySelector",
];

const SVELTE_STORES: [&str; 8] = [
    "writable",
    "readable",
    "derived",
    "subscribe",
    "set",
    "update",
    "get",
    "unsubscribe",
];

const SVELTE_ANIMATIONS: [&str; 10] = [
    "transition",
    "animate",
    "fade",
    "fly",
    "slide",
    "scale",
    "draw",
    "crossfade",
    "blur",
    "bounce",
];

pub fn should_skip(called: &str, _operand: &Option<String>) -> bool {
    SVELTE_LIFECYCLE.contains(&called)
        || WEB_API.contains(&called)
        || SVELTE_STORES.contains(&called)
        || SVELTE_ANIMATIONS.contains(&called)
}
