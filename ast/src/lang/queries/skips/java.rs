const STRING_METHODS: [&str; 20] = [
    "length",
    "isEmpty",
    "substring",
    "toLowerCase",
    "toUpperCase",
    "trim",
    "split",
    "replace",
    "replaceAll",
    "concat",
    "contains",
    "startsWith",
    "endsWith",
    "indexOf",
    "lastIndexOf",
    "charAt",
    "matches",
    "equals",
    "equalsIgnoreCase",
    "valueOf",
];

const LIST_METHODS: [&str; 15] = [
    "add",
    "remove",
    "get",
    "size",
    "isEmpty",
    "contains",
    "clear",
    "addAll",
    "removeAll",
    "indexOf",
    "set",
    "subList",
    "toArray",
    "iterator",
    "stream",
];

const COLLECTION_METHODS: [&str; 10] = [
    "size",
    "isEmpty",
    "contains",
    "iterator",
    "toArray",
    "add",
    "remove",
    "clear",
    "stream",
    "forEach",
];

const COMMON_CLASSES: [&str; 15] = [
    "System",
    "Math",
    "String",
    "Integer",
    "Double",
    "Long",
    "Boolean",
    "Collections",
    "Arrays",
    "Objects",
    "Optional",
    "Stream",
    "List",
    "Map",
    "Set",
];

pub fn should_skip(called: &str, operand: &Option<String>) -> bool {
    if let Some(op) = operand {
        if COMMON_CLASSES.contains(&op.as_str()) {
            return true;
        }
        if let Some(first_char) = op.chars().next() {
            if first_char.is_lowercase()
                && (STRING_METHODS.contains(&called)
                    || LIST_METHODS.contains(&called)
                    || COLLECTION_METHODS.contains(&called))
            {
                return true;
            }
        }
    }

    false
}
