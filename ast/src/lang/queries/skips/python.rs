const LIST_METHODS: [&str; 12] = [
    "append",
    "extend",
    "pop",
    "remove",
    "sort",
    "reverse",
    "clear",
    "copy",
    "count",
    "index",
    "insert",
    "len",
];

const DICT_METHODS: [&str; 12] = [
    "keys", "values", "items", "get", "pop", "update", "clear", "copy", "setdefault",
    "fromkeys", "popitem", "len",
];

const STRING_METHODS: [&str; 20] = [
    "split",
    "join",
    "strip",
    "lstrip",
    "rstrip",
    "replace",
    "upper",
    "lower",
    "capitalize",
    "title",
    "find",
    "index",
    "startswith",
    "endswith",
    "isdigit",
    "isalpha",
    "format",
    "encode",
    "decode",
    "len",
];

const BUILTINS: [&str; 15] = [
    "len",
    "print",
    "range",
    "enumerate",
    "zip",
    "map",
    "filter",
    "sorted",
    "reversed",
    "sum",
    "min",
    "max",
    "any",
    "all",
    "isinstance",
];

pub fn should_skip(called: &str, operand: &Option<String>) -> bool {
    if BUILTINS.contains(&called) {
        return true;
    }

    if let Some(op) = operand {
        if let Some(first_char) = op.chars().next() {
            if first_char.is_lowercase()
                && (LIST_METHODS.contains(&called)
                    || DICT_METHODS.contains(&called)
                    || STRING_METHODS.contains(&called))
            {
                return true;
            }
        }
    }

    false
}
