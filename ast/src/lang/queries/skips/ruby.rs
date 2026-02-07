const TEST_FRAMEWORK_METHODS: [&str; 21] = [
    "to",
    "not_to",
    "to_not",
    "eq",
    "eql",
    "be",
    "be_a",
    "be_an",
    "be_nil",
    "be_truthy",
    "be_falsey",
    "be_true",
    "be_false",
    "be_empty",
    "be_blank",
    "be_present",
    "include",
    "match",
    "raise_error",
    "change",
    "respond_to",
];

const ARRAY_METHODS: [&str; 20] = [
    "push",
    "pop",
    "shift",
    "unshift",
    "map",
    "select",
    "reject",
    "each",
    "compact",
    "flatten",
    "uniq",
    "sort",
    "reverse",
    "first",
    "last",
    "size",
    "length",
    "empty?",
    "include?",
    "join",
];

const STRING_METHODS: [&str; 15] = [
    "upcase",
    "downcase",
    "strip",
    "split",
    "gsub",
    "sub",
    "length",
    "size",
    "empty?",
    "include?",
    "start_with?",
    "end_with?",
    "chars",
    "bytes",
    "to_s",
];

const HASH_METHODS: [&str; 12] = [
    "keys",
    "values",
    "merge",
    "fetch",
    "dig",
    "each",
    "map",
    "select",
    "reject",
    "empty?",
    "size",
    "length",
];

const ENUMERABLE_METHODS: [&str; 10] = [
    "find",
    "inject",
    "reduce",
    "zip",
    "any?",
    "all?",
    "none?",
    "one?",
    "min",
    "max",
];

pub fn should_skip(called: &str, operand: &Option<String>) -> bool {
    if let Some(op) = operand {
        if let Some(first_char) = op.chars().next() {
            if first_char.is_lowercase() {
                return true;
            }
        }
    }

    TEST_FRAMEWORK_METHODS.iter().any(|&m| called.starts_with(m))
        || ARRAY_METHODS.contains(&called)
        || STRING_METHODS.contains(&called)
        || HASH_METHODS.contains(&called)
        || ENUMERABLE_METHODS.contains(&called)
        || called.starts_with("have_")
        || called == "expect"
        || called == "describe"
        || called == "it"
        || called == "context"
        || called == "before"
        || called == "after"
        || called == "let"
        || called == "subject"
}
