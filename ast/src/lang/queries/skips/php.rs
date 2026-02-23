const BUILTIN_FUNCTIONS: [&str; 13] = [
    "count",
    "isset",
    "empty",
    "in_array",
    "array_map",
    "array_filter",
    "array_reduce",
    "json_encode",
    "json_decode",
    "strlen",
    "explode",
    "implode",
    "trim",
];

pub fn should_skip(called: &str, _operand: &Option<String>) -> bool {
    BUILTIN_FUNCTIONS.contains(&called)
}
