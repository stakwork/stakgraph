const VEC_METHODS: [&str; 20] = [
    "push",
    "pop",
    "len",
    "is_empty",
    "iter",
    "iter_mut",
    "map",
    "filter",
    "collect",
    "into_iter",
    "get",
    "first",
    "last",
    "contains",
    "sort",
    "reverse",
    "retain",
    "dedup",
    "clear",
    "extend",
];

const OPTION_METHODS: [&str; 15] = [
    "unwrap",
    "expect",
    "map",
    "and_then",
    "or_else",
    "is_some",
    "is_none",
    "unwrap_or",
    "unwrap_or_else",
    "ok_or",
    "ok_or_else",
    "take",
    "filter",
    "flatten",
    "as_ref",
];

const RESULT_METHODS: [&str; 15] = [
    "unwrap",
    "expect",
    "map",
    "and_then",
    "or_else",
    "is_ok",
    "is_err",
    "unwrap_or",
    "unwrap_or_else",
    "map_err",
    "ok",
    "err",
    "as_ref",
    "unwrap_or_default",
    "expect_err",
];

const STRING_METHODS: [&str; 15] = [
    "len",
    "is_empty",
    "push_str",
    "to_string",
    "clone",
    "as_str",
    "into_bytes",
    "chars",
    "bytes",
    "split",
    "trim",
    "to_lowercase",
    "to_uppercase",
    "replace",
    "contains",
];

const ITERATOR_METHODS: [&str; 20] = [
    "map",
    "filter",
    "collect",
    "fold",
    "for_each",
    "any",
    "all",
    "find",
    "position",
    "count",
    "take",
    "skip",
    "chain",
    "zip",
    "enumerate",
    "flat_map",
    "flatten",
    "cloned",
    "copied",
    "sum",
];

const COMMON_TRAITS: [&str; 10] = [
    "clone",
    "to_owned",
    "into",
    "from",
    "as_ref",
    "as_mut",
    "default",
    "to_string",
    "fmt",
    "drop",
];

const PATH_METHODS: [&str; 25] = [
    "parent",
    "file_name",
    "extension",
    "file_stem",
    "to_path_buf",
    "to_str",
    "display",
    "exists",
    "is_file",
    "is_dir",
    "is_absolute",
    "is_relative",
    "canonicalize",
    "read_dir",
    "join",
    "with_extension",
    "with_file_name",
    "starts_with",
    "ends_with",
    "strip_prefix",
    "components",
    "ancestors",
    "metadata",
    "symlink_metadata",
    "read_link",
];

const OSSTR_METHODS: [&str; 5] = [
    "to_str",
    "to_string_lossy",
    "to_os_string",
    "len",
    "is_empty",
];

const COMMON_CONSTRUCTORS: [&str; 10] = [
    "new",
    "default",
    "Ok",
    "Err",
    "Some",
    "None",
    "Custom",
    "with_capacity",
    "from",
    "into",
];

const WALKDIR_METHODS: [&str; 10] = [
    "new",
    "min_depth",
    "max_depth",
    "follow_links",
    "max_open",
    "sort_by",
    "filter_entry",
    "into_iter",
    "path",
    "file_name",
];

pub fn should_skip(called: &str, operand: &Option<String>) -> bool {
    if let Some(op) = operand {
        if op == "std" || op == "core" || op.starts_with("std::") || op.starts_with("core::") {
            return true;
        }
        if op == "WalkDir" || op.ends_with("::WalkDir") {
            return true;
        }
        // Skip Path/PathBuf/DirEntry types
        if op == "Path"
            || op == "PathBuf"
            || op == "DirEntry"
            || op.ends_with("::Path")
            || op.ends_with("::PathBuf")
            || op.ends_with("::DirEntry")
        {
            return true;
        }
    }

    VEC_METHODS.contains(&called)
        || OPTION_METHODS.contains(&called)
        || RESULT_METHODS.contains(&called)
        || STRING_METHODS.contains(&called)
        || ITERATOR_METHODS.contains(&called)
        || COMMON_TRAITS.contains(&called)
        || PATH_METHODS.contains(&called)
        || OSSTR_METHODS.contains(&called)
        || COMMON_CONSTRUCTORS.contains(&called)
        || WALKDIR_METHODS.contains(&called)
}
