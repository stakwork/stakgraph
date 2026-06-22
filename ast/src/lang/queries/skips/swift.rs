const FRAMEWORK_OPERANDS: [&str; 10] = [
    "AF",
    "DispatchQueue",
    "URLSession",
    "JSONDecoder",
    "JSONEncoder",
    "JSONSerialization",
    "NotificationCenter",
    "UserDefaults",
    "Task",
    "super",
];

const STDLIB_BARE: [&str; 6] = [
    "print",
    "debugPrint",
    "fatalError",
    "precondition",
    "assert",
    "assertionFailure",
];

pub fn should_skip(called: &str, operand: &Option<String>) -> bool {
    if operand.is_none() && STDLIB_BARE.contains(&called) {
        return true;
    }
    if let Some(op) = operand {
        if FRAMEWORK_OPERANDS.contains(&op.as_str()) {
            return true;
        }
    }
    false
}
