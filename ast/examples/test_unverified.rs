fn main() {
    // These calls won't be found since we're parsing a single file
    some_external_function();
    another_missing_call();
    
    // This one exists
    local_function();
}

fn local_function() {
    println!("This exists");
    
    // More unverified calls
    unverified_helper();
}
