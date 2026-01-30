macro_rules! say_hello {
    () => {
        println!("Hello!");
    };
}

macro_rules! create_function {
    ($name:ident) => {
        pub fn $name() -> &'static str {
            stringify!($name)
        }
    };
}

macro_rules! log_expr {
    ($e:expr) => {
        println!("{} = {:?}", stringify!($e), $e)
    };
}

macro_rules! make_struct {
    ($name:ident { $($field:ident: $type:ty),* }) => {
        pub struct $name {
            $(pub $field: $type,)*
        }
    };
}

macro_rules! impl_display {
    ($type:ty, $fmt:expr) => {
        impl std::fmt::Display for $type {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, $fmt, self)
            }
        }
    };
}

/// This function uses macros
pub fn use_macros() {
    say_hello!();

    create_function!(generated_func);
    let name = generated_func();

    let value = 42;
    log_expr!(value);
    log_expr!(value + 10);
}

create_function!(test_func);
create_function!(another_func);

make_struct!(Config {
    host: String,
    port: u16
});
