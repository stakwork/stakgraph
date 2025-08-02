pub trait Greet {
    fn greet(&self) -> String;
}

pub struct Greeter {
    pub name: String,
}

impl Greeter {
    pub fn new(name: String) -> Self {
        Self { name }
    }
}

impl Greet for Greeter {
    fn greet(&self) -> String {
        format!("Hello, {}!", self.name)
    }
}
