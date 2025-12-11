pub trait Greet {
    fn greet(&self) -> String;
}

#[derive(Debug, Clone)]
pub struct Greeter {
    pub name: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct MultiAttrStruct {
    pub value: i32,
}

#[async_trait::async_trait]
impl Greet for Greeter {
    fn greet(&self) -> String {
        format!("Hello, {}!", self.name)
    }
}

pub trait Container {
    type Item;
    
    fn get(&self) -> Option<&Self::Item>;
    fn set(&mut self, item: Self::Item);
}

pub trait Processor<T>
where
    T: Clone + Send,
{
    fn process(&self, item: T) -> T;
    fn batch_process(&self, items: Vec<T>) -> Vec<T>;
}

pub trait Logger {
    fn log(&self, msg: &str) {
        println!("[LOG] {}", msg);
    }
    
    fn error(&self, msg: &str) {
        println!("[ERROR] {}", msg);
    }
}

pub struct StringContainer {
    value: Option<String>,
}

impl Container for StringContainer {
    type Item = String;
    
    fn get(&self) -> Option<&Self::Item> {
        self.value.as_ref()
    }
    
    fn set(&mut self, item: Self::Item) {
        self.value = Some(item);
    }
}

pub struct SimpleProcessor;

impl<T: Clone + Send> Processor<T> for SimpleProcessor {
    fn process(&self, item: T) -> T {
        item.clone()
    }
    
    fn batch_process(&self, items: Vec<T>) -> Vec<T> {
        items.into_iter().map(|i| self.process(i)).collect()
    }
}
