use serde::Serialize;

pub trait Greet {
    fn greet(&self) -> String;
}

#[derive(Debug, Clone)]
pub struct Greeter {
    pub name: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct MultiAttrStruct {
    pub value: i32,
}

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_greeter_greet() {
        let greeter = Greeter {
            name: "Alice".to_string(),
        };
        let result = greeter.greet();
        assert_eq!(result, "Hello, Alice!");
    }

    #[test]
    fn test_greeter_with_empty_name() {
        let greeter = Greeter {
            name: "".to_string(),
        };
        let result = greeter.greet();
        assert_eq!(result, "Hello, !");
    }

    #[test]
    fn test_string_container_new() {
        let container = StringContainer { value: None };
        assert!(container.get().is_none());
    }

    #[test]
    fn test_string_container_set_and_get() {
        let mut container = StringContainer { value: None };
        container.set("hello".to_string());
        
        let value = container.get();
        assert!(value.is_some());
        assert_eq!(value.unwrap(), "hello");
    }

    #[test]
    fn test_string_container_overwrite() {
        let mut container = StringContainer { value: Some("first".to_string()) };
        container.set("second".to_string());
        
        let value = container.get();
        assert_eq!(value.unwrap(), "second");
    }

    #[test]
    fn test_simple_processor_process() {
        let processor = SimpleProcessor;
        let result = processor.process(42);
        assert_eq!(result, 42);
    }

    #[test]
    fn test_simple_processor_batch() {
        let processor = SimpleProcessor;
        let items = vec![1, 2, 3, 4, 5];
        let result = processor.batch_process(items);
        assert_eq!(result, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_simple_processor_with_strings() {
        let processor = SimpleProcessor;
        let items = vec!["a".to_string(), "b".to_string()];
        let result = processor.batch_process(items);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_multi_attr_struct_creation() {
        let data = MultiAttrStruct { value: 100 };
        assert_eq!(data.value, 100);
    }

    #[test]
    fn test_multi_attr_struct_clone() {
        let data = MultiAttrStruct { value: 50 };
        let cloned = data.clone();
        assert_eq!(data, cloned);
    }
}
