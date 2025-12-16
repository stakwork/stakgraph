use serde::Serialize;
use std::collections::HashMap;
use std::fmt::Display;
use std::hash::Hash;

pub type Result<T> = std::result::Result<T, anyhow::Error>;
pub type UserId = u64;
pub type Handler = Box<dyn Fn(String) -> String>;
pub type UserMap = HashMap<UserId, String>;

pub fn process<T: Serialize>(data: T) -> Result<String> {
    let json = serde_json::to_string(&data)?;
    Ok(json)
}

pub fn transform<T, U>(input: T) -> U
where
    T: Clone + Display,
    U: From<T> + Default,
{
    U::from(input)
}

pub fn longest<'a>(x: &'a str, y: &'a str) -> &'a str {
    if x.len() > y.len() {
        x
    } else {
        y
    }
}

pub fn compare_and_display<T>(a: T, b: T) -> String
where
    T: PartialOrd + Display + Clone,
{
    if a > b {
        format!("{} is greater", a)
    } else {
        format!("{} is greater or equal", b)
    }
}

pub fn multi_generic<T, U, V>(t: T, u: U) -> V
where
    T: Into<V>,
    U: Display,
    V: Default + Clone,
{
    println!("Processing: {}", u);
    t.into()
}

pub fn closure_examples() {
    let simple = |x: i32| x + 1;
    let result = simple(5);

    let with_type: Box<dyn Fn(i32) -> i32> = Box::new(|x| x * 2);
    let doubled = with_type(10);

    let captures = {
        let value = 42;
        move |x: i32| x + value
    };
    let computed = captures(8);

    let handler: Handler = Box::new(|s| s.to_uppercase());
    let upper = handler("test".to_string());
}

pub const fn compute_constant() -> usize {
    1024
}

pub const fn array_size() -> usize {
    256
}

pub struct Cache<K, V>
where
    K: Hash + Eq,
{
    data: HashMap<K, V>,
}

impl<K, V> Cache<K, V>
where
    K: Hash + Eq,
{
    pub fn new() -> Self {
        Self {
            data: HashMap::new(),
        }
    }

    pub fn insert(&mut self, key: K, value: V) {
        self.data.insert(key, value);
    }

    pub fn get(&self, key: &K) -> Option<&V> {
        self.data.get(key)
    }
}

pub enum ApiResult<T, E> {
    Success(T),
    Failure(E),
    Pending,
}

impl<T, E> ApiResult<T, E> {
    pub fn is_success(&self) -> bool {
        matches!(self, ApiResult::Success(_))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_process_with_struct() {
        #[derive(Serialize)]
        struct TestData {
            name: String,
            value: i32,
        }
        
        let data = TestData {
            name: "test".to_string(),
            value: 42,
        };
        
        let result = process(data);
        assert!(result.is_ok());
        assert!(result.unwrap().contains("test"));
    }

    #[test]
    fn test_process_with_string() {
        let result = process("hello");
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "\"hello\"");
    }

    #[test]
    fn test_process_with_number() {
        let result = process(123);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "123");
    }

    #[test]
    fn test_longest_first_longer() {
        let result = longest("longest", "short");
        assert_eq!(result, "longest");
    }

    #[test]
    fn test_longest_second_longer() {
        let result = longest("short", "definitely longer");
        assert_eq!(result, "definitely longer");
    }

    #[test]
    fn test_longest_equal_length() {
        let result = longest("same", "size");
        assert_eq!(result, "size");
    }

    #[test]
    fn test_compare_and_display_greater() {
        let result = compare_and_display(10, 5);
        assert_eq!(result, "10 is greater");
    }

    #[test]
    fn test_compare_and_display_equal() {
        let result = compare_and_display(5, 5);
        assert_eq!(result, "5 is greater or equal");
    }

    #[test]
    fn test_compare_and_display_strings() {
        let result = compare_and_display("zebra", "apple");
        assert!(result.contains("greater"));
    }

    #[test]
    fn test_compute_constant() {
        let result = compute_constant();
        assert_eq!(result, 1024);
    }

    #[test]
    fn test_array_size() {
        let result = array_size();
        assert_eq!(result, 256);
    }

    #[test]
    fn test_cache_new() {
        let cache: Cache<String, i32> = Cache::new();
        assert_eq!(cache.data.len(), 0);
    }

    #[test]
    fn test_cache_insert_and_get() {
        let mut cache = Cache::new();
        cache.insert("key1".to_string(), 100);
        
        let value = cache.get(&"key1".to_string());
        assert!(value.is_some());
        assert_eq!(*value.unwrap(), 100);
    }

    #[test]
    fn test_cache_get_missing() {
        let cache: Cache<String, i32> = Cache::new();
        let value = cache.get(&"missing".to_string());
        assert!(value.is_none());
    }

    #[test]
    fn test_cache_multiple_inserts() {
        let mut cache = Cache::new();
        cache.insert(1, "one".to_string());
        cache.insert(2, "two".to_string());
        cache.insert(3, "three".to_string());
        
        assert_eq!(cache.get(&1), Some(&"one".to_string()));
        assert_eq!(cache.get(&2), Some(&"two".to_string()));
        assert_eq!(cache.get(&3), Some(&"three".to_string()));
    }

    #[test]
    fn test_api_result_is_success() {
        let success: ApiResult<i32, String> = ApiResult::Success(42);
        assert!(success.is_success());
    }

    #[test]
    fn test_api_result_is_failure() {
        let failure: ApiResult<i32, String> = ApiResult::Failure("error".to_string());
        assert!(!failure.is_success());
    }

    #[test]
    fn test_api_result_is_pending() {
        let pending: ApiResult<i32, String> = ApiResult::Pending;
        assert!(!pending.is_success());
    }

    #[test]
    fn test_closure_examples() {
        closure_examples();
    }
}
