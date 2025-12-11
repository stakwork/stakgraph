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
