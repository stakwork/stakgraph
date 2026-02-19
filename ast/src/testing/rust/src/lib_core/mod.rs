pub mod cache;
pub mod view;

pub trait Container<T> {
    fn get(&self) -> Option<&T>;
    fn set(&mut self, item: T);
}
