use std::cell::RefCell;
use std::collections::HashMap;
use std::hash::Hash;
use std::rc::Rc;

#[derive(Debug)]
pub struct Cache<K, V>
where
    K: Hash + Eq + Clone,
    V: Clone,
{
    store: RefCell<HashMap<K, V>>,
    capacity: usize,
}

impl<K, V> Cache<K, V>
where
    K: Hash + Eq + Clone,
    V: Clone,
{
    pub fn new(capacity: usize) -> Self {
        Self {
            store: RefCell::new(HashMap::new()),
            capacity,
        }
    }

    pub fn insert(&self, key: K, value: V) {
        let mut map = self.store.borrow_mut();
        if map.len() >= self.capacity {
            // naive eviction
            if let Some(k) = map.keys().next().cloned() {
                map.remove(&k);
            }
        }
        map.insert(key, value);
    }

    pub fn get(&self, key: &K) -> Option<V> {
        self.store.borrow().get(key).cloned()
    }
}
