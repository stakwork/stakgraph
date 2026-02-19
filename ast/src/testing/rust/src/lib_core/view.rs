use std::marker::PhantomData;

pub struct View<'a, T> {
    data: &'a T,
    _marker: PhantomData<&'a T>,
}

impl<'a, T> View<'a, T> {
    pub fn new(data: &'a T) -> Self {
        Self {
            data,
            _marker: PhantomData,
        }
    }

    pub fn access(&self) -> &T {
        self.data
    }
}
