#[repr(C)]
pub struct Register {
    value: u32,
}

impl Register {
    /// Simulates a volatile read
    pub unsafe fn read(&self) -> u32 {
        std::ptr::read_volatile(&self.value)
    }

    /// Simulates a volatile write
    pub unsafe fn write(&mut self, val: u32) {
        std::ptr::write_volatile(&mut self.value, val);
    }
}

pub fn init() {
    // mock init
}
