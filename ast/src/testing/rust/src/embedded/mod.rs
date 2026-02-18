pub mod interrupts;
pub mod registers;

pub fn init() {
    registers::init();
    interrupts::setup();
}
