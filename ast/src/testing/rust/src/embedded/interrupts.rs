// @ast node: Var "INTERRUPT_VECTOR"
// @ast node: Function "setup"
// @ast node: Function "handler"
pub static mut INTERRUPT_VECTOR: [usize; 256] = [0; 256];

pub fn setup() {
    unsafe {
        INTERRUPT_VECTOR[0] = handler as usize;
    }
}

fn handler() {
    // handle interrupt
}
