use std::io::{self, Write};

/// Wrapper around stdout that returns Result instead of panicking on broken pipe
pub struct Output {}

impl Output {
    pub fn new() -> Self {
        Self {}
    }

    pub fn writeln(&mut self, s: impl AsRef<str>) -> io::Result<()> {
        let mut stdout = io::stdout().lock();
        writeln!(stdout, "{}", s.as_ref())
    }

    pub fn newline(&mut self) -> io::Result<()> {
        let mut stdout = io::stdout().lock();
        writeln!(stdout)
    }
}
