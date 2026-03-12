use std::io::{self, Write};

enum OutputInner {
    Stdout,
    Buffer(Vec<u8>),
}

/// Wrapper around stdout (or an in-memory buffer) that returns Result instead of panicking on broken pipe
pub struct Output {
    inner: OutputInner,
}

impl Output {
    pub fn new() -> Self {
        Self {
            inner: OutputInner::Stdout,
        }
    }

    pub fn new_buffer() -> Self {
        Self {
            inner: OutputInner::Buffer(Vec::new()),
        }
    }

    pub fn into_string(self) -> String {
        match self.inner {
            OutputInner::Buffer(buf) => String::from_utf8_lossy(&buf).into_owned(),
            OutputInner::Stdout => String::new(),
        }
    }

    pub fn writeln(&mut self, s: impl AsRef<str>) -> io::Result<()> {
        match &mut self.inner {
            OutputInner::Stdout => {
                let mut stdout = io::stdout().lock();
                writeln!(stdout, "{}", s.as_ref())
            }
            OutputInner::Buffer(buf) => writeln!(buf, "{}", s.as_ref()),
        }
    }

    pub fn newline(&mut self) -> io::Result<()> {
        match &mut self.inner {
            OutputInner::Stdout => {
                let mut stdout = io::stdout().lock();
                writeln!(stdout)
            }
            OutputInner::Buffer(buf) => writeln!(buf),
        }
    }
}
