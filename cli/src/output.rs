use std::io::{self, Write};

use serde::Serialize;

enum OutputInner {
    Stdout,
    Buffer(Vec<u8>),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OutputMode {
    Human,
    Json,
}

impl OutputMode {
    pub fn from_json_flag(json: bool) -> Self {
        if json {
            Self::Json
        } else {
            Self::Human
        }
    }

    pub fn is_json(self) -> bool {
        matches!(self, Self::Json)
    }
}

#[derive(Serialize)]
pub struct JsonWarning {
    pub kind: String,
    pub message: String,
}

impl JsonWarning {
    pub fn new(kind: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            kind: kind.into(),
            message: message.into(),
        }
    }
}

#[derive(Serialize)]
pub struct JsonError {
    pub message: String,
}

impl JsonError {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

#[derive(Serialize)]
pub struct JsonSuccessEnvelope<T> {
    pub ok: bool,
    pub command: String,
    pub data: T,
    pub warnings: Vec<JsonWarning>,
}

#[derive(Serialize)]
pub struct JsonErrorEnvelope {
    pub ok: bool,
    pub command: String,
    pub error: JsonError,
}

pub fn write_json_success<T: Serialize>(
    out: &mut Output,
    command: &str,
    data: T,
    warnings: Vec<JsonWarning>,
) -> io::Result<()> {
    let payload = JsonSuccessEnvelope {
        ok: true,
        command: command.to_string(),
        data,
        warnings,
    };
    out.write_json(&payload)
}

pub fn write_json_error(out: &mut Output, command: &str, message: impl Into<String>) -> io::Result<()> {
    let payload = JsonErrorEnvelope {
        ok: false,
        command: command.to_string(),
        error: JsonError::new(message),
    };
    out.write_json(&payload)
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

    pub fn write_json<T: Serialize>(&mut self, value: &T) -> io::Result<()> {
        let json = serde_json::to_string(value)
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?;
        self.writeln(json)
    }
}
