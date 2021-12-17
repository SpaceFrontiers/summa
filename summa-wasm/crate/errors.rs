use std::io;

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("async_io: {0}")]
    AsyncIo(#[from] tantivy::error::AsyncIoError),
    #[error("core: {0}")]
    Core(#[from] summa_core::Error),
    #[error("index_error: {0}")]
    Index(#[from] tantivy::TantivyError),
    #[error("js_error: {0}")]
    Js(String),
    #[error("serialization_error: {0}")]
    Serialization(String),
}

impl From<serde_wasm_bindgen::Error> for Error {
    fn from(error: serde_wasm_bindgen::Error) -> Self {
        Error::Serialization(format!("{error:?}"))
    }
}

impl From<wasm_bindgen::JsValue> for Error {
    fn from(error: wasm_bindgen::JsValue) -> Self {
        Error::Js(format!("{error:?}"))
    }
}

impl From<Error> for io::Error {
    fn from(error: Error) -> Self {
        io::Error::new(io::ErrorKind::Other, error)
    }
}

impl From<strfmt::FmtError> for Error {
    fn from(error: strfmt::FmtError) -> Self {
        Error::Serialization(format!("{error:?}"))
    }
}

impl From<Error> for wasm_bindgen::JsValue {
    fn from(error: Error) -> Self {
        wasm_bindgen::JsValue::from(format!("{error:?}"))
    }
}

impl From<Error> for tantivy::error::AsyncIoError {
    fn from(error: Error) -> Self {
        tantivy::error::AsyncIoError::Io(error.into())
    }
}

pub type SummaWasmResult<T> = Result<T, Error>;
