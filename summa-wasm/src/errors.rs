use std::io;
use wasm_bindgen::JsValue;

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("index_error: {0}")]
    AsyncIo(tantivy::error::AsyncIoError),
    #[error("index_error: {0}")]
    Core(summa_core::Error),
    #[error("incorrect_payload")]
    IncorrectPayload,
    #[error("index_error: {0}")]
    Index(tantivy::TantivyError),
    #[error("js_error: {0}")]
    Js(String),
    #[error("serialization_error: {0}")]
    Serialization(String),
}

impl From<serde_wasm_bindgen::Error> for Error {
    fn from(error: serde_wasm_bindgen::Error) -> Self {
        Error::Serialization(format!("{:?}", error))
    }
}

impl From<wasm_bindgen::JsValue> for Error {
    fn from(error: wasm_bindgen::JsValue) -> Self {
        Error::Js(format!("{:?}", error))
    }
}

impl From<Error> for io::Error {
    fn from(error: Error) -> Self {
        io::Error::new(io::ErrorKind::Other, error)
    }
}

impl From<strfmt::FmtError> for Error {
    fn from(error: strfmt::FmtError) -> Self {
        Error::Serialization(format!("{:?}", error))
    }
}

impl From<Error> for JsValue {
    fn from(error: Error) -> Self {
        JsValue::from(format!("{:?}", error))
    }
}

impl From<Error> for tantivy::error::AsyncIoError {
    fn from(error: Error) -> Self {
        tantivy::error::AsyncIoError::Io(error.into())
    }
}

impl From<tantivy::error::AsyncIoError> for Error {
    fn from(error: tantivy::error::AsyncIoError) -> Self {
        Error::AsyncIo(error)
    }
}

impl From<tantivy::TantivyError> for Error {
    fn from(error: tantivy::TantivyError) -> Self {
        Error::Index(error)
    }
}

impl From<summa_core::Error> for Error {
    fn from(error: summa_core::Error) -> Self {
        Error::Core(error)
    }
}

pub type SummaWasmResult<T> = Result<T, Error>;
