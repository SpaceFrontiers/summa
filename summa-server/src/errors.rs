use std::convert::From;
use std::path::PathBuf;
use tracing::warn;

#[derive(thiserror::Error, Debug)]
pub enum ValidationError {
    #[error("aliased_error: {0}")]
    Aliased(String),
    #[error("existing_index_error: {0}")]
    ExistingIndex(String),
    #[error("invalid_schema_error: {0}")]
    InvalidSchema(String),
    #[error("missing_index_error: {0}")]
    MissingIndex(String),
    #[error("missing_default_field_error: {0}")]
    MissingDefaultField(String),
    #[error("missing_field_error: {0}")]
    MissingField(String),
    #[error("missing_multi_field_error: {0}")]
    MissingMultiField(String),
    #[error("missing_primary_key_error: {0:?}")]
    MissingPrimaryKey(Option<String>),
}

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("addr_parse_error: {0}")]
    AddrParse(std::net::AddrParseError),
    #[error("hyper_error: {0}")]
    Hyper(hyper::Error),
    #[error("hyper_http_error: {0}")]
    HttpHyper(hyper::http::Error),
    #[error("internal_error")]
    Internal,
    #[error("{0:?}")]
    IO((std::io::Error, Option<PathBuf>)),
    #[error("json_error: {0}")]
    Json(serde_json::Error),
    #[error("summa_core: {0}")]
    Core(summa_core::Error),
    #[error("tonic_error: {0}")]
    Tonic(tonic::transport::Error),
    #[error("upstream_http_status_error: {0}")]
    UpstreamHttpStatus(hyper::StatusCode, String),
    #[error("utf8_error: {0}")]
    Utf8(std::str::Utf8Error),
    #[error("{0}")]
    Validation(ValidationError),
}

impl From<ValidationError> for Error {
    fn from(error: ValidationError) -> Self {
        Error::Validation(error)
    }
}

impl From<hyper::Error> for Error {
    fn from(error: hyper::Error) -> Self {
        Error::Hyper(error)
    }
}

impl From<hyper::http::Error> for Error {
    fn from(error: hyper::http::Error) -> Self {
        Error::HttpHyper(error)
    }
}

impl From<summa_core::Error> for Error {
    fn from(error: summa_core::Error) -> Self {
        Error::Core(error)
    }
}

impl From<serde_json::Error> for Error {
    fn from(error: serde_json::Error) -> Self {
        Error::Json(error)
    }
}

impl From<std::io::Error> for Error {
    fn from(error: std::io::Error) -> Self {
        Error::IO((error, None))
    }
}

impl From<std::net::AddrParseError> for Error {
    fn from(error: std::net::AddrParseError) -> Self {
        Error::AddrParse(error)
    }
}

impl From<std::str::Utf8Error> for Error {
    fn from(error: std::str::Utf8Error) -> Self {
        Error::Utf8(error)
    }
}

impl From<tokio::task::JoinError> for Error {
    fn from(_error: tokio::task::JoinError) -> Self {
        Error::Internal
    }
}

impl From<tonic::transport::Error> for Error {
    fn from(error: tonic::transport::Error) -> Self {
        Error::Tonic(error)
    }
}

impl From<Error> for tonic::Status {
    fn from(error: Error) -> Self {
        warn!(action = "error", error = ?error);
        tonic::Status::new(
            match error {
                Error::IO((ref io_error, _)) => match io_error.kind() {
                    std::io::ErrorKind::PermissionDenied => tonic::Code::PermissionDenied,
                    _ => tonic::Code::Internal,
                },
                Error::Validation(ValidationError::MissingIndex(_)) => tonic::Code::NotFound,
                Error::Validation(_) => tonic::Code::InvalidArgument,
                _ => tonic::Code::Internal,
            },
            format!("{}", error),
        )
    }
}

impl From<ValidationError> for tonic::Status {
    fn from(error: ValidationError) -> Self {
        tonic::Status::from(Error::Validation(error))
    }
}

pub type SummaServerResult<T> = Result<T, Error>;
