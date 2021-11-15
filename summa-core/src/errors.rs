use std::convert::{From, Infallible};
use std::fmt::{Display, Formatter};
use std::path::PathBuf;

use tantivy::schema::FieldType;

#[derive(thiserror::Error, Debug)]
pub enum ValidationError {
    #[error("builder_error: {0}")]
    Builder(#[from] BuilderError),
    #[error("empty_argument_error: {0}")]
    EmptyArgument(String),
    #[error("existing_consumer_error: {0}")]
    ExistingConsumer(String),
    #[error("invalid_fast_field_type_error: ({field:?}, {field_type:?}, {tantivy_error:?})")]
    InvalidFastFieldType {
        field: String,
        field_type: FieldType,
        tantivy_error: tantivy::TantivyError,
    },
    #[error("invalid_http_header: <{0}: {1}>")]
    InvalidHttpHeader(String, String),
    #[error("invalid_http_method: {0}")]
    InvalidHttpMethod(String),
    #[error("invalid_primary_key_type_error: {0:?}")]
    InvalidPrimaryKeyType(FieldType),
    #[error("existing_path_error: {0}")]
    ExistingPath(PathBuf),
    #[error("missing_consumer_error: {0}")]
    MissingConsumer(String),
    #[error("missing_index_error: {0}")]
    MissingIndex(String),
    #[error("missing_field_error: {0}")]
    MissingField(String),
    #[error("missing_header_error: {0}")]
    MissingHeader(String),
    #[error("missing_path_error: {0}")]
    MissingPath(PathBuf),
    #[error("missing_primary_key_error: {0:?}")]
    MissingPrimaryKey(Option<String>),
    #[error("missing_range")]
    MissingRange,
    #[error("required_fast_field: {0}")]
    RequiredFastField(String),
    #[error("utf8_error: {0}")]
    Utf8(#[from] std::str::Utf8Error),
    #[error("template_error: {0}")]
    Template(#[from] strfmt::FmtError),
}

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("addr_parse_error: {0}")]
    AddrParse(#[from] std::net::AddrParseError),
    #[error("index_error: {0}")]
    AsyncIo(#[from] tantivy::error::AsyncIoError),
    #[error("config_error: {0}")]
    Config(#[from] config::ConfigError),
    #[error("document_parsing_error: {0}")]
    DocumentParsing(#[from] crate::components::DocumentParsingError),
    #[error("empty_query_error")]
    EmptyQuery,
    #[error("external: {0}")]
    External(String),
    #[error("fast_eval_error: {0:?}")]
    FastEval(#[from] fasteval2::Error),
    #[error("infallible")]
    Infallible,
    #[error("internal_error")]
    Internal,
    #[error("{0:?}: {1:?}")]
    InvalidFieldType(String, FieldType),
    #[error("{0:?}")]
    InvalidSyntax(String),
    #[error("{0:?} for {1:?}")]
    InvalidTantivySyntax(tantivy::query::QueryParserError, String),
    #[error("{0:?}")]
    IO((std::io::Error, Option<PathBuf>)),
    #[error("json_error: {0}")]
    Json(#[from] serde_json::Error),
    #[cfg(feature = "consume")]
    #[error("{0}")]
    Kafka(#[from] rdkafka::error::KafkaError),
    #[error("open_directory_error: {0}")]
    OpenDirectory(#[from] tantivy::directory::error::OpenDirectoryError),
    #[error("tantivy_error: {0}")]
    Tantivy(#[from] tantivy::TantivyError),
    #[error("proto")]
    Proto(#[from] summa_proto::errors::Error),
    #[error("unbound_document_error")]
    UnboundDocument,
    #[error("unknown_directory_error: {0}")]
    UnknownDirectory(String),
    #[error("{0}")]
    Validation(#[from] ValidationError),
    #[error("{0}")]
    Yaml(#[from] serde_yaml::Error),
}

impl From<BuilderError> for Error {
    fn from(error: BuilderError) -> Self {
        Error::Validation(ValidationError::Builder(error))
    }
}

impl From<tokio::task::JoinError> for Error {
    fn from(_error: tokio::task::JoinError) -> Self {
        Error::Internal
    }
}

impl From<std::io::Error> for Error {
    fn from(error: std::io::Error) -> Self {
        Error::IO((error, None))
    }
}

impl<T> From<async_broadcast::SendError<T>> for Error {
    fn from(_: async_broadcast::SendError<T>) -> Self {
        Error::Internal
    }
}

impl From<Infallible> for Error {
    fn from(_: Infallible) -> Self {
        Error::Infallible
    }
}

impl From<Error> for std::io::Error {
    fn from(error: Error) -> Self {
        std::io::Error::new(std::io::ErrorKind::Other, error)
    }
}

impl From<Error> for tantivy::error::AsyncIoError {
    fn from(error: Error) -> Self {
        tantivy::error::AsyncIoError::Io(error.into())
    }
}

#[derive(thiserror::Error, Debug)]
pub enum BuilderError {
    /// Uninitialized field
    UninitializedField(&'static str),
    /// Custom validation error
    ValidationError(String),
}

impl Display for BuilderError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            BuilderError::UninitializedField(s) => write!(f, "UninitializedField({})", s),
            BuilderError::ValidationError(s) => write!(f, "ValidationError({})", s),
        }
    }
}

impl From<String> for BuilderError {
    fn from(s: String) -> Self {
        Self::ValidationError(s)
    }
}

impl From<derive_builder::UninitializedFieldError> for BuilderError {
    fn from(error: derive_builder::UninitializedFieldError) -> Self {
        Self::ValidationError(error.to_string())
    }
}

pub type SummaResult<T> = Result<T, Error>;
