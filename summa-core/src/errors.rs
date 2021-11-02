use std::convert::{From, Infallible};
use std::path::PathBuf;
use tantivy::schema::FieldType;

#[derive(thiserror::Error, Debug)]
pub enum ValidationError {
    #[error("builder_error: {0}")]
    BuilderError(String),
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
    #[error("missing_path_error: {0}")]
    MissingPath(PathBuf),
    #[error("missing_primary_key_error: {0:?}")]
    MissingPrimaryKey(Option<String>),
    #[error("missing_range")]
    MissingRange,
    #[error("required_fast_field: {0}")]
    RequiredFastField(String),
    #[error("utf8_error: {0}")]
    Utf8(std::str::Utf8Error),
}

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("addr_parse_error: {0}")]
    AddrParse(std::net::AddrParseError),
    #[error("index_error: {0}")]
    AsyncIo(tantivy::error::AsyncIoError),
    #[error("config_error: {0}")]
    Config(config::ConfigError),
    #[error("document_parsing_error: {0}")]
    DocumentParsing(crate::components::DocumentParsingError),
    #[error("empty_query_error")]
    EmptyQuery,
    #[error("external: {0}")]
    External(String),
    #[error("fast_eval_error: {0:?}")]
    FastEval(fasteval2::Error),
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
    Json(serde_json::Error),
    #[cfg(feature = "index-updater")]
    #[error("{0}")]
    Kafka(rdkafka::error::KafkaError),
    #[error("open_directory_error: {0}")]
    OpenDirectory(tantivy::directory::error::OpenDirectoryError),
    #[error("tantivy_error: {0}")]
    Tantivy(tantivy::TantivyError),
    #[error("proto")]
    Proto(summa_proto::errors::Error),
    #[error("unbound_document_error")]
    UnboundDocument,
    #[error("unknown_directory_error: {0}")]
    UnknownDirectory(String),
    #[error("{0}")]
    Validation(ValidationError),
}

impl From<ValidationError> for Error {
    fn from(error: ValidationError) -> Self {
        Error::Validation(error)
    }
}

impl From<derive_builder::UninitializedFieldError> for ValidationError {
    fn from(ufe: derive_builder::UninitializedFieldError) -> ValidationError {
        ValidationError::BuilderError(ufe.to_string())
    }
}

impl From<crate::components::DocumentParsingError> for Error {
    fn from(error: crate::components::DocumentParsingError) -> Self {
        Error::DocumentParsing(error)
    }
}

impl From<serde_json::Error> for Error {
    fn from(error: serde_json::Error) -> Self {
        Error::Json(error)
    }
}

impl From<tokio::task::JoinError> for Error {
    fn from(_error: tokio::task::JoinError) -> Self {
        Error::Internal
    }
}

impl From<config::ConfigError> for Error {
    fn from(error: config::ConfigError) -> Self {
        Error::Config(error)
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

impl From<std::str::Utf8Error> for ValidationError {
    fn from(error: std::str::Utf8Error) -> Self {
        ValidationError::Utf8(error)
    }
}

impl From<tantivy::TantivyError> for Error {
    fn from(error: tantivy::TantivyError) -> Self {
        Error::Tantivy(error)
    }
}

impl From<summa_proto::errors::Error> for Error {
    fn from(error: summa_proto::errors::Error) -> Self {
        Error::Proto(error)
    }
}

impl From<fasteval2::Error> for Error {
    fn from(error: fasteval2::Error) -> Self {
        Error::FastEval(error)
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

impl From<tantivy::error::AsyncIoError> for Error {
    fn from(error: tantivy::error::AsyncIoError) -> Self {
        Error::AsyncIo(error)
    }
}

impl From<tantivy::directory::error::OpenDirectoryError> for Error {
    fn from(error: tantivy::directory::error::OpenDirectoryError) -> Self {
        Error::OpenDirectory(error)
    }
}

#[cfg(feature = "index-updater")]
impl From<rdkafka::error::KafkaError> for Error {
    fn from(error: rdkafka::error::KafkaError) -> Self {
        Error::Kafka(error)
    }
}

pub type SummaResult<T> = Result<T, Error>;
