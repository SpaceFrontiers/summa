use std::convert::{From, Infallible};
use std::path::PathBuf;
use tracing::warn;

#[derive(thiserror::Error, Debug)]
pub enum ValidationError {
    #[error("aliased_error: {0}")]
    AliasedError(String),
    #[error("empty_argument: {0}")]
    EmptyArgument(String),
    #[error("existing_config_error: {0}")]
    ExistingConfigError(PathBuf),
    #[error("existing_consumers_error: {0}")]
    ExistingConsumersError(String),
    #[error("existing_consumer_error: {0}")]
    ExistingConsumerError(String),
    #[error("existing_path_error: {0}")]
    ExistingPathError(PathBuf),
    #[error("invalid_memory_error: {0}")]
    InvalidMemoryError(u64),
    #[error("invalid_threads_number_error: {0}")]
    InvalidThreadsNumberError(u64),
    #[error("missing_consumer_error: {0}")]
    MissingConsumerError(String),
    #[error("missing_index_error: {0}")]
    MissingIndexError(String),
    #[error("missing_default_field_error: {0}")]
    MissingDefaultField(String),
    #[error("missing_path_error: {0}")]
    MissingPathError(String),
    #[error("missing_primary_key_error: {0:?}")]
    MissingPrimaryKeyError(Option<String>),
    #[error("utf8_error: {0}")]
    Utf8Error(std::str::Utf8Error),
}

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("addr_parse_error: {0}")]
    AddrParseError(std::net::AddrParseError),
    #[error("arc_index_writer_holder_leaked_error")]
    ArcIndexWriterHolderLeakedError,
    #[error("canceled_error")]
    CanceledError,
    #[error("config_error: {0}")]
    ConfigError(config::ConfigError),
    #[error("hyper_error: {0}")]
    HyperError(hyper::Error),
    #[error("infallible")]
    Infallible,
    #[error("internal_error")]
    InternalError,
    #[error("{0:?}")]
    InvalidSyntaxError((tantivy::query::QueryParserError, String)),
    #[error("invalid_config_error: {0}")]
    InvalidConfigError(String),
    #[error("{0:?}")]
    IOError((std::io::Error, Option<PathBuf>)),
    #[error("{0}")]
    KafkaError(rdkafka::error::KafkaError),
    #[error("parse_error: {0}")]
    ParseError(tantivy::schema::DocParsingError),
    #[error("tantivy_error: {0}")]
    TantivyError(tantivy::TantivyError),
    #[error("timeout_error")]
    TimeoutError,
    #[error("tonic_error: {0}")]
    TonicError(tonic::transport::Error),
    #[error("transition_state_error")]
    TransitionStateError,
    #[error("unbound_document_error")]
    UnboundDocumentError,
    #[error("unknown_directory_error: {0}")]
    UnknownDirectoryError(String),
    #[error("utf8_error: {0}")]
    Utf8Error(std::str::Utf8Error),
    #[error("vaildation_error: {0}")]
    ValidationError(ValidationError),
    #[error("yaml_error: {0}")]
    YamlError(serde_yaml::Error),
}

impl From<ValidationError> for Error {
    fn from(error: ValidationError) -> Self {
        Error::ValidationError(error)
    }
}

impl From<config::ConfigError> for Error {
    fn from(error: config::ConfigError) -> Self {
        Error::ConfigError(error)
    }
}

impl From<hyper::Error> for Error {
    fn from(error: hyper::Error) -> Self {
        Error::HyperError(error)
    }
}

impl From<rdkafka::error::KafkaError> for Error {
    fn from(error: rdkafka::error::KafkaError) -> Self {
        Error::KafkaError(error)
    }
}

impl From<serde_yaml::Error> for Error {
    fn from(error: serde_yaml::Error) -> Self {
        Error::YamlError(error)
    }
}

impl From<std::io::Error> for Error {
    fn from(error: std::io::Error) -> Self {
        Error::IOError((error, None))
    }
}

impl From<std::net::AddrParseError> for Error {
    fn from(error: std::net::AddrParseError) -> Self {
        Error::AddrParseError(error)
    }
}

impl From<std::str::Utf8Error> for Error {
    fn from(error: std::str::Utf8Error) -> Self {
        Error::ValidationError(ValidationError::Utf8Error(error))
    }
}

impl From<tantivy::TantivyError> for Error {
    fn from(error: tantivy::TantivyError) -> Self {
        Error::TantivyError(error)
    }
}

impl From<tokio::time::error::Elapsed> for Error {
    fn from(_error: tokio::time::error::Elapsed) -> Self {
        Error::TimeoutError
    }
}

impl From<tokio::task::JoinError> for Error {
    fn from(_error: tokio::task::JoinError) -> Self {
        Error::InternalError
    }
}

impl From<tonic::transport::Error> for Error {
    fn from(error: tonic::transport::Error) -> Self {
        Error::TonicError(error)
    }
}

impl From<Infallible> for Error {
    fn from(_: Infallible) -> Self {
        Error::Infallible
    }
}

impl From<Error> for tonic::Status {
    fn from(error: Error) -> Self {
        warn!(action = "error", error = ?error);
        tonic::Status::new(
            match error {
                Error::IOError((ref io_error, _)) => match io_error.kind() {
                    std::io::ErrorKind::PermissionDenied => tonic::Code::PermissionDenied,
                    _ => tonic::Code::Internal,
                },
                Error::TantivyError(_) => tonic::Code::InvalidArgument,
                Error::ValidationError(ValidationError::MissingConsumerError(_)) | Error::ValidationError(ValidationError::MissingIndexError(_)) => tonic::Code::NotFound,
                Error::ValidationError(_) => tonic::Code::InvalidArgument,
                _ => tonic::Code::Internal,
            },
            format!("{}", error),
        )
    }
}

impl From<ValidationError> for tonic::Status {
    fn from(error: ValidationError) -> Self {
        tonic::Status::from(Error::ValidationError(error))
    }
}

pub type SummaResult<T> = Result<T, Error>;
