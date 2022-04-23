use std::convert::{From, Infallible};
use std::fmt::{Display, Formatter};
use std::path::PathBuf;

use tantivy::schema::FieldType;

#[derive(thiserror::Error, Debug)]
pub enum ValidationError {
    #[error("builder_error: {0}")]
    Builder(#[from] BuilderError),
    #[error("invalid_fast_field_type_error: ({field:?}, {field_type:?}, {tantivy_error:?})")]
    InvalidFastFieldType {
        field: String,
        field_type: FieldType,
        tantivy_error: tantivy::TantivyError,
    },
    #[error("invalid_http_header: <{0}: {1}>")]
    InvalidHttpHeader(String, String),
    #[error("invalid_segments_number: {0}")]
    InvalidSegmentsNumber(u32),
    #[error("invalid_schema_error: {0}")]
    InvalidSchema(String),
    #[error("invalid_unique_field_type_error: {0:?}")]
    InvalidUniqueFieldType(FieldType),
    #[error("empty_argument_error: {0}")]
    EmptyArgument(String),
    #[error("existing_path_error: {0}")]
    ExistingPath(PathBuf),
    #[error("missing_index_error: {0}")]
    MissingIndex(String),
    #[error("missing_field_error: {0}")]
    MissingField(String),
    #[error("missing_header_error: {0}")]
    MissingHeader(String),
    #[error("missing_path_error: {0}")]
    MissingPath(PathBuf),
    #[error("missing_range")]
    MissingRange,
    #[error("missing_unique_field_error: {0:?}")]
    MissingUniqueField(String),
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
    #[error("anyhow_error: {0}")]
    Anyhow(#[from] anyhow::Error),
    #[error("config_error: {0}")]
    Config(#[from] config::ConfigError),
    #[error("document_parsing_error: {0}")]
    DocumentParsing(#[from] crate::components::DocumentParsingError),
    #[error("empty_query_error")]
    EmptyQuery,
    #[error("fast_eval_error: {0:?}")]
    FastEval(#[from] fasteval2::Error),
    #[cfg(feature = "hyper-external-request")]
    #[error("hyper_error: {0}")]
    Hyper(#[from] hyper::Error),
    #[cfg(feature = "hyper-external-request")]
    #[error("hyper_http_error: {0}")]
    HyperHttp(#[from] hyper::http::Error),
    #[error("infallible")]
    Infallible,
    #[error("internal_error")]
    Internal,
    #[error("invalid_aggregation")]
    InvalidAggregation,
    #[error("{0:?}: {1:?}")]
    InvalidFieldType(String, FieldType),
    #[error("{0:?} for {1:?}")]
    InvalidQuerySyntax(Box<crate::components::QueryParserError>, String),
    #[error("{0:?}")]
    InvalidSegmentId(String),
    #[error("{0:?}")]
    InvalidSyntax(String),
    #[error("{0:?}")]
    IO((std::io::Error, Option<PathBuf>)),
    #[error("json_error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("open_directory_error: {0}")]
    OpenDirectory(#[from] tantivy::directory::error::OpenDirectoryError),
    #[error("tantivy_error: {0}")]
    Tantivy(#[from] tantivy::TantivyError),
    #[error("read_only_index: {0}")]
    ReadOnlyIndex(String),
    #[error("request_error: {0}")]
    RequestError(#[from] crate::directories::RequestError),
    #[error("unbound_document_error")]
    UnboundDocument,
    #[error("unknown_directory_error: {0}")]
    UnknownDirectory(String),
    #[error("{0}")]
    Validation(Box<ValidationError>),
    #[error("{0}")]
    Yaml(#[from] serde_yaml::Error),
}

#[derive(thiserror::Error, Debug)]
pub enum BuilderError {
    /// Uninitialized field
    UninitializedField(&'static str),
    /// Custom validation error
    ValidationError(String),
}

impl From<BuilderError> for Error {
    fn from(error: BuilderError) -> Self {
        Error::Validation(Box::new(ValidationError::Builder(error)))
    }
}

impl From<ValidationError> for Error {
    fn from(error: ValidationError) -> Self {
        Error::Validation(Box::new(error))
    }
}

#[cfg(feature = "tokio-rt")]
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

impl Display for BuilderError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            BuilderError::UninitializedField(s) => write!(f, "UninitializedField({s})"),
            BuilderError::ValidationError(s) => write!(f, "ValidationError({s})"),
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
