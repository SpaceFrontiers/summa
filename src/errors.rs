use serde_json::json;
use std::convert::From;

#[derive(thiserror::Error, Debug)]
pub enum ConfigError {
    #[error("yaml_error")]
    YamlError(serde_yaml::Error),
}

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("bad_request_error")]
    BadRequestError,
    #[error("canceled_error")]
    CanceledError,
    #[error("config_error")]
    ConfigError(ConfigError),
    #[error("internal_error")]
    InternalError,
    #[error("invalid_syntax_error")]
    InvalidSyntaxError((tantivy::query::QueryParserError, String)),
    #[error("io_error")]
    IOError(std::io::Error),
    #[error("poison_error")]
    PoisonError,
    #[error("protobuf_error")]
    ProtobufError(protobuf::error::ProtobufError),
    #[error("not_found_error")]
    NotFoundError,
    #[error("tantivy_error")]
    TantivyError(tantivy::TantivyError),
    #[error("templar_error")]
    TemplarError(templar::error::TemplarError),
    #[error("timeout_error")]
    TimeoutError,
    #[error("unknown_schema_error")]
    UnknownSchemaError,
}

impl From<ConfigError> for Error {
    fn from(error: ConfigError) -> Self {
        Error::ConfigError(error)
    }
}

impl From<actix_web::error::BlockingError<Error>> for Error {
    fn from(error: actix_web::error::BlockingError<Error>) -> Self {
        match error {
            actix_web::error::BlockingError::Error(e) => e,
            actix_web::error::BlockingError::Canceled => Error::CanceledError,
        }
    }
}

impl From<protobuf::error::ProtobufError> for Error {
    fn from(error: protobuf::error::ProtobufError) -> Self {
        Error::ProtobufError(error)
    }
}

impl From<serde_json::error::Error> for Error {
    fn from(_error: serde_json::error::Error) -> Self {
        Error::InternalError
    }
}

impl From<std::io::Error> for Error {
    fn from(error: std::io::Error) -> Self {
        Error::IOError(error)
    }
}

impl<T> From<std::sync::PoisonError<T>> for Error {
    fn from(_error: std::sync::PoisonError<T>) -> Self {
        Error::PoisonError
    }
}

impl From<tantivy::TantivyError> for Error {
    fn from(error: tantivy::TantivyError) -> Self {
        Error::TantivyError(error)
    }
}

impl From<tokio::time::Elapsed> for Error {
    fn from(_error: tokio::time::Elapsed) -> Self {
        Error::TimeoutError
    }
}

impl From<templar::error::TemplarError> for Error {
    fn from(error: templar::error::TemplarError) -> Self {
        Error::TemplarError(error)
    }
}

impl From<tokio::task::JoinError> for Error {
    fn from(_error: tokio::task::JoinError) -> Self {
        Error::InternalError
    }
}

impl actix_web::error::ResponseError for Error {
    fn error_response(&self) -> actix_web::HttpResponse {
        // ToDo логировать в файл
        println!("{:?}", self);
        actix_http::ResponseBuilder::new(self.status_code())
            .set_header(actix_web::http::header::CONTENT_TYPE, "application/json")
            .json(json!({
                "code": self.to_string(),
                "status": "error",
            }))
    }

    fn status_code(&self) -> actix_web::http::StatusCode {
        match *self {
            Error::BadRequestError => actix_web::http::StatusCode::BAD_REQUEST,
            Error::CanceledError => actix_web::http::StatusCode::INTERNAL_SERVER_ERROR,
            Error::ConfigError(_) => actix_web::http::StatusCode::INTERNAL_SERVER_ERROR,
            Error::InternalError => actix_web::http::StatusCode::INTERNAL_SERVER_ERROR,
            Error::InvalidSyntaxError(_) => actix_web::http::StatusCode::BAD_REQUEST,
            Error::IOError(_) => actix_web::http::StatusCode::INTERNAL_SERVER_ERROR,
            Error::NotFoundError => actix_web::http::StatusCode::NOT_FOUND,
            Error::PoisonError => actix_web::http::StatusCode::INTERNAL_SERVER_ERROR,
            Error::ProtobufError(_) => actix_web::http::StatusCode::INTERNAL_SERVER_ERROR,
            Error::TantivyError(_) => actix_web::http::StatusCode::INTERNAL_SERVER_ERROR,
            Error::TemplarError(_) => actix_web::http::StatusCode::INTERNAL_SERVER_ERROR,
            Error::TimeoutError => actix_web::http::StatusCode::GATEWAY_TIMEOUT,
            Error::UnknownSchemaError => actix_web::http::StatusCode::INTERNAL_SERVER_ERROR,
        }
    }
}
