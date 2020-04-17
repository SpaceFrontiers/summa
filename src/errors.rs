use serde_json::json;
use std::convert::From;

#[derive(thiserror::Error, Debug)]
pub enum BadRequestError {
    #[error("doc_parsing_error")]
    DocParsingError(tantivy::schema::DocParsingError),
    #[error("unknown_content_type_error")]
    UnknownContentTypeError,
    #[error("utf8_error")]
    Utf8Error(std::str::Utf8Error),
}

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("bad_request_error")]
    BadRequestError(BadRequestError),
    #[error("canceled_error")]
    CanceledError,
    #[error("config_error")]
    ConfigError(config::ConfigError),
    #[error("crossbeam_send_error")]
    CrossbeamSendError,
    #[error("file_error")]
    FileError((std::io::Error, String)),
    #[error("internal_error")]
    InternalError,
    #[error("invalid_syntax_error")]
    InvalidSyntaxError((tantivy::query::QueryParserError, String)),
    #[error("io_error")]
    IOError(std::io::Error),
    #[error("poison_error")]
    PoisonError,
    #[error("not_found_error")]
    NotFoundError,
    #[error("tantivy_error")]
    TantivyError(tantivy::TantivyError),
    #[error("timeout_error")]
    TimeoutError,
    #[error("unknown_schema_error")]
    UnknownSchemaError,
    #[error("yaml_error")]
    YamlError(serde_yaml::Error),
}

impl From<config::ConfigError> for Error {
    fn from(error: config::ConfigError) -> Self {
        Error::ConfigError(error)
    }
}

impl<D> From<crossbeam_channel::SendError<D>> for Error {
    fn from(_error: crossbeam_channel::SendError<D>) -> Self {
        Error::CrossbeamSendError
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

impl From<std::str::Utf8Error> for Error {
    fn from(error: std::str::Utf8Error) -> Self {
        Error::BadRequestError(BadRequestError::Utf8Error(error))
    }
}

impl<T> From<std::sync::PoisonError<T>> for Error {
    fn from(_error: std::sync::PoisonError<T>) -> Self {
        Error::PoisonError
    }
}

impl From<tantivy::schema::DocParsingError> for Error {
    fn from(error: tantivy::schema::DocParsingError) -> Self {
        Error::BadRequestError(BadRequestError::DocParsingError(error))
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
            Error::BadRequestError(_) => actix_web::http::StatusCode::BAD_REQUEST,
            Error::InvalidSyntaxError(_) => actix_web::http::StatusCode::BAD_REQUEST,
            Error::NotFoundError => actix_web::http::StatusCode::NOT_FOUND,
            Error::TimeoutError => actix_web::http::StatusCode::GATEWAY_TIMEOUT,
            _ => actix_web::http::StatusCode::INTERNAL_SERVER_ERROR,
        }
    }
}
