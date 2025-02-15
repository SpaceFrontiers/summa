//! Mapping from hermes_core::Error to gRPC Status codes

use tonic::Status;

/// Convert a `hermes_core::Error` into the most appropriate gRPC `Status` code.
///
/// Every variant is matched explicitly so that adding a new variant to
/// `hermes_core::Error` causes a compile error here (no catch-all `_`).
pub fn hermes_error_to_status(e: hermes_core::Error) -> Status {
    match &e {
        // Client errors — INVALID_ARGUMENT
        hermes_core::Error::Schema(_) => Status::invalid_argument(e.to_string()),
        hermes_core::Error::Query(_) => Status::invalid_argument(e.to_string()),
        hermes_core::Error::Document(_) => Status::invalid_argument(e.to_string()),
        hermes_core::Error::Tokenizer(_) => Status::invalid_argument(e.to_string()),
        hermes_core::Error::InvalidFieldType { .. } => Status::invalid_argument(e.to_string()),

        // Not-found variants
        hermes_core::Error::FieldNotFound(_) => Status::not_found(e.to_string()),
        hermes_core::Error::DocumentNotFound(_) => Status::not_found(e.to_string()),

        // Conflict
        hermes_core::Error::DuplicatePrimaryKey(_) => Status::already_exists(e.to_string()),

        // Backpressure
        hermes_core::Error::QueueFull => Status::resource_exhausted(e.to_string()),

        // Precondition failures
        hermes_core::Error::IndexClosed => Status::failed_precondition(e.to_string()),

        // Infrastructure / transient
        hermes_core::Error::Io(_) => Status::unavailable(e.to_string()),

        // Server-side errors — INTERNAL
        hermes_core::Error::Corruption(_) => Status::internal(e.to_string()),
        hermes_core::Error::Serialization(_) => Status::internal(e.to_string()),
        hermes_core::Error::Internal(_) => Status::internal(e.to_string()),
    }
}
