//! Mapping from summa_core::Error to gRPC Status codes

use tonic::Status;

/// Convert a `summa_core::Error` into the most appropriate gRPC `Status` code.
///
/// Every variant is matched explicitly so that adding a new variant to
/// `summa_core::Error` causes a compile error here (no catch-all `_`).
pub fn summa_error_to_status(e: summa_core::Error) -> Status {
    match &e {
        // Client errors — INVALID_ARGUMENT
        summa_core::Error::Schema(_) => Status::invalid_argument(e.to_string()),
        summa_core::Error::Query(_) => Status::invalid_argument(e.to_string()),
        summa_core::Error::Document(_) => Status::invalid_argument(e.to_string()),
        summa_core::Error::Tokenizer(_) => Status::invalid_argument(e.to_string()),
        summa_core::Error::InvalidFieldType { .. } => Status::invalid_argument(e.to_string()),

        // Not-found variants
        summa_core::Error::FieldNotFound(_) => Status::not_found(e.to_string()),
        summa_core::Error::DocumentNotFound(_) => Status::not_found(e.to_string()),

        // Conflict
        summa_core::Error::DuplicatePrimaryKey(_) => Status::already_exists(e.to_string()),

        // Backpressure
        summa_core::Error::QueueFull | summa_core::Error::CommitInProgress => {
            Status::resource_exhausted(e.to_string())
        }
        summa_core::Error::CommitFlushTimeout { .. } => Status::deadline_exceeded(e.to_string()),

        // Precondition failures
        summa_core::Error::IndexClosed => Status::failed_precondition(e.to_string()),

        // Infrastructure / transient
        summa_core::Error::Io(_) => Status::unavailable(e.to_string()),

        // Server-side errors — INTERNAL
        summa_core::Error::Corruption(_) => Status::internal(e.to_string()),
        summa_core::Error::Serialization(_) => Status::internal(e.to_string()),
        summa_core::Error::Internal(_) => Status::internal(e.to_string()),
    }
}
