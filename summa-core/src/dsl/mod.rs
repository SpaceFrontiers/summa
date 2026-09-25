//! Schema and query-language front ends.
//!
//! The SDL parser builds a [`Schema`] from index definitions. The query
//! language parser turns user text into executable queries and can apply
//! field-routing rules before planning.

pub mod ql;
pub mod query_field_router;
mod schema;
pub mod sdl;

pub use ql::{ParsedQuery, QueryLanguageParser};
pub use query_field_router::{QueryFieldRouter, QueryRouterRule, RoutedQuery, RoutingMode};
pub use schema::*;
pub use sdl::{FieldDef, IndexDef, SdlParser, parse_sdl, parse_single_index};
