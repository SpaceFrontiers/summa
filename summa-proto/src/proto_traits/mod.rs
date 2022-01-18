/// Package contains casting routines for proto objects
pub mod collector;
pub mod query;
pub mod score;

pub mod shortcuts {
    pub use super::collector::shortcuts::*;
    pub use super::query::shortcuts::*;
}
