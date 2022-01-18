/// Package contains casting routines for proto objects
pub mod collector;
pub mod query;
pub mod score;

pub mod shortcuts {
    pub use collector::shortcuts::*;
    pub use query::shortcuts::*;
}
