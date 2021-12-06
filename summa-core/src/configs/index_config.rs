use std::collections::HashSet;

use serde::{Deserialize, Serialize};

use crate::errors::BuilderError;

#[derive(Builder, Clone, Serialize, Deserialize)]
#[builder(build_fn(error = "BuilderError"))]
pub struct IndexAttributes {
    pub created_at: u64,
    #[builder(default = "Vec::new()")]
    pub default_fields: Vec<String>,
    pub primary_key: Option<String>,
    #[builder(default = "HashSet::new()")]
    pub multi_fields: HashSet<String>,
}
