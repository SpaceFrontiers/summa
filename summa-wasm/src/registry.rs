//! Index registry for managing multiple indexes

use std::collections::HashMap;
use std::sync::Arc;

use parking_lot::RwLock;
use wasm_bindgen::prelude::*;

use crate::remote_index::RemoteIndex;

/// Index registry for managing multiple indexes
#[wasm_bindgen]
pub struct IndexRegistry {
    indexes: RwLock<HashMap<String, Arc<RemoteIndex>>>,
}

#[wasm_bindgen]
impl IndexRegistry {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            indexes: RwLock::new(HashMap::new()),
        }
    }

    /// Add a remote index
    #[wasm_bindgen]
    pub async fn add_remote(&self, name: String, base_url: String) -> Result<(), JsValue> {
        let mut index = RemoteIndex::new(base_url);
        index.load().await?;

        self.indexes.write().insert(name, Arc::new(index));
        Ok(())
    }

    /// Remove an index
    #[wasm_bindgen]
    pub fn remove(&self, name: &str) {
        self.indexes.write().remove(name);
    }

    /// List index names
    #[wasm_bindgen]
    pub fn list(&self) -> JsValue {
        let names: Vec<String> = self.indexes.read().keys().cloned().collect();
        serde_wasm_bindgen::to_value(&names).unwrap_or(JsValue::NULL)
    }

    /// Search an index by name
    #[wasm_bindgen]
    pub async fn search(
        &self,
        index_name: &str,
        text: String,
        limit: usize,
    ) -> Result<JsValue, JsValue> {
        let index = self
            .indexes
            .read()
            .get(index_name)
            .cloned()
            .ok_or_else(|| JsValue::from_str(&format!("Index '{}' not found", index_name)))?;

        index.search(text, limit).await
    }
}

impl Default for IndexRegistry {
    fn default() -> Self {
        Self::new()
    }
}
