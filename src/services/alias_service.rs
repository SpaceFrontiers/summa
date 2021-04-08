use crate::configs::RuntimeConfigHolder;
use parking_lot::{MappedRwLockReadGuard, RwLock, RwLockReadGuard};
use std::collections::HashMap;
use std::sync::Arc;

/// Managing aliases for indices.
///
/// Main goal of aliases is keeping a stable name for using in [Search API](crate::apis::search::SearchApiImpl)
#[derive(Clone, Debug)]
pub struct AliasService {
    runtime_config: Arc<RwLock<RuntimeConfigHolder>>,
}

impl AliasService {
    /// New `AliasService`
    pub fn new(runtime_config: &Arc<RwLock<RuntimeConfigHolder>>) -> AliasService {
        AliasService {
            runtime_config: runtime_config.clone(),
        }
    }

    /// All aliases
    pub fn indices_aliases(&self) -> MappedRwLockReadGuard<'_, HashMap<String, String>> {
        RwLockReadGuard::map(self.runtime_config.read(), |f| &f.aliases)
    }

    /// Copy aliases for the index
    pub fn get_index_aliases_for_index(&self, index_name: &str) -> Vec<String> {
        self.indices_aliases()
            .iter()
            .filter(|(_, v)| *v == index_name)
            .map(|(k, _)| k.clone())
            .collect::<Vec<String>>()
    }

    /// Find index by alias
    pub fn resolve_index_alias(&self, alias: &str) -> Option<String> {
        self.indices_aliases().get(alias).cloned()
    }

    /// Set new alias for index
    pub fn set_index_alias(&self, alias: &str, index_name: &str) -> Option<String> {
        self.runtime_config.write().autosave().aliases.insert(alias.to_string(), index_name.to_string())
    }

    /// Delete all aliases listed in `index_aliases`
    pub fn delete_index_aliases(&self, index_aliases: &Vec<String>) {
        let mut runtime_config = self.runtime_config.write();
        let mut runtime_config_autosave = runtime_config.autosave();
        for alias in index_aliases {
            runtime_config_autosave.aliases.remove(alias);
        }
    }
}
