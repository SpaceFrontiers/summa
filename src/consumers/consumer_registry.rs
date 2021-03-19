use crate::configurator::configs::{KafkaConsumerConfig, RuntimeConfigHolder};
use crate::errors::{BadRequestError, SummaResult};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;

#[derive(Clone, Debug)]
pub struct ConsumerRegistry {
    runtime_config: Arc<RwLock<RuntimeConfigHolder>>,
}

impl ConsumerRegistry {
    pub fn new(runtime_config: &Arc<RwLock<RuntimeConfigHolder>>) -> SummaResult<ConsumerRegistry> {
        Ok(ConsumerRegistry {
            runtime_config: runtime_config.clone(),
        })
    }

    pub fn insert_consumer_config(&self, consumer_name: &str, consumer_config: &KafkaConsumerConfig) -> Option<KafkaConsumerConfig> {
        self.runtime_config
            .write()
            .autosave()
            .consumer_configs
            .insert(consumer_name.to_string(), consumer_config.clone())
    }

    pub fn delete_consumer_config(&self, consumer_name: &str) -> SummaResult<KafkaConsumerConfig> {
        Ok(self
            .runtime_config
            .write()
            .autosave()
            .consumer_configs
            .remove(consumer_name)
            .ok_or_else(|| BadRequestError::NotFoundError(consumer_name.to_string()))?)
    }

    pub fn delete_consumer_configs(&self, consumer_names: &Vec<String>) {
        let mut runtime_config = self.runtime_config.write();
        let mut runtime_config_autosave = runtime_config.autosave();
        for consumer_name in consumer_names {
            runtime_config_autosave.consumer_configs.remove(consumer_name);
        }
    }

    pub fn get_consumer_config(&self, consumer_name: &str) -> SummaResult<KafkaConsumerConfig> {
        Ok(self
            .runtime_config
            .read()
            .consumer_configs
            .get(consumer_name)
            .ok_or_else(|| BadRequestError::NotFoundError(consumer_name.to_string()))?
            .clone())
    }

    pub fn consumer_configs(&self) -> HashMap<String, KafkaConsumerConfig> {
        self.runtime_config.read().consumer_configs.clone()
    }
    pub fn get_consumer_configs_for_index(&self, index_name: &str) -> HashMap<String, KafkaConsumerConfig> {
        self.runtime_config
            .read()
            .consumer_configs
            .iter()
            .filter_map(|(consumer_name, consumer_config)| {
                if consumer_config.index_name == index_name {
                    Some((consumer_name.to_string(), consumer_config.clone()))
                } else {
                    None
                }
            })
            .collect()
    }
}
