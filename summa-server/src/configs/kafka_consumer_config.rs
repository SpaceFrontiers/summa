use crate::errors::SummaServerResult;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ConsumerConfig {
    pub bootstrap_servers: Vec<String>,
    pub create_topics: bool,
    pub delete_topics: bool,
    pub group_id: String,
    pub max_poll_interval_ms: u32,
    pub session_timeout_ms: u32,
    pub topics: Vec<String>,
    pub threads: u32,
}

impl ConsumerConfig {
    pub fn new(bootstrap_servers: &[String], group_id: &str, mut threads: u32, topics: &[String]) -> SummaServerResult<ConsumerConfig> {
        if threads == 0 {
            threads = 1;
        }
        Ok(ConsumerConfig {
            bootstrap_servers: bootstrap_servers.to_owned(),
            create_topics: true,
            delete_topics: true,
            group_id: group_id.to_owned(),
            max_poll_interval_ms: 1800000,
            session_timeout_ms: 300000,
            threads,
            topics: topics.to_owned(),
        })
    }
}
