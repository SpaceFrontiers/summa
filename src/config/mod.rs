use colored::Colorize;
use serde::{Deserialize, Serialize};
use textwrap::indent;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CorsConfig {
    pub allowed_headers: Vec<String>,
    pub allowed_methods: Vec<String>,
    pub allowed_origins: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HttpConfig {
    pub address: String,
    pub cors: CorsConfig,
    pub keep_alive_secs: usize,
    pub max_body_size_mb: usize,
    pub port: usize,
    pub workers: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SearchEngineConfig {
    pub auto_commit: bool,
    pub data_path: String,
    pub default_page_size: usize,
    pub timeout_secs: u64,
    pub writer_memory_mb: usize,
    pub writer_threads: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Config {
    pub debug: bool,
    pub http: HttpConfig,
    pub log_path: String,
    pub search_engine: SearchEngineConfig,
}

impl std::fmt::Display for Config {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "{}\n{}",
            "Summa Config:".green().bold(),
            indent(&serde_yaml::to_string(&self).unwrap(), "  "),
        )
    }
}

impl Config {
    pub fn from_file(config_filepath: &str) -> Result<Self, crate::errors::Error> {
        let mut s = config::Config::new();
        s.merge(config::File::with_name(config_filepath))?;
        s.merge(config::Environment::with_prefix("SUMMA").separator("."))?;
        s.try_into()
            .map_err(|e| crate::errors::Error::ConfigError(e))
    }
}
