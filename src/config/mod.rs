use colored::Colorize;
use serde::{Deserialize, Serialize};
use textwrap::indent;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HttpConfig {
    pub bind_addr: String,
    pub keep_alive_secs: usize,
    pub max_body_size_mb: usize,
    pub workers: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SearchEngineConfig {
    pub data_path: String,
    pub default_page_size: usize,
    pub timeout_secs: u64,
    pub writer_memory_mb: usize,
    pub writer_threads: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Config {
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
            indent(
                &serde_yaml::to_string(&self)
                    .map_err(|e| crate::errors::ConfigError::YamlError(e))
                    .unwrap(),
                "  "
            ),
        )
    }
}

impl Config {
    pub fn from_reader<T: std::io::Read>(mut reader: T) -> Result<Self, crate::errors::Error> {
        let mut buffer = String::new();
        reader.read_to_string(&mut buffer)?;

        for (key, value) in std::env::vars() {
            let re = regex::Regex::new(&format!(r"\{{\{{\s*{}\s*\}}\}}", key)).unwrap();
            buffer = re
                .replace_all(&buffer, |_caps: &regex::Captures| &value)
                .to_string();
        }

        Ok(serde_yaml::from_str(&buffer).map_err(|e| crate::errors::ConfigError::YamlError(e))?)
    }
}
