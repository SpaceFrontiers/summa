use crate::configs::NetworkConfig;
use crate::errors::SummaResult;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt::Debug;
use std::marker::PhantomData;
use std::ops::Range;
use strfmt::strfmt;

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub struct Header {
    pub name: String,
    pub value: String,
}

#[async_trait]
pub trait ExternalRequest: Debug + Send + Sync {
    fn new(method: &str, url: &str, headers: &[Header]) -> Self
    where
        Self: Sized;
    fn request(&self) -> SummaResult<Vec<u8>>;
    async fn request_async(&self) -> SummaResult<Vec<u8>>;
}

pub trait ExternalRequestGenerator<TExternalRequest: ExternalRequest>: ExternalRequestGeneratorClone<TExternalRequest> + Debug + Send + Sync {
    fn new(network_config: NetworkConfig) -> Self
    where
        Self: Sized;
    fn generate(&self, file_name: &str, range: Range<usize>) -> SummaResult<TExternalRequest>;
}

pub trait ExternalRequestGeneratorClone<TExternalRequest: ExternalRequest> {
    fn box_clone(&self) -> Box<dyn ExternalRequestGenerator<TExternalRequest>>;
}

#[derive(Clone, Debug)]
pub struct DefaultExternalRequestGenerator<TExternalRequest: ExternalRequest + Clone> {
    network_config: NetworkConfig,
    _pd: PhantomData<TExternalRequest>,
}

impl<TExternalRequest: ExternalRequest + Clone + 'static> ExternalRequestGeneratorClone<TExternalRequest>
    for DefaultExternalRequestGenerator<TExternalRequest>
{
    fn box_clone(&self) -> Box<dyn ExternalRequestGenerator<TExternalRequest>> {
        Box::new((*self).clone())
    }
}

impl<TExternalRequest: ExternalRequest + Clone + 'static> ExternalRequestGenerator<TExternalRequest> for DefaultExternalRequestGenerator<TExternalRequest> {
    fn new(network_config: NetworkConfig) -> DefaultExternalRequestGenerator<TExternalRequest> {
        DefaultExternalRequestGenerator {
            network_config,
            _pd: PhantomData,
        }
    }

    fn generate(&self, file_name: &str, range: Range<usize>) -> SummaResult<TExternalRequest> {
        let mut vars = HashMap::new();
        let start = range.start.to_string();
        let end = (range.end - 1).to_string();
        vars.insert("file_name".to_string(), file_name);
        vars.insert("start".to_string(), &start);
        vars.insert("end".to_string(), &end);
        let mut headers = Vec::with_capacity(self.network_config.headers_template.len());
        for header in self.network_config.headers_template.iter() {
            headers.push(Header {
                name: header.name.clone(),
                value: strfmt(&header.value, &vars).unwrap(),
            });
        }
        Ok(TExternalRequest::new(
            &self.network_config.method,
            &strfmt(&self.network_config.url_template, &vars).unwrap(),
            &headers,
        ))
    }
}
