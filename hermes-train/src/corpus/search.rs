use std::collections::BTreeMap;
use std::env;
use std::thread;
use std::time::Duration;

use anyhow::{Context, Result, bail, ensure};
use serde::{Deserialize, Serialize};
use serde_json::{Number, Value};

use super::DiscoveryQuery;

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct SourceSnapshot {
    pub provider: String,
    pub revision: String,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct DiscoveryHit {
    pub record_key: String,
    pub score: f64,
    /// URI values are opaque metadata. No scheme or prefix is interpreted.
    pub uris: Vec<String>,
    pub metadata: BTreeMap<String, Value>,
    pub inline_text: Option<String>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct DiscoveryPage {
    pub hits: Vec<DiscoveryHit>,
    pub total_hits: Option<u64>,
    /// Identity returned by the remote backend for this exact page. Pipelines
    /// reject a missing or changed identity before accepting any page content.
    pub snapshot: SourceSnapshot,
}

/// Backend-neutral record discovery. Implementations return stable record
/// keys and lightweight provenance, never assumed canonical text.
pub trait SearchBackend: Send + Sync {
    fn name(&self) -> &str;
    /// Serializable, secret-free provider configuration for the immutable
    /// build manifest.
    fn configuration(&self) -> Result<Value>;
    /// Expected immutable source identity. Each returned page must carry the
    /// independently obtained remote proof in [`DiscoveryPage::snapshot`].
    fn snapshot(&self) -> Result<SourceSnapshot>;
    fn page_size(&self) -> usize;
    fn discover(
        &self,
        query: &DiscoveryQuery,
        offset: usize,
        limit: usize,
    ) -> Result<DiscoveryPage>;
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SearchApiConfig {
    pub endpoint: String,
    /// Complete provider request. Indexes, fields, clauses, filters, fusion
    /// options, projections, and provider-specific values live here rather
    /// than in pipeline code.
    pub request_template: Value,
    pub request_mapping: SearchApiRequestMapping,
    pub response_mapping: SearchApiResponseMapping,
    pub fusion: SearchApiFusionContract,
    pub snapshot: SearchApiSnapshotContract,
    pub page_size: usize,
    #[serde(default = "default_timeout_seconds")]
    pub timeout_seconds: u64,
    #[serde(default = "default_retries")]
    pub max_retries: usize,
    #[serde(default = "default_retry_initial_ms")]
    pub retry_initial_ms: u64,
    #[serde(default = "default_retry_max_ms")]
    pub retry_max_ms: u64,
    pub auth: Option<SearchApiAuthConfig>,
}

fn default_timeout_seconds() -> u64 {
    180
}

fn default_retries() -> usize {
    3
}

fn default_retry_initial_ms() -> u64 {
    1_000
}

fn default_retry_max_ms() -> u64 {
    30_000
}

impl SearchApiConfig {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.endpoint.starts_with("http://") || self.endpoint.starts_with("https://"),
            "search_api endpoint must use HTTP or HTTPS"
        );
        ensure!(self.page_size > 0, "search_api page_size must be positive");
        ensure!(
            self.timeout_seconds > 0,
            "search_api timeout_seconds must be positive"
        );
        ensure!(
            self.retry_initial_ms > 0 && self.retry_initial_ms <= self.retry_max_ms,
            "search_api retry delays must be positive and ordered"
        );
        self.request_mapping.validate(&self.request_template)?;
        self.response_mapping.validate()?;
        self.fusion.validate(&self.request_template)?;
        self.snapshot.validate(&self.request_template)?;
        if let Some(auth) = &self.auth {
            ensure!(
                !auth.header.trim().is_empty() && !auth.environment.trim().is_empty(),
                "search_api auth header and environment must not be empty"
            );
        }
        Ok(())
    }
}

/// Provider-neutral wire contract for an immutable remote index generation.
/// The expected revision is written into every request and both provider and
/// revision must be echoed by every response. Merely naming a revision in the
/// local recipe is deliberately insufficient proof.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SearchApiSnapshotContract {
    pub provider: String,
    pub revision: String,
    pub request_revision_pointer: String,
    pub response_provider_pointer: String,
    pub response_revision_pointer: String,
}

impl SearchApiSnapshotContract {
    fn identity(&self) -> SourceSnapshot {
        SourceSnapshot {
            provider: self.provider.clone(),
            revision: self.revision.clone(),
        }
    }

    fn validate(&self, template: &Value) -> Result<()> {
        ensure!(
            !self.provider.trim().is_empty() && !self.revision.trim().is_empty(),
            "search_api snapshot provider and revision must not be empty"
        );
        ensure_pointer(template, &self.request_revision_pointer)
            .context("invalid search_api snapshot request revision pointer")?;
        validate_json_pointer(&self.response_provider_pointer)
            .context("invalid search_api snapshot response provider pointer")?;
        validate_json_pointer(&self.response_revision_pointer)
            .context("invalid search_api snapshot response revision pointer")?;
        ensure!(
            self.response_provider_pointer != self.response_revision_pointer,
            "search_api snapshot provider and revision response pointers must be distinct"
        );
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SearchApiAuthConfig {
    pub header: String,
    /// Name of the environment variable containing the secret. The secret
    /// itself is never part of serializable configuration or manifests.
    pub environment: String,
    #[serde(default)]
    pub prefix: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SearchApiRequestMapping {
    pub query_pointer: String,
    pub offset_pointer: String,
    pub limit_pointer: String,
    #[serde(default)]
    pub parameter_pointers: BTreeMap<String, String>,
    /// All listed switches are overwritten with `false` for every request.
    /// The production recipe includes both rerank and cross-rerank switches.
    pub disabled_reranker_pointers: Vec<String>,
    pub return_documents_pointer: Option<String>,
}

impl SearchApiRequestMapping {
    fn validate(&self, template: &Value) -> Result<()> {
        for (name, pointer) in [
            ("query", &self.query_pointer),
            ("offset", &self.offset_pointer),
            ("limit", &self.limit_pointer),
        ] {
            ensure_pointer(template, pointer)
                .with_context(|| format!("invalid search_api {name} pointer"))?;
        }
        ensure!(
            !self.disabled_reranker_pointers.is_empty(),
            "search_api must configure at least one disabled reranker pointer"
        );
        for pointer in &self.disabled_reranker_pointers {
            ensure_pointer(template, pointer).with_context(|| {
                format!("invalid disabled search_api reranker pointer `{pointer}`")
            })?;
        }
        if let Some(pointer) = &self.return_documents_pointer {
            ensure_pointer(template, pointer)
                .with_context(|| format!("invalid return_documents pointer `{pointer}`"))?;
        }
        for (parameter, pointer) in &self.parameter_pointers {
            ensure!(
                !parameter.trim().is_empty(),
                "search_api parameter names must not be empty"
            );
            ensure_pointer(template, pointer).with_context(|| {
                format!("invalid pointer `{pointer}` for query parameter `{parameter}`")
            })?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SearchApiResponseMapping {
    pub hits_pointer: String,
    pub total_hits_pointer: Option<String>,
    /// The remaining pointers are relative to one hit object.
    pub record_key_pointer: String,
    pub score_pointer: Option<String>,
    pub uris_pointer: Option<String>,
    pub inline_text_pointer: Option<String>,
    #[serde(default)]
    pub metadata_pointers: BTreeMap<String, String>,
}

impl SearchApiResponseMapping {
    fn validate(&self) -> Result<()> {
        for (label, pointer) in [
            ("hits", self.hits_pointer.as_str()),
            ("record key", self.record_key_pointer.as_str()),
        ] {
            validate_json_pointer(pointer)
                .with_context(|| format!("invalid search_api {label} pointer"))?;
        }
        for pointer in self
            .total_hits_pointer
            .iter()
            .chain(self.score_pointer.iter())
            .chain(self.uris_pointer.iter())
            .chain(self.inline_text_pointer.iter())
            .chain(self.metadata_pointers.values())
        {
            validate_json_pointer(pointer)?;
        }
        Ok(())
    }
}

/// Declares that the configured Search API request executes sparse+dense
/// fusion. Field names remain configuration data because deployments use
/// different schemas and the remote API may own query embedding.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SearchApiFusionContract {
    pub marker_pointer: String,
    pub marker_value: Value,
    pub sparse: SearchApiFusionBranch,
    pub dense: SearchApiFusionBranch,
}

/// One provider-defined branch of a sparse+dense fusion request.  Both JSON
/// pointers are configuration because Search APIs do not share a request
/// schema.  The clause itself stays in `request_template`; the field value is
/// written on every request so the manifest cannot claim one field while the
/// remote request silently uses another.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SearchApiFusionBranch {
    pub clause_pointer: String,
    pub vector_field_pointer: String,
    pub vector_field: String,
}

impl SearchApiFusionContract {
    fn validate(&self, template: &Value) -> Result<()> {
        ensure_pointer(template, &self.marker_pointer)?;
        ensure!(
            template.pointer(&self.marker_pointer) == Some(&self.marker_value),
            "search_api fusion marker does not match request_template"
        );
        self.sparse.validate("sparse", template)?;
        self.dense.validate("dense", template)?;
        ensure!(
            self.sparse.clause_pointer != self.dense.clause_pointer,
            "search_api sparse and dense fusion clauses must be distinct"
        );
        ensure!(
            self.sparse.vector_field_pointer != self.dense.vector_field_pointer,
            "search_api sparse and dense vector-field pointers must be distinct"
        );
        Ok(())
    }
}

impl SearchApiFusionBranch {
    fn validate(&self, name: &str, template: &Value) -> Result<()> {
        ensure_pointer(template, &self.clause_pointer)
            .with_context(|| format!("invalid search_api {name} fusion clause pointer"))?;
        ensure!(
            !template
                .pointer(&self.clause_pointer)
                .is_none_or(Value::is_null),
            "search_api {name} fusion clause must not be null"
        );
        ensure_pointer(template, &self.vector_field_pointer)
            .with_context(|| format!("invalid search_api {name} vector-field pointer"))?;
        let field_prefix = if self.clause_pointer.is_empty() {
            "/".to_owned()
        } else {
            format!("{}/", self.clause_pointer.trim_end_matches('/'))
        };
        ensure!(
            self.vector_field_pointer.starts_with(&field_prefix),
            "search_api {name} vector-field pointer must be inside its fusion clause"
        );
        ensure!(
            !self.vector_field.trim().is_empty(),
            "search_api {name} vector field must not be empty"
        );
        Ok(())
    }
}

fn validate_json_pointer(pointer: &str) -> Result<()> {
    ensure!(
        pointer.is_empty() || pointer.starts_with('/'),
        "`{pointer}` is not an RFC 6901 JSON pointer"
    );
    Ok(())
}

fn ensure_pointer(value: &Value, pointer: &str) -> Result<()> {
    validate_json_pointer(pointer)?;
    ensure!(
        value.pointer(pointer).is_some(),
        "JSON pointer `{pointer}` does not exist in request_template"
    );
    Ok(())
}

pub trait SearchApiTransport: Send + Sync {
    fn post_json(&self, endpoint: &str, body: &Value) -> Result<Value>;
}

/// Blocking HTTP transport used by corpus builder processes. The builder is
/// normally run outside the GPU training loop; retries are bounded and all
/// authentication values are resolved only at construction time.
pub struct UreqSearchApiTransport {
    agent: ureq::Agent,
    auth: Option<(String, String)>,
    max_retries: usize,
    retry_initial_ms: u64,
    retry_max_ms: u64,
}

impl UreqSearchApiTransport {
    pub fn from_config(config: &SearchApiConfig) -> Result<Self> {
        config.validate()?;
        let timeout = Duration::from_secs(config.timeout_seconds);
        let agent = ureq::AgentBuilder::new().timeout(timeout).build();
        let auth = config
            .auth
            .as_ref()
            .map(|auth| {
                let secret = env::var(&auth.environment).with_context(|| {
                    format!(
                        "search_api authentication environment variable `{}` is not set",
                        auth.environment
                    )
                })?;
                Ok::<_, anyhow::Error>((auth.header.clone(), format!("{}{}", auth.prefix, secret)))
            })
            .transpose()?;
        Ok(Self {
            agent,
            auth,
            max_retries: config.max_retries,
            retry_initial_ms: config.retry_initial_ms,
            retry_max_ms: config.retry_max_ms,
        })
    }
}

impl SearchApiTransport for UreqSearchApiTransport {
    fn post_json(&self, endpoint: &str, body: &Value) -> Result<Value> {
        for attempt in 0..=self.max_retries {
            let mut request = self
                .agent
                .post(endpoint)
                .set("Content-Type", "application/json")
                .set("X-Request-Source", "hermes-train-corpus");
            if let Some((header, value)) = &self.auth {
                request = request.set(header, value);
            }
            match request.send_json(body) {
                Ok(response) => {
                    return response
                        .into_json::<Value>()
                        .context("search_api returned invalid JSON");
                }
                Err(ureq::Error::Status(status, response)) => {
                    let retryable = matches!(status, 429 | 500 | 502 | 503 | 504);
                    if !retryable || attempt == self.max_retries {
                        bail!(
                            "search_api request failed with HTTP {status}: {}",
                            response.status_text()
                        );
                    }
                }
                Err(ureq::Error::Transport(error)) => {
                    if attempt == self.max_retries {
                        return Err(error).context("search_api transport failed");
                    }
                }
            }
            let multiplier = 1_u64.checked_shl(attempt as u32).unwrap_or(u64::MAX);
            let delay = self
                .retry_initial_ms
                .saturating_mul(multiplier)
                .min(self.retry_max_ms);
            thread::sleep(Duration::from_millis(delay));
        }
        unreachable!("bounded search_api retry loop always returns")
    }
}

pub struct SearchApiClient<T = UreqSearchApiTransport> {
    config: SearchApiConfig,
    transport: T,
}

impl SearchApiClient<UreqSearchApiTransport> {
    pub fn connect(config: SearchApiConfig) -> Result<Self> {
        let transport = UreqSearchApiTransport::from_config(&config)?;
        Ok(Self { config, transport })
    }
}

impl<T> SearchApiClient<T>
where
    T: SearchApiTransport,
{
    pub fn with_transport(config: SearchApiConfig, transport: T) -> Result<Self> {
        config.validate()?;
        Ok(Self { config, transport })
    }

    pub fn request_for(
        &self,
        query: &DiscoveryQuery,
        offset: usize,
        limit: usize,
    ) -> Result<Value> {
        ensure!(limit > 0, "search_api request limit must be positive");
        let mut request = self.config.request_template.clone();
        set_pointer(
            &mut request,
            &self.config.request_mapping.query_pointer,
            Value::String(query.text.clone()),
        )?;
        set_pointer(
            &mut request,
            &self.config.request_mapping.offset_pointer,
            Value::Number(Number::from(u64::try_from(offset)?)),
        )?;
        set_pointer(
            &mut request,
            &self.config.request_mapping.limit_pointer,
            Value::Number(Number::from(u64::try_from(limit)?)),
        )?;
        set_pointer(
            &mut request,
            &self.config.snapshot.request_revision_pointer,
            Value::String(self.config.snapshot.revision.clone()),
        )?;
        for (name, value) in &query.parameters {
            let pointer = self
                .config
                .request_mapping
                .parameter_pointers
                .get(name)
                .with_context(|| {
                    format!(
                        "query `{}` uses unmapped search_api parameter `{name}`",
                        query.name
                    )
                })?;
            set_pointer(&mut request, pointer, value.clone())?;
        }
        for pointer in &self.config.request_mapping.disabled_reranker_pointers {
            set_pointer(&mut request, pointer, Value::Bool(false))?;
        }
        if let Some(pointer) = &self.config.request_mapping.return_documents_pointer {
            set_pointer(&mut request, pointer, Value::Bool(false))?;
        }
        set_pointer(
            &mut request,
            &self.config.fusion.marker_pointer,
            self.config.fusion.marker_value.clone(),
        )?;
        set_pointer(
            &mut request,
            &self.config.fusion.sparse.vector_field_pointer,
            Value::String(self.config.fusion.sparse.vector_field.clone()),
        )?;
        set_pointer(
            &mut request,
            &self.config.fusion.dense.vector_field_pointer,
            Value::String(self.config.fusion.dense.vector_field.clone()),
        )?;
        // These checks run after all query-specific substitutions. They make
        // accidental replacement/removal of either provider-defined clause a
        // request-time error rather than an implicit single-vector search.
        for (name, pointer) in [
            ("sparse", &self.config.fusion.sparse.clause_pointer),
            ("dense", &self.config.fusion.dense.clause_pointer),
        ] {
            ensure!(
                !request.pointer(pointer).is_none_or(Value::is_null),
                "search_api {name} fusion clause `{pointer}` is absent from request"
            );
        }
        Ok(request)
    }

    fn parse_response(&self, response: &Value) -> Result<DiscoveryPage> {
        let mapping = &self.config.response_mapping;
        let snapshot = SourceSnapshot {
            provider: scalar_string(
                response.pointer(&self.config.snapshot.response_provider_pointer),
            )
            .context("search_api response has no scalar snapshot provider proof")?,
            revision: scalar_string(
                response.pointer(&self.config.snapshot.response_revision_pointer),
            )
            .context("search_api response has no scalar snapshot revision proof")?,
        };
        ensure!(
            snapshot == self.config.snapshot.identity(),
            "search_api response snapshot mismatch: expected {}@{}, got {}@{}",
            self.config.snapshot.provider,
            self.config.snapshot.revision,
            snapshot.provider,
            snapshot.revision
        );
        let raw_hits = response
            .pointer(&mapping.hits_pointer)
            .and_then(Value::as_array)
            .with_context(|| {
                format!(
                    "search_api response `{}` is absent or not an array",
                    mapping.hits_pointer
                )
            })?;
        let mut hits = Vec::with_capacity(raw_hits.len());
        for raw in raw_hits {
            let record_key = scalar_string(raw.pointer(&mapping.record_key_pointer))
                .context("search_api hit has no scalar record key")?;
            ensure!(
                !record_key.is_empty(),
                "search_api hit record key must not be empty"
            );
            let score = mapping
                .score_pointer
                .as_ref()
                .and_then(|pointer| raw.pointer(pointer))
                .and_then(Value::as_f64)
                .unwrap_or(0.0);
            let score = if score.is_finite() { score } else { 0.0 };
            let uris = mapping
                .uris_pointer
                .as_ref()
                .and_then(|pointer| raw.pointer(pointer))
                .map(strings)
                .transpose()?
                .unwrap_or_default();
            let inline_text = mapping
                .inline_text_pointer
                .as_ref()
                .and_then(|pointer| raw.pointer(pointer))
                .and_then(Value::as_str)
                .map(str::to_owned);
            let metadata = mapping
                .metadata_pointers
                .iter()
                .filter_map(|(name, pointer)| {
                    raw.pointer(pointer)
                        .cloned()
                        .map(|value| (name.clone(), value))
                })
                .collect();
            hits.push(DiscoveryHit {
                record_key,
                score,
                uris,
                metadata,
                inline_text,
            });
        }
        let total_hits = mapping
            .total_hits_pointer
            .as_ref()
            .and_then(|pointer| response.pointer(pointer))
            .and_then(Value::as_u64);
        Ok(DiscoveryPage {
            hits,
            total_hits,
            snapshot,
        })
    }
}

impl<T> SearchBackend for SearchApiClient<T>
where
    T: SearchApiTransport,
{
    fn name(&self) -> &str {
        "search_api"
    }

    fn configuration(&self) -> Result<Value> {
        serde_json::to_value(&self.config).context("failed to serialize search_api configuration")
    }

    fn snapshot(&self) -> Result<SourceSnapshot> {
        Ok(self.config.snapshot.identity())
    }

    fn page_size(&self) -> usize {
        self.config.page_size
    }

    fn discover(
        &self,
        query: &DiscoveryQuery,
        offset: usize,
        limit: usize,
    ) -> Result<DiscoveryPage> {
        let request = self.request_for(query, offset, limit)?;
        let response = self
            .transport
            .post_json(&self.config.endpoint, &request)
            .with_context(|| format!("search_api query `{}` failed", query.name))?;
        self.parse_response(&response)
            .with_context(|| format!("invalid search_api response for query `{}`", query.name))
    }
}

fn set_pointer(root: &mut Value, pointer: &str, value: Value) -> Result<()> {
    let target = root
        .pointer_mut(pointer)
        .with_context(|| format!("JSON pointer `{pointer}` does not exist"))?;
    *target = value;
    Ok(())
}

fn scalar_string(value: Option<&Value>) -> Option<String> {
    match value? {
        Value::String(value) => Some(value.clone()),
        Value::Number(value) => Some(value.to_string()),
        Value::Bool(value) => Some(value.to_string()),
        _ => None,
    }
}

fn strings(value: &Value) -> Result<Vec<String>> {
    match value {
        Value::String(value) => Ok(vec![value.clone()]),
        Value::Array(values) => values
            .iter()
            .map(|value| {
                value
                    .as_str()
                    .map(str::to_owned)
                    .context("URI array contains a non-string value")
            })
            .collect(),
        _ => bail!("URI mapping must resolve to a string or string array"),
    }
}
