use std::collections::{BTreeMap, HashMap};
use std::env;
use std::net::IpAddr;
use std::str::FromStr;
use std::sync::Mutex;

use anyhow::{Context, Result, bail, ensure};
use native_tls::{Certificate, Protocol, TlsConnector};
use postgres::config::{Host, SslMode, SslNegotiation};
use postgres::{Client, Config as PostgresConfig, NoTls, Row};
use postgres_native_tls::{MakeTlsConnector, set_postgresql_alpn};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};

use super::{DiscoveryHit, SourceSnapshot};

#[derive(Clone, Debug, PartialEq)]
pub struct MaterializedRecord {
    pub record_key: String,
    pub text: String,
    /// URI values remain opaque; materializers never impose a scheme.
    pub uris: Vec<String>,
    pub metadata: BTreeMap<String, Value>,
}

pub trait RecordMaterializer: Send + Sync {
    fn name(&self) -> &str;
    /// Serializable, secret-free materializer configuration for the build
    /// manifest.
    fn configuration(&self) -> Result<Value>;
    fn snapshot(&self) -> Result<SourceSnapshot>;
    fn materialize(&self, hits: &[DiscoveryHit]) -> Result<Vec<MaterializedRecord>>;
}

/// Materializer for discovery engines which are themselves the canonical text
/// store. It fails when any hit lacks inline content.
#[derive(Default)]
pub struct InlineRecordMaterializer;

impl RecordMaterializer for InlineRecordMaterializer {
    fn name(&self) -> &str {
        "inline"
    }

    fn configuration(&self) -> Result<Value> {
        Ok(serde_json::json!({ "type": "inline" }))
    }

    fn snapshot(&self) -> Result<SourceSnapshot> {
        Ok(SourceSnapshot {
            provider: "inline".to_owned(),
            revision: "discovery-snapshot".to_owned(),
        })
    }

    fn materialize(&self, hits: &[DiscoveryHit]) -> Result<Vec<MaterializedRecord>> {
        hits.iter()
            .map(|hit| {
                let text = hit.inline_text.clone().with_context(|| {
                    format!("discovery hit `{}` has no inline text", hit.record_key)
                })?;
                Ok(MaterializedRecord {
                    record_key: hit.record_key.clone(),
                    text,
                    uris: hit.uris.clone(),
                    metadata: hit.metadata.clone(),
                })
            })
            .collect()
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PostgresRecordMaterializerConfig {
    /// Environment variable containing the PostgreSQL DSN. Only this variable
    /// name is serialized; its value never enters the corpus manifest.
    pub connection_environment: String,
    /// Prepared-style statement accepting one `text[]` parameter containing
    /// discovery keys. Column aliases are mapped below and are not inferred.
    pub statement: String,
    pub columns: PostgresColumnMapping,
    /// Must return a remotely owned, stable dataset generation. Volatile MVCC
    /// snapshot identifiers cannot support restart across processes.
    pub snapshot_statement: String,
    /// Transport policy is mandatory. Production databases should use
    /// `verified_tls`; plaintext is restricted to a local trusted proxy or
    /// Unix socket and requires an explicit acknowledgement.
    pub transport_security: PostgresTransportSecurity,
    #[serde(default = "default_true")]
    pub require_every_record: bool,
}

fn default_true() -> bool {
    true
}

impl PostgresRecordMaterializerConfig {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            !self.connection_environment.trim().is_empty(),
            "postgres connection_environment must not be empty"
        );
        ensure!(
            !self.statement.trim().is_empty(),
            "postgres materialization statement must not be empty"
        );
        ensure!(
            !self.snapshot_statement.trim().is_empty(),
            "postgres snapshot_statement must not be empty"
        );
        self.transport_security.validate()?;
        self.columns.validate()
    }
}

/// PostgreSQL transport policy. It is deliberately separate from the secret
/// DSN so the policy is present in the reproducible corpus configuration and a
/// DSN cannot silently weaken it.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "mode", rename_all = "snake_case", deny_unknown_fields)]
pub enum PostgresTransportSecurity {
    /// Require TLS, validate the certificate chain and verify the exact DNS/IP
    /// names configured here. The DSN must contain those names in the same
    /// order; `hostaddr` may additionally select their network addresses.
    VerifiedTls {
        server_names: Vec<String>,
        trust: PostgresTlsTrust,
    },
    /// Disable TLS only between this process and a trusted local proxy (whose
    /// upstream connection is expected to provide its own security), or for a
    /// local test database. Remote TCP hosts are rejected.
    PlaintextLocalProxy { acknowledge_plaintext: bool },
}

impl PostgresTransportSecurity {
    fn validate(&self) -> Result<()> {
        match self {
            Self::VerifiedTls {
                server_names,
                trust,
            } => {
                ensure!(
                    !server_names.is_empty(),
                    "postgres verified TLS requires at least one server name"
                );
                let mut unique = std::collections::HashSet::with_capacity(server_names.len());
                for name in server_names {
                    validate_server_name(name)?;
                    ensure!(
                        unique.insert(name),
                        "postgres verified TLS server names must be unique"
                    );
                }
                trust.validate()
            }
            Self::PlaintextLocalProxy {
                acknowledge_plaintext,
            } => {
                ensure!(
                    *acknowledge_plaintext,
                    "postgres plaintext local-proxy mode requires acknowledge_plaintext=true"
                );
                Ok(())
            }
        }
    }
}

/// Certificate roots used by verified TLS. `PinnedPem` disables platform roots
/// and binds the exact PEM bytes through SHA-256, making the trust input
/// reproducible without embedding certificate contents in the manifest.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "source", rename_all = "snake_case", deny_unknown_fields)]
pub enum PostgresTlsTrust {
    System,
    PinnedPem {
        certificate_pem_environment: String,
        certificate_sha256: String,
    },
}

impl PostgresTlsTrust {
    fn validate(&self) -> Result<()> {
        if let Self::PinnedPem {
            certificate_pem_environment,
            certificate_sha256,
        } = self
        {
            ensure!(
                !certificate_pem_environment.trim().is_empty(),
                "postgres TLS certificate_pem_environment must not be empty"
            );
            validate_sha256_label(certificate_sha256, "postgres TLS certificate")?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PostgresColumnMapping {
    pub record_key: String,
    pub text: String,
    pub uris: Option<String>,
    pub metadata: Option<String>,
}

impl PostgresColumnMapping {
    fn validate(&self) -> Result<()> {
        ensure!(
            !self.record_key.trim().is_empty() && !self.text.trim().is_empty(),
            "postgres record_key and text column mappings must not be empty"
        );
        ensure!(
            self.uris
                .as_ref()
                .is_none_or(|name| !name.trim().is_empty())
                && self
                    .metadata
                    .as_ref()
                    .is_none_or(|name| !name.trim().is_empty()),
            "optional postgres column mappings must not be empty"
        );
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct PostgresRecordRow {
    pub record_key: String,
    pub text: String,
    pub uris: Vec<String>,
    pub metadata: BTreeMap<String, Value>,
}

pub trait PostgresExecutor: Send + Sync {
    fn snapshot_revision(&self) -> Result<String>;
    fn fetch(
        &self,
        statement: &str,
        columns: &PostgresColumnMapping,
        record_keys: &[String],
    ) -> Result<Vec<PostgresRecordRow>>;
}

/// One read-only, repeatable-read transaction keeps every materialization
/// query on the same PostgreSQL snapshot for the lifetime of a build.
pub struct LivePostgresExecutor {
    client: Mutex<Client>,
    snapshot_revision: String,
}

impl LivePostgresExecutor {
    pub fn connect(config: &PostgresRecordMaterializerConfig) -> Result<Self> {
        config.validate()?;
        let dsn = env::var(&config.connection_environment).with_context(|| {
            format!(
                "postgres connection environment variable `{}` is not set",
                config.connection_environment
            )
        })?;
        let connection = configured_postgres_connection(&dsn, &config.transport_security)?;
        let mut client = match &config.transport_security {
            PostgresTransportSecurity::VerifiedTls { trust, .. } => {
                let connector = postgres_tls_connector(trust, connection.get_ssl_negotiation())?;
                connection.connect(connector)
            }
            PostgresTransportSecurity::PlaintextLocalProxy { .. } => connection.connect(NoTls),
        }
        .with_context(|| {
            format!(
                "failed to connect to postgres using `{}`",
                config.connection_environment
            )
        })?;
        client
            .batch_execute("BEGIN ISOLATION LEVEL REPEATABLE READ READ ONLY")
            .context("failed to start postgres corpus snapshot")?;
        let row = client
            .query_one(&config.snapshot_statement, &[])
            .context("failed to capture postgres source snapshot")?;
        let snapshot_revision = scalar_column(&row, 0)
            .context("postgres snapshot_statement must return one scalar value")?;
        Ok(Self {
            client: Mutex::new(client),
            snapshot_revision,
        })
    }
}

fn configured_postgres_connection(
    dsn: &str,
    transport: &PostgresTransportSecurity,
) -> Result<PostgresConfig> {
    transport.validate()?;
    let mut connection = PostgresConfig::from_str(dsn).context("invalid postgres DSN")?;
    match transport {
        PostgresTransportSecurity::VerifiedTls { server_names, .. } => {
            let actual_names = connection
                .get_hosts()
                .iter()
                .map(|host| match host {
                    Host::Tcp(name) => Ok(name.as_str()),
                    #[cfg(unix)]
                    Host::Unix(_) => bail!(
                        "postgres verified TLS cannot use a Unix socket; select plaintext_local_proxy for a trusted local socket"
                    ),
                })
                .collect::<Result<Vec<_>>>()?;
            ensure!(
                actual_names == server_names.iter().map(String::as_str).collect::<Vec<_>>(),
                "postgres DSN hosts must exactly match verified TLS server_names"
            );
            connection.ssl_mode(SslMode::Require);
        }
        PostgresTransportSecurity::PlaintextLocalProxy { .. } => {
            validate_local_postgres_destination(&connection)?;
            connection.ssl_mode(SslMode::Disable);
        }
    }
    Ok(connection)
}

fn postgres_tls_connector(
    trust: &PostgresTlsTrust,
    negotiation: SslNegotiation,
) -> Result<MakeTlsConnector> {
    let mut builder = TlsConnector::builder();
    // Native TLS already defaults to TLS 1.2, but set it explicitly so a
    // platform default cannot weaken this adapter.
    builder.min_protocol_version(Some(Protocol::Tlsv12));
    match trust {
        PostgresTlsTrust::System => {}
        PostgresTlsTrust::PinnedPem {
            certificate_pem_environment,
            certificate_sha256,
        } => {
            let pem = env::var(certificate_pem_environment).with_context(|| {
                format!(
                    "postgres TLS certificate environment variable `{certificate_pem_environment}` is not set"
                )
            })?;
            let actual = format!("sha256:{:x}", Sha256::digest(pem.as_bytes()));
            ensure!(
                actual == *certificate_sha256,
                "postgres TLS certificate digest does not match certificate_sha256"
            );
            let certificates = Certificate::stack_from_pem(pem.as_bytes())
                .context("postgres TLS certificate environment does not contain valid PEM")?;
            ensure!(
                !certificates.is_empty(),
                "postgres TLS certificate PEM must contain at least one certificate"
            );
            builder.disable_built_in_roots(true);
            for certificate in certificates {
                builder.add_root_certificate(certificate);
            }
        }
    }
    if negotiation == SslNegotiation::Direct {
        set_postgresql_alpn(&mut builder);
    }
    let connector = builder
        .build()
        .context("failed to configure verified postgres TLS")?;
    Ok(MakeTlsConnector::new(connector))
}

fn validate_local_postgres_destination(connection: &PostgresConfig) -> Result<()> {
    ensure!(
        !connection.get_hosts().is_empty(),
        "postgres plaintext local-proxy mode requires an explicit loopback IP or Unix socket host"
    );
    for host in connection.get_hosts() {
        match host {
            Host::Tcp(value) => {
                let address = value.parse::<IpAddr>().with_context(|| {
                    format!(
                        "postgres plaintext local-proxy host `{value}` must be a numeric loopback address"
                    )
                })?;
                ensure!(
                    address.is_loopback(),
                    "postgres plaintext local-proxy host `{value}` is not loopback"
                );
            }
            #[cfg(unix)]
            Host::Unix(_) => {}
        }
    }
    ensure!(
        connection.get_hostaddrs().iter().all(IpAddr::is_loopback),
        "postgres plaintext local-proxy hostaddr must be loopback"
    );
    Ok(())
}

fn validate_server_name(name: &str) -> Result<()> {
    ensure!(
        !name.is_empty() && name == name.trim(),
        "postgres verified TLS server names must not be empty or padded"
    );
    ensure!(
        !name.contains('*') && !name.contains('/') && !name.contains('\0'),
        "postgres verified TLS server names must be exact DNS names or IP addresses"
    );
    if name.parse::<IpAddr>().is_ok() {
        return Ok(());
    }
    ensure!(
        name.len() <= 253 && name.is_ascii() && name == name.to_ascii_lowercase(),
        "postgres verified TLS DNS names must be lowercase ASCII and at most 253 bytes"
    );
    let canonical = name.strip_suffix('.').unwrap_or(name);
    ensure!(
        !canonical.is_empty()
            && canonical.split('.').all(|label| {
                !label.is_empty()
                    && label.len() <= 63
                    && !label.starts_with('-')
                    && !label.ends_with('-')
                    && label
                        .bytes()
                        .all(|byte| byte.is_ascii_alphanumeric() || byte == b'-')
            }),
        "postgres verified TLS server name `{name}` is not a valid DNS name"
    );
    Ok(())
}

fn validate_sha256_label(value: &str, subject: &str) -> Result<()> {
    let Some(digest) = value.strip_prefix("sha256:") else {
        bail!("{subject} digest must use sha256:<64 lowercase hex>");
    };
    ensure!(
        digest.len() == 64
            && digest
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)),
        "{subject} digest must use sha256:<64 lowercase hex>"
    );
    Ok(())
}

impl PostgresExecutor for LivePostgresExecutor {
    fn snapshot_revision(&self) -> Result<String> {
        Ok(self.snapshot_revision.clone())
    }

    fn fetch(
        &self,
        statement: &str,
        columns: &PostgresColumnMapping,
        record_keys: &[String],
    ) -> Result<Vec<PostgresRecordRow>> {
        let mut client = self
            .client
            .lock()
            .map_err(|_| anyhow::anyhow!("postgres materializer lock is poisoned"))?;
        let rows = client
            .query(statement, &[&record_keys])
            .context("postgres record materialization query failed")?;
        rows.iter().map(|row| decode_row(row, columns)).collect()
    }
}

fn decode_row(row: &Row, columns: &PostgresColumnMapping) -> Result<PostgresRecordRow> {
    let record_key = row
        .try_get::<_, String>(columns.record_key.as_str())
        .with_context(|| format!("invalid postgres `{}` column", columns.record_key))?;
    let text = row
        .try_get::<_, String>(columns.text.as_str())
        .with_context(|| format!("invalid postgres `{}` column", columns.text))?;
    let uris = match &columns.uris {
        Some(column) => decode_uris(row, column)?,
        None => Vec::new(),
    };
    let metadata = match &columns.metadata {
        Some(column) => {
            let value = row
                .try_get::<_, Value>(column.as_str())
                .with_context(|| format!("invalid postgres `{column}` JSON column"))?;
            let Value::Object(values) = value else {
                bail!("postgres metadata column `{column}` must be a JSON object");
            };
            values.into_iter().collect()
        }
        None => BTreeMap::new(),
    };
    Ok(PostgresRecordRow {
        record_key,
        text,
        uris,
        metadata,
    })
}

fn decode_uris(row: &Row, column: &str) -> Result<Vec<String>> {
    if let Ok(values) = row.try_get::<_, Vec<String>>(column) {
        return Ok(values);
    }
    let value = row
        .try_get::<_, Value>(column)
        .with_context(|| format!("postgres URI column `{column}` must be text[] or JSON"))?;
    match value {
        Value::Null => Ok(Vec::new()),
        Value::String(value) => Ok(vec![value]),
        Value::Array(values) => values
            .into_iter()
            .map(|value| {
                value
                    .as_str()
                    .map(str::to_owned)
                    .with_context(|| format!("postgres URI column `{column}` has non-string item"))
            })
            .collect(),
        _ => bail!("postgres URI column `{column}` must be a string or array"),
    }
}

fn scalar_column(row: &Row, index: usize) -> Option<String> {
    row.try_get::<_, String>(index)
        .ok()
        .or_else(|| {
            row.try_get::<_, i64>(index)
                .ok()
                .map(|value| value.to_string())
        })
        .or_else(|| {
            row.try_get::<_, i32>(index)
                .ok()
                .map(|value| value.to_string())
        })
}

pub struct PostgresRecordMaterializer<E = LivePostgresExecutor> {
    config: PostgresRecordMaterializerConfig,
    executor: E,
}

impl PostgresRecordMaterializer<LivePostgresExecutor> {
    pub fn connect(config: PostgresRecordMaterializerConfig) -> Result<Self> {
        let executor = LivePostgresExecutor::connect(&config)?;
        Ok(Self { config, executor })
    }
}

impl<E> PostgresRecordMaterializer<E>
where
    E: PostgresExecutor,
{
    pub fn with_executor(config: PostgresRecordMaterializerConfig, executor: E) -> Result<Self> {
        config.validate()?;
        Ok(Self { config, executor })
    }
}

impl<E> RecordMaterializer for PostgresRecordMaterializer<E>
where
    E: PostgresExecutor,
{
    fn name(&self) -> &str {
        "postgres"
    }

    fn configuration(&self) -> Result<Value> {
        serde_json::to_value(&self.config)
            .context("failed to serialize postgres materializer configuration")
    }

    fn snapshot(&self) -> Result<SourceSnapshot> {
        Ok(SourceSnapshot {
            provider: "postgres".to_owned(),
            revision: self.executor.snapshot_revision()?,
        })
    }

    fn materialize(&self, hits: &[DiscoveryHit]) -> Result<Vec<MaterializedRecord>> {
        let keys: Vec<_> = hits.iter().map(|hit| hit.record_key.clone()).collect();
        let rows = self
            .executor
            .fetch(&self.config.statement, &self.config.columns, &keys)?;
        let hits_by_key: HashMap<_, _> = hits
            .iter()
            .map(|hit| (hit.record_key.as_str(), hit))
            .collect();
        let mut found = std::collections::HashSet::new();
        let mut records = Vec::with_capacity(rows.len());
        for row in rows {
            let hit = hits_by_key.get(row.record_key.as_str()).with_context(|| {
                format!(
                    "postgres returned unrequested record key `{}`",
                    row.record_key
                )
            })?;
            ensure!(
                found.insert(row.record_key.clone()),
                "postgres returned duplicate record key `{}`",
                row.record_key
            );
            let mut metadata = hit.metadata.clone();
            metadata.extend(row.metadata);
            records.push(MaterializedRecord {
                record_key: row.record_key,
                text: row.text,
                uris: if row.uris.is_empty() {
                    hit.uris.clone()
                } else {
                    row.uris
                },
                metadata,
            });
        }
        if self.config.require_every_record {
            ensure!(
                found.len() == hits.len(),
                "postgres materialized {} of {} requested records",
                found.len(),
                hits.len()
            );
        }
        Ok(records)
    }
}

#[cfg(test)]
mod transport_tests {
    use super::*;

    fn verified(server_names: &[&str]) -> PostgresTransportSecurity {
        PostgresTransportSecurity::VerifiedTls {
            server_names: server_names
                .iter()
                .map(|value| (*value).to_owned())
                .collect(),
            trust: PostgresTlsTrust::System,
        }
    }

    #[test]
    fn verified_tls_forces_encryption_and_binds_dsn_hosts() {
        let connection = configured_postgres_connection(
            "host=db.example.invalid hostaddr=192.0.2.20 user=corpus sslmode=disable",
            &verified(&["db.example.invalid"]),
        )
        .unwrap();
        assert_eq!(connection.get_ssl_mode(), SslMode::Require);
        assert_eq!(
            connection.get_hosts(),
            &[Host::Tcp("db.example.invalid".to_owned())]
        );

        let error = configured_postgres_connection(
            "host=other.example.invalid user=corpus",
            &verified(&["db.example.invalid"]),
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("exactly match"), "{error}");
    }

    #[test]
    fn plaintext_policy_forces_disable_and_rejects_non_loopback_tcp() {
        let policy = PostgresTransportSecurity::PlaintextLocalProxy {
            acknowledge_plaintext: true,
        };
        let connection = configured_postgres_connection(
            "host=127.0.0.1 port=5433 user=corpus sslmode=require",
            &policy,
        )
        .unwrap();
        assert_eq!(connection.get_ssl_mode(), SslMode::Disable);

        for dsn in [
            "host=192.0.2.1 user=corpus",
            "host=localhost user=corpus",
            "user=corpus",
        ] {
            assert!(configured_postgres_connection(dsn, &policy).is_err());
        }
    }

    #[test]
    fn verified_tls_connector_can_be_built_without_connecting() {
        postgres_tls_connector(&PostgresTlsTrust::System, SslNegotiation::Postgres).unwrap();
        postgres_tls_connector(&PostgresTlsTrust::System, SslNegotiation::Direct).unwrap();
    }
}
