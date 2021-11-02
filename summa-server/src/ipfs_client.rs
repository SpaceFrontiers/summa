use crate::errors::{Error, SummaServerResult};
use hyper::body::HttpBody;
use hyper::client::HttpConnector;
use hyper::header::HeaderName;
use hyper::http::uri::Builder;
use hyper::http::HeaderValue;
use hyper::{body, Body, Client, HeaderMap, Method, Request, Response, StatusCode, Uri};
use itertools::Itertools;
use izihawa_hyper_multipart::client::multipart;
use serde::{de, Deserialize, Deserializer};
use serde_json::Value;
use std::collections::HashMap;
use std::fmt::{Debug, Formatter};
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use summa_core::components::IndexFilePath;
use summa_core::configs::IpfsConfig;
use tokio::fs::File;
use tokio_util::compat::TokioAsyncReadCompatExt;
use tracing::info;

#[derive(Clone, Default)]
pub struct IpfsClient {
    http_connector: Client<HttpConnector>,
    ipfs_config: IpfsConfig,
}

#[derive(Clone, Deserialize)]
#[serde(rename_all = "PascalCase")]
pub struct AddedFile {
    pub name: String,
    pub hash: String,
    #[serde(deserialize_with = "desize")]
    pub size: u64,
}

#[derive(Clone, Deserialize)]
#[serde(rename_all = "PascalCase")]
pub struct Key {
    pub(crate) id: String,
    pub(crate) name: String,
}

impl Key {
    pub fn id(&self) -> &str {
        &self.id
    }
    pub fn name(&self) -> &str {
        &self.name
    }
}

#[derive(Clone, Deserialize)]
#[serde(rename_all = "PascalCase")]
pub struct KeyListResponse {
    pub(crate) keys: Vec<Key>,
}

#[derive(Clone, Deserialize)]
#[serde(rename_all = "PascalCase")]
pub struct NamePublishResponse {
    name: String,
    value: String,
}

#[derive(Clone, Deserialize)]
#[serde(rename_all = "PascalCase")]
pub struct NameResolveResponse {
    pub(crate) path: String,
}

#[derive(Clone, Default, Deserialize)]
#[serde(rename_all = "PascalCase")]
pub struct PinRmResponse {
    pins: Vec<String>,
}

#[derive(Clone, Deserialize)]
#[serde(rename_all = "PascalCase")]
pub struct GcRemovedItem {
    key: HashMap<String, String>,
}

fn desize<'de, D: Deserializer<'de>>(deserializer: D) -> Result<u64, D::Error> {
    Ok(match Value::deserialize(deserializer)? {
        Value::String(s) => s.parse().map_err(de::Error::custom)?,
        Value::Number(num) => num.as_u64().ok_or_else(|| de::Error::custom("Invalid number"))? as u64,
        _ => return Err(de::Error::custom("wrong type")),
    })
}

impl Debug for IpfsClient {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.ipfs_config.fmt(f)
    }
}

impl IpfsClient {
    pub fn new(ipfs_config: IpfsConfig) -> IpfsClient {
        let ipfs_client = Client::builder().build(HttpConnector::new());
        IpfsClient {
            http_connector: ipfs_client,
            ipfs_config,
        }
    }

    fn generate_uri(&self, path: &str, params: &[(&str, &str)]) -> SummaServerResult<Uri> {
        Ok(Builder::new()
            .scheme("http")
            .authority(self.ipfs_config.api_endpoint.to_string())
            .path_and_query(format!(
                "{}?{}",
                path,
                params.iter().map(|(name, value)| format!("{}={}", name, value)).join("&")
            ))
            .build()?)
    }

    pub async fn parse_response<B, E>(&self, response: Response<B>) -> SummaServerResult<String>
    where
        B: HttpBody<Error = E> + Send + 'static,
        E: Debug,
    {
        let status = response.status();
        let body = response.into_body();
        let text = String::from_utf8(
            body::to_bytes(body)
                .await
                .map_err(|e| Error::UpstreamHttpStatus(status, format!("{:?}", e)))?
                .to_vec(),
        )
        .map_err(|e| Error::Utf8(e.utf8_error()))?;
        if status != StatusCode::OK {
            return Err(Error::UpstreamHttpStatus(status, text));
        }
        Ok(text)
    }

    pub async fn request(&self, uri: Uri) -> SummaServerResult<Response<Body>> {
        let request = Request::builder().method(Method::POST).uri(uri.clone()).body(Body::empty())?;
        info!(action = "ipfs_request", request = ?request);
        Ok(self.http_connector.request(request).await?)
    }

    pub async fn key_gen(&self, name: &str) -> SummaServerResult<Key> {
        let response = self.request(self.generate_uri("/api/v0/key/gen", &[("arg", name)])?).await?;
        let text = self.parse_response(response).await?;
        Ok(serde_json::from_str(&text)?)
    }

    pub async fn key_list(&self) -> SummaServerResult<KeyListResponse> {
        let response = self.request(self.generate_uri("/api/v0/key/list", &[])?).await?;
        let text = self.parse_response(response).await?;
        Ok(serde_json::from_str(&text)?)
    }

    pub async fn name_publish(&self, hash: &str, key_name: &str) -> SummaServerResult<NamePublishResponse> {
        let response = self
            .request(self.generate_uri("/api/v0/name/publish", &[("arg", hash), ("key", key_name)])?)
            .await?;
        let text = self.parse_response(response).await?;
        Ok(serde_json::from_str(&text)?)
    }

    pub async fn name_resolve(&self, name: &str) -> SummaServerResult<NameResolveResponse> {
        let response = self.request(self.generate_uri("/api/v0/name/resolve", &[("arg", name)])?).await?;
        let text = self.parse_response(response).await?;
        Ok(serde_json::from_str(&text)?)
    }

    pub async fn pin_rm(&self, hash: &str) -> SummaServerResult<Vec<PinRmResponse>> {
        let response = self
            .request(self.generate_uri("/api/v0/pin/rm", &[("arg", hash), ("recursive", "true")])?)
            .await?;
        match self.parse_response(response).await {
            Err(Error::UpstreamHttpStatus(StatusCode::INTERNAL_SERVER_ERROR, _)) => Ok(vec![]),
            Ok(text) => Ok(serde_json::from_str(&text).unwrap()),
            Err(e) => Err(e),
        }
    }

    pub async fn repo_gc(&self) -> SummaServerResult<Vec<GcRemovedItem>> {
        let response = self.request(self.generate_uri("/api/v0/repo/gc", &[])?).await?;
        let text = self.parse_response(response).await?;
        Ok(text
            .lines()
            .filter(|line| !line.is_empty())
            .map(|line| serde_json::from_str(line).unwrap())
            .collect::<Vec<_>>())
    }

    pub async fn files_mkdir<P1: AsRef<Path> + Debug>(&self, directory: P1) -> SummaServerResult<()> {
        let response = self
            .request(self.generate_uri("/api/v0/files/mkdir", &[("arg", &directory.as_ref().to_string_lossy()), ("parent", "true")])?)
            .await?;
        self.parse_response(response).await?;
        Ok(())
    }

    pub async fn files_cp<P1: AsRef<Path> + Debug>(&self, directory: P1) -> SummaServerResult<()> {
        let response = self
            .request(self.generate_uri("/api/v0/files/cp", &[("arg", &directory.as_ref().to_string_lossy()), ("parent", "true")])?)
            .await?;
        self.parse_response(response).await?;
        Ok(())
    }

    pub async fn add<P1: AsRef<Path> + Debug>(&self, directory: P1, index_file_paths: &[IndexFilePath], no_copy: bool) -> SummaServerResult<Vec<AddedFile>> {
        let full_directory_path = directory.as_ref();
        let directory_name = PathBuf::from(full_directory_path.file_name().unwrap());
        let mut form = multipart::Form::default();

        info!(action = "prepare_request", directory = ?directory, index_file_paths = ?index_file_paths);

        let uri = self.generate_uri(
            "/api/v0/add",
            &[
                ("nocopy", if no_copy { "true" } else { "false" }),
                ("cid-version", "1"),
                ("hash", "blake2b-256"),
                ("chunker", "size-65536"),
            ],
        )?;
        for index_file_path in index_file_paths {
            let abs_path = full_directory_path.join(index_file_path.path());

            let headers = no_copy.then(|| {
                let mut headers = HeaderMap::new();
                headers.insert(
                    HeaderName::from_str("Abspath").unwrap(),
                    HeaderValue::from_str(&abs_path.to_string_lossy()).unwrap(),
                );
                headers
            });

            // ToDo: This part is worth to redesign. `meta.json` file should be created with appropriate permissions
            // ToDo: but it requires changes in `tempfile` crate to add such possibility and changes to `tantivy` to use
            // ToDo: the possibility
            tokio::fs::set_permissions(&abs_path, std::fs::Permissions::from_mode(0o644)).await.unwrap();
            form.add_async_reader_file(
                "file",
                File::open(abs_path).await.unwrap().compat(),
                directory_name.join(index_file_path.path()).to_string_lossy(),
                headers,
            );
        }

        let request = form
            .set_body_convert::<Body, multipart::Body>(Request::builder().method(Method::POST).uri(uri.clone()))
            .unwrap();

        let response = self.http_connector.request(request).await?;
        let text = self.parse_response(response).await?;
        let add_file_responses = text
            .lines()
            .filter(|line| !line.is_empty())
            .map(|line| serde_json::from_str::<AddedFile>(line).unwrap())
            .collect::<Vec<_>>();
        Ok(add_file_responses)
    }
}
