use crate::configs::IpfsConfig;
use crate::errors::SummaResult;
use crate::search_engine::IndexHolder;
use crate::utils::sync::Handler;
use hyper::client::HttpConnector;
use hyper::header::HeaderName;
use hyper::http::uri::Builder;
use hyper::http::HeaderValue;
use hyper::{body, Client, HeaderMap, Method, Request};
use izihawa_common_multipart::client::multipart::{content_disposition, RandomAsciiGenerator, CONTENT_TYPE_APPLICATION_X_DIRECTORY};
use izihawa_hyper_multipart::client::multipart;
use serde::Deserialize;
use std::fmt::{Debug, Formatter};
use std::path::Path;
use std::str::FromStr;
use tantivy::SegmentAttribute::ConjunctiveBool;
use tantivy::SegmentAttributes;
use tokio::fs::File;
use tokio_util::compat::TokioAsyncReadCompatExt;

#[derive(Clone, Default)]
pub struct BeaconService {
    ipfs_client: Client<HttpConnector>,
    ipfs_config: IpfsConfig,
}

#[derive(Clone, Deserialize)]
#[serde(rename_all = "PascalCase")]
pub struct AddFilesResponse {
    pub hash: String,
}

impl Debug for BeaconService {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.ipfs_config.fmt(f)
    }
}

impl BeaconService {
    pub fn new(ipfs_config: IpfsConfig) -> BeaconService {
        let ipfs_client = Client::builder().build(HttpConnector::new());
        BeaconService { ipfs_client, ipfs_config }
    }

    pub async fn publish_index(&self, index_holder: Handler<IndexHolder>) -> SummaResult<String> {
        let index_updater = index_holder.index_updater();
        let files = {
            let mut index_updater = index_updater.write().await;
            index_updater
                .vacuum(Some(SegmentAttributes::from_iter(vec![("is_freeze", ConjunctiveBool(true))].into_iter())))
                .await?;
            index_updater.get_frozen_segments().await?
        };
        let parent_folder = files[0].parent().unwrap();
        let add_files_response = self.add_files(parent_folder, &files).await?;
        Ok(add_files_response.hash)
    }

    async fn add_files<P1: AsRef<Path>, P2: AsRef<Path>>(&self, directory: P1, files: &[P2]) -> SummaResult<AddFilesResponse> {
        let directory_name = directory.as_ref().to_string_lossy().to_string();
        let content_disposition = content_disposition("file", Some(directory_name));

        let mut form = multipart::Form::with_headers::<RandomAsciiGenerator>(Some(CONTENT_TYPE_APPLICATION_X_DIRECTORY.to_string()), Some(content_disposition));

        for file in files {
            let mut headers = HeaderMap::new();
            let file_name = file.as_ref().to_string_lossy();
            let header_value = HeaderValue::from_str(file_name.as_ref()).unwrap();
            headers.insert(HeaderName::from_str("Abspath").unwrap(), header_value);
            form.add_async_reader_file("file", File::open(file).await.unwrap().compat(), file_name.as_ref(), Some(headers));
        }

        let uri = Builder::new()
            .scheme("http")
            .authority(self.ipfs_config.api_endpoint.to_string())
            .path_and_query("/api/v0/add?nocopy=true")
            .build()
            .unwrap();

        let req_builder = Request::builder().method(Method::POST).uri(uri);

        let req = form.set_body_convert::<hyper::Body, multipart::Body>(req_builder).unwrap();
        let resp = self.ipfs_client.request(req).await?;
        let add_response: AddFilesResponse = serde_json::from_slice(&body::to_bytes(resp.into_body()).await?).unwrap();
        Ok(add_response)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use tokio::io::AsyncWriteExt;

    #[tokio::test]
    async fn test_base() {
        let beacon_service = BeaconService::new(IpfsConfig::default());
        let root_path = tempdir::TempDir::new("test_base").unwrap();
        let test_file = root_path.path().join("test_file.txt");
        let mut file = File::create(&test_file).await.unwrap();
        file.write_all(b"Hello, world!").await.unwrap();
        beacon_service.add_files(PathBuf::from("test_base"), &[test_file]).await.unwrap();
    }
}
