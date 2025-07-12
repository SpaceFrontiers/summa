//! Resolve a model artifact location — weights, config, or tokenizer — that may
//! be local or remote to a local file path.
//!
//! Accepts a plain path (`./weights.safetensors`, `file:///…`) or a remote URI
//! (`s3://bucket/key`, `gs://bucket/key`, `http(s)://…`). Remote objects are
//! downloaded once into a cache dir (`$HERMES_CACHE`, else `$HOME/.hermes-cache`)
//! keyed by a hash of the full URI, and reused on later runs.
//!
//! Credentials follow the `object_store` provider chains: AWS_* env / IMDS for
//! S3, `GOOGLE_APPLICATION_CREDENTIALS` / gcloud ADC / metadata for GCS. Public
//! buckets over `http(s)://` need no credentials.

#[cfg(not(feature = "remote"))]
mod imp {
    use anyhow::{Result, bail};
    use std::path::PathBuf;

    pub fn resolve(uri: &str) -> Result<PathBuf> {
        if let Some(scheme) = super::remote_scheme(uri) {
            bail!(
                "'{uri}' is a remote ({scheme}) location but this binary was built without the \
                 `remote` feature; rebuild with `--features remote` or download the file first"
            );
        }
        Ok(PathBuf::from(super::strip_file_scheme(uri)))
    }
}

#[cfg(feature = "remote")]
mod imp {
    use anyhow::{Context, Result};
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    use std::path::PathBuf;

    pub fn resolve(uri: &str) -> Result<PathBuf> {
        match super::remote_scheme(uri) {
            None => Ok(PathBuf::from(super::strip_file_scheme(uri))),
            Some(_) => download_cached(uri),
        }
    }

    /// Cache location, in priority order:
    ///   1. `$HERMES_CACHE` (explicit override)
    ///   2. `$HOME/.hermes-cache` (the default — stable, discoverable)
    ///   3. OS cache dir `…/hermes-llm` (last resort if `$HOME` is unset)
    fn cache_dir() -> Result<PathBuf> {
        let base = if let Some(p) = std::env::var_os("HERMES_CACHE") {
            PathBuf::from(p)
        } else if let Some(home) = dirs::home_dir() {
            home.join(".hermes-cache")
        } else {
            dirs::cache_dir()
                .context("no $HOME or OS cache dir; set HERMES_CACHE")?
                .join("hermes-llm")
        };
        std::fs::create_dir_all(&base)
            .with_context(|| format!("creating cache dir {}", base.display()))?;
        Ok(base)
    }

    /// Cache path: `<hash-of-uri>-<original-filename>` so the extension (e.g.
    /// `.json`, `.safetensors`) is preserved for downstream loaders while the
    /// hash keeps distinct URIs (and buckets) from colliding.
    fn cache_path(uri: &str) -> Result<PathBuf> {
        let mut h = DefaultHasher::new();
        uri.hash(&mut h);
        let name = uri.rsplit(['/', '\\']).next().unwrap_or("artifact");
        let name = if name.is_empty() { "artifact" } else { name };
        Ok(cache_dir()?.join(format!("{:016x}-{name}", h.finish())))
    }

    fn download_cached(uri: &str) -> Result<PathBuf> {
        let dest = cache_path(uri)?;
        if dest.exists() {
            tracing::info!("using cached {} ({})", uri, dest.display());
            return Ok(dest);
        }
        tracing::info!("downloading {} …", uri);
        let bytes = fetch(uri)?;
        // Atomic publish: write to a temp sibling then rename, so a concurrent
        // or interrupted run never observes a partial cache file.
        let tmp = dest.with_extension("part");
        std::fs::write(&tmp, &bytes)
            .with_context(|| format!("writing cache temp {}", tmp.display()))?;
        std::fs::rename(&tmp, &dest)
            .with_context(|| format!("publishing cache file {}", dest.display()))?;
        tracing::info!(
            "downloaded {} ({:.1} MiB) → {}",
            uri,
            bytes.len() as f64 / (1024.0 * 1024.0),
            dest.display()
        );
        Ok(dest)
    }

    fn fetch(uri: &str) -> Result<Vec<u8>> {
        use object_store::ObjectStore;

        let url = url::Url::parse(uri).with_context(|| format!("parsing URI {uri}"))?;
        let (store, path): (Box<dyn ObjectStore>, object_store::path::Path) =
            object_store::parse_url(&url)
                .with_context(|| format!("no object-store backend for {uri}"))?;

        // object_store is async; run a single-threaded runtime for this blocking
        // CLI call rather than making the whole inference path async.
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .context("building download runtime")?;
        let bytes = rt.block_on(async move {
            let get = store
                .get(&path)
                .await
                .with_context(|| format!("GET {path}"))?;
            get.bytes().await.context("reading object body")
        })?;
        anyhow::Ok(bytes.to_vec())
    }
}

pub use imp::resolve;

/// The remote scheme of `uri` (`s3`, `gs`, `http`, `https`), or `None` for a
/// local path or `file://`. Bare Windows drive letters (`C:\…`) are local.
fn remote_scheme(uri: &str) -> Option<&'static str> {
    let lower = uri.to_ascii_lowercase();
    for scheme in ["s3://", "gs://", "gcs://", "http://", "https://"] {
        if lower.starts_with(scheme) {
            return Some(match scheme {
                "s3://" => "s3",
                "gs://" | "gcs://" => "gs",
                "http://" => "http",
                _ => "https",
            });
        }
    }
    None
}

fn strip_file_scheme(uri: &str) -> String {
    uri.strip_prefix("file://").unwrap_or(uri).to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn local_paths_pass_through() {
        assert_eq!(
            resolve("./w.safetensors").unwrap().to_str().unwrap(),
            "./w.safetensors"
        );
        assert_eq!(
            resolve("/abs/w.json").unwrap().to_str().unwrap(),
            "/abs/w.json"
        );
        assert_eq!(
            resolve("file:///abs/w.json").unwrap().to_str().unwrap(),
            "/abs/w.json"
        );
    }

    #[test]
    fn scheme_detection() {
        assert_eq!(remote_scheme("s3://b/k"), Some("s3"));
        assert_eq!(remote_scheme("gs://b/k"), Some("gs"));
        assert_eq!(remote_scheme("GS://B/K"), Some("gs"));
        assert_eq!(remote_scheme("https://h/k"), Some("https"));
        assert_eq!(remote_scheme("/local/path"), None);
        assert_eq!(remote_scheme("C:\\weights"), None);
    }

    #[cfg(not(feature = "remote"))]
    #[test]
    fn remote_without_feature_errors_clearly() {
        let err = resolve("s3://b/k").unwrap_err().to_string();
        assert!(err.contains("remote") && err.contains("feature"), "{err}");
    }
}
