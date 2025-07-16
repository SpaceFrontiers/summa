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
    use std::fs::File;
    use std::hash::{Hash, Hasher};
    use std::io::Write;
    use std::ops::Range;
    use std::path::PathBuf;

    pub(super) const DOWNLOAD_CHUNK_BYTES: u64 = 16 * 1024 * 1024;

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

    /// Return a URI suitable for diagnostics. Authentication material, signed
    /// query parameters, and fragments are deliberately omitted.
    pub(super) fn redacted_uri(uri: &str) -> String {
        let Ok(mut url) = url::Url::parse(uri) else {
            return "<invalid remote URI>".to_owned();
        };
        let _ = url.set_username("");
        let _ = url.set_password(None);
        url.set_query(None);
        url.set_fragment(None);
        url.to_string()
    }

    /// Cache filename: `<hash-of-full-uri>-<path-filename>`. Hashing the full
    /// URI keeps distinct signed URLs from colliding, while deriving the human
    /// readable suffix from the URL path prevents credentials and query
    /// parameters from becoming filenames.
    pub(super) fn cache_file_name(uri: &str) -> Result<String> {
        let mut h = DefaultHasher::new();
        uri.hash(&mut h);

        let safe_uri = redacted_uri(uri);
        let url = url::Url::parse(uri).with_context(|| format!("parsing URI {safe_uri}"))?;
        let name = url
            .path_segments()
            .and_then(|mut segments| segments.rfind(|segment| !segment.is_empty()))
            .filter(|name| name.len() <= 160)
            .unwrap_or("artifact");
        Ok(format!("{:016x}-{name}", h.finish()))
    }

    fn cache_path(uri: &str) -> Result<PathBuf> {
        Ok(cache_dir()?.join(cache_file_name(uri)?))
    }

    fn download_cached(uri: &str) -> Result<PathBuf> {
        let dest = cache_path(uri)?;
        let safe_uri = redacted_uri(uri);
        if dest.exists() {
            tracing::info!("using cached {} ({})", safe_uri, dest.display());
            return Ok(dest);
        }
        tracing::info!("downloading {} …", safe_uri);
        // Publish only after the complete object has been written and synced, so
        // an interrupted run never exposes a partial cache file.
        let tmp = dest.with_extension("part");
        let size = download(uri, &tmp)?;
        std::fs::rename(&tmp, &dest)
            .with_context(|| format!("publishing cache file {}", dest.display()))?;
        tracing::info!(
            "downloaded {} ({:.1} MiB) → {}",
            safe_uri,
            size as f64 / (1024.0 * 1024.0),
            dest.display()
        );
        Ok(dest)
    }

    pub(super) fn ranges(size: u64) -> impl Iterator<Item = Range<u64>> {
        (0..size)
            .step_by(DOWNLOAD_CHUNK_BYTES as usize)
            .map(move |start| start..(start + DOWNLOAD_CHUNK_BYTES).min(size))
    }

    fn download(uri: &str, dest: &std::path::Path) -> Result<u64> {
        use object_store::{ObjectStore, ObjectStoreExt};

        let safe_uri = redacted_uri(uri);
        let url = url::Url::parse(uri).with_context(|| format!("parsing URI {safe_uri}"))?;
        let (store, path): (Box<dyn ObjectStore>, object_store::path::Path) =
            object_store::parse_url(&url)
                .with_context(|| format!("no object-store backend for {safe_uri}"))?;

        // object_store is async; run a single-threaded runtime for this blocking
        // CLI call rather than making the whole inference path async.
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .context("building download runtime")?;
        let mut file = File::create(dest)
            .with_context(|| format!("creating cache temp {}", dest.display()))?;
        let size = rt.block_on(async move {
            let size = store
                .head(&path)
                .await
                .with_context(|| format!("HEAD {path}"))?
                .size;
            for range in ranges(size) {
                let bytes = store
                    .get_range(&path, range.clone())
                    .await
                    .with_context(|| format!("GET {path} bytes {}-{}", range.start, range.end))?;
                file.write_all(&bytes)
                    .with_context(|| format!("writing cache temp {}", dest.display()))?;
            }
            file.sync_all()
                .with_context(|| format!("syncing cache temp {}", dest.display()))?;
            anyhow::Ok(size)
        })?;
        Ok(size)
    }
}

pub use imp::resolve;

/// The remote scheme of `uri` (`s3`, `gs`, `http`, `https`), or `None` for a
/// local path or `file://`. Bare Windows drive letters (`C:\…`) are local.
fn remote_scheme(uri: &str) -> Option<&'static str> {
    [
        ("s3://", "s3"),
        ("gs://", "gs"),
        ("gcs://", "gs"),
        ("http://", "http"),
        ("https://", "https"),
    ]
    .into_iter()
    .find_map(|(prefix, scheme)| {
        uri.get(..prefix.len())
            .is_some_and(|head| head.eq_ignore_ascii_case(prefix))
            .then_some(scheme)
    })
}

fn strip_file_scheme(uri: &str) -> &str {
    uri.strip_prefix("file://").unwrap_or(uri)
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

    #[cfg(feature = "remote")]
    #[test]
    fn download_ranges_cover_object_once() {
        use super::imp::{DOWNLOAD_CHUNK_BYTES, ranges};

        let size = DOWNLOAD_CHUNK_BYTES * 2 + 7;
        let ranges = ranges(size).collect::<Vec<_>>();
        assert_eq!(
            ranges,
            [
                0..DOWNLOAD_CHUNK_BYTES,
                DOWNLOAD_CHUNK_BYTES..DOWNLOAD_CHUNK_BYTES * 2,
                DOWNLOAD_CHUNK_BYTES * 2..size,
            ]
        );
    }

    #[cfg(feature = "remote")]
    #[test]
    fn signed_http_uri_is_redacted_and_not_used_as_cache_filename() {
        use super::imp::{cache_file_name, redacted_uri};

        let uri = "https://alice:secret@example.com/models/weights.safetensors?X-Amz-Signature=private#fragment";
        let display = redacted_uri(uri);
        assert_eq!(display, "https://example.com/models/weights.safetensors");

        let name = cache_file_name(uri).unwrap();
        assert!(name.ends_with("-weights.safetensors"), "{name}");
        assert!(
            !name.contains("secret") && !name.contains("X-Amz"),
            "{name}"
        );
    }

    #[cfg(not(feature = "remote"))]
    #[test]
    fn remote_without_feature_errors_clearly() {
        let err = resolve("s3://b/k").unwrap_err().to_string();
        assert!(err.contains("remote") && err.contains("feature"), "{err}");
    }
}
