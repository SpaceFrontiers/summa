use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use super::pipeline::read_corpus_metadata_file;
use super::{
    CorpusBuildConfig, CorpusManifest, CorpusPipeline, CorpusTokenizer, PostgresRecordMaterializer,
    PostgresRecordMaterializerConfig, SearchApiClient, SearchApiConfig, SqliteDeduplicator,
};

/// Ready-to-run production recipe for the current Search API + PostgreSQL
/// deployment. Other providers use the same [`CorpusPipeline`] directly.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SearchApiPostgresCorpusRecipe {
    pub corpus: CorpusBuildConfig,
    pub search_api: SearchApiConfig,
    pub postgres: PostgresRecordMaterializerConfig,
}

impl SearchApiPostgresCorpusRecipe {
    pub fn load(path: &Path) -> Result<Self> {
        let bytes = read_corpus_metadata_file(path, "corpus recipe")?;
        let recipe: Self = serde_json::from_slice(&bytes)
            .with_context(|| format!("invalid corpus recipe JSON in {}", path.display()))?;
        recipe.corpus.validate()?;
        recipe.search_api.validate()?;
        recipe.postgres.validate()?;
        Ok(recipe)
    }

    /// Execute with live HTTP and PostgreSQL adapters. `work_directory` holds
    /// the transactional discovery/deduplication catalog and authoritative
    /// resume cursor; it must be durable local storage and is not copied into
    /// the immutable output or manifest.
    pub fn run(
        self,
        tokenizer: &dyn CorpusTokenizer,
        output_root: &Path,
        work_directory: &Path,
    ) -> Result<(PathBuf, CorpusManifest)> {
        fs::create_dir_all(work_directory).with_context(|| {
            format!(
                "failed to create corpus work directory {}",
                work_directory.display()
            )
        })?;
        let dedup_path = work_directory.join(format!("{}.dedup.sqlite", self.corpus.build_id));
        let search = SearchApiClient::connect(self.search_api)?;
        let materializer = PostgresRecordMaterializer::connect(self.postgres)?;
        let mut deduplicator =
            SqliteDeduplicator::open(&dedup_path, self.corpus.deduplication.clone())?;
        CorpusPipeline::new(
            self.corpus,
            &search,
            &materializer,
            tokenizer,
            &mut deduplicator,
        )?
        .run(output_root)
    }
}
