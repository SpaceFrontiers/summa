//! Inline publication for the portable writer, reusing core PK and mask formats.
use super::*;
use crate::Error;
use crate::index::primary_key::{
    PkSegmentData, PrimaryKeyIndex, document_key, load_pk_segment_data,
};
use crate::query::DocBitset;
use crate::segment::{SegmentFiles, deletion};

impl<D: DirectoryWriter + 'static> IndexWriter<D> {
    pub(super) fn ensure_healthy(&self) -> Result<()> {
        if self.publication_pending {
            return Err(Error::CommitInProgress);
        }
        if self.poisoned {
            return Err(Error::Internal(
                "pending transaction lost its builder; call abort before continuing".into(),
            ));
        }
        Ok(())
    }

    pub(super) async fn initialize_primary_key(&mut self) -> Result<()> {
        if let Some(field) = self.schema.primary_field() {
            let data = self.prepare_primary_key_refresh(&self.metadata).await?;
            self.primary_key = Some(PrimaryKeyIndex::build(field, data));
        }
        Ok(())
    }

    /// Stage an exact-key deletion of the document and every chunk. Commit to publish.
    pub fn delete_primary_key(&mut self, key: &str) -> Result<()> {
        self.ensure_healthy()?;
        self.primary_key
            .as_ref()
            .ok_or_else(|| Error::Schema("row deletion requires a primary key".into()))?
            .delete(key)?;
        Ok(())
    }

    /// Stage a full replacement, inserting absent keys. One pending insertion per key.
    pub async fn upsert_document(&mut self, doc: Document) -> Result<()> {
        self.ensure_healthy()?;
        self.validate_document(&doc)?;
        let field = self
            .schema
            .primary_field()
            .ok_or_else(|| Error::Schema("upserts require a primary key".into()))?;
        let key = document_key(&doc, field)?.to_owned();
        let staged = self.primary_key.as_ref().unwrap().delete(&key)?;
        if let Err(error) = self.add_document(doc).await {
            if staged {
                self.primary_key.as_ref().unwrap().rollback_delete(&key);
            }
            return Err(error);
        }
        Ok(())
    }

    /// Discard all pending additions, replacements and deletions. Published rows survive.
    pub async fn abort(&mut self) -> Result<()> {
        self.recover_publication().await?;
        self.builder = None;
        self.pending_segments.clear();
        if let Some(pk) = &mut self.primary_key {
            pk.clear_uncommitted();
        }
        self.poisoned = false;
        self.cleanup_outputs().await
    }

    /// Publish replacement segments and immutable masks in one metadata generation.
    /// A failed metadata save is retryable; a consumed builder requires abort.
    pub async fn commit(&mut self) -> Result<bool> {
        if self.recover_publication().await? {
            self.cleanup_outputs().await?;
            return Ok(true);
        }
        self.ensure_healthy()?;
        self.cleanup_outputs().await?;
        self.flush_builder().await?;
        let keys = self
            .primary_key
            .as_ref()
            .map_or_else(Vec::new, |pk| pk.pending_deletes());
        if self.pending_segments.is_empty() && keys.is_empty() {
            return Ok(false);
        }

        let mut next = self.metadata.clone();
        if !keys.is_empty() {
            let field = self.schema.primary_field().unwrap();
            // Committed data only: the replacement inserted in this transaction survives.
            for data in self.primary_key.as_ref().unwrap().segment_data() {
                let info = &self.metadata.segment_metas[&data.segment_id];
                let column = data
                    .fast_fields
                    .get(&field.0)
                    .ok_or_else(|| Error::Corruption("primary-key column is missing".into()))?;
                let ordinals = deletion::target_ordinals(
                    column,
                    info.num_docs,
                    keys.iter().map(String::as_str),
                    || Ok(()),
                )?;
                if ordinals.is_empty() {
                    continue;
                }
                let mut alive = data
                    .alive_docs
                    .as_ref()
                    .map_or_else(|| DocBitset::all(info.num_docs), |mask| (**mask).clone());
                if !deletion::clear_target_rows(column, &ordinals, &mut alive, || Ok(()))? {
                    continue;
                }
                let id = SegmentId::new();
                self.owned_outputs.insert(id.to_hex());
                let mask =
                    deletion::write(self.directory.as_ref(), id, info.num_docs, &alive).await?;
                next.segment_metas
                    .get_mut(&data.segment_id)
                    .unwrap()
                    .deletions = Some(mask);
            }
        }
        for (id, count) in &self.pending_segments {
            next.add_segment(id.clone(), *count);
        }
        next.publication_generation = next
            .publication_generation
            .checked_add(1)
            .ok_or_else(|| Error::Corruption("publication generation overflow".into()))?;
        // All fallible cache loads precede publication. A retry never replays a
        // published deletion against the replacement's new physical row.
        let data = self.prepare_primary_key_refresh(&next).await?;
        self.publication_pending = true;
        next.save(self.directory.as_ref()).await?;
        self.finish_publication(next, data);
        self.cleanup_outputs().await?;
        Ok(true)
    }

    // Cancellation may arrive after rename has taken effect. Reconcile with
    // the durable commit point before any output cleanup or mutation replay.
    async fn recover_publication(&mut self) -> Result<bool> {
        if !self.publication_pending {
            return Ok(false);
        }
        let durable = IndexMetadata::load(self.directory.as_ref()).await?;
        if durable.publication_generation == self.metadata.publication_generation {
            self.publication_pending = false;
            return Ok(false);
        }
        if Some(durable.publication_generation)
            != self.metadata.publication_generation.checked_add(1)
        {
            return Err(Error::Corruption(
                "unexpected portable publication generation; exclusive writer required".into(),
            ));
        }
        let data = self.prepare_primary_key_refresh(&durable).await?;
        self.finish_publication(durable, data);
        Ok(true)
    }

    fn finish_publication(&mut self, next: IndexMetadata, data: Vec<PkSegmentData>) {
        let owned = owned_ids(&next);
        let retired: Vec<_> = self
            .metadata
            .segment_metas
            .values()
            .filter_map(|info| info.deletions.as_ref())
            .filter(|mask| !owned.contains(mask.id.as_str()))
            .map(|mask| mask.id.clone())
            .collect();
        self.owned_outputs.retain(|id| !owned.contains(id.as_str()));
        self.metadata = next;
        self.pending_segments.clear();
        if let Some(pk) = &mut self.primary_key {
            pk.mark_deletes_published();
            pk.refresh_data(data, &self.metadata.segment_ids());
        }
        self.owned_outputs.extend(retired);
        self.publication_pending = false;
    }

    async fn prepare_primary_key_refresh(
        &self,
        metadata: &IndexMetadata,
    ) -> Result<Vec<PkSegmentData>> {
        let Some(field) = self.schema.primary_field() else {
            return Ok(vec![]);
        };
        let existing: FxHashMap<_, _> = self
            .primary_key
            .iter()
            .flat_map(|pk| pk.committed_visibility())
            .collect();
        let mut data = Vec::new();
        for (id, info) in &metadata.segment_metas {
            if existing
                .get(id.as_str())
                .is_some_and(|mask| *mask == info.deletions.as_ref())
            {
                continue;
            }
            let mut loaded = load_pk_segment_data(
                self.directory.as_ref(),
                id,
                &self.schema,
                info.deletions.clone().map(|mask| (info.num_docs, mask)),
            )
            .await?;
            let column = loaded
                .fast_fields
                .get(&field.0)
                .ok_or_else(|| Error::Corruption("primary-key column is missing".into()))?;
            deletion::target_ordinals(column, info.num_docs, std::iter::empty(), || Ok(()))?;
            loaded.prepare_live_keys(field);
            data.push(loaded);
        }
        Ok(data)
    }

    // Opening establishes exclusive ownership of this index directory. Reclaim
    // known segment artifacts left by a previous failed/cancelled writer before
    // they accumulate in RAM or get copied to the storage adapter again.
    pub(super) async fn reclaim_orphans(&mut self) -> Result<()> {
        let owned = owned_ids(&self.metadata);
        for path in self.directory.list_files(std::path::Path::new("")).await? {
            let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
                continue;
            };
            let Some(hex) = name
                .strip_prefix("seg_")
                .and_then(|name| name.split('.').next())
            else {
                continue;
            };
            let Some(id) = SegmentId::from_hex(hex) else {
                continue;
            };
            if !owned.contains(hex) && SegmentFiles::new(id.0).lifecycle_paths().contains(&path) {
                self.owned_outputs.insert(hex.to_owned());
            }
        }
        self.cleanup_outputs().await
    }

    // Portable searchers own decoded mask bytes; no reader reopens a retired
    // mask. Data segments are never retired by this writer. Only claimed output
    // IDs are cleaned, and a cancellation leaves the claim available for retry.
    async fn cleanup_outputs(&mut self) -> Result<()> {
        if self.owned_outputs.is_empty() {
            return Ok(());
        }
        let mut protected = owned_ids(&self.metadata);
        protected.extend(self.pending_segments.iter().map(|(id, _)| id.as_str()));
        let ids: Vec<_> = self
            .owned_outputs
            .iter()
            .filter(|id| !protected.contains(id.as_str()))
            .cloned()
            .collect();
        for id in ids {
            let segment = SegmentId::from_hex(&id)
                .ok_or_else(|| Error::Corruption("invalid output claim".into()))?;
            for path in SegmentFiles::new(segment.0).lifecycle_paths() {
                match self.directory.delete(&path).await {
                    Ok(()) => {}
                    Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                    Err(error) => return Err(error.into()),
                }
            }
            self.owned_outputs.remove(&id);
        }
        Ok(())
    }
}

fn owned_ids(metadata: &IndexMetadata) -> rustc_hash::FxHashSet<&str> {
    metadata
        .segment_metas
        .keys()
        .map(String::as_str)
        .chain(
            metadata
                .segment_metas
                .values()
                .filter_map(|info| info.deletions.as_ref().map(|mask| mask.id.as_str())),
        )
        .collect()
}
