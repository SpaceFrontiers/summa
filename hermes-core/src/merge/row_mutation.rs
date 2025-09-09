//! Row visibility publication uses the segment manager's existing transaction.
use super::*;
use crate::query::DocBitset;

impl<D: DirectoryWriter + 'static> SegmentManager<D> {
    pub(crate) async fn commit_with_deletes(
        self: &Arc<Self>,
        new_segments: &[(String, u32)],
        keys: Vec<String>,
    ) -> Result<()> {
        if keys.is_empty() {
            return self.commit(new_segments).await;
        }
        let field = self
            .schema
            .primary_field()
            .ok_or_else(|| Error::Schema("row deletion requires a primary key".into()))?;
        for (id, count) in new_segments {
            self.validate_completed_segment(id, *count).await?;
        }
        let mut st = Arc::clone(&self.state).lock_owned().await;
        let manager = Arc::clone(self);
        let new_segments = new_segments.to_vec();
        self.run_lifecycle_transaction(async move {
            let keys: Arc<HashSet<String>> = Arc::new(keys.into_iter().collect());
            let mut next = st.metadata.clone();
            let mut retired = Vec::new();
            let mut outputs = Vec::new();
            // Only committed rows are targets. The same transaction's replacement
            // insertions must survive the deletion of their old primary keys.
            for (segment_id, info) in &st.metadata.segment_metas {
                let sid = SegmentId::from_hex(segment_id)
                    .ok_or_else(|| Error::Corruption("invalid deletion source".into()))?;
                let mut fields = crate::segment::reader::loader::load_fast_fields_file(
                    manager.directory.as_ref(),
                    &SegmentFiles::new(sid.0),
                    &manager.schema,
                )
                .await?;
                let ff = fields
                    .remove(&field.0)
                    .ok_or_else(|| Error::Corruption("primary-key column is missing".into()))?;
                // Bounded by the admitted deletion-key set, independent of
                // the source dictionary/row count. Decode global ordinals in
                // batches instead of looking up and hashing each row's text.
                let cancellation = manager.active_operations.cancellation_flag();
                let lookup_cancellation = Arc::clone(&cancellation);
                let keys = Arc::clone(&keys);
                let num_docs = info.num_docs;
                // Publication holds the state lock. Run its bounded serial scan
                // directly on Tokio's blocking executor, never behind bulk BP.
                let (ff, ordinals) = tokio::task::spawn_blocking(move || {
                    let ordinals = crate::segment::deletion::target_ordinals(
                        &ff,
                        num_docs,
                        keys.iter().map(String::as_str),
                        || {
                            if lookup_cancellation.load(Ordering::Acquire) {
                                Err(Error::IndexClosed)
                            } else {
                                Ok(())
                            }
                        },
                    )?;
                    Ok::<_, Error>((ff, ordinals))
                })
                .await
                .map_err(|error| {
                    Error::Internal(format!("deletion key lookup failed: {error}"))
                })??;
                if ordinals.is_empty() {
                    continue;
                }
                let mut alive = match &info.deletions {
                    Some(meta) => {
                        (*meta.load(manager.directory.as_ref(), info.num_docs).await?).clone()
                    }
                    None => DocBitset::all(info.num_docs),
                };
                let (alive, changed) = tokio::task::spawn_blocking(move || {
                    let changed = crate::segment::deletion::clear_target_rows(
                        &ff,
                        &ordinals,
                        &mut alive,
                        || {
                            if cancellation.load(Ordering::Acquire) {
                                Err(Error::IndexClosed)
                            } else {
                                Ok(())
                            }
                        },
                    )?;
                    Ok::<_, Error>((alive, changed))
                })
                .await
                .map_err(|error| Error::Internal(format!("deletion worker failed: {error}")))??;
                if !changed {
                    continue;
                }
                let id = SegmentId::new();
                let claim = manager.protect_new_segment(id.to_hex())?;
                let cleanup = manager.output_cleanup_guard(id);
                // Install the unwind owner before writing the first byte.
                outputs.push((claim, cleanup));
                let deletion = crate::segment::deletion::write(
                    manager.directory.as_ref(),
                    id,
                    info.num_docs,
                    &alive,
                )
                .await?;
                if let Some(old) = &info.deletions {
                    retired.push(old.id.clone());
                }
                next.segment_metas.get_mut(segment_id).unwrap().deletions = Some(deletion);
            }
            for (id, count) in &new_segments {
                if !next.has_segment(id) {
                    next.add_segment(id.clone(), *count);
                }
            }
            next.publication_generation = next
                .publication_generation
                .checked_add(1)
                .ok_or_else(|| Error::Corruption("publication generation overflow".into()))?;
            next.save(manager.directory.as_ref()).await?;
            for id in next.owned_ids() {
                manager.tracker.register(&id);
            }
            let previous = manager.published_generation();
            manager
                .published_generation
                .store(Arc::new(PublishedIndexGeneration {
                    publication_id: next.publication_generation,
                    schema: Arc::clone(&previous.schema),
                    trained_vectors: previous.trained_vectors.clone(),
                }));
            retired.retain(|id| !next.owns_id(id));
            st.metadata = next;
            for (_, cleanup) in &mut outputs {
                cleanup.disarm();
            }
            let ready = manager.tracker.mark_for_deletion(&retired);
            drop(st);
            (manager.delete_fn)(ready);
            Ok(())
        })
        .await
    }
}

impl<D: DirectoryWriter + 'static> SegmentManager<D> {
    /// Rewrite one segment with a dense live-row space, including single-segment
    /// indexes. Returns false if it has no tombstones or is no longer current.
    pub async fn compact_segment(
        self: &Arc<Self>,
        segment_id: &str,
        memory_budget: usize,
    ) -> Result<bool> {
        self.compact_segment_admitted(segment_id, memory_budget, None)
            .await
    }

    /// Metadata-only candidates for the existing periodic optimizer. Zero
    /// threshold disables automatic work; invalid ratios fail before I/O.
    pub async fn compaction_candidates(&self, threshold: f64) -> Result<Vec<(String, u32, u32)>> {
        validate_threshold(threshold)?;
        if threshold == 0.0 || self.force_merge_active.load(Ordering::Acquire) > 0 {
            return Ok(Vec::new());
        }
        let active = self.active_operations.snapshot();
        let paused = self.paused_reorder_segments();
        let quarantined = self.quarantined_segments.lock().clone();
        let st = self.state.lock().await;
        let mut candidates: Vec<_> = st
            .metadata
            .segment_metas
            .iter()
            .filter(|(id, meta)| {
                meta.num_deleted_docs() > 0
                    && meta.deleted_ratio() >= threshold
                    && !active.contains(*id)
                    && !paused.contains(*id)
                    && !quarantined.contains(*id)
            })
            .map(|(id, meta)| (id.clone(), meta.num_docs, meta.num_deleted_docs()))
            .collect();
        candidates.sort_unstable_by(|a, b| {
            (f64::from(b.2) / f64::from(b.1))
                .total_cmp(&(f64::from(a.2) / f64::from(a.1)))
                .then_with(|| a.1.cmp(&b.1))
                .then_with(|| a.0.cmp(&b.0))
        });
        Ok(candidates)
    }

    /// Nonblocking maintenance admission, using the same ownership, pool,
    /// capacity and bounded retry state as existing optimizer work.
    pub async fn compact_segment_if_eligible(
        self: &Arc<Self>,
        segment_id: &str,
        threshold: f64,
        memory_budget: usize,
    ) -> Result<bool> {
        validate_threshold(threshold)?;
        if threshold == 0.0 {
            return Ok(false);
        }
        let result = self
            .compact_segment_admitted(segment_id, memory_budget, Some(threshold))
            .await;
        match &result {
            Ok(true) => self.clear_reorder_retry(segment_id),
            Err(error) if !matches!(error, Error::IndexClosed) => {
                self.pause_reorder_retries(segment_id, error).await
            }
            _ => {}
        }
        result
    }

    async fn compact_segment_admitted(
        self: &Arc<Self>,
        segment_id: &str,
        memory_budget: usize,
        threshold: Option<f64>,
    ) -> Result<bool> {
        if memory_budget < 1024 * 1024 {
            return Err(Error::Schema(
                "compaction memory budget must be at least 1 MiB".into(),
            ));
        }
        let sid = SegmentId::from_hex(segment_id)
            .ok_or_else(|| Error::Document("invalid compaction segment ID".into()))?;
        if !self.active_operations.is_accepting() {
            return Err(Error::IndexClosed);
        }
        if self.quarantined_segments.lock().contains(segment_id) {
            return Err(Error::Corruption("compaction source is quarantined".into()));
        }
        if threshold.is_some()
            && (self.force_merge_active.load(Ordering::Acquire) > 0
                || self.paused_reorder_segments().contains(segment_id))
        {
            return Ok(false);
        }
        let (global_permit, permit, reorder_permit) = if threshold.is_some() {
            let Ok(global) = Arc::clone(&self.global_merge_permits).try_acquire_owned() else {
                return Ok(false);
            };
            let Ok(local) = Arc::clone(&self.merge_permits).try_acquire_owned() else {
                return Ok(false);
            };
            let Ok(reorder) = self.reorder_permits.try_acquire_optimizer() else {
                return Ok(false);
            };
            (global, local, reorder)
        } else {
            let (global, local) = self.acquire_maintenance_capacity().await?;
            let reorder = tokio::select! {
                biased;
                () = self.active_operations.wait_for_shutdown() => return Err(Error::IndexClosed),
                permit = self.reorder_permits.acquire(ReorderPriority::Optimizer) => permit.map_err(|_| Error::IndexClosed)?,
            };
            (global, local, reorder)
        };
        let manager = Arc::clone(self);
        let source_id = segment_id.to_owned();
        self.run_lifecycle_transaction(async move {
            let _permit = permit;
            let _global_permit = global_permit;
            let _reorder_permit = reorder_permit;
            let start = std::time::Instant::now();
            let output = SegmentId::new();
            let (claim, snapshot, visibility, schema) = {
                let st = manager.state.lock().await;
                let Some(info) = st.metadata.segment_metas.get(&source_id) else {
                    return Ok(false);
                };
                if let Some(threshold) = threshold
                    && (manager.force_merge_active.load(Ordering::Acquire) > 0 || info.deleted_ratio() < threshold) {
                    return Ok(false);
                }
                let Some(visibility) = info.deletions.clone() else {
                    return Ok(false);
                };
                let Some(claim) = manager.active_operations.try_register(vec![source_id.clone(), output.to_hex()]) else {
                    return if threshold.is_some() { Ok(false) } else {
                        Err(Error::Internal("compaction source is busy; retry after maintenance finishes".into()))
                    };
                };
                let acquired = manager
                    .tracker
                    .acquire(&[source_id.clone(), visibility.id.clone()]);
                let snapshot = SegmentSnapshot::with_delete_fn(
                    Arc::clone(&manager.tracker),
                    acquired,
                    Arc::clone(&manager.delete_fn),
                );
                (
                    claim,
                    snapshot,
                    visibility,
                    manager.published_generation().schema.clone(),
                )
            };
            let _claim = claim;
            let _snapshot = snapshot;
            let mut cleanup = manager.output_cleanup_guard(output);
            let mut reader = SegmentReader::open(
                manager.directory.as_ref(),
                sid,
                schema.clone(),
                manager.term_cache_blocks,
            )
            .await?;
            reader
                .load_deletions(manager.directory.as_ref(), visibility.clone())
                .await?;
            let physical = reader.num_docs();
            let meta = manager
                .build_compacted(reader, output, memory_budget)
                .await?;
            let removed = visibility.num_deleted;
            if meta.num_docs.checked_add(removed) != Some(physical) {
                return Err(Error::Corruption("compaction row count mismatch".into()));
            }
            let expected = HashMap::from([(source_id.clone(), Some(visibility))]);
            manager
                .replace_segments(
                    std::slice::from_ref(&source_id),
                    output.to_hex(),
                    meta.num_docs,
                    ReplacementLayout::Compacted,
                    Some(&expected),
                )
                .await?;
            cleanup.disarm();
            log::info!("[compaction] index={} source={} physical_rows={} removed_rows={} removed_ratio={:.6} live_rows={} elapsed_secs={:.3}",
                manager.schema.index_label(), source_id, physical, removed,
                f64::from(removed) / f64::from(physical), meta.num_docs, start.elapsed().as_secs_f64());
            Ok(true)
        })
        .await
    }

    pub(super) async fn build_compacted(
        &self,
        reader: SegmentReader,
        output: SegmentId,
        memory_budget: usize,
    ) -> Result<crate::segment::SegmentMeta> {
        let directory = Arc::clone(&self.directory);
        let merger = crate::segment::SegmentMerger::new(self.published_generation().schema.clone())
            .with_posting_config(self.optimization, self.posting_codec)
            .with_cancellation(self.active_operations.cancellation_flag())
            .with_background_pool(Some(self.background_cpu_pool()));
        let runtime = tokio::runtime::Handle::current();
        let (meta, _) = tokio::task::spawn_blocking(move || {
            runtime.block_on(merger.compact(directory.as_ref(), &reader, output, memory_budget))
        })
        .await
        .map_err(|error| Error::Internal(format!("compaction worker failed: {error}")))??;
        Ok(meta)
    }
}

fn validate_threshold(threshold: f64) -> Result<()> {
    if !threshold.is_finite() || !(0.0..=1.0).contains(&threshold) {
        return Err(Error::Schema("compaction deleted ratio must be finite and in 0..=1 (0 disables automatic compaction)".into()));
    }
    Ok(())
}
