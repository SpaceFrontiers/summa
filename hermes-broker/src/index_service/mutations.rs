//! Primary-key routing for staged mutations; no retries or publication here.
use super::*;
use crate::proto::hermes::DocumentError;

enum MutationBatch {
    Delete(Vec<String>),
    Upsert(Vec<NamedDocument>),
}

impl BrokerIndexService {
    pub(super) async fn route_deletions(
        &self,
        request: Request<DeleteDocumentsRequest>,
    ) -> Result<DocumentMutationResponse, Status> {
        request.get_ref().validate_limits()?;
        self.ctx.check_admission()?;
        let timeout = forward_timeout(request.metadata());
        let req = request.into_inner();
        let route = self.ctx.write_route(&req.index_name)?;
        let groups = if route.is_partitioned() {
            let mut groups: Vec<(Vec<usize>, Vec<String>)> = (0..route.targets().len())
                .map(|_| (Vec::new(), Vec::new()))
                .collect();
            for (position, key) in req.primary_keys.into_iter().enumerate() {
                let group = partition::partition_of(key.as_bytes(), groups.len());
                groups[group].0.push(position);
                groups[group].1.push(key);
            }
            groups
        } else {
            vec![((0..req.primary_keys.len()).collect(), req.primary_keys)]
        };
        let groups = groups
            .into_iter()
            .map(|(positions, keys)| (positions, MutationBatch::Delete(keys)))
            .collect();
        self.send_mutations(&req.index_name, route, groups, vec![], timeout)
            .await
    }

    pub(super) async fn route_upserts(
        &self,
        request: Request<UpsertDocumentsRequest>,
    ) -> Result<DocumentMutationResponse, Status> {
        request.get_ref().validate_limits()?;
        self.ctx.check_admission()?;
        let timeout = forward_timeout(request.metadata());
        let req = request.into_inner();
        let route = self.ctx.write_route(&req.index_name)?;
        let (groups, errors) = if route.is_partitioned() {
            let primary = self
                .ctx
                .primary_key_field(&req.index_name, &route.targets()[0])
                .await?;
            let routed = partition::route_documents(req.documents, &primary, route.targets().len());
            (routed.groups, routed.unroutable)
        } else {
            (
                vec![((0..req.documents.len()).collect(), req.documents)],
                vec![],
            )
        };
        let groups = groups
            .into_iter()
            .map(|(positions, documents)| (positions, MutationBatch::Upsert(documents)))
            .collect();
        self.send_mutations(&req.index_name, route, groups, errors, timeout)
            .await
    }

    async fn send_mutations(
        &self,
        index_name: &str,
        route: Route,
        groups: Vec<(Vec<usize>, MutationBatch)>,
        mut errors: Vec<DocumentError>,
        timeout: Option<Duration>,
    ) -> Result<DocumentMutationResponse, Status> {
        let calls = groups
            .into_iter()
            .zip(route.targets())
            .filter(|((positions, _), _)| !positions.is_empty() || !route.is_partitioned())
            .map(|((positions, batch), target)| async move {
                let started = Instant::now();
                let mut client = target.channels.index.clone();
                let (rpc, result) = match batch {
                    MutationBatch::Delete(primary_keys) => {
                        let mut request = Request::new(DeleteDocumentsRequest {
                            index_name: index_name.into(),
                            primary_keys,
                        });
                        if let Some(timeout) = timeout {
                            request.set_timeout(timeout);
                        }
                        ("delete_documents", client.delete_documents(request).await)
                    }
                    MutationBatch::Upsert(documents) => {
                        let mut request = Request::new(UpsertDocumentsRequest {
                            index_name: index_name.into(),
                            documents,
                        });
                        if let Some(timeout) = timeout {
                            request.set_timeout(timeout);
                        }
                        ("upsert_documents", client.upsert_documents(request).await)
                    }
                };
                let code = result
                    .as_ref()
                    .map(|_| tonic::Code::Ok)
                    .unwrap_or_else(|s| s.code());
                record_backend(&target.backend_id, rpc, started, code);
                (target, positions, result)
            });
        let mut accepted_count = 0;
        for (target, positions, result) in futures::future::join_all(calls).await {
            let response = result
                .map_err(|status| {
                    if route.is_partitioned() {
                        partition::partition_failure(index_name, &target.shard, status)
                    } else {
                        status
                    }
                })?
                .into_inner();
            validate_response(&positions, &response)?;
            accepted_count += response.accepted_count;
            errors.extend(response.errors.into_iter().map(|mut error| {
                error.index = positions[error.index as usize] as u32;
                error
            }));
        }
        errors.sort_unstable_by_key(|error| error.index);
        Ok(DocumentMutationResponse {
            accepted_count,
            errors,
        })
    }
}

fn validate_response(
    positions: &[usize],
    response: &DocumentMutationResponse,
) -> Result<(), Status> {
    if response.accepted_count as usize + response.errors.len() != positions.len() {
        return Err(Status::data_loss(
            "backend mutation response does not account for every input",
        ));
    }
    let mut seen = vec![false; positions.len()];
    for error in &response.errors {
        let Some(seen) = seen.get_mut(error.index as usize) else {
            return Err(Status::data_loss(
                "backend mutation error index is out of bounds",
            ));
        };
        if std::mem::replace(seen, true) {
            return Err(Status::data_loss(
                "backend mutation response repeats an error index",
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn malformed_backend_mutation_accounting_is_rejected() {
        let response = |accepted_count, indices: &[u32]| DocumentMutationResponse {
            accepted_count,
            errors: indices
                .iter()
                .map(|index| DocumentError {
                    index: *index,
                    error: "rejected".into(),
                })
                .collect(),
        };
        assert!(validate_response(&[2, 7], &response(1, &[0])).is_ok());
        for invalid in [
            response(2, &[0]),
            response(0, &[0, 0]),
            response(1, &[2]),
            response(0, &[]),
        ] {
            assert_eq!(
                validate_response(&[2, 7], &invalid).unwrap_err().code(),
                tonic::Code::DataLoss
            );
        }
    }
}
