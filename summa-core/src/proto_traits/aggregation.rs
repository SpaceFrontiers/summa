use summa_proto::proto;
use tantivy::aggregation::agg_req::Aggregation;
use tantivy::aggregation::agg_result::{AggregationResult, BucketEntries, BucketEntry, BucketResult, MetricResult, RangeBucketEntry};
use tantivy::aggregation::Key;

use crate::errors::Error;
use crate::proto_traits::Wrapper;

impl TryFrom<Wrapper<proto::Aggregation>> for Aggregation {
    type Error = Error;

    fn try_from(value: Wrapper<proto::Aggregation>) -> Result<Self, Error> {
        match value.into_inner().aggregation {
            Some(aggregation) => Wrapper::from(aggregation).try_into(),
            _ => Err(Error::InvalidAggregation),
        }
    }
}

impl TryFrom<Wrapper<proto::aggregation::Aggregation>> for Aggregation {
    type Error = Error;

    fn try_from(aggregation: Wrapper<proto::aggregation::Aggregation>) -> Result<Self, Error> {
        Ok(serde_json::from_str(&serde_json::to_string(&aggregation.into_inner())?)?)
    }
}

impl From<AggregationResult> for Wrapper<proto::AggregationResult> {
    fn from(aggregation_result: AggregationResult) -> Self {
        Wrapper::from(proto::AggregationResult {
            aggregation_result: Some(match aggregation_result {
                AggregationResult::BucketResult(bucket_result) => proto::aggregation_result::AggregationResult::Bucket(proto::BucketResult {
                    bucket_result: Some(match bucket_result {
                        BucketResult::Range { buckets } => proto::bucket_result::BucketResult::Range(proto::RangeResult {
                            buckets: match buckets {
                                BucketEntries::Vec(vec) => vec.into_iter().map(|bucket| Wrapper::from(bucket).into_inner()).collect(),
                                BucketEntries::HashMap(hm) => hm.into_iter().map(|bucket| Wrapper::from(bucket.1).into_inner()).collect(),
                            },
                        }),
                        BucketResult::Histogram { buckets } => proto::bucket_result::BucketResult::Histogram(proto::HistogramResult {
                            buckets: match buckets {
                                BucketEntries::Vec(vec) => vec.into_iter().map(|bucket| Wrapper::from(bucket).into_inner()).collect(),
                                BucketEntries::HashMap(hm) => hm.into_iter().map(|bucket| Wrapper::from(bucket.1).into_inner()).collect(),
                            },
                        }),
                        BucketResult::Terms {
                            buckets,
                            sum_other_doc_count,
                            doc_count_error_upper_bound,
                        } => proto::bucket_result::BucketResult::Terms(proto::TermsResult {
                            buckets: buckets.into_iter().map(|bucket| Wrapper::from(bucket).into_inner()).collect(),
                            sum_other_doc_count,
                            doc_count_error_upper_bound,
                        }),
                    }),
                }),
                AggregationResult::MetricResult(metric_result) => proto::aggregation_result::AggregationResult::Metric(proto::MetricResult {
                    metric_result: Some(match metric_result {
                        MetricResult::Average(single_metric_result) => proto::metric_result::MetricResult::SingleMetric(proto::SingleMetricResult {
                            value: single_metric_result.value,
                        }),
                        MetricResult::Stats(stats) => proto::metric_result::MetricResult::Stats(proto::StatsResult {
                            avg: stats.avg,
                            count: stats.count,
                            max: stats.max,
                            min: stats.min,
                            sum: stats.sum,
                        }),
                        _ => unimplemented!(),
                    }),
                }),
            }),
        })
    }
}

impl From<Key> for Wrapper<proto::Key> {
    fn from(key: Key) -> Self {
        Wrapper::from(match key {
            Key::Str(s) => proto::Key {
                key: Some(proto::key::Key::Str(s)),
            },
            Key::F64(f) => proto::Key {
                key: Some(proto::key::Key::F64(f)),
            },
        })
    }
}

impl From<BucketEntry> for Wrapper<proto::BucketEntry> {
    fn from(bucket_entry: BucketEntry) -> Self {
        Wrapper::from(proto::BucketEntry {
            key: Some(Wrapper::from(bucket_entry.key).into_inner()),
            doc_count: bucket_entry.doc_count,
            sub_aggregation: bucket_entry
                .sub_aggregation
                .0
                .into_iter()
                .map(|(name, aggregation_result)| (name, Wrapper::from(aggregation_result).into_inner()))
                .collect(),
        })
    }
}

impl From<RangeBucketEntry> for Wrapper<proto::RangeBucketEntry> {
    fn from(range_bucket_entry: RangeBucketEntry) -> Self {
        Wrapper::from(proto::RangeBucketEntry {
            key: Some(Wrapper::from(range_bucket_entry.key).into_inner()),
            doc_count: range_bucket_entry.doc_count,
            sub_aggregation: range_bucket_entry
                .sub_aggregation
                .0
                .into_iter()
                .map(|(name, aggregation_result)| (name, Wrapper::from(aggregation_result).into_inner()))
                .collect(),
            from: range_bucket_entry.from,
            to: range_bucket_entry.to,
        })
    }
}
