use tantivy::aggregation::agg_req::{Aggregation, BucketAggregation, BucketAggregationType, MetricAggregation, RangeAggregation};
use tantivy::aggregation::agg_result::{AggregationResult, BucketEntry, BucketResult, MetricResult, RangeBucketEntry};
use tantivy::aggregation::bucket::{CustomOrder, HistogramAggregation, HistogramBounds, Order, OrderTarget, RangeAggregationRange, TermsAggregation};
use tantivy::aggregation::Key;
use tantivy::aggregation::metric::{AverageAggregation, StatsAggregation};
use crate::errors::{Error, SummaResult, ValidationError};
use crate::proto;

impl TryFrom<proto::Aggregation> for Aggregation {
    type Error = Error;

    fn try_from(value: proto::Aggregation) -> SummaResult<Self> {
        match value.aggregation {
            None => Err(ValidationError::InvalidAggregationError)?,
            Some(aggregation) => aggregation.try_into()
        }
    }
}
impl TryFrom<proto::aggregation::Aggregation> for Aggregation {
    type Error = Error;

    fn try_from(aggregation: proto::aggregation::Aggregation) -> SummaResult<Self> {
        Ok(match aggregation {
            proto::aggregation::Aggregation::Bucket(bucket_aggregation) => {
                Aggregation::Bucket(BucketAggregation {
                    bucket_agg: match bucket_aggregation.bucket_agg {
                        Some(proto::bucket_aggregation::BucketAgg::Histogram(histogram_aggregation)) => {
                            BucketAggregationType::Histogram(HistogramAggregation {
                                field: histogram_aggregation.field,
                                interval: histogram_aggregation.interval,
                                offset: histogram_aggregation.offset,
                                min_doc_count: histogram_aggregation.min_doc_count,
                                hard_bounds: histogram_aggregation.hard_bounds.map(|hard_bounds| {
                                    HistogramBounds {
                                        min: hard_bounds.min,
                                        max: hard_bounds.max,
                                    }
                                }),
                                extended_bounds: histogram_aggregation.extended_bounds.map(|extended_bounds| {
                                    HistogramBounds {
                                        min: extended_bounds.min,
                                        max: extended_bounds.max,
                                    }
                                }),
                            })
                        }
                        Some(proto::bucket_aggregation::BucketAgg::Range(range_aggregation)) => {
                            BucketAggregationType::Range(RangeAggregation {
                                field: range_aggregation.field,
                                ranges: range_aggregation.ranges.into_iter().map(|range| {
                                    RangeAggregationRange {
                                        from: range.from,
                                        to: range.to,
                                    }
                                }).collect(),
                            })
                        }
                        Some(proto::bucket_aggregation::BucketAgg::Terms(terms_aggregation)) => {
                            BucketAggregationType::Terms(TermsAggregation {
                                field: terms_aggregation.field,
                                size: terms_aggregation.size,
                                split_size: terms_aggregation.split_size,
                                segment_size: terms_aggregation.segment_size,
                                show_term_doc_count_error: terms_aggregation.show_term_doc_count_error,
                                min_doc_count: terms_aggregation.min_doc_count,
                                order: terms_aggregation.order.map(|order| {
                                    CustomOrder {
                                        target: match order.order_target {
                                            None | Some(proto::custom_order::OrderTarget::Key(_)) => OrderTarget::Key,
                                            Some(proto::custom_order::OrderTarget::Count(_)) => OrderTarget::Count,
                                            Some(proto::custom_order::OrderTarget::SubAggregation(sub_aggregation)) => OrderTarget::SubAggregation(sub_aggregation),
                                        },
                                        order: match proto::Order::from_i32(order.order) {
                                            None => Order::Asc,
                                            Some(order) => order.into()
                                        },
                                    }
                                }),
                            })
                        }
                        None => Err(ValidationError::InvalidAggregationError)?,
                    },
                    sub_aggregation: bucket_aggregation
                        .sub_aggregation
                        .into_iter()
                        .map(|(name, aggregation)| {
                            Ok((name, aggregation.try_into()?))
                        }).collect::<SummaResult<_>>()?,
                })
            }
            proto::aggregation::Aggregation::Metric(metric_aggregation) => {
                match metric_aggregation.metric_aggregation {
                    Some(proto::metric_aggregation::MetricAggregation::Average(average_aggregation)) => {
                        Aggregation::Metric(
                            MetricAggregation::Average {
                                0: AverageAggregation::from_field_name(average_aggregation.field)
                            }
                        )
                    }
                    Some(proto::metric_aggregation::MetricAggregation::Stats(stats_aggregation)) => {
                        Aggregation::Metric(
                            MetricAggregation::Stats {
                                0: StatsAggregation::from_field_name(stats_aggregation.field)
                            }
                        )
                    }
                    None => Err(ValidationError::InvalidAggregationError)?
                }
            }
        })
    }
}

impl Into<proto::AggregationResult> for AggregationResult {
    fn into(self) -> proto::AggregationResult {
        proto::AggregationResult {
            aggregation_result: Some(match self {
                AggregationResult::BucketResult(bucket_result) => {
                    proto::aggregation_result::AggregationResult::Bucket(proto::BucketResult {
                        bucket_result: Some(match bucket_result {
                            BucketResult::Range { buckets } => {
                                proto::bucket_result::BucketResult::Range(proto::RangeResult {
                                    buckets: buckets.into_iter().map(|bucket| bucket.into()).collect(),
                                })
                            }
                            BucketResult::Histogram { buckets } => {
                                proto::bucket_result::BucketResult::Histogram(proto::HistogramResult {
                                    buckets: buckets.into_iter().map(|bucket| bucket.into()).collect(),
                                })
                            }
                            BucketResult::Terms { buckets, sum_other_doc_count, doc_count_error_upper_bound } => {
                                proto::bucket_result::BucketResult::Terms(proto::TermsResult {
                                    buckets: buckets.into_iter().map(|bucket| bucket.into()).collect(),
                                    sum_other_doc_count,
                                    doc_count_error_upper_bound,
                                })
                            }
                        })
                    })
                }
                AggregationResult::MetricResult(metric_result) => {
                    proto::aggregation_result::AggregationResult::Metric(proto::MetricResult {
                        metric_result: Some(match metric_result {
                            MetricResult::Average(single_metric_result) => {
                                proto::metric_result::MetricResult::SingleMetric(proto::SingleMetricResult {
                                    value: single_metric_result.value,
                                })
                            }
                            MetricResult::Stats(stats) => {
                                proto::metric_result::MetricResult::Stats(proto::StatsResult {
                                    avg: stats.avg,
                                    count: stats.count as u64,
                                    max: stats.max,
                                    min: stats.min,
                                    standard_deviation: stats.standard_deviation,
                                    sum: stats.sum,
                                })
                            }
                        })
                    })
                }
            })
        }
    }
}

impl Into<proto::Key> for Key {
    fn into(self) -> proto::Key {
        match self {
            Key::Str(s) => proto::Key { key: Some(proto::key::Key::Str(s)) },
            Key::F64(f) => proto::Key { key: Some(proto::key::Key::F64(f)) },
        }
    }
}

impl Into<proto::BucketEntry> for BucketEntry {
    fn into(self) -> proto::BucketEntry {
        proto::BucketEntry {
            key: Some(self.key.into()),
            doc_count: self.doc_count,
            sub_aggregation: self.sub_aggregation.0.into_iter().map(|(name, aggregation_result)| {
                (name, aggregation_result.into())
            }).collect(),
        }
    }
}

impl Into<proto::RangeBucketEntry> for RangeBucketEntry {
    fn into(self) -> proto::RangeBucketEntry {
        proto::RangeBucketEntry {
            key: Some(self.key.into()),
            doc_count: self.doc_count,
            sub_aggregation: self.sub_aggregation.0.into_iter().map(|(name, aggregation_result)| {
                (name, aggregation_result.into())
            }).collect(),
            from: self.from,
            to: self.to,
        }
    }
}

