from typing import ClassVar as _ClassVar
from typing import Iterable as _Iterable
from typing import Mapping as _Mapping
from typing import Optional as _Optional
from typing import Union as _Union

import utils_pb2 as _utils_pb2
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper

DESCRIPTOR: _descriptor.FileDescriptor
must: Occur
must_not: Occur
should: Occur

class Aggregation(_message.Message):
    __slots__ = ["bucket", "metric"]
    BUCKET_FIELD_NUMBER: _ClassVar[int]
    METRIC_FIELD_NUMBER: _ClassVar[int]
    bucket: BucketAggregation
    metric: MetricAggregation
    def __init__(self, bucket: _Optional[_Union[BucketAggregation, _Mapping]] = ..., metric: _Optional[_Union[MetricAggregation, _Mapping]] = ...) -> None: ...

class AggregationCollector(_message.Message):
    __slots__ = ["aggregations"]
    class AggregationsEntry(_message.Message):
        __slots__ = ["key", "value"]
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: Aggregation
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[Aggregation, _Mapping]] = ...) -> None: ...
    AGGREGATIONS_FIELD_NUMBER: _ClassVar[int]
    aggregations: _containers.MessageMap[str, Aggregation]
    def __init__(self, aggregations: _Optional[_Mapping[str, Aggregation]] = ...) -> None: ...

class AggregationCollectorOutput(_message.Message):
    __slots__ = ["aggregation_results"]
    class AggregationResultsEntry(_message.Message):
        __slots__ = ["key", "value"]
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: AggregationResult
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[AggregationResult, _Mapping]] = ...) -> None: ...
    AGGREGATION_RESULTS_FIELD_NUMBER: _ClassVar[int]
    aggregation_results: _containers.MessageMap[str, AggregationResult]
    def __init__(self, aggregation_results: _Optional[_Mapping[str, AggregationResult]] = ...) -> None: ...

class AggregationResult(_message.Message):
    __slots__ = ["bucket", "metric"]
    BUCKET_FIELD_NUMBER: _ClassVar[int]
    METRIC_FIELD_NUMBER: _ClassVar[int]
    bucket: BucketResult
    metric: MetricResult
    def __init__(self, bucket: _Optional[_Union[BucketResult, _Mapping]] = ..., metric: _Optional[_Union[MetricResult, _Mapping]] = ...) -> None: ...

class AllQuery(_message.Message):
    __slots__ = []
    def __init__(self) -> None: ...

class AverageAggregation(_message.Message):
    __slots__ = ["field"]
    FIELD_FIELD_NUMBER: _ClassVar[int]
    field: str
    def __init__(self, field: _Optional[str] = ...) -> None: ...

class BooleanQuery(_message.Message):
    __slots__ = ["subqueries"]
    SUBQUERIES_FIELD_NUMBER: _ClassVar[int]
    subqueries: _containers.RepeatedCompositeFieldContainer[BooleanSubquery]
    def __init__(self, subqueries: _Optional[_Iterable[_Union[BooleanSubquery, _Mapping]]] = ...) -> None: ...

class BooleanSubquery(_message.Message):
    __slots__ = ["occur", "query"]
    OCCUR_FIELD_NUMBER: _ClassVar[int]
    QUERY_FIELD_NUMBER: _ClassVar[int]
    occur: Occur
    query: Query
    def __init__(self, occur: _Optional[_Union[Occur, str]] = ..., query: _Optional[_Union[Query, _Mapping]] = ...) -> None: ...

class BoostQuery(_message.Message):
    __slots__ = ["query", "score"]
    QUERY_FIELD_NUMBER: _ClassVar[int]
    SCORE_FIELD_NUMBER: _ClassVar[int]
    query: Query
    score: str
    def __init__(self, query: _Optional[_Union[Query, _Mapping]] = ..., score: _Optional[str] = ...) -> None: ...

class BucketAggregation(_message.Message):
    __slots__ = ["histogram", "range", "sub_aggregation", "terms"]
    class SubAggregationEntry(_message.Message):
        __slots__ = ["key", "value"]
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: Aggregation
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[Aggregation, _Mapping]] = ...) -> None: ...
    HISTOGRAM_FIELD_NUMBER: _ClassVar[int]
    RANGE_FIELD_NUMBER: _ClassVar[int]
    SUB_AGGREGATION_FIELD_NUMBER: _ClassVar[int]
    TERMS_FIELD_NUMBER: _ClassVar[int]
    histogram: HistogramAggregation
    range: RangeAggregation
    sub_aggregation: _containers.MessageMap[str, Aggregation]
    terms: TermsAggregation
    def __init__(self, range: _Optional[_Union[RangeAggregation, _Mapping]] = ..., histogram: _Optional[_Union[HistogramAggregation, _Mapping]] = ..., terms: _Optional[_Union[TermsAggregation, _Mapping]] = ..., sub_aggregation: _Optional[_Mapping[str, Aggregation]] = ...) -> None: ...

class BucketEntry(_message.Message):
    __slots__ = ["doc_count", "key", "sub_aggregation"]
    class SubAggregationEntry(_message.Message):
        __slots__ = ["key", "value"]
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: AggregationResult
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[AggregationResult, _Mapping]] = ...) -> None: ...
    DOC_COUNT_FIELD_NUMBER: _ClassVar[int]
    KEY_FIELD_NUMBER: _ClassVar[int]
    SUB_AGGREGATION_FIELD_NUMBER: _ClassVar[int]
    doc_count: int
    key: Key
    sub_aggregation: _containers.MessageMap[str, AggregationResult]
    def __init__(self, key: _Optional[_Union[Key, _Mapping]] = ..., doc_count: _Optional[int] = ..., sub_aggregation: _Optional[_Mapping[str, AggregationResult]] = ...) -> None: ...

class BucketResult(_message.Message):
    __slots__ = ["histogram", "range", "terms"]
    HISTOGRAM_FIELD_NUMBER: _ClassVar[int]
    RANGE_FIELD_NUMBER: _ClassVar[int]
    TERMS_FIELD_NUMBER: _ClassVar[int]
    histogram: HistogramResult
    range: RangeResult
    terms: TermsResult
    def __init__(self, range: _Optional[_Union[RangeResult, _Mapping]] = ..., histogram: _Optional[_Union[HistogramResult, _Mapping]] = ..., terms: _Optional[_Union[TermsResult, _Mapping]] = ...) -> None: ...

class Collector(_message.Message):
    __slots__ = ["aggregation", "count", "facet", "reservoir_sampling", "top_docs"]
    AGGREGATION_FIELD_NUMBER: _ClassVar[int]
    COUNT_FIELD_NUMBER: _ClassVar[int]
    FACET_FIELD_NUMBER: _ClassVar[int]
    RESERVOIR_SAMPLING_FIELD_NUMBER: _ClassVar[int]
    TOP_DOCS_FIELD_NUMBER: _ClassVar[int]
    aggregation: AggregationCollector
    count: CountCollector
    facet: FacetCollector
    reservoir_sampling: ReservoirSamplingCollector
    top_docs: TopDocsCollector
    def __init__(self, top_docs: _Optional[_Union[TopDocsCollector, _Mapping]] = ..., reservoir_sampling: _Optional[_Union[ReservoirSamplingCollector, _Mapping]] = ..., count: _Optional[_Union[CountCollector, _Mapping]] = ..., facet: _Optional[_Union[FacetCollector, _Mapping]] = ..., aggregation: _Optional[_Union[AggregationCollector, _Mapping]] = ...) -> None: ...

class CollectorOutput(_message.Message):
    __slots__ = ["aggregation", "count", "documents", "facet"]
    AGGREGATION_FIELD_NUMBER: _ClassVar[int]
    COUNT_FIELD_NUMBER: _ClassVar[int]
    DOCUMENTS_FIELD_NUMBER: _ClassVar[int]
    FACET_FIELD_NUMBER: _ClassVar[int]
    aggregation: AggregationCollectorOutput
    count: CountCollectorOutput
    documents: DocumentsCollectorOutput
    facet: FacetCollectorOutput
    def __init__(self, documents: _Optional[_Union[DocumentsCollectorOutput, _Mapping]] = ..., count: _Optional[_Union[CountCollectorOutput, _Mapping]] = ..., facet: _Optional[_Union[FacetCollectorOutput, _Mapping]] = ..., aggregation: _Optional[_Union[AggregationCollectorOutput, _Mapping]] = ...) -> None: ...

class CountCollector(_message.Message):
    __slots__ = []
    def __init__(self) -> None: ...

class CountCollectorOutput(_message.Message):
    __slots__ = ["count"]
    COUNT_FIELD_NUMBER: _ClassVar[int]
    count: int
    def __init__(self, count: _Optional[int] = ...) -> None: ...

class CustomOrder(_message.Message):
    __slots__ = ["count", "key", "order", "sub_aggregation"]
    COUNT_FIELD_NUMBER: _ClassVar[int]
    KEY_FIELD_NUMBER: _ClassVar[int]
    ORDER_FIELD_NUMBER: _ClassVar[int]
    SUB_AGGREGATION_FIELD_NUMBER: _ClassVar[int]
    count: _utils_pb2.Empty
    key: _utils_pb2.Empty
    order: _utils_pb2.Order
    sub_aggregation: str
    def __init__(self, key: _Optional[_Union[_utils_pb2.Empty, _Mapping]] = ..., count: _Optional[_Union[_utils_pb2.Empty, _Mapping]] = ..., sub_aggregation: _Optional[str] = ..., order: _Optional[_Union[_utils_pb2.Order, str]] = ...) -> None: ...

class DisjunctionMaxQuery(_message.Message):
    __slots__ = ["disjuncts", "tie_breaker"]
    DISJUNCTS_FIELD_NUMBER: _ClassVar[int]
    TIE_BREAKER_FIELD_NUMBER: _ClassVar[int]
    disjuncts: _containers.RepeatedCompositeFieldContainer[Query]
    tie_breaker: str
    def __init__(self, disjuncts: _Optional[_Iterable[_Union[Query, _Mapping]]] = ..., tie_breaker: _Optional[str] = ...) -> None: ...

class DocumentsCollectorOutput(_message.Message):
    __slots__ = ["has_next", "scored_documents"]
    HAS_NEXT_FIELD_NUMBER: _ClassVar[int]
    SCORED_DOCUMENTS_FIELD_NUMBER: _ClassVar[int]
    has_next: bool
    scored_documents: _containers.RepeatedCompositeFieldContainer[ScoredDocument]
    def __init__(self, scored_documents: _Optional[_Iterable[_Union[ScoredDocument, _Mapping]]] = ..., has_next: bool = ...) -> None: ...

class EmptyQuery(_message.Message):
    __slots__ = []
    def __init__(self) -> None: ...

class ExactMatchesPromoter(_message.Message):
    __slots__ = ["boost", "slop"]
    BOOST_FIELD_NUMBER: _ClassVar[int]
    SLOP_FIELD_NUMBER: _ClassVar[int]
    boost: float
    slop: int
    def __init__(self, slop: _Optional[int] = ..., boost: _Optional[float] = ...) -> None: ...

class ExistsQuery(_message.Message):
    __slots__ = ["field"]
    FIELD_FIELD_NUMBER: _ClassVar[int]
    field: str
    def __init__(self, field: _Optional[str] = ...) -> None: ...

class FacetCollector(_message.Message):
    __slots__ = ["facets", "field"]
    FACETS_FIELD_NUMBER: _ClassVar[int]
    FIELD_FIELD_NUMBER: _ClassVar[int]
    facets: _containers.RepeatedScalarFieldContainer[str]
    field: str
    def __init__(self, field: _Optional[str] = ..., facets: _Optional[_Iterable[str]] = ...) -> None: ...

class FacetCollectorOutput(_message.Message):
    __slots__ = ["facet_counts"]
    class FacetCountsEntry(_message.Message):
        __slots__ = ["key", "value"]
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: int
        def __init__(self, key: _Optional[str] = ..., value: _Optional[int] = ...) -> None: ...
    FACET_COUNTS_FIELD_NUMBER: _ClassVar[int]
    facet_counts: _containers.ScalarMap[str, int]
    def __init__(self, facet_counts: _Optional[_Mapping[str, int]] = ...) -> None: ...

class Highlight(_message.Message):
    __slots__ = ["to"]
    FROM_FIELD_NUMBER: _ClassVar[int]
    TO_FIELD_NUMBER: _ClassVar[int]
    to: int
    def __init__(self, to: _Optional[int] = ..., **kwargs) -> None: ...

class HistogramAggregation(_message.Message):
    __slots__ = ["extended_bounds", "field", "hard_bounds", "interval", "min_doc_count", "offset"]
    EXTENDED_BOUNDS_FIELD_NUMBER: _ClassVar[int]
    FIELD_FIELD_NUMBER: _ClassVar[int]
    HARD_BOUNDS_FIELD_NUMBER: _ClassVar[int]
    INTERVAL_FIELD_NUMBER: _ClassVar[int]
    MIN_DOC_COUNT_FIELD_NUMBER: _ClassVar[int]
    OFFSET_FIELD_NUMBER: _ClassVar[int]
    extended_bounds: HistogramBounds
    field: str
    hard_bounds: HistogramBounds
    interval: float
    min_doc_count: int
    offset: float
    def __init__(self, field: _Optional[str] = ..., interval: _Optional[float] = ..., offset: _Optional[float] = ..., min_doc_count: _Optional[int] = ..., hard_bounds: _Optional[_Union[HistogramBounds, _Mapping]] = ..., extended_bounds: _Optional[_Union[HistogramBounds, _Mapping]] = ...) -> None: ...

class HistogramBounds(_message.Message):
    __slots__ = ["max", "min"]
    MAX_FIELD_NUMBER: _ClassVar[int]
    MIN_FIELD_NUMBER: _ClassVar[int]
    max: float
    min: float
    def __init__(self, min: _Optional[float] = ..., max: _Optional[float] = ...) -> None: ...

class HistogramResult(_message.Message):
    __slots__ = ["buckets"]
    BUCKETS_FIELD_NUMBER: _ClassVar[int]
    buckets: _containers.RepeatedCompositeFieldContainer[BucketEntry]
    def __init__(self, buckets: _Optional[_Iterable[_Union[BucketEntry, _Mapping]]] = ...) -> None: ...

class Key(_message.Message):
    __slots__ = ["f64", "str"]
    F64_FIELD_NUMBER: _ClassVar[int]
    STR_FIELD_NUMBER: _ClassVar[int]
    f64: float
    str: str
    def __init__(self, str: _Optional[str] = ..., f64: _Optional[float] = ...) -> None: ...

class MatchQuery(_message.Message):
    __slots__ = ["boolean_should_mode", "default_fields", "disjuction_max_mode", "exact_matches_promoter", "field_boosts", "value"]
    class FieldBoostsEntry(_message.Message):
        __slots__ = ["key", "value"]
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: float
        def __init__(self, key: _Optional[str] = ..., value: _Optional[float] = ...) -> None: ...
    BOOLEAN_SHOULD_MODE_FIELD_NUMBER: _ClassVar[int]
    DEFAULT_FIELDS_FIELD_NUMBER: _ClassVar[int]
    DISJUCTION_MAX_MODE_FIELD_NUMBER: _ClassVar[int]
    EXACT_MATCHES_PROMOTER_FIELD_NUMBER: _ClassVar[int]
    FIELD_BOOSTS_FIELD_NUMBER: _ClassVar[int]
    VALUE_FIELD_NUMBER: _ClassVar[int]
    boolean_should_mode: MatchQueryBooleanShouldMode
    default_fields: _containers.RepeatedScalarFieldContainer[str]
    disjuction_max_mode: MatchQueryDisjuctionMaxMode
    exact_matches_promoter: ExactMatchesPromoter
    field_boosts: _containers.ScalarMap[str, float]
    value: str
    def __init__(self, value: _Optional[str] = ..., default_fields: _Optional[_Iterable[str]] = ..., boolean_should_mode: _Optional[_Union[MatchQueryBooleanShouldMode, _Mapping]] = ..., disjuction_max_mode: _Optional[_Union[MatchQueryDisjuctionMaxMode, _Mapping]] = ..., field_boosts: _Optional[_Mapping[str, float]] = ..., exact_matches_promoter: _Optional[_Union[ExactMatchesPromoter, _Mapping]] = ...) -> None: ...

class MatchQueryBooleanShouldMode(_message.Message):
    __slots__ = []
    def __init__(self) -> None: ...

class MatchQueryDisjuctionMaxMode(_message.Message):
    __slots__ = ["tie_breaker"]
    TIE_BREAKER_FIELD_NUMBER: _ClassVar[int]
    tie_breaker: float
    def __init__(self, tie_breaker: _Optional[float] = ...) -> None: ...

class MetricAggregation(_message.Message):
    __slots__ = ["average", "stats"]
    AVERAGE_FIELD_NUMBER: _ClassVar[int]
    STATS_FIELD_NUMBER: _ClassVar[int]
    average: AverageAggregation
    stats: StatsAggregation
    def __init__(self, average: _Optional[_Union[AverageAggregation, _Mapping]] = ..., stats: _Optional[_Union[StatsAggregation, _Mapping]] = ...) -> None: ...

class MetricResult(_message.Message):
    __slots__ = ["single_metric", "stats"]
    SINGLE_METRIC_FIELD_NUMBER: _ClassVar[int]
    STATS_FIELD_NUMBER: _ClassVar[int]
    single_metric: SingleMetricResult
    stats: StatsResult
    def __init__(self, single_metric: _Optional[_Union[SingleMetricResult, _Mapping]] = ..., stats: _Optional[_Union[StatsResult, _Mapping]] = ...) -> None: ...

class MoreLikeThisQuery(_message.Message):
    __slots__ = ["boost", "document", "max_doc_frequency", "max_query_terms", "max_word_length", "min_doc_frequency", "min_term_frequency", "min_word_length", "stop_words"]
    BOOST_FIELD_NUMBER: _ClassVar[int]
    DOCUMENT_FIELD_NUMBER: _ClassVar[int]
    MAX_DOC_FREQUENCY_FIELD_NUMBER: _ClassVar[int]
    MAX_QUERY_TERMS_FIELD_NUMBER: _ClassVar[int]
    MAX_WORD_LENGTH_FIELD_NUMBER: _ClassVar[int]
    MIN_DOC_FREQUENCY_FIELD_NUMBER: _ClassVar[int]
    MIN_TERM_FREQUENCY_FIELD_NUMBER: _ClassVar[int]
    MIN_WORD_LENGTH_FIELD_NUMBER: _ClassVar[int]
    STOP_WORDS_FIELD_NUMBER: _ClassVar[int]
    boost: str
    document: str
    max_doc_frequency: int
    max_query_terms: int
    max_word_length: int
    min_doc_frequency: int
    min_term_frequency: int
    min_word_length: int
    stop_words: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, document: _Optional[str] = ..., min_doc_frequency: _Optional[int] = ..., max_doc_frequency: _Optional[int] = ..., min_term_frequency: _Optional[int] = ..., max_query_terms: _Optional[int] = ..., min_word_length: _Optional[int] = ..., max_word_length: _Optional[int] = ..., boost: _Optional[str] = ..., stop_words: _Optional[_Iterable[str]] = ...) -> None: ...

class PhraseQuery(_message.Message):
    __slots__ = ["field", "slop", "value"]
    FIELD_FIELD_NUMBER: _ClassVar[int]
    SLOP_FIELD_NUMBER: _ClassVar[int]
    VALUE_FIELD_NUMBER: _ClassVar[int]
    field: str
    slop: int
    value: str
    def __init__(self, field: _Optional[str] = ..., value: _Optional[str] = ..., slop: _Optional[int] = ...) -> None: ...

class Query(_message.Message):
    __slots__ = ["all", "boolean", "boost", "disjunction_max", "empty", "exists", "match", "more_like_this", "phrase", "range", "regex", "term"]
    ALL_FIELD_NUMBER: _ClassVar[int]
    BOOLEAN_FIELD_NUMBER: _ClassVar[int]
    BOOST_FIELD_NUMBER: _ClassVar[int]
    DISJUNCTION_MAX_FIELD_NUMBER: _ClassVar[int]
    EMPTY_FIELD_NUMBER: _ClassVar[int]
    EXISTS_FIELD_NUMBER: _ClassVar[int]
    MATCH_FIELD_NUMBER: _ClassVar[int]
    MORE_LIKE_THIS_FIELD_NUMBER: _ClassVar[int]
    PHRASE_FIELD_NUMBER: _ClassVar[int]
    RANGE_FIELD_NUMBER: _ClassVar[int]
    REGEX_FIELD_NUMBER: _ClassVar[int]
    TERM_FIELD_NUMBER: _ClassVar[int]
    all: AllQuery
    boolean: BooleanQuery
    boost: BoostQuery
    disjunction_max: DisjunctionMaxQuery
    empty: EmptyQuery
    exists: ExistsQuery
    match: MatchQuery
    more_like_this: MoreLikeThisQuery
    phrase: PhraseQuery
    range: RangeQuery
    regex: RegexQuery
    term: TermQuery
    def __init__(self, boolean: _Optional[_Union[BooleanQuery, _Mapping]] = ..., match: _Optional[_Union[MatchQuery, _Mapping]] = ..., regex: _Optional[_Union[RegexQuery, _Mapping]] = ..., term: _Optional[_Union[TermQuery, _Mapping]] = ..., phrase: _Optional[_Union[PhraseQuery, _Mapping]] = ..., range: _Optional[_Union[RangeQuery, _Mapping]] = ..., all: _Optional[_Union[AllQuery, _Mapping]] = ..., more_like_this: _Optional[_Union[MoreLikeThisQuery, _Mapping]] = ..., boost: _Optional[_Union[BoostQuery, _Mapping]] = ..., disjunction_max: _Optional[_Union[DisjunctionMaxQuery, _Mapping]] = ..., empty: _Optional[_Union[EmptyQuery, _Mapping]] = ..., exists: _Optional[_Union[ExistsQuery, _Mapping]] = ...) -> None: ...

class RandomDocument(_message.Message):
    __slots__ = ["document", "index_alias", "score"]
    DOCUMENT_FIELD_NUMBER: _ClassVar[int]
    INDEX_ALIAS_FIELD_NUMBER: _ClassVar[int]
    SCORE_FIELD_NUMBER: _ClassVar[int]
    document: str
    index_alias: str
    score: Score
    def __init__(self, document: _Optional[str] = ..., score: _Optional[_Union[Score, _Mapping]] = ..., index_alias: _Optional[str] = ...) -> None: ...

class Range(_message.Message):
    __slots__ = ["including_left", "including_right", "left", "right"]
    INCLUDING_LEFT_FIELD_NUMBER: _ClassVar[int]
    INCLUDING_RIGHT_FIELD_NUMBER: _ClassVar[int]
    LEFT_FIELD_NUMBER: _ClassVar[int]
    RIGHT_FIELD_NUMBER: _ClassVar[int]
    including_left: bool
    including_right: bool
    left: str
    right: str
    def __init__(self, left: _Optional[str] = ..., right: _Optional[str] = ..., including_left: bool = ..., including_right: bool = ...) -> None: ...

class RangeAggregation(_message.Message):
    __slots__ = ["field", "ranges"]
    FIELD_FIELD_NUMBER: _ClassVar[int]
    RANGES_FIELD_NUMBER: _ClassVar[int]
    field: str
    ranges: _containers.RepeatedCompositeFieldContainer[RangeAggregationRange]
    def __init__(self, field: _Optional[str] = ..., ranges: _Optional[_Iterable[_Union[RangeAggregationRange, _Mapping]]] = ...) -> None: ...

class RangeAggregationRange(_message.Message):
    __slots__ = ["key", "to"]
    FROM_FIELD_NUMBER: _ClassVar[int]
    KEY_FIELD_NUMBER: _ClassVar[int]
    TO_FIELD_NUMBER: _ClassVar[int]
    key: str
    to: float
    def __init__(self, to: _Optional[float] = ..., key: _Optional[str] = ..., **kwargs) -> None: ...

class RangeBucketEntry(_message.Message):
    __slots__ = ["doc_count", "key", "sub_aggregation", "to"]
    class SubAggregationEntry(_message.Message):
        __slots__ = ["key", "value"]
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: AggregationResult
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[AggregationResult, _Mapping]] = ...) -> None: ...
    DOC_COUNT_FIELD_NUMBER: _ClassVar[int]
    FROM_FIELD_NUMBER: _ClassVar[int]
    KEY_FIELD_NUMBER: _ClassVar[int]
    SUB_AGGREGATION_FIELD_NUMBER: _ClassVar[int]
    TO_FIELD_NUMBER: _ClassVar[int]
    doc_count: int
    key: Key
    sub_aggregation: _containers.MessageMap[str, AggregationResult]
    to: float
    def __init__(self, key: _Optional[_Union[Key, _Mapping]] = ..., doc_count: _Optional[int] = ..., sub_aggregation: _Optional[_Mapping[str, AggregationResult]] = ..., to: _Optional[float] = ..., **kwargs) -> None: ...

class RangeQuery(_message.Message):
    __slots__ = ["field", "value"]
    FIELD_FIELD_NUMBER: _ClassVar[int]
    VALUE_FIELD_NUMBER: _ClassVar[int]
    field: str
    value: Range
    def __init__(self, field: _Optional[str] = ..., value: _Optional[_Union[Range, _Mapping]] = ...) -> None: ...

class RangeResult(_message.Message):
    __slots__ = ["buckets"]
    BUCKETS_FIELD_NUMBER: _ClassVar[int]
    buckets: _containers.RepeatedCompositeFieldContainer[RangeBucketEntry]
    def __init__(self, buckets: _Optional[_Iterable[_Union[RangeBucketEntry, _Mapping]]] = ...) -> None: ...

class RegexQuery(_message.Message):
    __slots__ = ["field", "value"]
    FIELD_FIELD_NUMBER: _ClassVar[int]
    VALUE_FIELD_NUMBER: _ClassVar[int]
    field: str
    value: str
    def __init__(self, field: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...

class ReservoirSamplingCollector(_message.Message):
    __slots__ = ["fields", "limit"]
    FIELDS_FIELD_NUMBER: _ClassVar[int]
    LIMIT_FIELD_NUMBER: _ClassVar[int]
    fields: _containers.RepeatedScalarFieldContainer[str]
    limit: int
    def __init__(self, limit: _Optional[int] = ..., fields: _Optional[_Iterable[str]] = ...) -> None: ...

class ReservoirSamplingCollectorOutput(_message.Message):
    __slots__ = ["documents"]
    DOCUMENTS_FIELD_NUMBER: _ClassVar[int]
    documents: _containers.RepeatedCompositeFieldContainer[RandomDocument]
    def __init__(self, documents: _Optional[_Iterable[_Union[RandomDocument, _Mapping]]] = ...) -> None: ...

class Score(_message.Message):
    __slots__ = ["f64_score", "u64_score"]
    F64_SCORE_FIELD_NUMBER: _ClassVar[int]
    U64_SCORE_FIELD_NUMBER: _ClassVar[int]
    f64_score: float
    u64_score: int
    def __init__(self, f64_score: _Optional[float] = ..., u64_score: _Optional[int] = ...) -> None: ...

class ScoredDocument(_message.Message):
    __slots__ = ["document", "index_alias", "position", "score", "snippets"]
    class SnippetsEntry(_message.Message):
        __slots__ = ["key", "value"]
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: Snippet
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[Snippet, _Mapping]] = ...) -> None: ...
    DOCUMENT_FIELD_NUMBER: _ClassVar[int]
    INDEX_ALIAS_FIELD_NUMBER: _ClassVar[int]
    POSITION_FIELD_NUMBER: _ClassVar[int]
    SCORE_FIELD_NUMBER: _ClassVar[int]
    SNIPPETS_FIELD_NUMBER: _ClassVar[int]
    document: str
    index_alias: str
    position: int
    score: Score
    snippets: _containers.MessageMap[str, Snippet]
    def __init__(self, document: _Optional[str] = ..., score: _Optional[_Union[Score, _Mapping]] = ..., position: _Optional[int] = ..., snippets: _Optional[_Mapping[str, Snippet]] = ..., index_alias: _Optional[str] = ...) -> None: ...

class Scorer(_message.Message):
    __slots__ = ["eval_expr", "order_by"]
    EVAL_EXPR_FIELD_NUMBER: _ClassVar[int]
    ORDER_BY_FIELD_NUMBER: _ClassVar[int]
    eval_expr: str
    order_by: str
    def __init__(self, eval_expr: _Optional[str] = ..., order_by: _Optional[str] = ...) -> None: ...

class SearchResponse(_message.Message):
    __slots__ = ["collector_outputs", "elapsed_secs"]
    COLLECTOR_OUTPUTS_FIELD_NUMBER: _ClassVar[int]
    ELAPSED_SECS_FIELD_NUMBER: _ClassVar[int]
    collector_outputs: _containers.RepeatedCompositeFieldContainer[CollectorOutput]
    elapsed_secs: float
    def __init__(self, elapsed_secs: _Optional[float] = ..., collector_outputs: _Optional[_Iterable[_Union[CollectorOutput, _Mapping]]] = ...) -> None: ...

class SingleMetricResult(_message.Message):
    __slots__ = ["value"]
    VALUE_FIELD_NUMBER: _ClassVar[int]
    value: float
    def __init__(self, value: _Optional[float] = ...) -> None: ...

class Snippet(_message.Message):
    __slots__ = ["fragment", "highlights", "html"]
    FRAGMENT_FIELD_NUMBER: _ClassVar[int]
    HIGHLIGHTS_FIELD_NUMBER: _ClassVar[int]
    HTML_FIELD_NUMBER: _ClassVar[int]
    fragment: bytes
    highlights: _containers.RepeatedCompositeFieldContainer[Highlight]
    html: str
    def __init__(self, fragment: _Optional[bytes] = ..., highlights: _Optional[_Iterable[_Union[Highlight, _Mapping]]] = ..., html: _Optional[str] = ...) -> None: ...

class StatsAggregation(_message.Message):
    __slots__ = ["field"]
    FIELD_FIELD_NUMBER: _ClassVar[int]
    field: str
    def __init__(self, field: _Optional[str] = ...) -> None: ...

class StatsResult(_message.Message):
    __slots__ = ["avg", "count", "max", "min", "sum"]
    AVG_FIELD_NUMBER: _ClassVar[int]
    COUNT_FIELD_NUMBER: _ClassVar[int]
    MAX_FIELD_NUMBER: _ClassVar[int]
    MIN_FIELD_NUMBER: _ClassVar[int]
    SUM_FIELD_NUMBER: _ClassVar[int]
    avg: float
    count: int
    max: float
    min: float
    sum: float
    def __init__(self, count: _Optional[int] = ..., sum: _Optional[float] = ..., min: _Optional[float] = ..., max: _Optional[float] = ..., avg: _Optional[float] = ...) -> None: ...

class TermQuery(_message.Message):
    __slots__ = ["field", "value"]
    FIELD_FIELD_NUMBER: _ClassVar[int]
    VALUE_FIELD_NUMBER: _ClassVar[int]
    field: str
    value: str
    def __init__(self, field: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...

class TermsAggregation(_message.Message):
    __slots__ = ["field", "min_doc_count", "order", "segment_size", "show_term_doc_count_error", "size", "split_size"]
    FIELD_FIELD_NUMBER: _ClassVar[int]
    MIN_DOC_COUNT_FIELD_NUMBER: _ClassVar[int]
    ORDER_FIELD_NUMBER: _ClassVar[int]
    SEGMENT_SIZE_FIELD_NUMBER: _ClassVar[int]
    SHOW_TERM_DOC_COUNT_ERROR_FIELD_NUMBER: _ClassVar[int]
    SIZE_FIELD_NUMBER: _ClassVar[int]
    SPLIT_SIZE_FIELD_NUMBER: _ClassVar[int]
    field: str
    min_doc_count: int
    order: CustomOrder
    segment_size: int
    show_term_doc_count_error: bool
    size: int
    split_size: int
    def __init__(self, field: _Optional[str] = ..., size: _Optional[int] = ..., split_size: _Optional[int] = ..., segment_size: _Optional[int] = ..., show_term_doc_count_error: bool = ..., min_doc_count: _Optional[int] = ..., order: _Optional[_Union[CustomOrder, _Mapping]] = ...) -> None: ...

class TermsResult(_message.Message):
    __slots__ = ["buckets", "doc_count_error_upper_bound", "sum_other_doc_count"]
    BUCKETS_FIELD_NUMBER: _ClassVar[int]
    DOC_COUNT_ERROR_UPPER_BOUND_FIELD_NUMBER: _ClassVar[int]
    SUM_OTHER_DOC_COUNT_FIELD_NUMBER: _ClassVar[int]
    buckets: _containers.RepeatedCompositeFieldContainer[BucketEntry]
    doc_count_error_upper_bound: int
    sum_other_doc_count: int
    def __init__(self, buckets: _Optional[_Iterable[_Union[BucketEntry, _Mapping]]] = ..., sum_other_doc_count: _Optional[int] = ..., doc_count_error_upper_bound: _Optional[int] = ...) -> None: ...

class TopDocsCollector(_message.Message):
    __slots__ = ["explain", "fields", "limit", "offset", "scorer", "snippet_configs"]
    class SnippetConfigsEntry(_message.Message):
        __slots__ = ["key", "value"]
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: int
        def __init__(self, key: _Optional[str] = ..., value: _Optional[int] = ...) -> None: ...
    EXPLAIN_FIELD_NUMBER: _ClassVar[int]
    FIELDS_FIELD_NUMBER: _ClassVar[int]
    LIMIT_FIELD_NUMBER: _ClassVar[int]
    OFFSET_FIELD_NUMBER: _ClassVar[int]
    SCORER_FIELD_NUMBER: _ClassVar[int]
    SNIPPET_CONFIGS_FIELD_NUMBER: _ClassVar[int]
    explain: bool
    fields: _containers.RepeatedScalarFieldContainer[str]
    limit: int
    offset: int
    scorer: Scorer
    snippet_configs: _containers.ScalarMap[str, int]
    def __init__(self, limit: _Optional[int] = ..., offset: _Optional[int] = ..., scorer: _Optional[_Union[Scorer, _Mapping]] = ..., snippet_configs: _Optional[_Mapping[str, int]] = ..., explain: bool = ..., fields: _Optional[_Iterable[str]] = ...) -> None: ...

class Occur(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = []
