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

class Occur(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = []
    should: _ClassVar[Occur]
    must: _ClassVar[Occur]
    must_not: _ClassVar[Occur]
should: Occur
must: Occur
must_not: Occur

class TermFieldMapperConfig(_message.Message):
    __slots__ = ["fields"]
    FIELDS_FIELD_NUMBER: _ClassVar[int]
    fields: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, fields: _Optional[_Iterable[str]] = ...) -> None: ...

class MatchQueryBooleanShouldMode(_message.Message):
    __slots__ = []
    def __init__(self) -> None: ...

class MatchQueryDisjuctionMaxMode(_message.Message):
    __slots__ = ["tie_breaker"]
    TIE_BREAKER_FIELD_NUMBER: _ClassVar[int]
    tie_breaker: float
    def __init__(self, tie_breaker: _Optional[float] = ...) -> None: ...

class ExactMatchesPromoter(_message.Message):
    __slots__ = ["slop", "boost", "fields"]
    SLOP_FIELD_NUMBER: _ClassVar[int]
    BOOST_FIELD_NUMBER: _ClassVar[int]
    FIELDS_FIELD_NUMBER: _ClassVar[int]
    slop: int
    boost: float
    fields: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, slop: _Optional[int] = ..., boost: _Optional[float] = ..., fields: _Optional[_Iterable[str]] = ...) -> None: ...

class NerMatchesPromoter(_message.Message):
    __slots__ = ["boost", "fields"]
    BOOST_FIELD_NUMBER: _ClassVar[int]
    FIELDS_FIELD_NUMBER: _ClassVar[int]
    boost: float
    fields: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, boost: _Optional[float] = ..., fields: _Optional[_Iterable[str]] = ...) -> None: ...

class MorphologyConfig(_message.Message):
    __slots__ = ["derive_tenses_coefficient"]
    DERIVE_TENSES_COEFFICIENT_FIELD_NUMBER: _ClassVar[int]
    derive_tenses_coefficient: float
    def __init__(self, derive_tenses_coefficient: _Optional[float] = ...) -> None: ...

class QueryParserConfig(_message.Message):
    __slots__ = ["field_aliases", "field_boosts", "term_field_mapper_configs", "term_limit", "default_fields", "boolean_should_mode", "disjuction_max_mode", "exact_matches_promoter", "removed_fields", "morphology_configs", "query_language"]
    class FieldAliasesEntry(_message.Message):
        __slots__ = ["key", "value"]
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    class FieldBoostsEntry(_message.Message):
        __slots__ = ["key", "value"]
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: float
        def __init__(self, key: _Optional[str] = ..., value: _Optional[float] = ...) -> None: ...
    class TermFieldMapperConfigsEntry(_message.Message):
        __slots__ = ["key", "value"]
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: TermFieldMapperConfig
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[TermFieldMapperConfig, _Mapping]] = ...) -> None: ...
    class MorphologyConfigsEntry(_message.Message):
        __slots__ = ["key", "value"]
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: MorphologyConfig
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[MorphologyConfig, _Mapping]] = ...) -> None: ...
    FIELD_ALIASES_FIELD_NUMBER: _ClassVar[int]
    FIELD_BOOSTS_FIELD_NUMBER: _ClassVar[int]
    TERM_FIELD_MAPPER_CONFIGS_FIELD_NUMBER: _ClassVar[int]
    TERM_LIMIT_FIELD_NUMBER: _ClassVar[int]
    DEFAULT_FIELDS_FIELD_NUMBER: _ClassVar[int]
    BOOLEAN_SHOULD_MODE_FIELD_NUMBER: _ClassVar[int]
    DISJUCTION_MAX_MODE_FIELD_NUMBER: _ClassVar[int]
    EXACT_MATCHES_PROMOTER_FIELD_NUMBER: _ClassVar[int]
    REMOVED_FIELDS_FIELD_NUMBER: _ClassVar[int]
    MORPHOLOGY_CONFIGS_FIELD_NUMBER: _ClassVar[int]
    QUERY_LANGUAGE_FIELD_NUMBER: _ClassVar[int]
    field_aliases: _containers.ScalarMap[str, str]
    field_boosts: _containers.ScalarMap[str, float]
    term_field_mapper_configs: _containers.MessageMap[str, TermFieldMapperConfig]
    term_limit: int
    default_fields: _containers.RepeatedScalarFieldContainer[str]
    boolean_should_mode: MatchQueryBooleanShouldMode
    disjuction_max_mode: MatchQueryDisjuctionMaxMode
    exact_matches_promoter: ExactMatchesPromoter
    removed_fields: _containers.RepeatedScalarFieldContainer[str]
    morphology_configs: _containers.MessageMap[str, MorphologyConfig]
    query_language: str
    def __init__(self, field_aliases: _Optional[_Mapping[str, str]] = ..., field_boosts: _Optional[_Mapping[str, float]] = ..., term_field_mapper_configs: _Optional[_Mapping[str, TermFieldMapperConfig]] = ..., term_limit: _Optional[int] = ..., default_fields: _Optional[_Iterable[str]] = ..., boolean_should_mode: _Optional[_Union[MatchQueryBooleanShouldMode, _Mapping]] = ..., disjuction_max_mode: _Optional[_Union[MatchQueryDisjuctionMaxMode, _Mapping]] = ..., exact_matches_promoter: _Optional[_Union[ExactMatchesPromoter, _Mapping]] = ..., removed_fields: _Optional[_Iterable[str]] = ..., morphology_configs: _Optional[_Mapping[str, MorphologyConfig]] = ..., query_language: _Optional[str] = ...) -> None: ...

class SearchResponse(_message.Message):
    __slots__ = ["elapsed_secs", "collector_outputs"]
    ELAPSED_SECS_FIELD_NUMBER: _ClassVar[int]
    COLLECTOR_OUTPUTS_FIELD_NUMBER: _ClassVar[int]
    elapsed_secs: float
    collector_outputs: _containers.RepeatedCompositeFieldContainer[CollectorOutput]
    def __init__(self, elapsed_secs: _Optional[float] = ..., collector_outputs: _Optional[_Iterable[_Union[CollectorOutput, _Mapping]]] = ...) -> None: ...

class Query(_message.Message):
    __slots__ = ["boolean", "match", "regex", "term", "phrase", "range", "all", "more_like_this", "boost", "disjunction_max", "empty", "exists"]
    BOOLEAN_FIELD_NUMBER: _ClassVar[int]
    MATCH_FIELD_NUMBER: _ClassVar[int]
    REGEX_FIELD_NUMBER: _ClassVar[int]
    TERM_FIELD_NUMBER: _ClassVar[int]
    PHRASE_FIELD_NUMBER: _ClassVar[int]
    RANGE_FIELD_NUMBER: _ClassVar[int]
    ALL_FIELD_NUMBER: _ClassVar[int]
    MORE_LIKE_THIS_FIELD_NUMBER: _ClassVar[int]
    BOOST_FIELD_NUMBER: _ClassVar[int]
    DISJUNCTION_MAX_FIELD_NUMBER: _ClassVar[int]
    EMPTY_FIELD_NUMBER: _ClassVar[int]
    EXISTS_FIELD_NUMBER: _ClassVar[int]
    boolean: BooleanQuery
    match: MatchQuery
    regex: RegexQuery
    term: TermQuery
    phrase: PhraseQuery
    range: RangeQuery
    all: AllQuery
    more_like_this: MoreLikeThisQuery
    boost: BoostQuery
    disjunction_max: DisjunctionMaxQuery
    empty: EmptyQuery
    exists: ExistsQuery
    def __init__(self, boolean: _Optional[_Union[BooleanQuery, _Mapping]] = ..., match: _Optional[_Union[MatchQuery, _Mapping]] = ..., regex: _Optional[_Union[RegexQuery, _Mapping]] = ..., term: _Optional[_Union[TermQuery, _Mapping]] = ..., phrase: _Optional[_Union[PhraseQuery, _Mapping]] = ..., range: _Optional[_Union[RangeQuery, _Mapping]] = ..., all: _Optional[_Union[AllQuery, _Mapping]] = ..., more_like_this: _Optional[_Union[MoreLikeThisQuery, _Mapping]] = ..., boost: _Optional[_Union[BoostQuery, _Mapping]] = ..., disjunction_max: _Optional[_Union[DisjunctionMaxQuery, _Mapping]] = ..., empty: _Optional[_Union[EmptyQuery, _Mapping]] = ..., exists: _Optional[_Union[ExistsQuery, _Mapping]] = ...) -> None: ...

class AllQuery(_message.Message):
    __slots__ = []
    def __init__(self) -> None: ...

class EmptyQuery(_message.Message):
    __slots__ = []
    def __init__(self) -> None: ...

class BoostQuery(_message.Message):
    __slots__ = ["query", "score"]
    QUERY_FIELD_NUMBER: _ClassVar[int]
    SCORE_FIELD_NUMBER: _ClassVar[int]
    query: Query
    score: str
    def __init__(self, query: _Optional[_Union[Query, _Mapping]] = ..., score: _Optional[str] = ...) -> None: ...

class DisjunctionMaxQuery(_message.Message):
    __slots__ = ["disjuncts", "tie_breaker"]
    DISJUNCTS_FIELD_NUMBER: _ClassVar[int]
    TIE_BREAKER_FIELD_NUMBER: _ClassVar[int]
    disjuncts: _containers.RepeatedCompositeFieldContainer[Query]
    tie_breaker: str
    def __init__(self, disjuncts: _Optional[_Iterable[_Union[Query, _Mapping]]] = ..., tie_breaker: _Optional[str] = ...) -> None: ...

class MoreLikeThisQuery(_message.Message):
    __slots__ = ["document", "min_doc_frequency", "max_doc_frequency", "min_term_frequency", "max_query_terms", "min_word_length", "max_word_length", "boost", "stop_words"]
    DOCUMENT_FIELD_NUMBER: _ClassVar[int]
    MIN_DOC_FREQUENCY_FIELD_NUMBER: _ClassVar[int]
    MAX_DOC_FREQUENCY_FIELD_NUMBER: _ClassVar[int]
    MIN_TERM_FREQUENCY_FIELD_NUMBER: _ClassVar[int]
    MAX_QUERY_TERMS_FIELD_NUMBER: _ClassVar[int]
    MIN_WORD_LENGTH_FIELD_NUMBER: _ClassVar[int]
    MAX_WORD_LENGTH_FIELD_NUMBER: _ClassVar[int]
    BOOST_FIELD_NUMBER: _ClassVar[int]
    STOP_WORDS_FIELD_NUMBER: _ClassVar[int]
    document: str
    min_doc_frequency: int
    max_doc_frequency: int
    min_term_frequency: int
    max_query_terms: int
    min_word_length: int
    max_word_length: int
    boost: str
    stop_words: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, document: _Optional[str] = ..., min_doc_frequency: _Optional[int] = ..., max_doc_frequency: _Optional[int] = ..., min_term_frequency: _Optional[int] = ..., max_query_terms: _Optional[int] = ..., min_word_length: _Optional[int] = ..., max_word_length: _Optional[int] = ..., boost: _Optional[str] = ..., stop_words: _Optional[_Iterable[str]] = ...) -> None: ...

class PhraseQuery(_message.Message):
    __slots__ = ["field", "value", "slop"]
    FIELD_FIELD_NUMBER: _ClassVar[int]
    VALUE_FIELD_NUMBER: _ClassVar[int]
    SLOP_FIELD_NUMBER: _ClassVar[int]
    field: str
    value: str
    slop: int
    def __init__(self, field: _Optional[str] = ..., value: _Optional[str] = ..., slop: _Optional[int] = ...) -> None: ...

class RangeQuery(_message.Message):
    __slots__ = ["field", "value"]
    FIELD_FIELD_NUMBER: _ClassVar[int]
    VALUE_FIELD_NUMBER: _ClassVar[int]
    field: str
    value: Range
    def __init__(self, field: _Optional[str] = ..., value: _Optional[_Union[Range, _Mapping]] = ...) -> None: ...

class MatchQuery(_message.Message):
    __slots__ = ["value", "query_parser_config"]
    VALUE_FIELD_NUMBER: _ClassVar[int]
    QUERY_PARSER_CONFIG_FIELD_NUMBER: _ClassVar[int]
    value: str
    query_parser_config: QueryParserConfig
    def __init__(self, value: _Optional[str] = ..., query_parser_config: _Optional[_Union[QueryParserConfig, _Mapping]] = ...) -> None: ...

class BooleanSubquery(_message.Message):
    __slots__ = ["occur", "query"]
    OCCUR_FIELD_NUMBER: _ClassVar[int]
    QUERY_FIELD_NUMBER: _ClassVar[int]
    occur: Occur
    query: Query
    def __init__(self, occur: _Optional[_Union[Occur, str]] = ..., query: _Optional[_Union[Query, _Mapping]] = ...) -> None: ...

class BooleanQuery(_message.Message):
    __slots__ = ["subqueries"]
    SUBQUERIES_FIELD_NUMBER: _ClassVar[int]
    subqueries: _containers.RepeatedCompositeFieldContainer[BooleanSubquery]
    def __init__(self, subqueries: _Optional[_Iterable[_Union[BooleanSubquery, _Mapping]]] = ...) -> None: ...

class RegexQuery(_message.Message):
    __slots__ = ["field", "value"]
    FIELD_FIELD_NUMBER: _ClassVar[int]
    VALUE_FIELD_NUMBER: _ClassVar[int]
    field: str
    value: str
    def __init__(self, field: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...

class TermQuery(_message.Message):
    __slots__ = ["field", "value"]
    FIELD_FIELD_NUMBER: _ClassVar[int]
    VALUE_FIELD_NUMBER: _ClassVar[int]
    field: str
    value: str
    def __init__(self, field: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...

class ExistsQuery(_message.Message):
    __slots__ = ["field"]
    FIELD_FIELD_NUMBER: _ClassVar[int]
    field: str
    def __init__(self, field: _Optional[str] = ...) -> None: ...

class Aggregation(_message.Message):
    __slots__ = ["bucket", "metric"]
    BUCKET_FIELD_NUMBER: _ClassVar[int]
    METRIC_FIELD_NUMBER: _ClassVar[int]
    bucket: BucketAggregation
    metric: MetricAggregation
    def __init__(self, bucket: _Optional[_Union[BucketAggregation, _Mapping]] = ..., metric: _Optional[_Union[MetricAggregation, _Mapping]] = ...) -> None: ...

class BucketAggregation(_message.Message):
    __slots__ = ["range", "histogram", "terms", "sub_aggregation"]
    class SubAggregationEntry(_message.Message):
        __slots__ = ["key", "value"]
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: Aggregation
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[Aggregation, _Mapping]] = ...) -> None: ...
    RANGE_FIELD_NUMBER: _ClassVar[int]
    HISTOGRAM_FIELD_NUMBER: _ClassVar[int]
    TERMS_FIELD_NUMBER: _ClassVar[int]
    SUB_AGGREGATION_FIELD_NUMBER: _ClassVar[int]
    range: RangeAggregation
    histogram: HistogramAggregation
    terms: TermsAggregation
    sub_aggregation: _containers.MessageMap[str, Aggregation]
    def __init__(self, range: _Optional[_Union[RangeAggregation, _Mapping]] = ..., histogram: _Optional[_Union[HistogramAggregation, _Mapping]] = ..., terms: _Optional[_Union[TermsAggregation, _Mapping]] = ..., sub_aggregation: _Optional[_Mapping[str, Aggregation]] = ...) -> None: ...

class RangeAggregation(_message.Message):
    __slots__ = ["field", "ranges"]
    FIELD_FIELD_NUMBER: _ClassVar[int]
    RANGES_FIELD_NUMBER: _ClassVar[int]
    field: str
    ranges: _containers.RepeatedCompositeFieldContainer[RangeAggregationRange]
    def __init__(self, field: _Optional[str] = ..., ranges: _Optional[_Iterable[_Union[RangeAggregationRange, _Mapping]]] = ...) -> None: ...

class RangeAggregationRange(_message.Message):
    __slots__ = ["to", "key"]
    FROM_FIELD_NUMBER: _ClassVar[int]
    TO_FIELD_NUMBER: _ClassVar[int]
    KEY_FIELD_NUMBER: _ClassVar[int]
    to: float
    key: str
    def __init__(self, to: _Optional[float] = ..., key: _Optional[str] = ..., **kwargs) -> None: ...

class HistogramAggregation(_message.Message):
    __slots__ = ["field", "interval", "offset", "min_doc_count", "hard_bounds", "extended_bounds"]
    FIELD_FIELD_NUMBER: _ClassVar[int]
    INTERVAL_FIELD_NUMBER: _ClassVar[int]
    OFFSET_FIELD_NUMBER: _ClassVar[int]
    MIN_DOC_COUNT_FIELD_NUMBER: _ClassVar[int]
    HARD_BOUNDS_FIELD_NUMBER: _ClassVar[int]
    EXTENDED_BOUNDS_FIELD_NUMBER: _ClassVar[int]
    field: str
    interval: float
    offset: float
    min_doc_count: int
    hard_bounds: HistogramBounds
    extended_bounds: HistogramBounds
    def __init__(self, field: _Optional[str] = ..., interval: _Optional[float] = ..., offset: _Optional[float] = ..., min_doc_count: _Optional[int] = ..., hard_bounds: _Optional[_Union[HistogramBounds, _Mapping]] = ..., extended_bounds: _Optional[_Union[HistogramBounds, _Mapping]] = ...) -> None: ...

class HistogramBounds(_message.Message):
    __slots__ = ["min", "max"]
    MIN_FIELD_NUMBER: _ClassVar[int]
    MAX_FIELD_NUMBER: _ClassVar[int]
    min: float
    max: float
    def __init__(self, min: _Optional[float] = ..., max: _Optional[float] = ...) -> None: ...

class TermsAggregation(_message.Message):
    __slots__ = ["field", "size", "split_size", "segment_size", "show_term_doc_count_error", "min_doc_count", "order"]
    FIELD_FIELD_NUMBER: _ClassVar[int]
    SIZE_FIELD_NUMBER: _ClassVar[int]
    SPLIT_SIZE_FIELD_NUMBER: _ClassVar[int]
    SEGMENT_SIZE_FIELD_NUMBER: _ClassVar[int]
    SHOW_TERM_DOC_COUNT_ERROR_FIELD_NUMBER: _ClassVar[int]
    MIN_DOC_COUNT_FIELD_NUMBER: _ClassVar[int]
    ORDER_FIELD_NUMBER: _ClassVar[int]
    field: str
    size: int
    split_size: int
    segment_size: int
    show_term_doc_count_error: bool
    min_doc_count: int
    order: CustomOrder
    def __init__(self, field: _Optional[str] = ..., size: _Optional[int] = ..., split_size: _Optional[int] = ..., segment_size: _Optional[int] = ..., show_term_doc_count_error: bool = ..., min_doc_count: _Optional[int] = ..., order: _Optional[_Union[CustomOrder, _Mapping]] = ...) -> None: ...

class CustomOrder(_message.Message):
    __slots__ = ["key", "count", "sub_aggregation", "order"]
    KEY_FIELD_NUMBER: _ClassVar[int]
    COUNT_FIELD_NUMBER: _ClassVar[int]
    SUB_AGGREGATION_FIELD_NUMBER: _ClassVar[int]
    ORDER_FIELD_NUMBER: _ClassVar[int]
    key: _utils_pb2.Empty
    count: _utils_pb2.Empty
    sub_aggregation: str
    order: _utils_pb2.Order
    def __init__(self, key: _Optional[_Union[_utils_pb2.Empty, _Mapping]] = ..., count: _Optional[_Union[_utils_pb2.Empty, _Mapping]] = ..., sub_aggregation: _Optional[str] = ..., order: _Optional[_Union[_utils_pb2.Order, str]] = ...) -> None: ...

class MetricAggregation(_message.Message):
    __slots__ = ["average", "stats"]
    AVERAGE_FIELD_NUMBER: _ClassVar[int]
    STATS_FIELD_NUMBER: _ClassVar[int]
    average: AverageAggregation
    stats: StatsAggregation
    def __init__(self, average: _Optional[_Union[AverageAggregation, _Mapping]] = ..., stats: _Optional[_Union[StatsAggregation, _Mapping]] = ...) -> None: ...

class AverageAggregation(_message.Message):
    __slots__ = ["field"]
    FIELD_FIELD_NUMBER: _ClassVar[int]
    field: str
    def __init__(self, field: _Optional[str] = ...) -> None: ...

class StatsAggregation(_message.Message):
    __slots__ = ["field"]
    FIELD_FIELD_NUMBER: _ClassVar[int]
    field: str
    def __init__(self, field: _Optional[str] = ...) -> None: ...

class BucketEntry(_message.Message):
    __slots__ = ["key", "doc_count", "sub_aggregation"]
    class SubAggregationEntry(_message.Message):
        __slots__ = ["key", "value"]
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: AggregationResult
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[AggregationResult, _Mapping]] = ...) -> None: ...
    KEY_FIELD_NUMBER: _ClassVar[int]
    DOC_COUNT_FIELD_NUMBER: _ClassVar[int]
    SUB_AGGREGATION_FIELD_NUMBER: _ClassVar[int]
    key: Key
    doc_count: int
    sub_aggregation: _containers.MessageMap[str, AggregationResult]
    def __init__(self, key: _Optional[_Union[Key, _Mapping]] = ..., doc_count: _Optional[int] = ..., sub_aggregation: _Optional[_Mapping[str, AggregationResult]] = ...) -> None: ...

class Key(_message.Message):
    __slots__ = ["str", "f64"]
    STR_FIELD_NUMBER: _ClassVar[int]
    F64_FIELD_NUMBER: _ClassVar[int]
    str: str
    f64: float
    def __init__(self, str: _Optional[str] = ..., f64: _Optional[float] = ...) -> None: ...

class Range(_message.Message):
    __slots__ = ["left", "right", "including_left", "including_right"]
    LEFT_FIELD_NUMBER: _ClassVar[int]
    RIGHT_FIELD_NUMBER: _ClassVar[int]
    INCLUDING_LEFT_FIELD_NUMBER: _ClassVar[int]
    INCLUDING_RIGHT_FIELD_NUMBER: _ClassVar[int]
    left: str
    right: str
    including_left: bool
    including_right: bool
    def __init__(self, left: _Optional[str] = ..., right: _Optional[str] = ..., including_left: bool = ..., including_right: bool = ...) -> None: ...

class RangeBucketEntry(_message.Message):
    __slots__ = ["key", "doc_count", "sub_aggregation", "to"]
    class SubAggregationEntry(_message.Message):
        __slots__ = ["key", "value"]
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: AggregationResult
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[AggregationResult, _Mapping]] = ...) -> None: ...
    KEY_FIELD_NUMBER: _ClassVar[int]
    DOC_COUNT_FIELD_NUMBER: _ClassVar[int]
    SUB_AGGREGATION_FIELD_NUMBER: _ClassVar[int]
    FROM_FIELD_NUMBER: _ClassVar[int]
    TO_FIELD_NUMBER: _ClassVar[int]
    key: Key
    doc_count: int
    sub_aggregation: _containers.MessageMap[str, AggregationResult]
    to: float
    def __init__(self, key: _Optional[_Union[Key, _Mapping]] = ..., doc_count: _Optional[int] = ..., sub_aggregation: _Optional[_Mapping[str, AggregationResult]] = ..., to: _Optional[float] = ..., **kwargs) -> None: ...

class Score(_message.Message):
    __slots__ = ["f64_score", "u64_score"]
    F64_SCORE_FIELD_NUMBER: _ClassVar[int]
    U64_SCORE_FIELD_NUMBER: _ClassVar[int]
    f64_score: float
    u64_score: int
    def __init__(self, f64_score: _Optional[float] = ..., u64_score: _Optional[int] = ...) -> None: ...

class Highlight(_message.Message):
    __slots__ = ["to"]
    FROM_FIELD_NUMBER: _ClassVar[int]
    TO_FIELD_NUMBER: _ClassVar[int]
    to: int
    def __init__(self, to: _Optional[int] = ..., **kwargs) -> None: ...

class Snippet(_message.Message):
    __slots__ = ["fragment", "highlights", "html"]
    FRAGMENT_FIELD_NUMBER: _ClassVar[int]
    HIGHLIGHTS_FIELD_NUMBER: _ClassVar[int]
    HTML_FIELD_NUMBER: _ClassVar[int]
    fragment: bytes
    highlights: _containers.RepeatedCompositeFieldContainer[Highlight]
    html: str
    def __init__(self, fragment: _Optional[bytes] = ..., highlights: _Optional[_Iterable[_Union[Highlight, _Mapping]]] = ..., html: _Optional[str] = ...) -> None: ...

class ScoredDocument(_message.Message):
    __slots__ = ["document", "score", "position", "snippets", "index_alias"]
    class SnippetsEntry(_message.Message):
        __slots__ = ["key", "value"]
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: Snippet
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[Snippet, _Mapping]] = ...) -> None: ...
    DOCUMENT_FIELD_NUMBER: _ClassVar[int]
    SCORE_FIELD_NUMBER: _ClassVar[int]
    POSITION_FIELD_NUMBER: _ClassVar[int]
    SNIPPETS_FIELD_NUMBER: _ClassVar[int]
    INDEX_ALIAS_FIELD_NUMBER: _ClassVar[int]
    document: str
    score: Score
    position: int
    snippets: _containers.MessageMap[str, Snippet]
    index_alias: str
    def __init__(self, document: _Optional[str] = ..., score: _Optional[_Union[Score, _Mapping]] = ..., position: _Optional[int] = ..., snippets: _Optional[_Mapping[str, Snippet]] = ..., index_alias: _Optional[str] = ...) -> None: ...

class Scorer(_message.Message):
    __slots__ = ["eval_expr", "order_by"]
    EVAL_EXPR_FIELD_NUMBER: _ClassVar[int]
    ORDER_BY_FIELD_NUMBER: _ClassVar[int]
    eval_expr: str
    order_by: str
    def __init__(self, eval_expr: _Optional[str] = ..., order_by: _Optional[str] = ...) -> None: ...

class Collector(_message.Message):
    __slots__ = ["top_docs", "reservoir_sampling", "count", "facet", "aggregation"]
    TOP_DOCS_FIELD_NUMBER: _ClassVar[int]
    RESERVOIR_SAMPLING_FIELD_NUMBER: _ClassVar[int]
    COUNT_FIELD_NUMBER: _ClassVar[int]
    FACET_FIELD_NUMBER: _ClassVar[int]
    AGGREGATION_FIELD_NUMBER: _ClassVar[int]
    top_docs: TopDocsCollector
    reservoir_sampling: ReservoirSamplingCollector
    count: CountCollector
    facet: FacetCollector
    aggregation: AggregationCollector
    def __init__(self, top_docs: _Optional[_Union[TopDocsCollector, _Mapping]] = ..., reservoir_sampling: _Optional[_Union[ReservoirSamplingCollector, _Mapping]] = ..., count: _Optional[_Union[CountCollector, _Mapping]] = ..., facet: _Optional[_Union[FacetCollector, _Mapping]] = ..., aggregation: _Optional[_Union[AggregationCollector, _Mapping]] = ...) -> None: ...

class CollectorOutput(_message.Message):
    __slots__ = ["documents", "count", "facet", "aggregation"]
    DOCUMENTS_FIELD_NUMBER: _ClassVar[int]
    COUNT_FIELD_NUMBER: _ClassVar[int]
    FACET_FIELD_NUMBER: _ClassVar[int]
    AGGREGATION_FIELD_NUMBER: _ClassVar[int]
    documents: DocumentsCollectorOutput
    count: CountCollectorOutput
    facet: FacetCollectorOutput
    aggregation: AggregationCollectorOutput
    def __init__(self, documents: _Optional[_Union[DocumentsCollectorOutput, _Mapping]] = ..., count: _Optional[_Union[CountCollectorOutput, _Mapping]] = ..., facet: _Optional[_Union[FacetCollectorOutput, _Mapping]] = ..., aggregation: _Optional[_Union[AggregationCollectorOutput, _Mapping]] = ...) -> None: ...

class CountCollector(_message.Message):
    __slots__ = []
    def __init__(self) -> None: ...

class CountCollectorOutput(_message.Message):
    __slots__ = ["count"]
    COUNT_FIELD_NUMBER: _ClassVar[int]
    count: int
    def __init__(self, count: _Optional[int] = ...) -> None: ...

class FacetCollector(_message.Message):
    __slots__ = ["field", "facets"]
    FIELD_FIELD_NUMBER: _ClassVar[int]
    FACETS_FIELD_NUMBER: _ClassVar[int]
    field: str
    facets: _containers.RepeatedScalarFieldContainer[str]
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

class ReservoirSamplingCollector(_message.Message):
    __slots__ = ["limit", "fields"]
    LIMIT_FIELD_NUMBER: _ClassVar[int]
    FIELDS_FIELD_NUMBER: _ClassVar[int]
    limit: int
    fields: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, limit: _Optional[int] = ..., fields: _Optional[_Iterable[str]] = ...) -> None: ...

class RandomDocument(_message.Message):
    __slots__ = ["document", "score", "index_alias"]
    DOCUMENT_FIELD_NUMBER: _ClassVar[int]
    SCORE_FIELD_NUMBER: _ClassVar[int]
    INDEX_ALIAS_FIELD_NUMBER: _ClassVar[int]
    document: str
    score: Score
    index_alias: str
    def __init__(self, document: _Optional[str] = ..., score: _Optional[_Union[Score, _Mapping]] = ..., index_alias: _Optional[str] = ...) -> None: ...

class ReservoirSamplingCollectorOutput(_message.Message):
    __slots__ = ["documents"]
    DOCUMENTS_FIELD_NUMBER: _ClassVar[int]
    documents: _containers.RepeatedCompositeFieldContainer[RandomDocument]
    def __init__(self, documents: _Optional[_Iterable[_Union[RandomDocument, _Mapping]]] = ...) -> None: ...

class TopDocsCollector(_message.Message):
    __slots__ = ["limit", "offset", "scorer", "snippet_configs", "explain", "fields"]
    class SnippetConfigsEntry(_message.Message):
        __slots__ = ["key", "value"]
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: int
        def __init__(self, key: _Optional[str] = ..., value: _Optional[int] = ...) -> None: ...
    LIMIT_FIELD_NUMBER: _ClassVar[int]
    OFFSET_FIELD_NUMBER: _ClassVar[int]
    SCORER_FIELD_NUMBER: _ClassVar[int]
    SNIPPET_CONFIGS_FIELD_NUMBER: _ClassVar[int]
    EXPLAIN_FIELD_NUMBER: _ClassVar[int]
    FIELDS_FIELD_NUMBER: _ClassVar[int]
    limit: int
    offset: int
    scorer: Scorer
    snippet_configs: _containers.ScalarMap[str, int]
    explain: bool
    fields: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, limit: _Optional[int] = ..., offset: _Optional[int] = ..., scorer: _Optional[_Union[Scorer, _Mapping]] = ..., snippet_configs: _Optional[_Mapping[str, int]] = ..., explain: bool = ..., fields: _Optional[_Iterable[str]] = ...) -> None: ...

class DocumentsCollectorOutput(_message.Message):
    __slots__ = ["scored_documents", "has_next"]
    SCORED_DOCUMENTS_FIELD_NUMBER: _ClassVar[int]
    HAS_NEXT_FIELD_NUMBER: _ClassVar[int]
    scored_documents: _containers.RepeatedCompositeFieldContainer[ScoredDocument]
    has_next: bool
    def __init__(self, scored_documents: _Optional[_Iterable[_Union[ScoredDocument, _Mapping]]] = ..., has_next: bool = ...) -> None: ...

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

class BucketResult(_message.Message):
    __slots__ = ["range", "histogram", "terms"]
    RANGE_FIELD_NUMBER: _ClassVar[int]
    HISTOGRAM_FIELD_NUMBER: _ClassVar[int]
    TERMS_FIELD_NUMBER: _ClassVar[int]
    range: RangeResult
    histogram: HistogramResult
    terms: TermsResult
    def __init__(self, range: _Optional[_Union[RangeResult, _Mapping]] = ..., histogram: _Optional[_Union[HistogramResult, _Mapping]] = ..., terms: _Optional[_Union[TermsResult, _Mapping]] = ...) -> None: ...

class RangeResult(_message.Message):
    __slots__ = ["buckets"]
    BUCKETS_FIELD_NUMBER: _ClassVar[int]
    buckets: _containers.RepeatedCompositeFieldContainer[RangeBucketEntry]
    def __init__(self, buckets: _Optional[_Iterable[_Union[RangeBucketEntry, _Mapping]]] = ...) -> None: ...

class HistogramResult(_message.Message):
    __slots__ = ["buckets"]
    BUCKETS_FIELD_NUMBER: _ClassVar[int]
    buckets: _containers.RepeatedCompositeFieldContainer[BucketEntry]
    def __init__(self, buckets: _Optional[_Iterable[_Union[BucketEntry, _Mapping]]] = ...) -> None: ...

class TermsResult(_message.Message):
    __slots__ = ["buckets", "sum_other_doc_count", "doc_count_error_upper_bound"]
    BUCKETS_FIELD_NUMBER: _ClassVar[int]
    SUM_OTHER_DOC_COUNT_FIELD_NUMBER: _ClassVar[int]
    DOC_COUNT_ERROR_UPPER_BOUND_FIELD_NUMBER: _ClassVar[int]
    buckets: _containers.RepeatedCompositeFieldContainer[BucketEntry]
    sum_other_doc_count: int
    doc_count_error_upper_bound: int
    def __init__(self, buckets: _Optional[_Iterable[_Union[BucketEntry, _Mapping]]] = ..., sum_other_doc_count: _Optional[int] = ..., doc_count_error_upper_bound: _Optional[int] = ...) -> None: ...

class MetricResult(_message.Message):
    __slots__ = ["single_metric", "stats"]
    SINGLE_METRIC_FIELD_NUMBER: _ClassVar[int]
    STATS_FIELD_NUMBER: _ClassVar[int]
    single_metric: SingleMetricResult
    stats: StatsResult
    def __init__(self, single_metric: _Optional[_Union[SingleMetricResult, _Mapping]] = ..., stats: _Optional[_Union[StatsResult, _Mapping]] = ...) -> None: ...

class SingleMetricResult(_message.Message):
    __slots__ = ["value"]
    VALUE_FIELD_NUMBER: _ClassVar[int]
    value: float
    def __init__(self, value: _Optional[float] = ...) -> None: ...

class StatsResult(_message.Message):
    __slots__ = ["count", "sum", "min", "max", "avg"]
    COUNT_FIELD_NUMBER: _ClassVar[int]
    SUM_FIELD_NUMBER: _ClassVar[int]
    MIN_FIELD_NUMBER: _ClassVar[int]
    MAX_FIELD_NUMBER: _ClassVar[int]
    AVG_FIELD_NUMBER: _ClassVar[int]
    count: int
    sum: float
    min: float
    max: float
    avg: float
    def __init__(self, count: _Optional[int] = ..., sum: _Optional[float] = ..., min: _Optional[float] = ..., max: _Optional[float] = ..., avg: _Optional[float] = ...) -> None: ...
