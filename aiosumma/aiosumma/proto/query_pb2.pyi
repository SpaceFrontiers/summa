from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class Occur(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    should: _ClassVar[Occur]
    must: _ClassVar[Occur]
    must_not: _ClassVar[Occur]
should: Occur
must: Occur
must_not: Occur

class TermFieldMapperConfig(_message.Message):
    __slots__ = ("fields",)
    FIELDS_FIELD_NUMBER: _ClassVar[int]
    fields: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, fields: _Optional[_Iterable[str]] = ...) -> None: ...

class MatchQueryBooleanShouldMode(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class MatchQueryDisjuctionMaxMode(_message.Message):
    __slots__ = ("tie_breaker",)
    TIE_BREAKER_FIELD_NUMBER: _ClassVar[int]
    tie_breaker: float
    def __init__(self, tie_breaker: _Optional[float] = ...) -> None: ...

class ExactMatchesPromoter(_message.Message):
    __slots__ = ("slop", "boost", "fields")
    SLOP_FIELD_NUMBER: _ClassVar[int]
    BOOST_FIELD_NUMBER: _ClassVar[int]
    FIELDS_FIELD_NUMBER: _ClassVar[int]
    slop: int
    boost: float
    fields: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, slop: _Optional[int] = ..., boost: _Optional[float] = ..., fields: _Optional[_Iterable[str]] = ...) -> None: ...

class NerMatchesPromoter(_message.Message):
    __slots__ = ("boost", "fields")
    BOOST_FIELD_NUMBER: _ClassVar[int]
    FIELDS_FIELD_NUMBER: _ClassVar[int]
    boost: float
    fields: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, boost: _Optional[float] = ..., fields: _Optional[_Iterable[str]] = ...) -> None: ...

class MorphologyConfig(_message.Message):
    __slots__ = ("derive_tenses_coefficient",)
    DERIVE_TENSES_COEFFICIENT_FIELD_NUMBER: _ClassVar[int]
    derive_tenses_coefficient: float
    def __init__(self, derive_tenses_coefficient: _Optional[float] = ...) -> None: ...

class QueryParserConfig(_message.Message):
    __slots__ = ("field_aliases", "field_boosts", "term_field_mapper_configs", "term_limit", "default_fields", "boolean_should_mode", "disjuction_max_mode", "exact_matches_promoter", "excluded_fields", "morphology_configs", "query_language")
    class FieldAliasesEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    class FieldBoostsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: float
        def __init__(self, key: _Optional[str] = ..., value: _Optional[float] = ...) -> None: ...
    class TermFieldMapperConfigsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: TermFieldMapperConfig
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[TermFieldMapperConfig, _Mapping]] = ...) -> None: ...
    class MorphologyConfigsEntry(_message.Message):
        __slots__ = ("key", "value")
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
    EXCLUDED_FIELDS_FIELD_NUMBER: _ClassVar[int]
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
    excluded_fields: _containers.RepeatedScalarFieldContainer[str]
    morphology_configs: _containers.MessageMap[str, MorphologyConfig]
    query_language: str
    def __init__(self, field_aliases: _Optional[_Mapping[str, str]] = ..., field_boosts: _Optional[_Mapping[str, float]] = ..., term_field_mapper_configs: _Optional[_Mapping[str, TermFieldMapperConfig]] = ..., term_limit: _Optional[int] = ..., default_fields: _Optional[_Iterable[str]] = ..., boolean_should_mode: _Optional[_Union[MatchQueryBooleanShouldMode, _Mapping]] = ..., disjuction_max_mode: _Optional[_Union[MatchQueryDisjuctionMaxMode, _Mapping]] = ..., exact_matches_promoter: _Optional[_Union[ExactMatchesPromoter, _Mapping]] = ..., excluded_fields: _Optional[_Iterable[str]] = ..., morphology_configs: _Optional[_Mapping[str, MorphologyConfig]] = ..., query_language: _Optional[str] = ...) -> None: ...

class SearchRequest(_message.Message):
    __slots__ = ("index_alias", "query", "collectors", "is_fieldnorms_scoring_enabled", "load_cache", "store_cache")
    INDEX_ALIAS_FIELD_NUMBER: _ClassVar[int]
    QUERY_FIELD_NUMBER: _ClassVar[int]
    COLLECTORS_FIELD_NUMBER: _ClassVar[int]
    IS_FIELDNORMS_SCORING_ENABLED_FIELD_NUMBER: _ClassVar[int]
    LOAD_CACHE_FIELD_NUMBER: _ClassVar[int]
    STORE_CACHE_FIELD_NUMBER: _ClassVar[int]
    index_alias: str
    query: Query
    collectors: _containers.RepeatedCompositeFieldContainer[Collector]
    is_fieldnorms_scoring_enabled: bool
    load_cache: bool
    store_cache: bool
    def __init__(self, index_alias: _Optional[str] = ..., query: _Optional[_Union[Query, _Mapping]] = ..., collectors: _Optional[_Iterable[_Union[Collector, _Mapping]]] = ..., is_fieldnorms_scoring_enabled: bool = ..., load_cache: bool = ..., store_cache: bool = ...) -> None: ...

class SearchResponse(_message.Message):
    __slots__ = ("elapsed_secs", "collector_outputs")
    ELAPSED_SECS_FIELD_NUMBER: _ClassVar[int]
    COLLECTOR_OUTPUTS_FIELD_NUMBER: _ClassVar[int]
    elapsed_secs: float
    collector_outputs: _containers.RepeatedCompositeFieldContainer[CollectorOutput]
    def __init__(self, elapsed_secs: _Optional[float] = ..., collector_outputs: _Optional[_Iterable[_Union[CollectorOutput, _Mapping]]] = ...) -> None: ...

class Query(_message.Message):
    __slots__ = ("boolean", "match", "regex", "term", "phrase", "range", "all", "more_like_this", "boost", "disjunction_max", "empty", "exists")
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
    __slots__ = ()
    def __init__(self) -> None: ...

class EmptyQuery(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class BoostQuery(_message.Message):
    __slots__ = ("query", "score")
    QUERY_FIELD_NUMBER: _ClassVar[int]
    SCORE_FIELD_NUMBER: _ClassVar[int]
    query: Query
    score: str
    def __init__(self, query: _Optional[_Union[Query, _Mapping]] = ..., score: _Optional[str] = ...) -> None: ...

class DisjunctionMaxQuery(_message.Message):
    __slots__ = ("disjuncts", "tie_breaker")
    DISJUNCTS_FIELD_NUMBER: _ClassVar[int]
    TIE_BREAKER_FIELD_NUMBER: _ClassVar[int]
    disjuncts: _containers.RepeatedCompositeFieldContainer[Query]
    tie_breaker: str
    def __init__(self, disjuncts: _Optional[_Iterable[_Union[Query, _Mapping]]] = ..., tie_breaker: _Optional[str] = ...) -> None: ...

class MoreLikeThisQuery(_message.Message):
    __slots__ = ("document", "min_doc_frequency", "max_doc_frequency", "min_term_frequency", "max_query_terms", "min_word_length", "max_word_length", "boost", "stop_words")
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
    __slots__ = ("field", "value", "slop")
    FIELD_FIELD_NUMBER: _ClassVar[int]
    VALUE_FIELD_NUMBER: _ClassVar[int]
    SLOP_FIELD_NUMBER: _ClassVar[int]
    field: str
    value: str
    slop: int
    def __init__(self, field: _Optional[str] = ..., value: _Optional[str] = ..., slop: _Optional[int] = ...) -> None: ...

class RangeQuery(_message.Message):
    __slots__ = ("field", "value")
    FIELD_FIELD_NUMBER: _ClassVar[int]
    VALUE_FIELD_NUMBER: _ClassVar[int]
    field: str
    value: Range
    def __init__(self, field: _Optional[str] = ..., value: _Optional[_Union[Range, _Mapping]] = ...) -> None: ...

class MatchQuery(_message.Message):
    __slots__ = ("value", "query_parser_config")
    VALUE_FIELD_NUMBER: _ClassVar[int]
    QUERY_PARSER_CONFIG_FIELD_NUMBER: _ClassVar[int]
    value: str
    query_parser_config: QueryParserConfig
    def __init__(self, value: _Optional[str] = ..., query_parser_config: _Optional[_Union[QueryParserConfig, _Mapping]] = ...) -> None: ...

class BooleanSubquery(_message.Message):
    __slots__ = ("occur", "query")
    OCCUR_FIELD_NUMBER: _ClassVar[int]
    QUERY_FIELD_NUMBER: _ClassVar[int]
    occur: Occur
    query: Query
    def __init__(self, occur: _Optional[_Union[Occur, str]] = ..., query: _Optional[_Union[Query, _Mapping]] = ...) -> None: ...

class BooleanQuery(_message.Message):
    __slots__ = ("subqueries",)
    SUBQUERIES_FIELD_NUMBER: _ClassVar[int]
    subqueries: _containers.RepeatedCompositeFieldContainer[BooleanSubquery]
    def __init__(self, subqueries: _Optional[_Iterable[_Union[BooleanSubquery, _Mapping]]] = ...) -> None: ...

class RegexQuery(_message.Message):
    __slots__ = ("field", "value")
    FIELD_FIELD_NUMBER: _ClassVar[int]
    VALUE_FIELD_NUMBER: _ClassVar[int]
    field: str
    value: str
    def __init__(self, field: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...

class TermQuery(_message.Message):
    __slots__ = ("field", "value")
    FIELD_FIELD_NUMBER: _ClassVar[int]
    VALUE_FIELD_NUMBER: _ClassVar[int]
    field: str
    value: str
    def __init__(self, field: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...

class ExistsQuery(_message.Message):
    __slots__ = ("field",)
    FIELD_FIELD_NUMBER: _ClassVar[int]
    field: str
    def __init__(self, field: _Optional[str] = ...) -> None: ...

class Range(_message.Message):
    __slots__ = ("left", "right", "including_left", "including_right")
    LEFT_FIELD_NUMBER: _ClassVar[int]
    RIGHT_FIELD_NUMBER: _ClassVar[int]
    INCLUDING_LEFT_FIELD_NUMBER: _ClassVar[int]
    INCLUDING_RIGHT_FIELD_NUMBER: _ClassVar[int]
    left: str
    right: str
    including_left: bool
    including_right: bool
    def __init__(self, left: _Optional[str] = ..., right: _Optional[str] = ..., including_left: bool = ..., including_right: bool = ...) -> None: ...

class Score(_message.Message):
    __slots__ = ("f64_score", "u64_score")
    F64_SCORE_FIELD_NUMBER: _ClassVar[int]
    U64_SCORE_FIELD_NUMBER: _ClassVar[int]
    f64_score: float
    u64_score: int
    def __init__(self, f64_score: _Optional[float] = ..., u64_score: _Optional[int] = ...) -> None: ...

class Highlight(_message.Message):
    __slots__ = ("to",)
    FROM_FIELD_NUMBER: _ClassVar[int]
    TO_FIELD_NUMBER: _ClassVar[int]
    to: int
    def __init__(self, to: _Optional[int] = ..., **kwargs) -> None: ...

class Snippet(_message.Message):
    __slots__ = ("fragment", "highlights", "html")
    FRAGMENT_FIELD_NUMBER: _ClassVar[int]
    HIGHLIGHTS_FIELD_NUMBER: _ClassVar[int]
    HTML_FIELD_NUMBER: _ClassVar[int]
    fragment: bytes
    highlights: _containers.RepeatedCompositeFieldContainer[Highlight]
    html: str
    def __init__(self, fragment: _Optional[bytes] = ..., highlights: _Optional[_Iterable[_Union[Highlight, _Mapping]]] = ..., html: _Optional[str] = ...) -> None: ...

class ScoredDocument(_message.Message):
    __slots__ = ("document", "score", "position", "snippets", "index_alias")
    class SnippetsEntry(_message.Message):
        __slots__ = ("key", "value")
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
    __slots__ = ("eval_expr", "order_by")
    EVAL_EXPR_FIELD_NUMBER: _ClassVar[int]
    ORDER_BY_FIELD_NUMBER: _ClassVar[int]
    eval_expr: str
    order_by: str
    def __init__(self, eval_expr: _Optional[str] = ..., order_by: _Optional[str] = ...) -> None: ...

class Collector(_message.Message):
    __slots__ = ("top_docs", "reservoir_sampling", "count", "facet", "aggregation")
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
    __slots__ = ("documents", "count", "facet", "aggregation")
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
    __slots__ = ()
    def __init__(self) -> None: ...

class CountCollectorOutput(_message.Message):
    __slots__ = ("count",)
    COUNT_FIELD_NUMBER: _ClassVar[int]
    count: int
    def __init__(self, count: _Optional[int] = ...) -> None: ...

class FacetCollector(_message.Message):
    __slots__ = ("field", "facets")
    FIELD_FIELD_NUMBER: _ClassVar[int]
    FACETS_FIELD_NUMBER: _ClassVar[int]
    field: str
    facets: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, field: _Optional[str] = ..., facets: _Optional[_Iterable[str]] = ...) -> None: ...

class FacetCollectorOutput(_message.Message):
    __slots__ = ("facet_counts",)
    class FacetCountsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: int
        def __init__(self, key: _Optional[str] = ..., value: _Optional[int] = ...) -> None: ...
    FACET_COUNTS_FIELD_NUMBER: _ClassVar[int]
    facet_counts: _containers.ScalarMap[str, int]
    def __init__(self, facet_counts: _Optional[_Mapping[str, int]] = ...) -> None: ...

class ReservoirSamplingCollector(_message.Message):
    __slots__ = ("limit", "fields", "excluded_fields")
    LIMIT_FIELD_NUMBER: _ClassVar[int]
    FIELDS_FIELD_NUMBER: _ClassVar[int]
    EXCLUDED_FIELDS_FIELD_NUMBER: _ClassVar[int]
    limit: int
    fields: _containers.RepeatedScalarFieldContainer[str]
    excluded_fields: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, limit: _Optional[int] = ..., fields: _Optional[_Iterable[str]] = ..., excluded_fields: _Optional[_Iterable[str]] = ...) -> None: ...

class RandomDocument(_message.Message):
    __slots__ = ("document", "score", "index_alias")
    DOCUMENT_FIELD_NUMBER: _ClassVar[int]
    SCORE_FIELD_NUMBER: _ClassVar[int]
    INDEX_ALIAS_FIELD_NUMBER: _ClassVar[int]
    document: str
    score: Score
    index_alias: str
    def __init__(self, document: _Optional[str] = ..., score: _Optional[_Union[Score, _Mapping]] = ..., index_alias: _Optional[str] = ...) -> None: ...

class ReservoirSamplingCollectorOutput(_message.Message):
    __slots__ = ("documents",)
    DOCUMENTS_FIELD_NUMBER: _ClassVar[int]
    documents: _containers.RepeatedCompositeFieldContainer[RandomDocument]
    def __init__(self, documents: _Optional[_Iterable[_Union[RandomDocument, _Mapping]]] = ...) -> None: ...

class TopDocsCollector(_message.Message):
    __slots__ = ("limit", "offset", "scorer", "snippet_configs", "explain", "fields", "excluded_fields")
    class SnippetConfigsEntry(_message.Message):
        __slots__ = ("key", "value")
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
    EXCLUDED_FIELDS_FIELD_NUMBER: _ClassVar[int]
    limit: int
    offset: int
    scorer: Scorer
    snippet_configs: _containers.ScalarMap[str, int]
    explain: bool
    fields: _containers.RepeatedScalarFieldContainer[str]
    excluded_fields: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, limit: _Optional[int] = ..., offset: _Optional[int] = ..., scorer: _Optional[_Union[Scorer, _Mapping]] = ..., snippet_configs: _Optional[_Mapping[str, int]] = ..., explain: bool = ..., fields: _Optional[_Iterable[str]] = ..., excluded_fields: _Optional[_Iterable[str]] = ...) -> None: ...

class DocumentsCollectorOutput(_message.Message):
    __slots__ = ("scored_documents", "has_next")
    SCORED_DOCUMENTS_FIELD_NUMBER: _ClassVar[int]
    HAS_NEXT_FIELD_NUMBER: _ClassVar[int]
    scored_documents: _containers.RepeatedCompositeFieldContainer[ScoredDocument]
    has_next: bool
    def __init__(self, scored_documents: _Optional[_Iterable[_Union[ScoredDocument, _Mapping]]] = ..., has_next: bool = ...) -> None: ...

class AggregationCollector(_message.Message):
    __slots__ = ("aggregations",)
    AGGREGATIONS_FIELD_NUMBER: _ClassVar[int]
    aggregations: str
    def __init__(self, aggregations: _Optional[str] = ...) -> None: ...

class AggregationCollectorOutput(_message.Message):
    __slots__ = ("aggregation_results",)
    AGGREGATION_RESULTS_FIELD_NUMBER: _ClassVar[int]
    aggregation_results: str
    def __init__(self, aggregation_results: _Optional[str] = ...) -> None: ...
