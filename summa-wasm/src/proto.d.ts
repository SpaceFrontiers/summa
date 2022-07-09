import * as $protobuf from "protobufjs";
import Long = require("long");
/** Namespace summa. */
export namespace summa {

    /** Namespace proto. */
    namespace proto {

        /** Properties of a SearchRequest. */
        interface ISearchRequest {

            /** SearchRequest index_queries */
            index_queries?: (summa.proto.IIndexQuery[]|null);

            /** SearchRequest tags */
            tags?: ({ [k: string]: string }|null);
        }

        /** Represents a SearchRequest. */
        class SearchRequest implements ISearchRequest {

            /**
             * Constructs a new SearchRequest.
             * @param [properties] Properties to set
             */
            constructor(properties?: summa.proto.ISearchRequest);

            /** SearchRequest index_queries. */
            public index_queries: summa.proto.IIndexQuery[];

            /** SearchRequest tags. */
            public tags: { [k: string]: string };

            /**
             * Creates a new SearchRequest instance using the specified properties.
             * @param [properties] Properties to set
             * @returns SearchRequest instance
             */
            public static create(properties?: summa.proto.ISearchRequest): summa.proto.SearchRequest;
        }

        /** Properties of an IndexQuery. */
        interface IIndexQuery {

            /** IndexQuery index_alias */
            index_alias?: (string|null);

            /** IndexQuery query */
            query?: (summa.proto.IQuery|null);

            /** IndexQuery collectors */
            collectors?: (summa.proto.ICollector[]|null);

            /** IndexQuery is_fieldnorms_scoring_enabled */
            is_fieldnorms_scoring_enabled?: (boolean|null);
        }

        /** Represents an IndexQuery. */
        class IndexQuery implements IIndexQuery {

            /**
             * Constructs a new IndexQuery.
             * @param [properties] Properties to set
             */
            constructor(properties?: summa.proto.IIndexQuery);

            /** IndexQuery index_alias. */
            public index_alias: string;

            /** IndexQuery query. */
            public query?: (summa.proto.IQuery|null);

            /** IndexQuery collectors. */
            public collectors: summa.proto.ICollector[];

            /** IndexQuery is_fieldnorms_scoring_enabled. */
            public is_fieldnorms_scoring_enabled?: (boolean|null);

            /** IndexQuery _is_fieldnorms_scoring_enabled. */
            public _is_fieldnorms_scoring_enabled?: "is_fieldnorms_scoring_enabled";

            /**
             * Creates a new IndexQuery instance using the specified properties.
             * @param [properties] Properties to set
             * @returns IndexQuery instance
             */
            public static create(properties?: summa.proto.IIndexQuery): summa.proto.IndexQuery;
        }

        /** Properties of a TermFieldMapperConfig. */
        interface ITermFieldMapperConfig {

            /** TermFieldMapperConfig fields */
            fields?: (string[]|null);
        }

        /** Represents a TermFieldMapperConfig. */
        class TermFieldMapperConfig implements ITermFieldMapperConfig {

            /**
             * Constructs a new TermFieldMapperConfig.
             * @param [properties] Properties to set
             */
            constructor(properties?: summa.proto.ITermFieldMapperConfig);

            /** TermFieldMapperConfig fields. */
            public fields: string[];

            /**
             * Creates a new TermFieldMapperConfig instance using the specified properties.
             * @param [properties] Properties to set
             * @returns TermFieldMapperConfig instance
             */
            public static create(properties?: summa.proto.ITermFieldMapperConfig): summa.proto.TermFieldMapperConfig;
        }

        /** Properties of a MatchQueryBooleanShouldMode. */
        interface IMatchQueryBooleanShouldMode {
        }

        /** Represents a MatchQueryBooleanShouldMode. */
        class MatchQueryBooleanShouldMode implements IMatchQueryBooleanShouldMode {

            /**
             * Constructs a new MatchQueryBooleanShouldMode.
             * @param [properties] Properties to set
             */
            constructor(properties?: summa.proto.IMatchQueryBooleanShouldMode);

            /**
             * Creates a new MatchQueryBooleanShouldMode instance using the specified properties.
             * @param [properties] Properties to set
             * @returns MatchQueryBooleanShouldMode instance
             */
            public static create(properties?: summa.proto.IMatchQueryBooleanShouldMode): summa.proto.MatchQueryBooleanShouldMode;
        }

        /** Properties of a MatchQueryDisjuctionMaxMode. */
        interface IMatchQueryDisjuctionMaxMode {

            /** MatchQueryDisjuctionMaxMode tie_breaker */
            tie_breaker?: (number|null);
        }

        /** Represents a MatchQueryDisjuctionMaxMode. */
        class MatchQueryDisjuctionMaxMode implements IMatchQueryDisjuctionMaxMode {

            /**
             * Constructs a new MatchQueryDisjuctionMaxMode.
             * @param [properties] Properties to set
             */
            constructor(properties?: summa.proto.IMatchQueryDisjuctionMaxMode);

            /** MatchQueryDisjuctionMaxMode tie_breaker. */
            public tie_breaker: number;

            /**
             * Creates a new MatchQueryDisjuctionMaxMode instance using the specified properties.
             * @param [properties] Properties to set
             * @returns MatchQueryDisjuctionMaxMode instance
             */
            public static create(properties?: summa.proto.IMatchQueryDisjuctionMaxMode): summa.proto.MatchQueryDisjuctionMaxMode;
        }

        /** Properties of an ExactMatchesPromoter. */
        interface IExactMatchesPromoter {

            /** ExactMatchesPromoter slop */
            slop?: (number|null);

            /** ExactMatchesPromoter boost */
            boost?: (number|null);

            /** ExactMatchesPromoter fields */
            fields?: (string[]|null);
        }

        /** Represents an ExactMatchesPromoter. */
        class ExactMatchesPromoter implements IExactMatchesPromoter {

            /**
             * Constructs a new ExactMatchesPromoter.
             * @param [properties] Properties to set
             */
            constructor(properties?: summa.proto.IExactMatchesPromoter);

            /** ExactMatchesPromoter slop. */
            public slop: number;

            /** ExactMatchesPromoter boost. */
            public boost?: (number|null);

            /** ExactMatchesPromoter fields. */
            public fields: string[];

            /** ExactMatchesPromoter _boost. */
            public _boost?: "boost";

            /**
             * Creates a new ExactMatchesPromoter instance using the specified properties.
             * @param [properties] Properties to set
             * @returns ExactMatchesPromoter instance
             */
            public static create(properties?: summa.proto.IExactMatchesPromoter): summa.proto.ExactMatchesPromoter;
        }

        /** Properties of a NerMatchesPromoter. */
        interface INerMatchesPromoter {

            /** NerMatchesPromoter boost */
            boost?: (number|null);

            /** NerMatchesPromoter fields */
            fields?: (string[]|null);
        }

        /** Represents a NerMatchesPromoter. */
        class NerMatchesPromoter implements INerMatchesPromoter {

            /**
             * Constructs a new NerMatchesPromoter.
             * @param [properties] Properties to set
             */
            constructor(properties?: summa.proto.INerMatchesPromoter);

            /** NerMatchesPromoter boost. */
            public boost?: (number|null);

            /** NerMatchesPromoter fields. */
            public fields: string[];

            /** NerMatchesPromoter _boost. */
            public _boost?: "boost";

            /**
             * Creates a new NerMatchesPromoter instance using the specified properties.
             * @param [properties] Properties to set
             * @returns NerMatchesPromoter instance
             */
            public static create(properties?: summa.proto.INerMatchesPromoter): summa.proto.NerMatchesPromoter;
        }

        /** Properties of a MorphologyConfig. */
        interface IMorphologyConfig {

            /** MorphologyConfig derive_tenses_coefficient */
            derive_tenses_coefficient?: (number|null);
        }

        /** Represents a MorphologyConfig. */
        class MorphologyConfig implements IMorphologyConfig {

            /**
             * Constructs a new MorphologyConfig.
             * @param [properties] Properties to set
             */
            constructor(properties?: summa.proto.IMorphologyConfig);

            /** MorphologyConfig derive_tenses_coefficient. */
            public derive_tenses_coefficient?: (number|null);

            /** MorphologyConfig _derive_tenses_coefficient. */
            public _derive_tenses_coefficient?: "derive_tenses_coefficient";

            /**
             * Creates a new MorphologyConfig instance using the specified properties.
             * @param [properties] Properties to set
             * @returns MorphologyConfig instance
             */
            public static create(properties?: summa.proto.IMorphologyConfig): summa.proto.MorphologyConfig;
        }

        /** Properties of a QueryParserConfig. */
        interface IQueryParserConfig {

            /** QueryParserConfig field_aliases */
            field_aliases?: ({ [k: string]: string }|null);

            /** QueryParserConfig field_boosts */
            field_boosts?: ({ [k: string]: number }|null);

            /** QueryParserConfig term_field_mapper_configs */
            term_field_mapper_configs?: ({ [k: string]: summa.proto.ITermFieldMapperConfig }|null);

            /** QueryParserConfig term_limit */
            term_limit?: (number|null);

            /** QueryParserConfig default_fields */
            default_fields?: (string[]|null);

            /** QueryParserConfig boolean_should_mode */
            boolean_should_mode?: (summa.proto.IMatchQueryBooleanShouldMode|null);

            /** QueryParserConfig disjuction_max_mode */
            disjuction_max_mode?: (summa.proto.IMatchQueryDisjuctionMaxMode|null);

            /** QueryParserConfig exact_matches_promoter */
            exact_matches_promoter?: (summa.proto.IExactMatchesPromoter|null);

            /** QueryParserConfig removed_fields */
            removed_fields?: (string[]|null);

            /** QueryParserConfig morphology_configs */
            morphology_configs?: ({ [k: string]: summa.proto.IMorphologyConfig }|null);

            /** QueryParserConfig query_language */
            query_language?: (string|null);
        }

        /** Represents a QueryParserConfig. */
        class QueryParserConfig implements IQueryParserConfig {

            /**
             * Constructs a new QueryParserConfig.
             * @param [properties] Properties to set
             */
            constructor(properties?: summa.proto.IQueryParserConfig);

            /** QueryParserConfig field_aliases. */
            public field_aliases: { [k: string]: string };

            /** QueryParserConfig field_boosts. */
            public field_boosts: { [k: string]: number };

            /** QueryParserConfig term_field_mapper_configs. */
            public term_field_mapper_configs: { [k: string]: summa.proto.ITermFieldMapperConfig };

            /** QueryParserConfig term_limit. */
            public term_limit: number;

            /** QueryParserConfig default_fields. */
            public default_fields: string[];

            /** QueryParserConfig boolean_should_mode. */
            public boolean_should_mode?: (summa.proto.IMatchQueryBooleanShouldMode|null);

            /** QueryParserConfig disjuction_max_mode. */
            public disjuction_max_mode?: (summa.proto.IMatchQueryDisjuctionMaxMode|null);

            /** QueryParserConfig exact_matches_promoter. */
            public exact_matches_promoter?: (summa.proto.IExactMatchesPromoter|null);

            /** QueryParserConfig removed_fields. */
            public removed_fields: string[];

            /** QueryParserConfig morphology_configs. */
            public morphology_configs: { [k: string]: summa.proto.IMorphologyConfig };

            /** QueryParserConfig query_language. */
            public query_language?: (string|null);

            /** QueryParserConfig default_mode. */
            public default_mode?: ("boolean_should_mode"|"disjuction_max_mode");

            /** QueryParserConfig _query_language. */
            public _query_language?: "query_language";

            /**
             * Creates a new QueryParserConfig instance using the specified properties.
             * @param [properties] Properties to set
             * @returns QueryParserConfig instance
             */
            public static create(properties?: summa.proto.IQueryParserConfig): summa.proto.QueryParserConfig;
        }

        /** Properties of a SearchResponse. */
        interface ISearchResponse {

            /** SearchResponse elapsed_secs */
            elapsed_secs?: (number|null);

            /** SearchResponse collector_outputs */
            collector_outputs?: (summa.proto.ICollectorOutput[]|null);
        }

        /** Represents a SearchResponse. */
        class SearchResponse implements ISearchResponse {

            /**
             * Constructs a new SearchResponse.
             * @param [properties] Properties to set
             */
            constructor(properties?: summa.proto.ISearchResponse);

            /** SearchResponse elapsed_secs. */
            public elapsed_secs: number;

            /** SearchResponse collector_outputs. */
            public collector_outputs: summa.proto.ICollectorOutput[];

            /**
             * Creates a new SearchResponse instance using the specified properties.
             * @param [properties] Properties to set
             * @returns SearchResponse instance
             */
            public static create(properties?: summa.proto.ISearchResponse): summa.proto.SearchResponse;
        }

        /** Properties of a Query. */
        interface IQuery {

            /** Query boolean */
            boolean?: (summa.proto.IBooleanQuery|null);

            /** Query match */
            match?: (summa.proto.IMatchQuery|null);

            /** Query regex */
            regex?: (summa.proto.IRegexQuery|null);

            /** Query term */
            term?: (summa.proto.ITermQuery|null);

            /** Query phrase */
            phrase?: (summa.proto.IPhraseQuery|null);

            /** Query range */
            range?: (summa.proto.IRangeQuery|null);

            /** Query all */
            all?: (summa.proto.IAllQuery|null);

            /** Query more_like_this */
            more_like_this?: (summa.proto.IMoreLikeThisQuery|null);

            /** Query boost */
            boost?: (summa.proto.IBoostQuery|null);

            /** Query disjunction_max */
            disjunction_max?: (summa.proto.IDisjunctionMaxQuery|null);

            /** Query empty */
            empty?: (summa.proto.IEmptyQuery|null);

            /** Query exists */
            exists?: (summa.proto.IExistsQuery|null);
        }

        /** Represents a Query. */
        class Query implements IQuery {

            /**
             * Constructs a new Query.
             * @param [properties] Properties to set
             */
            constructor(properties?: summa.proto.IQuery);

            /** Query boolean. */
            public boolean?: (summa.proto.IBooleanQuery|null);

            /** Query match. */
            public match?: (summa.proto.IMatchQuery|null);

            /** Query regex. */
            public regex?: (summa.proto.IRegexQuery|null);

            /** Query term. */
            public term?: (summa.proto.ITermQuery|null);

            /** Query phrase. */
            public phrase?: (summa.proto.IPhraseQuery|null);

            /** Query range. */
            public range?: (summa.proto.IRangeQuery|null);

            /** Query all. */
            public all?: (summa.proto.IAllQuery|null);

            /** Query more_like_this. */
            public more_like_this?: (summa.proto.IMoreLikeThisQuery|null);

            /** Query boost. */
            public boost?: (summa.proto.IBoostQuery|null);

            /** Query disjunction_max. */
            public disjunction_max?: (summa.proto.IDisjunctionMaxQuery|null);

            /** Query empty. */
            public empty?: (summa.proto.IEmptyQuery|null);

            /** Query exists. */
            public exists?: (summa.proto.IExistsQuery|null);

            /** Query query. */
            public query?: ("boolean"|"match"|"regex"|"term"|"phrase"|"range"|"all"|"more_like_this"|"boost"|"disjunction_max"|"empty"|"exists");

            /**
             * Creates a new Query instance using the specified properties.
             * @param [properties] Properties to set
             * @returns Query instance
             */
            public static create(properties?: summa.proto.IQuery): summa.proto.Query;
        }

        /** Properties of an AllQuery. */
        interface IAllQuery {
        }

        /** Represents an AllQuery. */
        class AllQuery implements IAllQuery {

            /**
             * Constructs a new AllQuery.
             * @param [properties] Properties to set
             */
            constructor(properties?: summa.proto.IAllQuery);

            /**
             * Creates a new AllQuery instance using the specified properties.
             * @param [properties] Properties to set
             * @returns AllQuery instance
             */
            public static create(properties?: summa.proto.IAllQuery): summa.proto.AllQuery;
        }

        /** Properties of an EmptyQuery. */
        interface IEmptyQuery {
        }

        /** Represents an EmptyQuery. */
        class EmptyQuery implements IEmptyQuery {

            /**
             * Constructs a new EmptyQuery.
             * @param [properties] Properties to set
             */
            constructor(properties?: summa.proto.IEmptyQuery);

            /**
             * Creates a new EmptyQuery instance using the specified properties.
             * @param [properties] Properties to set
             * @returns EmptyQuery instance
             */
            public static create(properties?: summa.proto.IEmptyQuery): summa.proto.EmptyQuery;
        }

        /** Properties of a BoostQuery. */
        interface IBoostQuery {

            /** BoostQuery query */
            query?: (summa.proto.IQuery|null);

            /** BoostQuery score */
            score?: (string|null);
        }

        /** Represents a BoostQuery. */
        class BoostQuery implements IBoostQuery {

            /**
             * Constructs a new BoostQuery.
             * @param [properties] Properties to set
             */
            constructor(properties?: summa.proto.IBoostQuery);

            /** BoostQuery query. */
            public query?: (summa.proto.IQuery|null);

            /** BoostQuery score. */
            public score: string;

            /**
             * Creates a new BoostQuery instance using the specified properties.
             * @param [properties] Properties to set
             * @returns BoostQuery instance
             */
            public static create(properties?: summa.proto.IBoostQuery): summa.proto.BoostQuery;
        }

        /** Properties of a DisjunctionMaxQuery. */
        interface IDisjunctionMaxQuery {

            /** DisjunctionMaxQuery disjuncts */
            disjuncts?: (summa.proto.IQuery[]|null);

            /** DisjunctionMaxQuery tie_breaker */
            tie_breaker?: (string|null);
        }

        /** Represents a DisjunctionMaxQuery. */
        class DisjunctionMaxQuery implements IDisjunctionMaxQuery {

            /**
             * Constructs a new DisjunctionMaxQuery.
             * @param [properties] Properties to set
             */
            constructor(properties?: summa.proto.IDisjunctionMaxQuery);

            /** DisjunctionMaxQuery disjuncts. */
            public disjuncts: summa.proto.IQuery[];

            /** DisjunctionMaxQuery tie_breaker. */
            public tie_breaker: string;

            /**
             * Creates a new DisjunctionMaxQuery instance using the specified properties.
             * @param [properties] Properties to set
             * @returns DisjunctionMaxQuery instance
             */
            public static create(properties?: summa.proto.IDisjunctionMaxQuery): summa.proto.DisjunctionMaxQuery;
        }

        /** Properties of a MoreLikeThisQuery. */
        interface IMoreLikeThisQuery {

            /** MoreLikeThisQuery document */
            document?: (string|null);

            /** MoreLikeThisQuery min_doc_frequency */
            min_doc_frequency?: (number|Long|null);

            /** MoreLikeThisQuery max_doc_frequency */
            max_doc_frequency?: (number|Long|null);

            /** MoreLikeThisQuery min_term_frequency */
            min_term_frequency?: (number|Long|null);

            /** MoreLikeThisQuery max_query_terms */
            max_query_terms?: (number|Long|null);

            /** MoreLikeThisQuery min_word_length */
            min_word_length?: (number|Long|null);

            /** MoreLikeThisQuery max_word_length */
            max_word_length?: (number|Long|null);

            /** MoreLikeThisQuery boost */
            boost?: (string|null);

            /** MoreLikeThisQuery stop_words */
            stop_words?: (string[]|null);
        }

        /** Represents a MoreLikeThisQuery. */
        class MoreLikeThisQuery implements IMoreLikeThisQuery {

            /**
             * Constructs a new MoreLikeThisQuery.
             * @param [properties] Properties to set
             */
            constructor(properties?: summa.proto.IMoreLikeThisQuery);

            /** MoreLikeThisQuery document. */
            public document: string;

            /** MoreLikeThisQuery min_doc_frequency. */
            public min_doc_frequency?: (number|Long|null);

            /** MoreLikeThisQuery max_doc_frequency. */
            public max_doc_frequency?: (number|Long|null);

            /** MoreLikeThisQuery min_term_frequency. */
            public min_term_frequency?: (number|Long|null);

            /** MoreLikeThisQuery max_query_terms. */
            public max_query_terms?: (number|Long|null);

            /** MoreLikeThisQuery min_word_length. */
            public min_word_length?: (number|Long|null);

            /** MoreLikeThisQuery max_word_length. */
            public max_word_length?: (number|Long|null);

            /** MoreLikeThisQuery boost. */
            public boost?: (string|null);

            /** MoreLikeThisQuery stop_words. */
            public stop_words: string[];

            /** MoreLikeThisQuery _min_doc_frequency. */
            public _min_doc_frequency?: "min_doc_frequency";

            /** MoreLikeThisQuery _max_doc_frequency. */
            public _max_doc_frequency?: "max_doc_frequency";

            /** MoreLikeThisQuery _min_term_frequency. */
            public _min_term_frequency?: "min_term_frequency";

            /** MoreLikeThisQuery _max_query_terms. */
            public _max_query_terms?: "max_query_terms";

            /** MoreLikeThisQuery _min_word_length. */
            public _min_word_length?: "min_word_length";

            /** MoreLikeThisQuery _max_word_length. */
            public _max_word_length?: "max_word_length";

            /** MoreLikeThisQuery _boost. */
            public _boost?: "boost";

            /**
             * Creates a new MoreLikeThisQuery instance using the specified properties.
             * @param [properties] Properties to set
             * @returns MoreLikeThisQuery instance
             */
            public static create(properties?: summa.proto.IMoreLikeThisQuery): summa.proto.MoreLikeThisQuery;
        }

        /** Properties of a PhraseQuery. */
        interface IPhraseQuery {

            /** PhraseQuery field */
            field?: (string|null);

            /** PhraseQuery value */
            value?: (string|null);

            /** PhraseQuery slop */
            slop?: (number|null);
        }

        /** Represents a PhraseQuery. */
        class PhraseQuery implements IPhraseQuery {

            /**
             * Constructs a new PhraseQuery.
             * @param [properties] Properties to set
             */
            constructor(properties?: summa.proto.IPhraseQuery);

            /** PhraseQuery field. */
            public field: string;

            /** PhraseQuery value. */
            public value: string;

            /** PhraseQuery slop. */
            public slop: number;

            /**
             * Creates a new PhraseQuery instance using the specified properties.
             * @param [properties] Properties to set
             * @returns PhraseQuery instance
             */
            public static create(properties?: summa.proto.IPhraseQuery): summa.proto.PhraseQuery;
        }

        /** Properties of a RangeQuery. */
        interface IRangeQuery {

            /** RangeQuery field */
            field?: (string|null);

            /** RangeQuery value */
            value?: (summa.proto.IRange|null);
        }

        /** Represents a RangeQuery. */
        class RangeQuery implements IRangeQuery {

            /**
             * Constructs a new RangeQuery.
             * @param [properties] Properties to set
             */
            constructor(properties?: summa.proto.IRangeQuery);

            /** RangeQuery field. */
            public field: string;

            /** RangeQuery value. */
            public value?: (summa.proto.IRange|null);

            /**
             * Creates a new RangeQuery instance using the specified properties.
             * @param [properties] Properties to set
             * @returns RangeQuery instance
             */
            public static create(properties?: summa.proto.IRangeQuery): summa.proto.RangeQuery;
        }

        /** Properties of a MatchQuery. */
        interface IMatchQuery {

            /** MatchQuery value */
            value?: (string|null);

            /** MatchQuery query_parser_config */
            query_parser_config?: (summa.proto.IQueryParserConfig|null);
        }

        /** Represents a MatchQuery. */
        class MatchQuery implements IMatchQuery {

            /**
             * Constructs a new MatchQuery.
             * @param [properties] Properties to set
             */
            constructor(properties?: summa.proto.IMatchQuery);

            /** MatchQuery value. */
            public value: string;

            /** MatchQuery query_parser_config. */
            public query_parser_config?: (summa.proto.IQueryParserConfig|null);

            /** MatchQuery _query_parser_config. */
            public _query_parser_config?: "query_parser_config";

            /**
             * Creates a new MatchQuery instance using the specified properties.
             * @param [properties] Properties to set
             * @returns MatchQuery instance
             */
            public static create(properties?: summa.proto.IMatchQuery): summa.proto.MatchQuery;
        }

        /** Properties of a BooleanSubquery. */
        interface IBooleanSubquery {

            /** BooleanSubquery occur */
            occur?: (summa.proto.Occur|null);

            /** BooleanSubquery query */
            query?: (summa.proto.IQuery|null);
        }

        /** Represents a BooleanSubquery. */
        class BooleanSubquery implements IBooleanSubquery {

            /**
             * Constructs a new BooleanSubquery.
             * @param [properties] Properties to set
             */
            constructor(properties?: summa.proto.IBooleanSubquery);

            /** BooleanSubquery occur. */
            public occur: summa.proto.Occur;

            /** BooleanSubquery query. */
            public query?: (summa.proto.IQuery|null);

            /**
             * Creates a new BooleanSubquery instance using the specified properties.
             * @param [properties] Properties to set
             * @returns BooleanSubquery instance
             */
            public static create(properties?: summa.proto.IBooleanSubquery): summa.proto.BooleanSubquery;
        }

        /** Properties of a BooleanQuery. */
        interface IBooleanQuery {

            /** BooleanQuery subqueries */
            subqueries?: (summa.proto.IBooleanSubquery[]|null);
        }

        /** Represents a BooleanQuery. */
        class BooleanQuery implements IBooleanQuery {

            /**
             * Constructs a new BooleanQuery.
             * @param [properties] Properties to set
             */
            constructor(properties?: summa.proto.IBooleanQuery);

            /** BooleanQuery subqueries. */
            public subqueries: summa.proto.IBooleanSubquery[];

            /**
             * Creates a new BooleanQuery instance using the specified properties.
             * @param [properties] Properties to set
             * @returns BooleanQuery instance
             */
            public static create(properties?: summa.proto.IBooleanQuery): summa.proto.BooleanQuery;
        }

        /** Properties of a RegexQuery. */
        interface IRegexQuery {

            /** RegexQuery field */
            field?: (string|null);

            /** RegexQuery value */
            value?: (string|null);
        }

        /** Represents a RegexQuery. */
        class RegexQuery implements IRegexQuery {

            /**
             * Constructs a new RegexQuery.
             * @param [properties] Properties to set
             */
            constructor(properties?: summa.proto.IRegexQuery);

            /** RegexQuery field. */
            public field: string;

            /** RegexQuery value. */
            public value: string;

            /**
             * Creates a new RegexQuery instance using the specified properties.
             * @param [properties] Properties to set
             * @returns RegexQuery instance
             */
            public static create(properties?: summa.proto.IRegexQuery): summa.proto.RegexQuery;
        }

        /** Properties of a TermQuery. */
        interface ITermQuery {

            /** TermQuery field */
            field?: (string|null);

            /** TermQuery value */
            value?: (string|null);
        }

        /** Represents a TermQuery. */
        class TermQuery implements ITermQuery {

            /**
             * Constructs a new TermQuery.
             * @param [properties] Properties to set
             */
            constructor(properties?: summa.proto.ITermQuery);

            /** TermQuery field. */
            public field: string;

            /** TermQuery value. */
            public value: string;

            /**
             * Creates a new TermQuery instance using the specified properties.
             * @param [properties] Properties to set
             * @returns TermQuery instance
             */
            public static create(properties?: summa.proto.ITermQuery): summa.proto.TermQuery;
        }

        /** Properties of an ExistsQuery. */
        interface IExistsQuery {

            /** ExistsQuery field */
            field?: (string|null);
        }

        /** Represents an ExistsQuery. */
        class ExistsQuery implements IExistsQuery {

            /**
             * Constructs a new ExistsQuery.
             * @param [properties] Properties to set
             */
            constructor(properties?: summa.proto.IExistsQuery);

            /** ExistsQuery field. */
            public field: string;

            /**
             * Creates a new ExistsQuery instance using the specified properties.
             * @param [properties] Properties to set
             * @returns ExistsQuery instance
             */
            public static create(properties?: summa.proto.IExistsQuery): summa.proto.ExistsQuery;
        }

        /** Properties of an Aggregation. */
        interface IAggregation {

            /** Aggregation bucket */
            bucket?: (summa.proto.IBucketAggregation|null);

            /** Aggregation metric */
            metric?: (summa.proto.IMetricAggregation|null);
        }

        /** Represents an Aggregation. */
        class Aggregation implements IAggregation {

            /**
             * Constructs a new Aggregation.
             * @param [properties] Properties to set
             */
            constructor(properties?: summa.proto.IAggregation);

            /** Aggregation bucket. */
            public bucket?: (summa.proto.IBucketAggregation|null);

            /** Aggregation metric. */
            public metric?: (summa.proto.IMetricAggregation|null);

            /** Aggregation aggregation. */
            public aggregation?: ("bucket"|"metric");

            /**
             * Creates a new Aggregation instance using the specified properties.
             * @param [properties] Properties to set
             * @returns Aggregation instance
             */
            public static create(properties?: summa.proto.IAggregation): summa.proto.Aggregation;
        }

        /** Properties of a BucketAggregation. */
        interface IBucketAggregation {

            /** BucketAggregation range */
            range?: (summa.proto.IRangeAggregation|null);

            /** BucketAggregation histogram */
            histogram?: (summa.proto.IHistogramAggregation|null);

            /** BucketAggregation terms */
            terms?: (summa.proto.ITermsAggregation|null);

            /** BucketAggregation sub_aggregation */
            sub_aggregation?: ({ [k: string]: summa.proto.IAggregation }|null);
        }

        /** Represents a BucketAggregation. */
        class BucketAggregation implements IBucketAggregation {

            /**
             * Constructs a new BucketAggregation.
             * @param [properties] Properties to set
             */
            constructor(properties?: summa.proto.IBucketAggregation);

            /** BucketAggregation range. */
            public range?: (summa.proto.IRangeAggregation|null);

            /** BucketAggregation histogram. */
            public histogram?: (summa.proto.IHistogramAggregation|null);

            /** BucketAggregation terms. */
            public terms?: (summa.proto.ITermsAggregation|null);

            /** BucketAggregation sub_aggregation. */
            public sub_aggregation: { [k: string]: summa.proto.IAggregation };

            /** BucketAggregation bucket_agg. */
            public bucket_agg?: ("range"|"histogram"|"terms");

            /**
             * Creates a new BucketAggregation instance using the specified properties.
             * @param [properties] Properties to set
             * @returns BucketAggregation instance
             */
            public static create(properties?: summa.proto.IBucketAggregation): summa.proto.BucketAggregation;
        }

        /** Properties of a RangeAggregation. */
        interface IRangeAggregation {

            /** RangeAggregation field */
            field?: (string|null);

            /** RangeAggregation ranges */
            ranges?: (summa.proto.IRangeAggregationRange[]|null);
        }

        /** Represents a RangeAggregation. */
        class RangeAggregation implements IRangeAggregation {

            /**
             * Constructs a new RangeAggregation.
             * @param [properties] Properties to set
             */
            constructor(properties?: summa.proto.IRangeAggregation);

            /** RangeAggregation field. */
            public field: string;

            /** RangeAggregation ranges. */
            public ranges: summa.proto.IRangeAggregationRange[];

            /**
             * Creates a new RangeAggregation instance using the specified properties.
             * @param [properties] Properties to set
             * @returns RangeAggregation instance
             */
            public static create(properties?: summa.proto.IRangeAggregation): summa.proto.RangeAggregation;
        }

        /** Properties of a RangeAggregationRange. */
        interface IRangeAggregationRange {

            /** RangeAggregationRange from */
            from?: (number|null);

            /** RangeAggregationRange to */
            to?: (number|null);

            /** RangeAggregationRange key */
            key?: (string|null);
        }

        /** Represents a RangeAggregationRange. */
        class RangeAggregationRange implements IRangeAggregationRange {

            /**
             * Constructs a new RangeAggregationRange.
             * @param [properties] Properties to set
             */
            constructor(properties?: summa.proto.IRangeAggregationRange);

            /** RangeAggregationRange from. */
            public from?: (number|null);

            /** RangeAggregationRange to. */
            public to?: (number|null);

            /** RangeAggregationRange key. */
            public key?: (string|null);

            /** RangeAggregationRange _from. */
            public _from?: "from";

            /** RangeAggregationRange _to. */
            public _to?: "to";

            /** RangeAggregationRange _key. */
            public _key?: "key";

            /**
             * Creates a new RangeAggregationRange instance using the specified properties.
             * @param [properties] Properties to set
             * @returns RangeAggregationRange instance
             */
            public static create(properties?: summa.proto.IRangeAggregationRange): summa.proto.RangeAggregationRange;
        }

        /** Properties of a HistogramAggregation. */
        interface IHistogramAggregation {

            /** HistogramAggregation field */
            field?: (string|null);

            /** HistogramAggregation interval */
            interval?: (number|null);

            /** HistogramAggregation offset */
            offset?: (number|null);

            /** HistogramAggregation min_doc_count */
            min_doc_count?: (number|Long|null);

            /** HistogramAggregation hard_bounds */
            hard_bounds?: (summa.proto.IHistogramBounds|null);

            /** HistogramAggregation extended_bounds */
            extended_bounds?: (summa.proto.IHistogramBounds|null);
        }

        /** Represents a HistogramAggregation. */
        class HistogramAggregation implements IHistogramAggregation {

            /**
             * Constructs a new HistogramAggregation.
             * @param [properties] Properties to set
             */
            constructor(properties?: summa.proto.IHistogramAggregation);

            /** HistogramAggregation field. */
            public field: string;

            /** HistogramAggregation interval. */
            public interval: number;

            /** HistogramAggregation offset. */
            public offset?: (number|null);

            /** HistogramAggregation min_doc_count. */
            public min_doc_count?: (number|Long|null);

            /** HistogramAggregation hard_bounds. */
            public hard_bounds?: (summa.proto.IHistogramBounds|null);

            /** HistogramAggregation extended_bounds. */
            public extended_bounds?: (summa.proto.IHistogramBounds|null);

            /** HistogramAggregation _offset. */
            public _offset?: "offset";

            /** HistogramAggregation _min_doc_count. */
            public _min_doc_count?: "min_doc_count";

            /** HistogramAggregation _hard_bounds. */
            public _hard_bounds?: "hard_bounds";

            /** HistogramAggregation _extended_bounds. */
            public _extended_bounds?: "extended_bounds";

            /**
             * Creates a new HistogramAggregation instance using the specified properties.
             * @param [properties] Properties to set
             * @returns HistogramAggregation instance
             */
            public static create(properties?: summa.proto.IHistogramAggregation): summa.proto.HistogramAggregation;
        }

        /** Properties of a HistogramBounds. */
        interface IHistogramBounds {

            /** HistogramBounds min */
            min?: (number|null);

            /** HistogramBounds max */
            max?: (number|null);
        }

        /** Represents a HistogramBounds. */
        class HistogramBounds implements IHistogramBounds {

            /**
             * Constructs a new HistogramBounds.
             * @param [properties] Properties to set
             */
            constructor(properties?: summa.proto.IHistogramBounds);

            /** HistogramBounds min. */
            public min: number;

            /** HistogramBounds max. */
            public max: number;

            /**
             * Creates a new HistogramBounds instance using the specified properties.
             * @param [properties] Properties to set
             * @returns HistogramBounds instance
             */
            public static create(properties?: summa.proto.IHistogramBounds): summa.proto.HistogramBounds;
        }

        /** Properties of a TermsAggregation. */
        interface ITermsAggregation {

            /** TermsAggregation field */
            field?: (string|null);

            /** TermsAggregation size */
            size?: (number|null);

            /** TermsAggregation split_size */
            split_size?: (number|null);

            /** TermsAggregation segment_size */
            segment_size?: (number|null);

            /** TermsAggregation show_term_doc_count_error */
            show_term_doc_count_error?: (boolean|null);

            /** TermsAggregation min_doc_count */
            min_doc_count?: (number|Long|null);

            /** TermsAggregation order */
            order?: (summa.proto.ICustomOrder|null);
        }

        /** Represents a TermsAggregation. */
        class TermsAggregation implements ITermsAggregation {

            /**
             * Constructs a new TermsAggregation.
             * @param [properties] Properties to set
             */
            constructor(properties?: summa.proto.ITermsAggregation);

            /** TermsAggregation field. */
            public field: string;

            /** TermsAggregation size. */
            public size?: (number|null);

            /** TermsAggregation split_size. */
            public split_size?: (number|null);

            /** TermsAggregation segment_size. */
            public segment_size?: (number|null);

            /** TermsAggregation show_term_doc_count_error. */
            public show_term_doc_count_error?: (boolean|null);

            /** TermsAggregation min_doc_count. */
            public min_doc_count?: (number|Long|null);

            /** TermsAggregation order. */
            public order?: (summa.proto.ICustomOrder|null);

            /** TermsAggregation _size. */
            public _size?: "size";

            /** TermsAggregation _split_size. */
            public _split_size?: "split_size";

            /** TermsAggregation _segment_size. */
            public _segment_size?: "segment_size";

            /** TermsAggregation _show_term_doc_count_error. */
            public _show_term_doc_count_error?: "show_term_doc_count_error";

            /** TermsAggregation _min_doc_count. */
            public _min_doc_count?: "min_doc_count";

            /** TermsAggregation _order. */
            public _order?: "order";

            /**
             * Creates a new TermsAggregation instance using the specified properties.
             * @param [properties] Properties to set
             * @returns TermsAggregation instance
             */
            public static create(properties?: summa.proto.ITermsAggregation): summa.proto.TermsAggregation;
        }

        /** Properties of a CustomOrder. */
        interface ICustomOrder {

            /** CustomOrder key */
            key?: (summa.proto.IEmpty|null);

            /** CustomOrder count */
            count?: (summa.proto.IEmpty|null);

            /** CustomOrder sub_aggregation */
            sub_aggregation?: (string|null);

            /** CustomOrder order */
            order?: (summa.proto.Order|null);
        }

        /** Represents a CustomOrder. */
        class CustomOrder implements ICustomOrder {

            /**
             * Constructs a new CustomOrder.
             * @param [properties] Properties to set
             */
            constructor(properties?: summa.proto.ICustomOrder);

            /** CustomOrder key. */
            public key?: (summa.proto.IEmpty|null);

            /** CustomOrder count. */
            public count?: (summa.proto.IEmpty|null);

            /** CustomOrder sub_aggregation. */
            public sub_aggregation?: (string|null);

            /** CustomOrder order. */
            public order: summa.proto.Order;

            /** CustomOrder order_target. */
            public order_target?: ("key"|"count"|"sub_aggregation");

            /**
             * Creates a new CustomOrder instance using the specified properties.
             * @param [properties] Properties to set
             * @returns CustomOrder instance
             */
            public static create(properties?: summa.proto.ICustomOrder): summa.proto.CustomOrder;
        }

        /** Properties of a MetricAggregation. */
        interface IMetricAggregation {

            /** MetricAggregation average */
            average?: (summa.proto.IAverageAggregation|null);

            /** MetricAggregation stats */
            stats?: (summa.proto.IStatsAggregation|null);
        }

        /** Represents a MetricAggregation. */
        class MetricAggregation implements IMetricAggregation {

            /**
             * Constructs a new MetricAggregation.
             * @param [properties] Properties to set
             */
            constructor(properties?: summa.proto.IMetricAggregation);

            /** MetricAggregation average. */
            public average?: (summa.proto.IAverageAggregation|null);

            /** MetricAggregation stats. */
            public stats?: (summa.proto.IStatsAggregation|null);

            /** MetricAggregation metric_aggregation. */
            public metric_aggregation?: ("average"|"stats");

            /**
             * Creates a new MetricAggregation instance using the specified properties.
             * @param [properties] Properties to set
             * @returns MetricAggregation instance
             */
            public static create(properties?: summa.proto.IMetricAggregation): summa.proto.MetricAggregation;
        }

        /** Properties of an AverageAggregation. */
        interface IAverageAggregation {

            /** AverageAggregation field */
            field?: (string|null);
        }

        /** Represents an AverageAggregation. */
        class AverageAggregation implements IAverageAggregation {

            /**
             * Constructs a new AverageAggregation.
             * @param [properties] Properties to set
             */
            constructor(properties?: summa.proto.IAverageAggregation);

            /** AverageAggregation field. */
            public field: string;

            /**
             * Creates a new AverageAggregation instance using the specified properties.
             * @param [properties] Properties to set
             * @returns AverageAggregation instance
             */
            public static create(properties?: summa.proto.IAverageAggregation): summa.proto.AverageAggregation;
        }

        /** Properties of a StatsAggregation. */
        interface IStatsAggregation {

            /** StatsAggregation field */
            field?: (string|null);
        }

        /** Represents a StatsAggregation. */
        class StatsAggregation implements IStatsAggregation {

            /**
             * Constructs a new StatsAggregation.
             * @param [properties] Properties to set
             */
            constructor(properties?: summa.proto.IStatsAggregation);

            /** StatsAggregation field. */
            public field: string;

            /**
             * Creates a new StatsAggregation instance using the specified properties.
             * @param [properties] Properties to set
             * @returns StatsAggregation instance
             */
            public static create(properties?: summa.proto.IStatsAggregation): summa.proto.StatsAggregation;
        }

        /** Properties of a BucketEntry. */
        interface IBucketEntry {

            /** BucketEntry key */
            key?: (summa.proto.IKey|null);

            /** BucketEntry doc_count */
            doc_count?: (number|Long|null);

            /** BucketEntry sub_aggregation */
            sub_aggregation?: ({ [k: string]: summa.proto.IAggregationResult }|null);
        }

        /** Represents a BucketEntry. */
        class BucketEntry implements IBucketEntry {

            /**
             * Constructs a new BucketEntry.
             * @param [properties] Properties to set
             */
            constructor(properties?: summa.proto.IBucketEntry);

            /** BucketEntry key. */
            public key?: (summa.proto.IKey|null);

            /** BucketEntry doc_count. */
            public doc_count: (number|Long);

            /** BucketEntry sub_aggregation. */
            public sub_aggregation: { [k: string]: summa.proto.IAggregationResult };

            /**
             * Creates a new BucketEntry instance using the specified properties.
             * @param [properties] Properties to set
             * @returns BucketEntry instance
             */
            public static create(properties?: summa.proto.IBucketEntry): summa.proto.BucketEntry;
        }

        /** Properties of a Key. */
        interface IKey {

            /** Key str */
            str?: (string|null);

            /** Key f64 */
            f64?: (number|null);
        }

        /** Represents a Key. */
        class Key implements IKey {

            /**
             * Constructs a new Key.
             * @param [properties] Properties to set
             */
            constructor(properties?: summa.proto.IKey);

            /** Key str. */
            public str?: (string|null);

            /** Key f64. */
            public f64?: (number|null);

            /** Key key. */
            public key?: ("str"|"f64");

            /**
             * Creates a new Key instance using the specified properties.
             * @param [properties] Properties to set
             * @returns Key instance
             */
            public static create(properties?: summa.proto.IKey): summa.proto.Key;
        }

        /** Occur enum. */
        enum Occur {
            should = 0,
            must = 1,
            must_not = 2
        }

        /** Properties of a Range. */
        interface IRange {

            /** Range left */
            left?: (string|null);

            /** Range right */
            right?: (string|null);

            /** Range including_left */
            including_left?: (boolean|null);

            /** Range including_right */
            including_right?: (boolean|null);
        }

        /** Represents a Range. */
        class Range implements IRange {

            /**
             * Constructs a new Range.
             * @param [properties] Properties to set
             */
            constructor(properties?: summa.proto.IRange);

            /** Range left. */
            public left: string;

            /** Range right. */
            public right: string;

            /** Range including_left. */
            public including_left: boolean;

            /** Range including_right. */
            public including_right: boolean;

            /**
             * Creates a new Range instance using the specified properties.
             * @param [properties] Properties to set
             * @returns Range instance
             */
            public static create(properties?: summa.proto.IRange): summa.proto.Range;
        }

        /** Properties of a RangeBucketEntry. */
        interface IRangeBucketEntry {

            /** RangeBucketEntry key */
            key?: (summa.proto.IKey|null);

            /** RangeBucketEntry doc_count */
            doc_count?: (number|Long|null);

            /** RangeBucketEntry sub_aggregation */
            sub_aggregation?: ({ [k: string]: summa.proto.IAggregationResult }|null);

            /** RangeBucketEntry from */
            from?: (number|null);

            /** RangeBucketEntry to */
            to?: (number|null);
        }

        /** Represents a RangeBucketEntry. */
        class RangeBucketEntry implements IRangeBucketEntry {

            /**
             * Constructs a new RangeBucketEntry.
             * @param [properties] Properties to set
             */
            constructor(properties?: summa.proto.IRangeBucketEntry);

            /** RangeBucketEntry key. */
            public key?: (summa.proto.IKey|null);

            /** RangeBucketEntry doc_count. */
            public doc_count: (number|Long);

            /** RangeBucketEntry sub_aggregation. */
            public sub_aggregation: { [k: string]: summa.proto.IAggregationResult };

            /** RangeBucketEntry from. */
            public from?: (number|null);

            /** RangeBucketEntry to. */
            public to?: (number|null);

            /** RangeBucketEntry _from. */
            public _from?: "from";

            /** RangeBucketEntry _to. */
            public _to?: "to";

            /**
             * Creates a new RangeBucketEntry instance using the specified properties.
             * @param [properties] Properties to set
             * @returns RangeBucketEntry instance
             */
            public static create(properties?: summa.proto.IRangeBucketEntry): summa.proto.RangeBucketEntry;
        }

        /** Properties of a Score. */
        interface IScore {

            /** Score f64_score */
            f64_score?: (number|null);

            /** Score u64_score */
            u64_score?: (number|Long|null);
        }

        /** Represents a Score. */
        class Score implements IScore {

            /**
             * Constructs a new Score.
             * @param [properties] Properties to set
             */
            constructor(properties?: summa.proto.IScore);

            /** Score f64_score. */
            public f64_score?: (number|null);

            /** Score u64_score. */
            public u64_score?: (number|Long|null);

            /** Score score. */
            public score?: ("f64_score"|"u64_score");

            /**
             * Creates a new Score instance using the specified properties.
             * @param [properties] Properties to set
             * @returns Score instance
             */
            public static create(properties?: summa.proto.IScore): summa.proto.Score;
        }

        /** Properties of a Highlight. */
        interface IHighlight {

            /** Highlight from */
            from?: (number|null);

            /** Highlight to */
            to?: (number|null);
        }

        /** Represents a Highlight. */
        class Highlight implements IHighlight {

            /**
             * Constructs a new Highlight.
             * @param [properties] Properties to set
             */
            constructor(properties?: summa.proto.IHighlight);

            /** Highlight from. */
            public from: number;

            /** Highlight to. */
            public to: number;

            /**
             * Creates a new Highlight instance using the specified properties.
             * @param [properties] Properties to set
             * @returns Highlight instance
             */
            public static create(properties?: summa.proto.IHighlight): summa.proto.Highlight;
        }

        /** Properties of a Snippet. */
        interface ISnippet {

            /** Snippet fragment */
            fragment?: (Uint8Array|null);

            /** Snippet highlights */
            highlights?: (summa.proto.IHighlight[]|null);

            /** Snippet html */
            html?: (string|null);
        }

        /** Represents a Snippet. */
        class Snippet implements ISnippet {

            /**
             * Constructs a new Snippet.
             * @param [properties] Properties to set
             */
            constructor(properties?: summa.proto.ISnippet);

            /** Snippet fragment. */
            public fragment: Uint8Array;

            /** Snippet highlights. */
            public highlights: summa.proto.IHighlight[];

            /** Snippet html. */
            public html: string;

            /**
             * Creates a new Snippet instance using the specified properties.
             * @param [properties] Properties to set
             * @returns Snippet instance
             */
            public static create(properties?: summa.proto.ISnippet): summa.proto.Snippet;
        }

        /** Properties of a ScoredDocument. */
        interface IScoredDocument {

            /** ScoredDocument document */
            document?: (string|null);

            /** ScoredDocument score */
            score?: (summa.proto.IScore|null);

            /** ScoredDocument position */
            position?: (number|null);

            /** ScoredDocument snippets */
            snippets?: ({ [k: string]: summa.proto.ISnippet }|null);

            /** ScoredDocument index_alias */
            index_alias?: (string|null);
        }

        /** Represents a ScoredDocument. */
        class ScoredDocument implements IScoredDocument {

            /**
             * Constructs a new ScoredDocument.
             * @param [properties] Properties to set
             */
            constructor(properties?: summa.proto.IScoredDocument);

            /** ScoredDocument document. */
            public document: string;

            /** ScoredDocument score. */
            public score?: (summa.proto.IScore|null);

            /** ScoredDocument position. */
            public position: number;

            /** ScoredDocument snippets. */
            public snippets: { [k: string]: summa.proto.ISnippet };

            /** ScoredDocument index_alias. */
            public index_alias: string;

            /**
             * Creates a new ScoredDocument instance using the specified properties.
             * @param [properties] Properties to set
             * @returns ScoredDocument instance
             */
            public static create(properties?: summa.proto.IScoredDocument): summa.proto.ScoredDocument;
        }

        /** Properties of a Scorer. */
        interface IScorer {

            /** Scorer eval_expr */
            eval_expr?: (string|null);

            /** Scorer order_by */
            order_by?: (string|null);
        }

        /** Represents a Scorer. */
        class Scorer implements IScorer {

            /**
             * Constructs a new Scorer.
             * @param [properties] Properties to set
             */
            constructor(properties?: summa.proto.IScorer);

            /** Scorer eval_expr. */
            public eval_expr?: (string|null);

            /** Scorer order_by. */
            public order_by?: (string|null);

            /** Scorer scorer. */
            public scorer?: ("eval_expr"|"order_by");

            /**
             * Creates a new Scorer instance using the specified properties.
             * @param [properties] Properties to set
             * @returns Scorer instance
             */
            public static create(properties?: summa.proto.IScorer): summa.proto.Scorer;
        }

        /** Properties of a Collector. */
        interface ICollector {

            /** Collector top_docs */
            top_docs?: (summa.proto.ITopDocsCollector|null);

            /** Collector reservoir_sampling */
            reservoir_sampling?: (summa.proto.IReservoirSamplingCollector|null);

            /** Collector count */
            count?: (summa.proto.ICountCollector|null);

            /** Collector facet */
            facet?: (summa.proto.IFacetCollector|null);

            /** Collector aggregation */
            aggregation?: (summa.proto.IAggregationCollector|null);
        }

        /** Represents a Collector. */
        class Collector implements ICollector {

            /**
             * Constructs a new Collector.
             * @param [properties] Properties to set
             */
            constructor(properties?: summa.proto.ICollector);

            /** Collector top_docs. */
            public top_docs?: (summa.proto.ITopDocsCollector|null);

            /** Collector reservoir_sampling. */
            public reservoir_sampling?: (summa.proto.IReservoirSamplingCollector|null);

            /** Collector count. */
            public count?: (summa.proto.ICountCollector|null);

            /** Collector facet. */
            public facet?: (summa.proto.IFacetCollector|null);

            /** Collector aggregation. */
            public aggregation?: (summa.proto.IAggregationCollector|null);

            /** Collector collector. */
            public collector?: ("top_docs"|"reservoir_sampling"|"count"|"facet"|"aggregation");

            /**
             * Creates a new Collector instance using the specified properties.
             * @param [properties] Properties to set
             * @returns Collector instance
             */
            public static create(properties?: summa.proto.ICollector): summa.proto.Collector;
        }

        /** Properties of a CollectorOutput. */
        interface ICollectorOutput {

            /** CollectorOutput documents */
            documents?: (summa.proto.IDocumentsCollectorOutput|null);

            /** CollectorOutput count */
            count?: (summa.proto.ICountCollectorOutput|null);

            /** CollectorOutput facet */
            facet?: (summa.proto.IFacetCollectorOutput|null);

            /** CollectorOutput aggregation */
            aggregation?: (summa.proto.IAggregationCollectorOutput|null);
        }

        /** Represents a CollectorOutput. */
        class CollectorOutput implements ICollectorOutput {

            /**
             * Constructs a new CollectorOutput.
             * @param [properties] Properties to set
             */
            constructor(properties?: summa.proto.ICollectorOutput);

            /** CollectorOutput documents. */
            public documents?: (summa.proto.IDocumentsCollectorOutput|null);

            /** CollectorOutput count. */
            public count?: (summa.proto.ICountCollectorOutput|null);

            /** CollectorOutput facet. */
            public facet?: (summa.proto.IFacetCollectorOutput|null);

            /** CollectorOutput aggregation. */
            public aggregation?: (summa.proto.IAggregationCollectorOutput|null);

            /** CollectorOutput collector_output. */
            public collector_output?: ("documents"|"count"|"facet"|"aggregation");

            /**
             * Creates a new CollectorOutput instance using the specified properties.
             * @param [properties] Properties to set
             * @returns CollectorOutput instance
             */
            public static create(properties?: summa.proto.ICollectorOutput): summa.proto.CollectorOutput;
        }

        /** Properties of a CountCollector. */
        interface ICountCollector {
        }

        /** Represents a CountCollector. */
        class CountCollector implements ICountCollector {

            /**
             * Constructs a new CountCollector.
             * @param [properties] Properties to set
             */
            constructor(properties?: summa.proto.ICountCollector);

            /**
             * Creates a new CountCollector instance using the specified properties.
             * @param [properties] Properties to set
             * @returns CountCollector instance
             */
            public static create(properties?: summa.proto.ICountCollector): summa.proto.CountCollector;
        }

        /** Properties of a CountCollectorOutput. */
        interface ICountCollectorOutput {

            /** CountCollectorOutput count */
            count?: (number|null);
        }

        /** Represents a CountCollectorOutput. */
        class CountCollectorOutput implements ICountCollectorOutput {

            /**
             * Constructs a new CountCollectorOutput.
             * @param [properties] Properties to set
             */
            constructor(properties?: summa.proto.ICountCollectorOutput);

            /** CountCollectorOutput count. */
            public count: number;

            /**
             * Creates a new CountCollectorOutput instance using the specified properties.
             * @param [properties] Properties to set
             * @returns CountCollectorOutput instance
             */
            public static create(properties?: summa.proto.ICountCollectorOutput): summa.proto.CountCollectorOutput;
        }

        /** Properties of a FacetCollector. */
        interface IFacetCollector {

            /** FacetCollector field */
            field?: (string|null);

            /** FacetCollector facets */
            facets?: (string[]|null);
        }

        /** Represents a FacetCollector. */
        class FacetCollector implements IFacetCollector {

            /**
             * Constructs a new FacetCollector.
             * @param [properties] Properties to set
             */
            constructor(properties?: summa.proto.IFacetCollector);

            /** FacetCollector field. */
            public field: string;

            /** FacetCollector facets. */
            public facets: string[];

            /**
             * Creates a new FacetCollector instance using the specified properties.
             * @param [properties] Properties to set
             * @returns FacetCollector instance
             */
            public static create(properties?: summa.proto.IFacetCollector): summa.proto.FacetCollector;
        }

        /** Properties of a FacetCollectorOutput. */
        interface IFacetCollectorOutput {

            /** FacetCollectorOutput facet_counts */
            facet_counts?: ({ [k: string]: (number|Long) }|null);
        }

        /** Represents a FacetCollectorOutput. */
        class FacetCollectorOutput implements IFacetCollectorOutput {

            /**
             * Constructs a new FacetCollectorOutput.
             * @param [properties] Properties to set
             */
            constructor(properties?: summa.proto.IFacetCollectorOutput);

            /** FacetCollectorOutput facet_counts. */
            public facet_counts: { [k: string]: (number|Long) };

            /**
             * Creates a new FacetCollectorOutput instance using the specified properties.
             * @param [properties] Properties to set
             * @returns FacetCollectorOutput instance
             */
            public static create(properties?: summa.proto.IFacetCollectorOutput): summa.proto.FacetCollectorOutput;
        }

        /** Properties of a ReservoirSamplingCollector. */
        interface IReservoirSamplingCollector {

            /** ReservoirSamplingCollector limit */
            limit?: (number|null);

            /** ReservoirSamplingCollector fields */
            fields?: (string[]|null);
        }

        /** Represents a ReservoirSamplingCollector. */
        class ReservoirSamplingCollector implements IReservoirSamplingCollector {

            /**
             * Constructs a new ReservoirSamplingCollector.
             * @param [properties] Properties to set
             */
            constructor(properties?: summa.proto.IReservoirSamplingCollector);

            /** ReservoirSamplingCollector limit. */
            public limit: number;

            /** ReservoirSamplingCollector fields. */
            public fields: string[];

            /**
             * Creates a new ReservoirSamplingCollector instance using the specified properties.
             * @param [properties] Properties to set
             * @returns ReservoirSamplingCollector instance
             */
            public static create(properties?: summa.proto.IReservoirSamplingCollector): summa.proto.ReservoirSamplingCollector;
        }

        /** Properties of a RandomDocument. */
        interface IRandomDocument {

            /** RandomDocument document */
            document?: (string|null);

            /** RandomDocument score */
            score?: (summa.proto.IScore|null);

            /** RandomDocument index_alias */
            index_alias?: (string|null);
        }

        /** Represents a RandomDocument. */
        class RandomDocument implements IRandomDocument {

            /**
             * Constructs a new RandomDocument.
             * @param [properties] Properties to set
             */
            constructor(properties?: summa.proto.IRandomDocument);

            /** RandomDocument document. */
            public document: string;

            /** RandomDocument score. */
            public score?: (summa.proto.IScore|null);

            /** RandomDocument index_alias. */
            public index_alias: string;

            /**
             * Creates a new RandomDocument instance using the specified properties.
             * @param [properties] Properties to set
             * @returns RandomDocument instance
             */
            public static create(properties?: summa.proto.IRandomDocument): summa.proto.RandomDocument;
        }

        /** Properties of a ReservoirSamplingCollectorOutput. */
        interface IReservoirSamplingCollectorOutput {

            /** ReservoirSamplingCollectorOutput documents */
            documents?: (summa.proto.IRandomDocument[]|null);
        }

        /** Represents a ReservoirSamplingCollectorOutput. */
        class ReservoirSamplingCollectorOutput implements IReservoirSamplingCollectorOutput {

            /**
             * Constructs a new ReservoirSamplingCollectorOutput.
             * @param [properties] Properties to set
             */
            constructor(properties?: summa.proto.IReservoirSamplingCollectorOutput);

            /** ReservoirSamplingCollectorOutput documents. */
            public documents: summa.proto.IRandomDocument[];

            /**
             * Creates a new ReservoirSamplingCollectorOutput instance using the specified properties.
             * @param [properties] Properties to set
             * @returns ReservoirSamplingCollectorOutput instance
             */
            public static create(properties?: summa.proto.IReservoirSamplingCollectorOutput): summa.proto.ReservoirSamplingCollectorOutput;
        }

        /** Properties of a TopDocsCollector. */
        interface ITopDocsCollector {

            /** TopDocsCollector limit */
            limit?: (number|null);

            /** TopDocsCollector offset */
            offset?: (number|null);

            /** TopDocsCollector scorer */
            scorer?: (summa.proto.IScorer|null);

            /** TopDocsCollector snippet_configs */
            snippet_configs?: ({ [k: string]: number }|null);

            /** TopDocsCollector explain */
            explain?: (boolean|null);

            /** TopDocsCollector fields */
            fields?: (string[]|null);
        }

        /** Represents a TopDocsCollector. */
        class TopDocsCollector implements ITopDocsCollector {

            /**
             * Constructs a new TopDocsCollector.
             * @param [properties] Properties to set
             */
            constructor(properties?: summa.proto.ITopDocsCollector);

            /** TopDocsCollector limit. */
            public limit: number;

            /** TopDocsCollector offset. */
            public offset: number;

            /** TopDocsCollector scorer. */
            public scorer?: (summa.proto.IScorer|null);

            /** TopDocsCollector snippet_configs. */
            public snippet_configs: { [k: string]: number };

            /** TopDocsCollector explain. */
            public explain: boolean;

            /** TopDocsCollector fields. */
            public fields: string[];

            /** TopDocsCollector _scorer. */
            public _scorer?: "scorer";

            /**
             * Creates a new TopDocsCollector instance using the specified properties.
             * @param [properties] Properties to set
             * @returns TopDocsCollector instance
             */
            public static create(properties?: summa.proto.ITopDocsCollector): summa.proto.TopDocsCollector;
        }

        /** Properties of a DocumentsCollectorOutput. */
        interface IDocumentsCollectorOutput {

            /** DocumentsCollectorOutput scored_documents */
            scored_documents?: (summa.proto.IScoredDocument[]|null);

            /** DocumentsCollectorOutput has_next */
            has_next?: (boolean|null);
        }

        /** Represents a DocumentsCollectorOutput. */
        class DocumentsCollectorOutput implements IDocumentsCollectorOutput {

            /**
             * Constructs a new DocumentsCollectorOutput.
             * @param [properties] Properties to set
             */
            constructor(properties?: summa.proto.IDocumentsCollectorOutput);

            /** DocumentsCollectorOutput scored_documents. */
            public scored_documents: summa.proto.IScoredDocument[];

            /** DocumentsCollectorOutput has_next. */
            public has_next: boolean;

            /**
             * Creates a new DocumentsCollectorOutput instance using the specified properties.
             * @param [properties] Properties to set
             * @returns DocumentsCollectorOutput instance
             */
            public static create(properties?: summa.proto.IDocumentsCollectorOutput): summa.proto.DocumentsCollectorOutput;
        }

        /** Properties of an AggregationCollector. */
        interface IAggregationCollector {

            /** AggregationCollector aggregations */
            aggregations?: ({ [k: string]: summa.proto.IAggregation }|null);
        }

        /** Represents an AggregationCollector. */
        class AggregationCollector implements IAggregationCollector {

            /**
             * Constructs a new AggregationCollector.
             * @param [properties] Properties to set
             */
            constructor(properties?: summa.proto.IAggregationCollector);

            /** AggregationCollector aggregations. */
            public aggregations: { [k: string]: summa.proto.IAggregation };

            /**
             * Creates a new AggregationCollector instance using the specified properties.
             * @param [properties] Properties to set
             * @returns AggregationCollector instance
             */
            public static create(properties?: summa.proto.IAggregationCollector): summa.proto.AggregationCollector;
        }

        /** Properties of an AggregationCollectorOutput. */
        interface IAggregationCollectorOutput {

            /** AggregationCollectorOutput aggregation_results */
            aggregation_results?: ({ [k: string]: summa.proto.IAggregationResult }|null);
        }

        /** Represents an AggregationCollectorOutput. */
        class AggregationCollectorOutput implements IAggregationCollectorOutput {

            /**
             * Constructs a new AggregationCollectorOutput.
             * @param [properties] Properties to set
             */
            constructor(properties?: summa.proto.IAggregationCollectorOutput);

            /** AggregationCollectorOutput aggregation_results. */
            public aggregation_results: { [k: string]: summa.proto.IAggregationResult };

            /**
             * Creates a new AggregationCollectorOutput instance using the specified properties.
             * @param [properties] Properties to set
             * @returns AggregationCollectorOutput instance
             */
            public static create(properties?: summa.proto.IAggregationCollectorOutput): summa.proto.AggregationCollectorOutput;
        }

        /** Properties of an AggregationResult. */
        interface IAggregationResult {

            /** AggregationResult bucket */
            bucket?: (summa.proto.IBucketResult|null);

            /** AggregationResult metric */
            metric?: (summa.proto.IMetricResult|null);
        }

        /** Represents an AggregationResult. */
        class AggregationResult implements IAggregationResult {

            /**
             * Constructs a new AggregationResult.
             * @param [properties] Properties to set
             */
            constructor(properties?: summa.proto.IAggregationResult);

            /** AggregationResult bucket. */
            public bucket?: (summa.proto.IBucketResult|null);

            /** AggregationResult metric. */
            public metric?: (summa.proto.IMetricResult|null);

            /** AggregationResult aggregation_result. */
            public aggregation_result?: ("bucket"|"metric");

            /**
             * Creates a new AggregationResult instance using the specified properties.
             * @param [properties] Properties to set
             * @returns AggregationResult instance
             */
            public static create(properties?: summa.proto.IAggregationResult): summa.proto.AggregationResult;
        }

        /** Properties of a BucketResult. */
        interface IBucketResult {

            /** BucketResult range */
            range?: (summa.proto.IRangeResult|null);

            /** BucketResult histogram */
            histogram?: (summa.proto.IHistogramResult|null);

            /** BucketResult terms */
            terms?: (summa.proto.ITermsResult|null);
        }

        /** Represents a BucketResult. */
        class BucketResult implements IBucketResult {

            /**
             * Constructs a new BucketResult.
             * @param [properties] Properties to set
             */
            constructor(properties?: summa.proto.IBucketResult);

            /** BucketResult range. */
            public range?: (summa.proto.IRangeResult|null);

            /** BucketResult histogram. */
            public histogram?: (summa.proto.IHistogramResult|null);

            /** BucketResult terms. */
            public terms?: (summa.proto.ITermsResult|null);

            /** BucketResult bucket_result. */
            public bucket_result?: ("range"|"histogram"|"terms");

            /**
             * Creates a new BucketResult instance using the specified properties.
             * @param [properties] Properties to set
             * @returns BucketResult instance
             */
            public static create(properties?: summa.proto.IBucketResult): summa.proto.BucketResult;
        }

        /** Properties of a RangeResult. */
        interface IRangeResult {

            /** RangeResult buckets */
            buckets?: (summa.proto.IRangeBucketEntry[]|null);
        }

        /** Represents a RangeResult. */
        class RangeResult implements IRangeResult {

            /**
             * Constructs a new RangeResult.
             * @param [properties] Properties to set
             */
            constructor(properties?: summa.proto.IRangeResult);

            /** RangeResult buckets. */
            public buckets: summa.proto.IRangeBucketEntry[];

            /**
             * Creates a new RangeResult instance using the specified properties.
             * @param [properties] Properties to set
             * @returns RangeResult instance
             */
            public static create(properties?: summa.proto.IRangeResult): summa.proto.RangeResult;
        }

        /** Properties of a HistogramResult. */
        interface IHistogramResult {

            /** HistogramResult buckets */
            buckets?: (summa.proto.IBucketEntry[]|null);
        }

        /** Represents a HistogramResult. */
        class HistogramResult implements IHistogramResult {

            /**
             * Constructs a new HistogramResult.
             * @param [properties] Properties to set
             */
            constructor(properties?: summa.proto.IHistogramResult);

            /** HistogramResult buckets. */
            public buckets: summa.proto.IBucketEntry[];

            /**
             * Creates a new HistogramResult instance using the specified properties.
             * @param [properties] Properties to set
             * @returns HistogramResult instance
             */
            public static create(properties?: summa.proto.IHistogramResult): summa.proto.HistogramResult;
        }

        /** Properties of a TermsResult. */
        interface ITermsResult {

            /** TermsResult buckets */
            buckets?: (summa.proto.IBucketEntry[]|null);

            /** TermsResult sum_other_doc_count */
            sum_other_doc_count?: (number|Long|null);

            /** TermsResult doc_count_error_upper_bound */
            doc_count_error_upper_bound?: (number|Long|null);
        }

        /** Represents a TermsResult. */
        class TermsResult implements ITermsResult {

            /**
             * Constructs a new TermsResult.
             * @param [properties] Properties to set
             */
            constructor(properties?: summa.proto.ITermsResult);

            /** TermsResult buckets. */
            public buckets: summa.proto.IBucketEntry[];

            /** TermsResult sum_other_doc_count. */
            public sum_other_doc_count: (number|Long);

            /** TermsResult doc_count_error_upper_bound. */
            public doc_count_error_upper_bound?: (number|Long|null);

            /** TermsResult _doc_count_error_upper_bound. */
            public _doc_count_error_upper_bound?: "doc_count_error_upper_bound";

            /**
             * Creates a new TermsResult instance using the specified properties.
             * @param [properties] Properties to set
             * @returns TermsResult instance
             */
            public static create(properties?: summa.proto.ITermsResult): summa.proto.TermsResult;
        }

        /** Properties of a MetricResult. */
        interface IMetricResult {

            /** MetricResult single_metric */
            single_metric?: (summa.proto.ISingleMetricResult|null);

            /** MetricResult stats */
            stats?: (summa.proto.IStatsResult|null);
        }

        /** Represents a MetricResult. */
        class MetricResult implements IMetricResult {

            /**
             * Constructs a new MetricResult.
             * @param [properties] Properties to set
             */
            constructor(properties?: summa.proto.IMetricResult);

            /** MetricResult single_metric. */
            public single_metric?: (summa.proto.ISingleMetricResult|null);

            /** MetricResult stats. */
            public stats?: (summa.proto.IStatsResult|null);

            /** MetricResult metric_result. */
            public metric_result?: ("single_metric"|"stats");

            /**
             * Creates a new MetricResult instance using the specified properties.
             * @param [properties] Properties to set
             * @returns MetricResult instance
             */
            public static create(properties?: summa.proto.IMetricResult): summa.proto.MetricResult;
        }

        /** Properties of a SingleMetricResult. */
        interface ISingleMetricResult {

            /** SingleMetricResult value */
            value?: (number|null);
        }

        /** Represents a SingleMetricResult. */
        class SingleMetricResult implements ISingleMetricResult {

            /**
             * Constructs a new SingleMetricResult.
             * @param [properties] Properties to set
             */
            constructor(properties?: summa.proto.ISingleMetricResult);

            /** SingleMetricResult value. */
            public value?: (number|null);

            /** SingleMetricResult _value. */
            public _value?: "value";

            /**
             * Creates a new SingleMetricResult instance using the specified properties.
             * @param [properties] Properties to set
             * @returns SingleMetricResult instance
             */
            public static create(properties?: summa.proto.ISingleMetricResult): summa.proto.SingleMetricResult;
        }

        /** Properties of a StatsResult. */
        interface IStatsResult {

            /** StatsResult count */
            count?: (number|Long|null);

            /** StatsResult sum */
            sum?: (number|null);

            /** StatsResult min */
            min?: (number|null);

            /** StatsResult max */
            max?: (number|null);

            /** StatsResult avg */
            avg?: (number|null);
        }

        /** Represents a StatsResult. */
        class StatsResult implements IStatsResult {

            /**
             * Constructs a new StatsResult.
             * @param [properties] Properties to set
             */
            constructor(properties?: summa.proto.IStatsResult);

            /** StatsResult count. */
            public count: (number|Long);

            /** StatsResult sum. */
            public sum: number;

            /** StatsResult min. */
            public min?: (number|null);

            /** StatsResult max. */
            public max?: (number|null);

            /** StatsResult avg. */
            public avg?: (number|null);

            /** StatsResult _min. */
            public _min?: "min";

            /** StatsResult _max. */
            public _max?: "max";

            /** StatsResult _avg. */
            public _avg?: "avg";

            /**
             * Creates a new StatsResult instance using the specified properties.
             * @param [properties] Properties to set
             * @returns StatsResult instance
             */
            public static create(properties?: summa.proto.IStatsResult): summa.proto.StatsResult;
        }

        /** Order enum. */
        enum Order {
            Asc = 0,
            Desc = 1
        }

        /** Properties of an Empty. */
        interface IEmpty {
        }

        /** Represents an Empty. */
        class Empty implements IEmpty {

            /**
             * Constructs a new Empty.
             * @param [properties] Properties to set
             */
            constructor(properties?: summa.proto.IEmpty);

            /**
             * Creates a new Empty instance using the specified properties.
             * @param [properties] Properties to set
             * @returns Empty instance
             */
            public static create(properties?: summa.proto.IEmpty): summa.proto.Empty;
        }

        /** Properties of a PrimaryKey. */
        interface IPrimaryKey {

            /** PrimaryKey str */
            str?: (string|null);

            /** PrimaryKey i64 */
            i64?: (number|Long|null);
        }

        /** Represents a PrimaryKey. */
        class PrimaryKey implements IPrimaryKey {

            /**
             * Constructs a new PrimaryKey.
             * @param [properties] Properties to set
             */
            constructor(properties?: summa.proto.IPrimaryKey);

            /** PrimaryKey str. */
            public str?: (string|null);

            /** PrimaryKey i64. */
            public i64?: (number|Long|null);

            /** PrimaryKey value. */
            public value?: ("str"|"i64");

            /**
             * Creates a new PrimaryKey instance using the specified properties.
             * @param [properties] Properties to set
             * @returns PrimaryKey instance
             */
            public static create(properties?: summa.proto.IPrimaryKey): summa.proto.PrimaryKey;
        }

        /** Properties of a MergePolicy. */
        interface IMergePolicy {

            /** MergePolicy log */
            log?: (summa.proto.ILogMergePolicy|null);

            /** MergePolicy temporal */
            temporal?: (summa.proto.ITemporalMergePolicy|null);
        }

        /** Represents a MergePolicy. */
        class MergePolicy implements IMergePolicy {

            /**
             * Constructs a new MergePolicy.
             * @param [properties] Properties to set
             */
            constructor(properties?: summa.proto.IMergePolicy);

            /** MergePolicy log. */
            public log?: (summa.proto.ILogMergePolicy|null);

            /** MergePolicy temporal. */
            public temporal?: (summa.proto.ITemporalMergePolicy|null);

            /** MergePolicy merge_policy. */
            public merge_policy?: ("log"|"temporal");

            /**
             * Creates a new MergePolicy instance using the specified properties.
             * @param [properties] Properties to set
             * @returns MergePolicy instance
             */
            public static create(properties?: summa.proto.IMergePolicy): summa.proto.MergePolicy;
        }

        /** Properties of an AttachFileEngineRequest. */
        interface IAttachFileEngineRequest {
        }

        /** Represents an AttachFileEngineRequest. */
        class AttachFileEngineRequest implements IAttachFileEngineRequest {

            /**
             * Constructs a new AttachFileEngineRequest.
             * @param [properties] Properties to set
             */
            constructor(properties?: summa.proto.IAttachFileEngineRequest);

            /**
             * Creates a new AttachFileEngineRequest instance using the specified properties.
             * @param [properties] Properties to set
             * @returns AttachFileEngineRequest instance
             */
            public static create(properties?: summa.proto.IAttachFileEngineRequest): summa.proto.AttachFileEngineRequest;
        }

        /** Properties of an AttachRemoteEngineRequest. */
        interface IAttachRemoteEngineRequest {

            /** AttachRemoteEngineRequest config */
            config?: (summa.proto.IRemoteEngineConfig|null);
        }

        /** Represents an AttachRemoteEngineRequest. */
        class AttachRemoteEngineRequest implements IAttachRemoteEngineRequest {

            /**
             * Constructs a new AttachRemoteEngineRequest.
             * @param [properties] Properties to set
             */
            constructor(properties?: summa.proto.IAttachRemoteEngineRequest);

            /** AttachRemoteEngineRequest config. */
            public config?: (summa.proto.IRemoteEngineConfig|null);

            /**
             * Creates a new AttachRemoteEngineRequest instance using the specified properties.
             * @param [properties] Properties to set
             * @returns AttachRemoteEngineRequest instance
             */
            public static create(properties?: summa.proto.IAttachRemoteEngineRequest): summa.proto.AttachRemoteEngineRequest;
        }

        /** Properties of an AttachIndexRequest. */
        interface IAttachIndexRequest {

            /** AttachIndexRequest index_name */
            index_name?: (string|null);

            /** AttachIndexRequest file */
            file?: (summa.proto.IAttachFileEngineRequest|null);

            /** AttachIndexRequest remote */
            remote?: (summa.proto.IAttachRemoteEngineRequest|null);

            /** AttachIndexRequest merge_policy */
            merge_policy?: (summa.proto.IMergePolicy|null);

            /** AttachIndexRequest query_parser_config */
            query_parser_config?: (summa.proto.IQueryParserConfig|null);
        }

        /** Represents an AttachIndexRequest. */
        class AttachIndexRequest implements IAttachIndexRequest {

            /**
             * Constructs a new AttachIndexRequest.
             * @param [properties] Properties to set
             */
            constructor(properties?: summa.proto.IAttachIndexRequest);

            /** AttachIndexRequest index_name. */
            public index_name: string;

            /** AttachIndexRequest file. */
            public file?: (summa.proto.IAttachFileEngineRequest|null);

            /** AttachIndexRequest remote. */
            public remote?: (summa.proto.IAttachRemoteEngineRequest|null);

            /** AttachIndexRequest merge_policy. */
            public merge_policy?: (summa.proto.IMergePolicy|null);

            /** AttachIndexRequest query_parser_config. */
            public query_parser_config?: (summa.proto.IQueryParserConfig|null);

            /** AttachIndexRequest index_engine. */
            public index_engine?: ("file"|"remote");

            /**
             * Creates a new AttachIndexRequest instance using the specified properties.
             * @param [properties] Properties to set
             * @returns AttachIndexRequest instance
             */
            public static create(properties?: summa.proto.IAttachIndexRequest): summa.proto.AttachIndexRequest;
        }

        /** Properties of an AttachIndexResponse. */
        interface IAttachIndexResponse {

            /** AttachIndexResponse index */
            index?: (summa.proto.IIndexDescription|null);
        }

        /** Represents an AttachIndexResponse. */
        class AttachIndexResponse implements IAttachIndexResponse {

            /**
             * Constructs a new AttachIndexResponse.
             * @param [properties] Properties to set
             */
            constructor(properties?: summa.proto.IAttachIndexResponse);

            /** AttachIndexResponse index. */
            public index?: (summa.proto.IIndexDescription|null);

            /**
             * Creates a new AttachIndexResponse instance using the specified properties.
             * @param [properties] Properties to set
             * @returns AttachIndexResponse instance
             */
            public static create(properties?: summa.proto.IAttachIndexResponse): summa.proto.AttachIndexResponse;
        }

        /** Properties of a CommitIndexRequest. */
        interface ICommitIndexRequest {

            /** CommitIndexRequest index_name */
            index_name?: (string|null);
        }

        /** Represents a CommitIndexRequest. */
        class CommitIndexRequest implements ICommitIndexRequest {

            /**
             * Constructs a new CommitIndexRequest.
             * @param [properties] Properties to set
             */
            constructor(properties?: summa.proto.ICommitIndexRequest);

            /** CommitIndexRequest index_name. */
            public index_name: string;

            /**
             * Creates a new CommitIndexRequest instance using the specified properties.
             * @param [properties] Properties to set
             * @returns CommitIndexRequest instance
             */
            public static create(properties?: summa.proto.ICommitIndexRequest): summa.proto.CommitIndexRequest;
        }

        /** Properties of a CommitIndexResponse. */
        interface ICommitIndexResponse {

            /** CommitIndexResponse elapsed_secs */
            elapsed_secs?: (number|null);
        }

        /** Represents a CommitIndexResponse. */
        class CommitIndexResponse implements ICommitIndexResponse {

            /**
             * Constructs a new CommitIndexResponse.
             * @param [properties] Properties to set
             */
            constructor(properties?: summa.proto.ICommitIndexResponse);

            /** CommitIndexResponse elapsed_secs. */
            public elapsed_secs: number;

            /**
             * Creates a new CommitIndexResponse instance using the specified properties.
             * @param [properties] Properties to set
             * @returns CommitIndexResponse instance
             */
            public static create(properties?: summa.proto.ICommitIndexResponse): summa.proto.CommitIndexResponse;
        }

        /** Properties of a CopyDocumentsRequest. */
        interface ICopyDocumentsRequest {

            /** CopyDocumentsRequest source_index_name */
            source_index_name?: (string|null);

            /** CopyDocumentsRequest target_index_name */
            target_index_name?: (string|null);

            /** CopyDocumentsRequest conflict_strategy */
            conflict_strategy?: (summa.proto.ConflictStrategy|null);
        }

        /** Represents a CopyDocumentsRequest. */
        class CopyDocumentsRequest implements ICopyDocumentsRequest {

            /**
             * Constructs a new CopyDocumentsRequest.
             * @param [properties] Properties to set
             */
            constructor(properties?: summa.proto.ICopyDocumentsRequest);

            /** CopyDocumentsRequest source_index_name. */
            public source_index_name: string;

            /** CopyDocumentsRequest target_index_name. */
            public target_index_name: string;

            /** CopyDocumentsRequest conflict_strategy. */
            public conflict_strategy?: (summa.proto.ConflictStrategy|null);

            /** CopyDocumentsRequest _conflict_strategy. */
            public _conflict_strategy?: "conflict_strategy";

            /**
             * Creates a new CopyDocumentsRequest instance using the specified properties.
             * @param [properties] Properties to set
             * @returns CopyDocumentsRequest instance
             */
            public static create(properties?: summa.proto.ICopyDocumentsRequest): summa.proto.CopyDocumentsRequest;
        }

        /** Properties of a CopyDocumentsResponse. */
        interface ICopyDocumentsResponse {

            /** CopyDocumentsResponse elapsed_secs */
            elapsed_secs?: (number|null);

            /** CopyDocumentsResponse copied_documents */
            copied_documents?: (number|null);
        }

        /** Represents a CopyDocumentsResponse. */
        class CopyDocumentsResponse implements ICopyDocumentsResponse {

            /**
             * Constructs a new CopyDocumentsResponse.
             * @param [properties] Properties to set
             */
            constructor(properties?: summa.proto.ICopyDocumentsResponse);

            /** CopyDocumentsResponse elapsed_secs. */
            public elapsed_secs: number;

            /** CopyDocumentsResponse copied_documents. */
            public copied_documents: number;

            /**
             * Creates a new CopyDocumentsResponse instance using the specified properties.
             * @param [properties] Properties to set
             * @returns CopyDocumentsResponse instance
             */
            public static create(properties?: summa.proto.ICopyDocumentsResponse): summa.proto.CopyDocumentsResponse;
        }

        /** Properties of a CopyIndexRequest. */
        interface ICopyIndexRequest {

            /** CopyIndexRequest source_index_name */
            source_index_name?: (string|null);

            /** CopyIndexRequest target_index_name */
            target_index_name?: (string|null);

            /** CopyIndexRequest file */
            file?: (summa.proto.ICreateFileEngineRequest|null);

            /** CopyIndexRequest memory */
            memory?: (summa.proto.ICreateMemoryEngineRequest|null);

            /** CopyIndexRequest merge_policy */
            merge_policy?: (summa.proto.IMergePolicy|null);
        }

        /** Represents a CopyIndexRequest. */
        class CopyIndexRequest implements ICopyIndexRequest {

            /**
             * Constructs a new CopyIndexRequest.
             * @param [properties] Properties to set
             */
            constructor(properties?: summa.proto.ICopyIndexRequest);

            /** CopyIndexRequest source_index_name. */
            public source_index_name: string;

            /** CopyIndexRequest target_index_name. */
            public target_index_name: string;

            /** CopyIndexRequest file. */
            public file?: (summa.proto.ICreateFileEngineRequest|null);

            /** CopyIndexRequest memory. */
            public memory?: (summa.proto.ICreateMemoryEngineRequest|null);

            /** CopyIndexRequest merge_policy. */
            public merge_policy?: (summa.proto.IMergePolicy|null);

            /** CopyIndexRequest target_index_engine. */
            public target_index_engine?: ("file"|"memory");

            /**
             * Creates a new CopyIndexRequest instance using the specified properties.
             * @param [properties] Properties to set
             * @returns CopyIndexRequest instance
             */
            public static create(properties?: summa.proto.ICopyIndexRequest): summa.proto.CopyIndexRequest;
        }

        /** Properties of a CopyIndexResponse. */
        interface ICopyIndexResponse {

            /** CopyIndexResponse index */
            index?: (summa.proto.IIndexDescription|null);
        }

        /** Represents a CopyIndexResponse. */
        class CopyIndexResponse implements ICopyIndexResponse {

            /**
             * Constructs a new CopyIndexResponse.
             * @param [properties] Properties to set
             */
            constructor(properties?: summa.proto.ICopyIndexResponse);

            /** CopyIndexResponse index. */
            public index?: (summa.proto.IIndexDescription|null);

            /**
             * Creates a new CopyIndexResponse instance using the specified properties.
             * @param [properties] Properties to set
             * @returns CopyIndexResponse instance
             */
            public static create(properties?: summa.proto.ICopyIndexResponse): summa.proto.CopyIndexResponse;
        }

        /** Properties of a SortByField. */
        interface ISortByField {

            /** SortByField field */
            field?: (string|null);

            /** SortByField order */
            order?: (summa.proto.Order|null);
        }

        /** Represents a SortByField. */
        class SortByField implements ISortByField {

            /**
             * Constructs a new SortByField.
             * @param [properties] Properties to set
             */
            constructor(properties?: summa.proto.ISortByField);

            /** SortByField field. */
            public field: string;

            /** SortByField order. */
            public order: summa.proto.Order;

            /**
             * Creates a new SortByField instance using the specified properties.
             * @param [properties] Properties to set
             * @returns SortByField instance
             */
            public static create(properties?: summa.proto.ISortByField): summa.proto.SortByField;
        }

        /** Properties of a CreateFileEngineRequest. */
        interface ICreateFileEngineRequest {
        }

        /** Represents a CreateFileEngineRequest. */
        class CreateFileEngineRequest implements ICreateFileEngineRequest {

            /**
             * Constructs a new CreateFileEngineRequest.
             * @param [properties] Properties to set
             */
            constructor(properties?: summa.proto.ICreateFileEngineRequest);

            /**
             * Creates a new CreateFileEngineRequest instance using the specified properties.
             * @param [properties] Properties to set
             * @returns CreateFileEngineRequest instance
             */
            public static create(properties?: summa.proto.ICreateFileEngineRequest): summa.proto.CreateFileEngineRequest;
        }

        /** Properties of a CreateMemoryEngineRequest. */
        interface ICreateMemoryEngineRequest {
        }

        /** Represents a CreateMemoryEngineRequest. */
        class CreateMemoryEngineRequest implements ICreateMemoryEngineRequest {

            /**
             * Constructs a new CreateMemoryEngineRequest.
             * @param [properties] Properties to set
             */
            constructor(properties?: summa.proto.ICreateMemoryEngineRequest);

            /**
             * Creates a new CreateMemoryEngineRequest instance using the specified properties.
             * @param [properties] Properties to set
             * @returns CreateMemoryEngineRequest instance
             */
            public static create(properties?: summa.proto.ICreateMemoryEngineRequest): summa.proto.CreateMemoryEngineRequest;
        }

        /** ConflictStrategy enum. */
        enum ConflictStrategy {
            DO_NOTHING = 0,
            OVERWRITE_ALWAYS = 1,
            OVERWRITE = 2,
            MERGE = 3
        }

        /** Properties of a MappedField. */
        interface IMappedField {

            /** MappedField source_field */
            source_field?: (string|null);

            /** MappedField target_field */
            target_field?: (string|null);
        }

        /** Represents a MappedField. */
        class MappedField implements IMappedField {

            /**
             * Constructs a new MappedField.
             * @param [properties] Properties to set
             */
            constructor(properties?: summa.proto.IMappedField);

            /** MappedField source_field. */
            public source_field: string;

            /** MappedField target_field. */
            public target_field: string;

            /**
             * Creates a new MappedField instance using the specified properties.
             * @param [properties] Properties to set
             * @returns MappedField instance
             */
            public static create(properties?: summa.proto.IMappedField): summa.proto.MappedField;
        }

        /** Properties of an IndexAttributes. */
        interface IIndexAttributes {

            /** IndexAttributes created_at */
            created_at?: (number|Long|null);

            /** IndexAttributes unique_fields */
            unique_fields?: (string[]|null);

            /** IndexAttributes multi_fields */
            multi_fields?: (string[]|null);

            /** IndexAttributes description */
            description?: (string|null);

            /** IndexAttributes conflict_strategy */
            conflict_strategy?: (summa.proto.ConflictStrategy|null);

            /** IndexAttributes mapped_fields */
            mapped_fields?: (summa.proto.IMappedField[]|null);
        }

        /** Represents an IndexAttributes. */
        class IndexAttributes implements IIndexAttributes {

            /**
             * Constructs a new IndexAttributes.
             * @param [properties] Properties to set
             */
            constructor(properties?: summa.proto.IIndexAttributes);

            /** IndexAttributes created_at. */
            public created_at: (number|Long);

            /** IndexAttributes unique_fields. */
            public unique_fields: string[];

            /** IndexAttributes multi_fields. */
            public multi_fields: string[];

            /** IndexAttributes description. */
            public description?: (string|null);

            /** IndexAttributes conflict_strategy. */
            public conflict_strategy: summa.proto.ConflictStrategy;

            /** IndexAttributes mapped_fields. */
            public mapped_fields: summa.proto.IMappedField[];

            /** IndexAttributes _description. */
            public _description?: "description";

            /**
             * Creates a new IndexAttributes instance using the specified properties.
             * @param [properties] Properties to set
             * @returns IndexAttributes instance
             */
            public static create(properties?: summa.proto.IIndexAttributes): summa.proto.IndexAttributes;
        }

        /** Properties of a CreateIndexRequest. */
        interface ICreateIndexRequest {

            /** CreateIndexRequest index_name */
            index_name?: (string|null);

            /** CreateIndexRequest file */
            file?: (summa.proto.ICreateFileEngineRequest|null);

            /** CreateIndexRequest memory */
            memory?: (summa.proto.ICreateMemoryEngineRequest|null);

            /** CreateIndexRequest schema */
            schema?: (string|null);

            /** CreateIndexRequest compression */
            compression?: (summa.proto.Compression|null);

            /** CreateIndexRequest blocksize */
            blocksize?: (number|null);

            /** CreateIndexRequest sort_by_field */
            sort_by_field?: (summa.proto.ISortByField|null);

            /** CreateIndexRequest index_attributes */
            index_attributes?: (summa.proto.IIndexAttributes|null);

            /** CreateIndexRequest merge_policy */
            merge_policy?: (summa.proto.IMergePolicy|null);

            /** CreateIndexRequest query_parser_config */
            query_parser_config?: (summa.proto.IQueryParserConfig|null);
        }

        /** Represents a CreateIndexRequest. */
        class CreateIndexRequest implements ICreateIndexRequest {

            /**
             * Constructs a new CreateIndexRequest.
             * @param [properties] Properties to set
             */
            constructor(properties?: summa.proto.ICreateIndexRequest);

            /** CreateIndexRequest index_name. */
            public index_name: string;

            /** CreateIndexRequest file. */
            public file?: (summa.proto.ICreateFileEngineRequest|null);

            /** CreateIndexRequest memory. */
            public memory?: (summa.proto.ICreateMemoryEngineRequest|null);

            /** CreateIndexRequest schema. */
            public schema: string;

            /** CreateIndexRequest compression. */
            public compression: summa.proto.Compression;

            /** CreateIndexRequest blocksize. */
            public blocksize?: (number|null);

            /** CreateIndexRequest sort_by_field. */
            public sort_by_field?: (summa.proto.ISortByField|null);

            /** CreateIndexRequest index_attributes. */
            public index_attributes?: (summa.proto.IIndexAttributes|null);

            /** CreateIndexRequest merge_policy. */
            public merge_policy?: (summa.proto.IMergePolicy|null);

            /** CreateIndexRequest query_parser_config. */
            public query_parser_config?: (summa.proto.IQueryParserConfig|null);

            /** CreateIndexRequest index_engine. */
            public index_engine?: ("file"|"memory");

            /** CreateIndexRequest _blocksize. */
            public _blocksize?: "blocksize";

            /** CreateIndexRequest _sort_by_field. */
            public _sort_by_field?: "sort_by_field";

            /**
             * Creates a new CreateIndexRequest instance using the specified properties.
             * @param [properties] Properties to set
             * @returns CreateIndexRequest instance
             */
            public static create(properties?: summa.proto.ICreateIndexRequest): summa.proto.CreateIndexRequest;
        }

        /** Properties of a CreateIndexResponse. */
        interface ICreateIndexResponse {

            /** CreateIndexResponse index */
            index?: (summa.proto.IIndexDescription|null);
        }

        /** Represents a CreateIndexResponse. */
        class CreateIndexResponse implements ICreateIndexResponse {

            /**
             * Constructs a new CreateIndexResponse.
             * @param [properties] Properties to set
             */
            constructor(properties?: summa.proto.ICreateIndexResponse);

            /** CreateIndexResponse index. */
            public index?: (summa.proto.IIndexDescription|null);

            /**
             * Creates a new CreateIndexResponse instance using the specified properties.
             * @param [properties] Properties to set
             * @returns CreateIndexResponse instance
             */
            public static create(properties?: summa.proto.ICreateIndexResponse): summa.proto.CreateIndexResponse;
        }

        /** Properties of a DeleteDocumentsRequest. */
        interface IDeleteDocumentsRequest {

            /** DeleteDocumentsRequest index_name */
            index_name?: (string|null);

            /** DeleteDocumentsRequest query */
            query?: (summa.proto.IQuery|null);
        }

        /** Represents a DeleteDocumentsRequest. */
        class DeleteDocumentsRequest implements IDeleteDocumentsRequest {

            /**
             * Constructs a new DeleteDocumentsRequest.
             * @param [properties] Properties to set
             */
            constructor(properties?: summa.proto.IDeleteDocumentsRequest);

            /** DeleteDocumentsRequest index_name. */
            public index_name: string;

            /** DeleteDocumentsRequest query. */
            public query?: (summa.proto.IQuery|null);

            /**
             * Creates a new DeleteDocumentsRequest instance using the specified properties.
             * @param [properties] Properties to set
             * @returns DeleteDocumentsRequest instance
             */
            public static create(properties?: summa.proto.IDeleteDocumentsRequest): summa.proto.DeleteDocumentsRequest;
        }

        /** Properties of a DeleteDocumentsResponse. */
        interface IDeleteDocumentsResponse {

            /** DeleteDocumentsResponse deleted_documents */
            deleted_documents?: (number|Long|null);
        }

        /** Represents a DeleteDocumentsResponse. */
        class DeleteDocumentsResponse implements IDeleteDocumentsResponse {

            /**
             * Constructs a new DeleteDocumentsResponse.
             * @param [properties] Properties to set
             */
            constructor(properties?: summa.proto.IDeleteDocumentsResponse);

            /** DeleteDocumentsResponse deleted_documents. */
            public deleted_documents: (number|Long);

            /**
             * Creates a new DeleteDocumentsResponse instance using the specified properties.
             * @param [properties] Properties to set
             * @returns DeleteDocumentsResponse instance
             */
            public static create(properties?: summa.proto.IDeleteDocumentsResponse): summa.proto.DeleteDocumentsResponse;
        }

        /** Properties of a DeleteIndexRequest. */
        interface IDeleteIndexRequest {

            /** DeleteIndexRequest index_name */
            index_name?: (string|null);
        }

        /** Represents a DeleteIndexRequest. */
        class DeleteIndexRequest implements IDeleteIndexRequest {

            /**
             * Constructs a new DeleteIndexRequest.
             * @param [properties] Properties to set
             */
            constructor(properties?: summa.proto.IDeleteIndexRequest);

            /** DeleteIndexRequest index_name. */
            public index_name: string;

            /**
             * Creates a new DeleteIndexRequest instance using the specified properties.
             * @param [properties] Properties to set
             * @returns DeleteIndexRequest instance
             */
            public static create(properties?: summa.proto.IDeleteIndexRequest): summa.proto.DeleteIndexRequest;
        }

        /** Properties of a DeleteIndexResponse. */
        interface IDeleteIndexResponse {

            /** DeleteIndexResponse deleted_index_name */
            deleted_index_name?: (string|null);
        }

        /** Represents a DeleteIndexResponse. */
        class DeleteIndexResponse implements IDeleteIndexResponse {

            /**
             * Constructs a new DeleteIndexResponse.
             * @param [properties] Properties to set
             */
            constructor(properties?: summa.proto.IDeleteIndexResponse);

            /** DeleteIndexResponse deleted_index_name. */
            public deleted_index_name: string;

            /**
             * Creates a new DeleteIndexResponse instance using the specified properties.
             * @param [properties] Properties to set
             * @returns DeleteIndexResponse instance
             */
            public static create(properties?: summa.proto.IDeleteIndexResponse): summa.proto.DeleteIndexResponse;
        }

        /** Properties of a GetIndicesAliasesRequest. */
        interface IGetIndicesAliasesRequest {
        }

        /** Represents a GetIndicesAliasesRequest. */
        class GetIndicesAliasesRequest implements IGetIndicesAliasesRequest {

            /**
             * Constructs a new GetIndicesAliasesRequest.
             * @param [properties] Properties to set
             */
            constructor(properties?: summa.proto.IGetIndicesAliasesRequest);

            /**
             * Creates a new GetIndicesAliasesRequest instance using the specified properties.
             * @param [properties] Properties to set
             * @returns GetIndicesAliasesRequest instance
             */
            public static create(properties?: summa.proto.IGetIndicesAliasesRequest): summa.proto.GetIndicesAliasesRequest;
        }

        /** Properties of a GetIndicesAliasesResponse. */
        interface IGetIndicesAliasesResponse {

            /** GetIndicesAliasesResponse indices_aliases */
            indices_aliases?: ({ [k: string]: string }|null);
        }

        /** Represents a GetIndicesAliasesResponse. */
        class GetIndicesAliasesResponse implements IGetIndicesAliasesResponse {

            /**
             * Constructs a new GetIndicesAliasesResponse.
             * @param [properties] Properties to set
             */
            constructor(properties?: summa.proto.IGetIndicesAliasesResponse);

            /** GetIndicesAliasesResponse indices_aliases. */
            public indices_aliases: { [k: string]: string };

            /**
             * Creates a new GetIndicesAliasesResponse instance using the specified properties.
             * @param [properties] Properties to set
             * @returns GetIndicesAliasesResponse instance
             */
            public static create(properties?: summa.proto.IGetIndicesAliasesResponse): summa.proto.GetIndicesAliasesResponse;
        }

        /** Properties of a GetIndexRequest. */
        interface IGetIndexRequest {

            /** GetIndexRequest index_name */
            index_name?: (string|null);
        }

        /** Represents a GetIndexRequest. */
        class GetIndexRequest implements IGetIndexRequest {

            /**
             * Constructs a new GetIndexRequest.
             * @param [properties] Properties to set
             */
            constructor(properties?: summa.proto.IGetIndexRequest);

            /** GetIndexRequest index_name. */
            public index_name: string;

            /**
             * Creates a new GetIndexRequest instance using the specified properties.
             * @param [properties] Properties to set
             * @returns GetIndexRequest instance
             */
            public static create(properties?: summa.proto.IGetIndexRequest): summa.proto.GetIndexRequest;
        }

        /** Properties of a GetIndexResponse. */
        interface IGetIndexResponse {

            /** GetIndexResponse index */
            index?: (summa.proto.IIndexDescription|null);
        }

        /** Represents a GetIndexResponse. */
        class GetIndexResponse implements IGetIndexResponse {

            /**
             * Constructs a new GetIndexResponse.
             * @param [properties] Properties to set
             */
            constructor(properties?: summa.proto.IGetIndexResponse);

            /** GetIndexResponse index. */
            public index?: (summa.proto.IIndexDescription|null);

            /**
             * Creates a new GetIndexResponse instance using the specified properties.
             * @param [properties] Properties to set
             * @returns GetIndexResponse instance
             */
            public static create(properties?: summa.proto.IGetIndexResponse): summa.proto.GetIndexResponse;
        }

        /** Properties of a GetIndicesRequest. */
        interface IGetIndicesRequest {
        }

        /** Represents a GetIndicesRequest. */
        class GetIndicesRequest implements IGetIndicesRequest {

            /**
             * Constructs a new GetIndicesRequest.
             * @param [properties] Properties to set
             */
            constructor(properties?: summa.proto.IGetIndicesRequest);

            /**
             * Creates a new GetIndicesRequest instance using the specified properties.
             * @param [properties] Properties to set
             * @returns GetIndicesRequest instance
             */
            public static create(properties?: summa.proto.IGetIndicesRequest): summa.proto.GetIndicesRequest;
        }

        /** Properties of a GetIndicesResponse. */
        interface IGetIndicesResponse {

            /** GetIndicesResponse index_names */
            index_names?: (string[]|null);
        }

        /** Represents a GetIndicesResponse. */
        class GetIndicesResponse implements IGetIndicesResponse {

            /**
             * Constructs a new GetIndicesResponse.
             * @param [properties] Properties to set
             */
            constructor(properties?: summa.proto.IGetIndicesResponse);

            /** GetIndicesResponse index_names. */
            public index_names: string[];

            /**
             * Creates a new GetIndicesResponse instance using the specified properties.
             * @param [properties] Properties to set
             * @returns GetIndicesResponse instance
             */
            public static create(properties?: summa.proto.IGetIndicesResponse): summa.proto.GetIndicesResponse;
        }

        /** Properties of an IndexDocumentStreamRequest. */
        interface IIndexDocumentStreamRequest {

            /** IndexDocumentStreamRequest index_name */
            index_name?: (string|null);

            /** IndexDocumentStreamRequest documents */
            documents?: (Uint8Array[]|null);

            /** IndexDocumentStreamRequest conflict_strategy */
            conflict_strategy?: (summa.proto.ConflictStrategy|null);
        }

        /** Represents an IndexDocumentStreamRequest. */
        class IndexDocumentStreamRequest implements IIndexDocumentStreamRequest {

            /**
             * Constructs a new IndexDocumentStreamRequest.
             * @param [properties] Properties to set
             */
            constructor(properties?: summa.proto.IIndexDocumentStreamRequest);

            /** IndexDocumentStreamRequest index_name. */
            public index_name: string;

            /** IndexDocumentStreamRequest documents. */
            public documents: Uint8Array[];

            /** IndexDocumentStreamRequest conflict_strategy. */
            public conflict_strategy?: (summa.proto.ConflictStrategy|null);

            /** IndexDocumentStreamRequest _conflict_strategy. */
            public _conflict_strategy?: "conflict_strategy";

            /**
             * Creates a new IndexDocumentStreamRequest instance using the specified properties.
             * @param [properties] Properties to set
             * @returns IndexDocumentStreamRequest instance
             */
            public static create(properties?: summa.proto.IIndexDocumentStreamRequest): summa.proto.IndexDocumentStreamRequest;
        }

        /** Properties of an IndexDocumentStreamResponse. */
        interface IIndexDocumentStreamResponse {

            /** IndexDocumentStreamResponse elapsed_secs */
            elapsed_secs?: (number|null);

            /** IndexDocumentStreamResponse success_docs */
            success_docs?: (number|Long|null);

            /** IndexDocumentStreamResponse failed_docs */
            failed_docs?: (number|Long|null);
        }

        /** Represents an IndexDocumentStreamResponse. */
        class IndexDocumentStreamResponse implements IIndexDocumentStreamResponse {

            /**
             * Constructs a new IndexDocumentStreamResponse.
             * @param [properties] Properties to set
             */
            constructor(properties?: summa.proto.IIndexDocumentStreamResponse);

            /** IndexDocumentStreamResponse elapsed_secs. */
            public elapsed_secs: number;

            /** IndexDocumentStreamResponse success_docs. */
            public success_docs: (number|Long);

            /** IndexDocumentStreamResponse failed_docs. */
            public failed_docs: (number|Long);

            /**
             * Creates a new IndexDocumentStreamResponse instance using the specified properties.
             * @param [properties] Properties to set
             * @returns IndexDocumentStreamResponse instance
             */
            public static create(properties?: summa.proto.IIndexDocumentStreamResponse): summa.proto.IndexDocumentStreamResponse;
        }

        /** Properties of an IndexDocumentRequest. */
        interface IIndexDocumentRequest {

            /** IndexDocumentRequest index_name */
            index_name?: (string|null);

            /** IndexDocumentRequest document */
            document?: (Uint8Array|null);
        }

        /** Represents an IndexDocumentRequest. */
        class IndexDocumentRequest implements IIndexDocumentRequest {

            /**
             * Constructs a new IndexDocumentRequest.
             * @param [properties] Properties to set
             */
            constructor(properties?: summa.proto.IIndexDocumentRequest);

            /** IndexDocumentRequest index_name. */
            public index_name: string;

            /** IndexDocumentRequest document. */
            public document: Uint8Array;

            /**
             * Creates a new IndexDocumentRequest instance using the specified properties.
             * @param [properties] Properties to set
             * @returns IndexDocumentRequest instance
             */
            public static create(properties?: summa.proto.IIndexDocumentRequest): summa.proto.IndexDocumentRequest;
        }

        /** Properties of an IndexDocumentResponse. */
        interface IIndexDocumentResponse {
        }

        /** Represents an IndexDocumentResponse. */
        class IndexDocumentResponse implements IIndexDocumentResponse {

            /**
             * Constructs a new IndexDocumentResponse.
             * @param [properties] Properties to set
             */
            constructor(properties?: summa.proto.IIndexDocumentResponse);

            /**
             * Creates a new IndexDocumentResponse instance using the specified properties.
             * @param [properties] Properties to set
             * @returns IndexDocumentResponse instance
             */
            public static create(properties?: summa.proto.IIndexDocumentResponse): summa.proto.IndexDocumentResponse;
        }

        /** Properties of a MergeSegmentsRequest. */
        interface IMergeSegmentsRequest {

            /** MergeSegmentsRequest index_name */
            index_name?: (string|null);

            /** MergeSegmentsRequest segment_ids */
            segment_ids?: (string[]|null);
        }

        /** Represents a MergeSegmentsRequest. */
        class MergeSegmentsRequest implements IMergeSegmentsRequest {

            /**
             * Constructs a new MergeSegmentsRequest.
             * @param [properties] Properties to set
             */
            constructor(properties?: summa.proto.IMergeSegmentsRequest);

            /** MergeSegmentsRequest index_name. */
            public index_name: string;

            /** MergeSegmentsRequest segment_ids. */
            public segment_ids: string[];

            /**
             * Creates a new MergeSegmentsRequest instance using the specified properties.
             * @param [properties] Properties to set
             * @returns MergeSegmentsRequest instance
             */
            public static create(properties?: summa.proto.IMergeSegmentsRequest): summa.proto.MergeSegmentsRequest;
        }

        /** Properties of a MergeSegmentsResponse. */
        interface IMergeSegmentsResponse {

            /** MergeSegmentsResponse segment_id */
            segment_id?: (string|null);
        }

        /** Represents a MergeSegmentsResponse. */
        class MergeSegmentsResponse implements IMergeSegmentsResponse {

            /**
             * Constructs a new MergeSegmentsResponse.
             * @param [properties] Properties to set
             */
            constructor(properties?: summa.proto.IMergeSegmentsResponse);

            /** MergeSegmentsResponse segment_id. */
            public segment_id?: (string|null);

            /** MergeSegmentsResponse _segment_id. */
            public _segment_id?: "segment_id";

            /**
             * Creates a new MergeSegmentsResponse instance using the specified properties.
             * @param [properties] Properties to set
             * @returns MergeSegmentsResponse instance
             */
            public static create(properties?: summa.proto.IMergeSegmentsResponse): summa.proto.MergeSegmentsResponse;
        }

        /** Properties of a SetIndexAliasRequest. */
        interface ISetIndexAliasRequest {

            /** SetIndexAliasRequest index_alias */
            index_alias?: (string|null);

            /** SetIndexAliasRequest index_name */
            index_name?: (string|null);
        }

        /** Represents a SetIndexAliasRequest. */
        class SetIndexAliasRequest implements ISetIndexAliasRequest {

            /**
             * Constructs a new SetIndexAliasRequest.
             * @param [properties] Properties to set
             */
            constructor(properties?: summa.proto.ISetIndexAliasRequest);

            /** SetIndexAliasRequest index_alias. */
            public index_alias: string;

            /** SetIndexAliasRequest index_name. */
            public index_name: string;

            /**
             * Creates a new SetIndexAliasRequest instance using the specified properties.
             * @param [properties] Properties to set
             * @returns SetIndexAliasRequest instance
             */
            public static create(properties?: summa.proto.ISetIndexAliasRequest): summa.proto.SetIndexAliasRequest;
        }

        /** Properties of a SetIndexAliasResponse. */
        interface ISetIndexAliasResponse {

            /** SetIndexAliasResponse old_index_name */
            old_index_name?: (string|null);
        }

        /** Represents a SetIndexAliasResponse. */
        class SetIndexAliasResponse implements ISetIndexAliasResponse {

            /**
             * Constructs a new SetIndexAliasResponse.
             * @param [properties] Properties to set
             */
            constructor(properties?: summa.proto.ISetIndexAliasResponse);

            /** SetIndexAliasResponse old_index_name. */
            public old_index_name?: (string|null);

            /** SetIndexAliasResponse _old_index_name. */
            public _old_index_name?: "old_index_name";

            /**
             * Creates a new SetIndexAliasResponse instance using the specified properties.
             * @param [properties] Properties to set
             * @returns SetIndexAliasResponse instance
             */
            public static create(properties?: summa.proto.ISetIndexAliasResponse): summa.proto.SetIndexAliasResponse;
        }

        /** Properties of a DocumentsRequest. */
        interface IDocumentsRequest {

            /** DocumentsRequest index_name */
            index_name?: (string|null);

            /** DocumentsRequest fields */
            fields?: (string[]|null);
        }

        /** Represents a DocumentsRequest. */
        class DocumentsRequest implements IDocumentsRequest {

            /**
             * Constructs a new DocumentsRequest.
             * @param [properties] Properties to set
             */
            constructor(properties?: summa.proto.IDocumentsRequest);

            /** DocumentsRequest index_name. */
            public index_name: string;

            /** DocumentsRequest fields. */
            public fields: string[];

            /**
             * Creates a new DocumentsRequest instance using the specified properties.
             * @param [properties] Properties to set
             * @returns DocumentsRequest instance
             */
            public static create(properties?: summa.proto.IDocumentsRequest): summa.proto.DocumentsRequest;
        }

        /** Properties of a DocumentsResponse. */
        interface IDocumentsResponse {

            /** DocumentsResponse document */
            document?: (string|null);
        }

        /** Represents a DocumentsResponse. */
        class DocumentsResponse implements IDocumentsResponse {

            /**
             * Constructs a new DocumentsResponse.
             * @param [properties] Properties to set
             */
            constructor(properties?: summa.proto.IDocumentsResponse);

            /** DocumentsResponse document. */
            public document: string;

            /**
             * Creates a new DocumentsResponse instance using the specified properties.
             * @param [properties] Properties to set
             * @returns DocumentsResponse instance
             */
            public static create(properties?: summa.proto.IDocumentsResponse): summa.proto.DocumentsResponse;
        }

        /** Properties of a VacuumIndexRequest. */
        interface IVacuumIndexRequest {

            /** VacuumIndexRequest index_name */
            index_name?: (string|null);

            /** VacuumIndexRequest excluded_segments */
            excluded_segments?: (string[]|null);
        }

        /** Represents a VacuumIndexRequest. */
        class VacuumIndexRequest implements IVacuumIndexRequest {

            /**
             * Constructs a new VacuumIndexRequest.
             * @param [properties] Properties to set
             */
            constructor(properties?: summa.proto.IVacuumIndexRequest);

            /** VacuumIndexRequest index_name. */
            public index_name: string;

            /** VacuumIndexRequest excluded_segments. */
            public excluded_segments: string[];

            /**
             * Creates a new VacuumIndexRequest instance using the specified properties.
             * @param [properties] Properties to set
             * @returns VacuumIndexRequest instance
             */
            public static create(properties?: summa.proto.IVacuumIndexRequest): summa.proto.VacuumIndexRequest;
        }

        /** Properties of a VacuumIndexResponse. */
        interface IVacuumIndexResponse {

            /** VacuumIndexResponse freed_space_bytes */
            freed_space_bytes?: (number|Long|null);
        }

        /** Represents a VacuumIndexResponse. */
        class VacuumIndexResponse implements IVacuumIndexResponse {

            /**
             * Constructs a new VacuumIndexResponse.
             * @param [properties] Properties to set
             */
            constructor(properties?: summa.proto.IVacuumIndexResponse);

            /** VacuumIndexResponse freed_space_bytes. */
            public freed_space_bytes: (number|Long);

            /**
             * Creates a new VacuumIndexResponse instance using the specified properties.
             * @param [properties] Properties to set
             * @returns VacuumIndexResponse instance
             */
            public static create(properties?: summa.proto.IVacuumIndexResponse): summa.proto.VacuumIndexResponse;
        }

        /** Properties of a WarmupIndexRequest. */
        interface IWarmupIndexRequest {

            /** WarmupIndexRequest index_name */
            index_name?: (string|null);

            /** WarmupIndexRequest is_full */
            is_full?: (boolean|null);
        }

        /** Represents a WarmupIndexRequest. */
        class WarmupIndexRequest implements IWarmupIndexRequest {

            /**
             * Constructs a new WarmupIndexRequest.
             * @param [properties] Properties to set
             */
            constructor(properties?: summa.proto.IWarmupIndexRequest);

            /** WarmupIndexRequest index_name. */
            public index_name: string;

            /** WarmupIndexRequest is_full. */
            public is_full: boolean;

            /**
             * Creates a new WarmupIndexRequest instance using the specified properties.
             * @param [properties] Properties to set
             * @returns WarmupIndexRequest instance
             */
            public static create(properties?: summa.proto.IWarmupIndexRequest): summa.proto.WarmupIndexRequest;
        }

        /** Properties of a WarmupIndexResponse. */
        interface IWarmupIndexResponse {

            /** WarmupIndexResponse elapsed_secs */
            elapsed_secs?: (number|null);
        }

        /** Represents a WarmupIndexResponse. */
        class WarmupIndexResponse implements IWarmupIndexResponse {

            /**
             * Constructs a new WarmupIndexResponse.
             * @param [properties] Properties to set
             */
            constructor(properties?: summa.proto.IWarmupIndexResponse);

            /** WarmupIndexResponse elapsed_secs. */
            public elapsed_secs: number;

            /**
             * Creates a new WarmupIndexResponse instance using the specified properties.
             * @param [properties] Properties to set
             * @returns WarmupIndexResponse instance
             */
            public static create(properties?: summa.proto.IWarmupIndexResponse): summa.proto.WarmupIndexResponse;
        }

        /** Compression enum. */
        enum Compression {
            None = 0,
            Brotli = 1,
            Lz4 = 2,
            Snappy = 3,
            Zstd = 4,
            Zstd7 = 5,
            Zstd9 = 6,
            Zstd14 = 7,
            Zstd19 = 8,
            Zstd22 = 9
        }

        /** Properties of a FileEngineConfig. */
        interface IFileEngineConfig {

            /** FileEngineConfig path */
            path?: (string|null);
        }

        /** Represents a FileEngineConfig. */
        class FileEngineConfig implements IFileEngineConfig {

            /**
             * Constructs a new FileEngineConfig.
             * @param [properties] Properties to set
             */
            constructor(properties?: summa.proto.IFileEngineConfig);

            /** FileEngineConfig path. */
            public path: string;

            /**
             * Creates a new FileEngineConfig instance using the specified properties.
             * @param [properties] Properties to set
             * @returns FileEngineConfig instance
             */
            public static create(properties?: summa.proto.IFileEngineConfig): summa.proto.FileEngineConfig;
        }

        /** Properties of a MemoryEngineConfig. */
        interface IMemoryEngineConfig {

            /** MemoryEngineConfig schema */
            schema?: (string|null);
        }

        /** Represents a MemoryEngineConfig. */
        class MemoryEngineConfig implements IMemoryEngineConfig {

            /**
             * Constructs a new MemoryEngineConfig.
             * @param [properties] Properties to set
             */
            constructor(properties?: summa.proto.IMemoryEngineConfig);

            /** MemoryEngineConfig schema. */
            public schema: string;

            /**
             * Creates a new MemoryEngineConfig instance using the specified properties.
             * @param [properties] Properties to set
             * @returns MemoryEngineConfig instance
             */
            public static create(properties?: summa.proto.IMemoryEngineConfig): summa.proto.MemoryEngineConfig;
        }

        /** Properties of a CacheConfig. */
        interface ICacheConfig {

            /** CacheConfig cache_size */
            cache_size?: (number|Long|null);
        }

        /** Represents a CacheConfig. */
        class CacheConfig implements ICacheConfig {

            /**
             * Constructs a new CacheConfig.
             * @param [properties] Properties to set
             */
            constructor(properties?: summa.proto.ICacheConfig);

            /** CacheConfig cache_size. */
            public cache_size: (number|Long);

            /**
             * Creates a new CacheConfig instance using the specified properties.
             * @param [properties] Properties to set
             * @returns CacheConfig instance
             */
            public static create(properties?: summa.proto.ICacheConfig): summa.proto.CacheConfig;
        }

        /** Properties of a RemoteEngineConfig. */
        interface IRemoteEngineConfig {

            /** RemoteEngineConfig method */
            method?: (string|null);

            /** RemoteEngineConfig url_template */
            url_template?: (string|null);

            /** RemoteEngineConfig headers_template */
            headers_template?: ({ [k: string]: string }|null);

            /** RemoteEngineConfig cache_config */
            cache_config?: (summa.proto.ICacheConfig|null);

            /** RemoteEngineConfig timeout_ms */
            timeout_ms?: (number|null);
        }

        /** Represents a RemoteEngineConfig. */
        class RemoteEngineConfig implements IRemoteEngineConfig {

            /**
             * Constructs a new RemoteEngineConfig.
             * @param [properties] Properties to set
             */
            constructor(properties?: summa.proto.IRemoteEngineConfig);

            /** RemoteEngineConfig method. */
            public method: string;

            /** RemoteEngineConfig url_template. */
            public url_template: string;

            /** RemoteEngineConfig headers_template. */
            public headers_template: { [k: string]: string };

            /** RemoteEngineConfig cache_config. */
            public cache_config?: (summa.proto.ICacheConfig|null);

            /** RemoteEngineConfig timeout_ms. */
            public timeout_ms?: (number|null);

            /** RemoteEngineConfig _timeout_ms. */
            public _timeout_ms?: "timeout_ms";

            /**
             * Creates a new RemoteEngineConfig instance using the specified properties.
             * @param [properties] Properties to set
             * @returns RemoteEngineConfig instance
             */
            public static create(properties?: summa.proto.IRemoteEngineConfig): summa.proto.RemoteEngineConfig;
        }

        /** Properties of a LogMergePolicy. */
        interface ILogMergePolicy {

            /** LogMergePolicy is_frozen */
            is_frozen?: (boolean|null);
        }

        /** Represents a LogMergePolicy. */
        class LogMergePolicy implements ILogMergePolicy {

            /**
             * Constructs a new LogMergePolicy.
             * @param [properties] Properties to set
             */
            constructor(properties?: summa.proto.ILogMergePolicy);

            /** LogMergePolicy is_frozen. */
            public is_frozen: boolean;

            /**
             * Creates a new LogMergePolicy instance using the specified properties.
             * @param [properties] Properties to set
             * @returns LogMergePolicy instance
             */
            public static create(properties?: summa.proto.ILogMergePolicy): summa.proto.LogMergePolicy;
        }

        /** Properties of a TemporalMergePolicy. */
        interface ITemporalMergePolicy {

            /** TemporalMergePolicy merge_older_then_secs */
            merge_older_then_secs?: (number|Long|null);
        }

        /** Represents a TemporalMergePolicy. */
        class TemporalMergePolicy implements ITemporalMergePolicy {

            /**
             * Constructs a new TemporalMergePolicy.
             * @param [properties] Properties to set
             */
            constructor(properties?: summa.proto.ITemporalMergePolicy);

            /** TemporalMergePolicy merge_older_then_secs. */
            public merge_older_then_secs: (number|Long);

            /**
             * Creates a new TemporalMergePolicy instance using the specified properties.
             * @param [properties] Properties to set
             * @returns TemporalMergePolicy instance
             */
            public static create(properties?: summa.proto.ITemporalMergePolicy): summa.proto.TemporalMergePolicy;
        }

        /** Properties of an IndexEngineConfig. */
        interface IIndexEngineConfig {

            /** IndexEngineConfig file */
            file?: (summa.proto.IFileEngineConfig|null);

            /** IndexEngineConfig memory */
            memory?: (summa.proto.IMemoryEngineConfig|null);

            /** IndexEngineConfig remote */
            remote?: (summa.proto.IRemoteEngineConfig|null);

            /** IndexEngineConfig merge_policy */
            merge_policy?: (summa.proto.IMergePolicy|null);

            /** IndexEngineConfig query_parser_config */
            query_parser_config?: (summa.proto.IQueryParserConfig|null);
        }

        /** Represents an IndexEngineConfig. */
        class IndexEngineConfig implements IIndexEngineConfig {

            /**
             * Constructs a new IndexEngineConfig.
             * @param [properties] Properties to set
             */
            constructor(properties?: summa.proto.IIndexEngineConfig);

            /** IndexEngineConfig file. */
            public file?: (summa.proto.IFileEngineConfig|null);

            /** IndexEngineConfig memory. */
            public memory?: (summa.proto.IMemoryEngineConfig|null);

            /** IndexEngineConfig remote. */
            public remote?: (summa.proto.IRemoteEngineConfig|null);

            /** IndexEngineConfig merge_policy. */
            public merge_policy?: (summa.proto.IMergePolicy|null);

            /** IndexEngineConfig query_parser_config. */
            public query_parser_config?: (summa.proto.IQueryParserConfig|null);

            /** IndexEngineConfig config. */
            public config?: ("file"|"memory"|"remote");

            /**
             * Creates a new IndexEngineConfig instance using the specified properties.
             * @param [properties] Properties to set
             * @returns IndexEngineConfig instance
             */
            public static create(properties?: summa.proto.IIndexEngineConfig): summa.proto.IndexEngineConfig;
        }

        /** Properties of an IndexDescription. */
        interface IIndexDescription {

            /** IndexDescription index_name */
            index_name?: (string|null);

            /** IndexDescription index_aliases */
            index_aliases?: (string[]|null);

            /** IndexDescription index_engine */
            index_engine?: (summa.proto.IIndexEngineConfig|null);

            /** IndexDescription num_docs */
            num_docs?: (number|Long|null);

            /** IndexDescription compression */
            compression?: (summa.proto.Compression|null);

            /** IndexDescription index_attributes */
            index_attributes?: (summa.proto.IIndexAttributes|null);
        }

        /** Represents an IndexDescription. */
        class IndexDescription implements IIndexDescription {

            /**
             * Constructs a new IndexDescription.
             * @param [properties] Properties to set
             */
            constructor(properties?: summa.proto.IIndexDescription);

            /** IndexDescription index_name. */
            public index_name: string;

            /** IndexDescription index_aliases. */
            public index_aliases: string[];

            /** IndexDescription index_engine. */
            public index_engine?: (summa.proto.IIndexEngineConfig|null);

            /** IndexDescription num_docs. */
            public num_docs: (number|Long);

            /** IndexDescription compression. */
            public compression: summa.proto.Compression;

            /** IndexDescription index_attributes. */
            public index_attributes?: (summa.proto.IIndexAttributes|null);

            /**
             * Creates a new IndexDescription instance using the specified properties.
             * @param [properties] Properties to set
             * @returns IndexDescription instance
             */
            public static create(properties?: summa.proto.IIndexDescription): summa.proto.IndexDescription;
        }

        /** Properties of an IndexDocumentOperation. */
        interface IIndexDocumentOperation {

            /** IndexDocumentOperation document */
            document?: (Uint8Array|null);
        }

        /** Represents an IndexDocumentOperation. */
        class IndexDocumentOperation implements IIndexDocumentOperation {

            /**
             * Constructs a new IndexDocumentOperation.
             * @param [properties] Properties to set
             */
            constructor(properties?: summa.proto.IIndexDocumentOperation);

            /** IndexDocumentOperation document. */
            public document: Uint8Array;

            /**
             * Creates a new IndexDocumentOperation instance using the specified properties.
             * @param [properties] Properties to set
             * @returns IndexDocumentOperation instance
             */
            public static create(properties?: summa.proto.IIndexDocumentOperation): summa.proto.IndexDocumentOperation;
        }

        /** Properties of an IndexOperation. */
        interface IIndexOperation {

            /** IndexOperation index_document */
            index_document?: (summa.proto.IIndexDocumentOperation|null);
        }

        /** Represents an IndexOperation. */
        class IndexOperation implements IIndexOperation {

            /**
             * Constructs a new IndexOperation.
             * @param [properties] Properties to set
             */
            constructor(properties?: summa.proto.IIndexOperation);

            /** IndexOperation index_document. */
            public index_document?: (summa.proto.IIndexDocumentOperation|null);

            /** IndexOperation operation. */
            public operation?: "index_document";

            /**
             * Creates a new IndexOperation instance using the specified properties.
             * @param [properties] Properties to set
             * @returns IndexOperation instance
             */
            public static create(properties?: summa.proto.IIndexOperation): summa.proto.IndexOperation;
        }
    }
}
