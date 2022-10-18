/*eslint-disable block-scoped-var, id-length, no-control-regex, no-magic-numbers, no-prototype-builtins, no-redeclare, no-shadow, no-var, sort-vars*/
import * as $protobuf from "protobufjs/minimal.js";

// Common aliases
const $util = $protobuf.util;

// Exported root namespace
const $root = $protobuf.roots["default"] || ($protobuf.roots["default"] = {});

export const summa = $root.summa = (() => {

    /**
     * Namespace summa.
     * @exports summa
     * @namespace
     */
    const summa = {};

    summa.proto = (function() {

        /**
         * Namespace proto.
         * @memberof summa
         * @namespace
         */
        const proto = {};

        proto.SearchRequest = (function() {

            /**
             * Properties of a SearchRequest.
             * @memberof summa.proto
             * @interface ISearchRequest
             * @property {string|null} [index_alias] SearchRequest index_alias
             * @property {summa.proto.IQuery|null} [query] SearchRequest query
             * @property {Array.<summa.proto.ICollector>|null} [collectors] SearchRequest collectors
             * @property {boolean|null} [is_fieldnorms_scoring_enabled] SearchRequest is_fieldnorms_scoring_enabled
             */

            /**
             * Constructs a new SearchRequest.
             * @memberof summa.proto
             * @classdesc Represents a SearchRequest.
             * @implements ISearchRequest
             * @constructor
             * @param {summa.proto.ISearchRequest=} [properties] Properties to set
             */
            function SearchRequest(properties) {
                this.collectors = [];
                if (properties)
                    for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }

            /**
             * SearchRequest index_alias.
             * @member {string} index_alias
             * @memberof summa.proto.SearchRequest
             * @instance
             */
            SearchRequest.prototype.index_alias = "";

            /**
             * SearchRequest query.
             * @member {summa.proto.IQuery|null|undefined} query
             * @memberof summa.proto.SearchRequest
             * @instance
             */
            SearchRequest.prototype.query = null;

            /**
             * SearchRequest collectors.
             * @member {Array.<summa.proto.ICollector>} collectors
             * @memberof summa.proto.SearchRequest
             * @instance
             */
            SearchRequest.prototype.collectors = $util.emptyArray;

            /**
             * SearchRequest is_fieldnorms_scoring_enabled.
             * @member {boolean|null|undefined} is_fieldnorms_scoring_enabled
             * @memberof summa.proto.SearchRequest
             * @instance
             */
            SearchRequest.prototype.is_fieldnorms_scoring_enabled = null;

            // OneOf field names bound to virtual getters and setters
            let $oneOfFields;

            /**
             * SearchRequest _is_fieldnorms_scoring_enabled.
             * @member {"is_fieldnorms_scoring_enabled"|undefined} _is_fieldnorms_scoring_enabled
             * @memberof summa.proto.SearchRequest
             * @instance
             */
            Object.defineProperty(SearchRequest.prototype, "_is_fieldnorms_scoring_enabled", {
                get: $util.oneOfGetter($oneOfFields = ["is_fieldnorms_scoring_enabled"]),
                set: $util.oneOfSetter($oneOfFields)
            });

            /**
             * Creates a new SearchRequest instance using the specified properties.
             * @function create
             * @memberof summa.proto.SearchRequest
             * @static
             * @param {summa.proto.ISearchRequest=} [properties] Properties to set
             * @returns {summa.proto.SearchRequest} SearchRequest instance
             */
            SearchRequest.create = function create(properties) {
                return new SearchRequest(properties);
            };

            return SearchRequest;
        })();

        proto.TermFieldMapperConfig = (function() {

            /**
             * Properties of a TermFieldMapperConfig.
             * @memberof summa.proto
             * @interface ITermFieldMapperConfig
             * @property {Array.<string>|null} [fields] TermFieldMapperConfig fields
             */

            /**
             * Constructs a new TermFieldMapperConfig.
             * @memberof summa.proto
             * @classdesc Represents a TermFieldMapperConfig.
             * @implements ITermFieldMapperConfig
             * @constructor
             * @param {summa.proto.ITermFieldMapperConfig=} [properties] Properties to set
             */
            function TermFieldMapperConfig(properties) {
                this.fields = [];
                if (properties)
                    for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }

            /**
             * TermFieldMapperConfig fields.
             * @member {Array.<string>} fields
             * @memberof summa.proto.TermFieldMapperConfig
             * @instance
             */
            TermFieldMapperConfig.prototype.fields = $util.emptyArray;

            /**
             * Creates a new TermFieldMapperConfig instance using the specified properties.
             * @function create
             * @memberof summa.proto.TermFieldMapperConfig
             * @static
             * @param {summa.proto.ITermFieldMapperConfig=} [properties] Properties to set
             * @returns {summa.proto.TermFieldMapperConfig} TermFieldMapperConfig instance
             */
            TermFieldMapperConfig.create = function create(properties) {
                return new TermFieldMapperConfig(properties);
            };
            return TermFieldMapperConfig;
        })();

        proto.MatchQueryBooleanShouldMode = (function() {

            /**
             * Properties of a MatchQueryBooleanShouldMode.
             * @memberof summa.proto
             * @interface IMatchQueryBooleanShouldMode
             */

            /**
             * Constructs a new MatchQueryBooleanShouldMode.
             * @memberof summa.proto
             * @classdesc Represents a MatchQueryBooleanShouldMode.
             * @implements IMatchQueryBooleanShouldMode
             * @constructor
             * @param {summa.proto.IMatchQueryBooleanShouldMode=} [properties] Properties to set
             */
            function MatchQueryBooleanShouldMode(properties) {
                if (properties)
                    for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }

            /**
             * Creates a new MatchQueryBooleanShouldMode instance using the specified properties.
             * @function create
             * @memberof summa.proto.MatchQueryBooleanShouldMode
             * @static
             * @param {summa.proto.IMatchQueryBooleanShouldMode=} [properties] Properties to set
             * @returns {summa.proto.MatchQueryBooleanShouldMode} MatchQueryBooleanShouldMode instance
             */
            MatchQueryBooleanShouldMode.create = function create(properties) {
                return new MatchQueryBooleanShouldMode(properties);
            };

            return MatchQueryBooleanShouldMode;
        })();

        proto.MatchQueryDisjuctionMaxMode = (function() {

            /**
             * Properties of a MatchQueryDisjuctionMaxMode.
             * @memberof summa.proto
             * @interface IMatchQueryDisjuctionMaxMode
             * @property {number|null} [tie_breaker] MatchQueryDisjuctionMaxMode tie_breaker
             */

            /**
             * Constructs a new MatchQueryDisjuctionMaxMode.
             * @memberof summa.proto
             * @classdesc Represents a MatchQueryDisjuctionMaxMode.
             * @implements IMatchQueryDisjuctionMaxMode
             * @constructor
             * @param {summa.proto.IMatchQueryDisjuctionMaxMode=} [properties] Properties to set
             */
            function MatchQueryDisjuctionMaxMode(properties) {
                if (properties)
                    for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }

            /**
             * MatchQueryDisjuctionMaxMode tie_breaker.
             * @member {number} tie_breaker
             * @memberof summa.proto.MatchQueryDisjuctionMaxMode
             * @instance
             */
            MatchQueryDisjuctionMaxMode.prototype.tie_breaker = 0;

            /**
             * Creates a new MatchQueryDisjuctionMaxMode instance using the specified properties.
             * @function create
             * @memberof summa.proto.MatchQueryDisjuctionMaxMode
             * @static
             * @param {summa.proto.IMatchQueryDisjuctionMaxMode=} [properties] Properties to set
             * @returns {summa.proto.MatchQueryDisjuctionMaxMode} MatchQueryDisjuctionMaxMode instance
             */
            MatchQueryDisjuctionMaxMode.create = function create(properties) {
                return new MatchQueryDisjuctionMaxMode(properties);
            };

            return MatchQueryDisjuctionMaxMode;
        })();

        proto.ExactMatchesPromoter = (function() {

            /**
             * Properties of an ExactMatchesPromoter.
             * @memberof summa.proto
             * @interface IExactMatchesPromoter
             * @property {number|null} [slop] ExactMatchesPromoter slop
             * @property {number|null} [boost] ExactMatchesPromoter boost
             * @property {Array.<string>|null} [fields] ExactMatchesPromoter fields
             */

            /**
             * Constructs a new ExactMatchesPromoter.
             * @memberof summa.proto
             * @classdesc Represents an ExactMatchesPromoter.
             * @implements IExactMatchesPromoter
             * @constructor
             * @param {summa.proto.IExactMatchesPromoter=} [properties] Properties to set
             */
            function ExactMatchesPromoter(properties) {
                this.fields = [];
                if (properties)
                    for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }

            /**
             * ExactMatchesPromoter slop.
             * @member {number} slop
             * @memberof summa.proto.ExactMatchesPromoter
             * @instance
             */
            ExactMatchesPromoter.prototype.slop = 0;

            /**
             * ExactMatchesPromoter boost.
             * @member {number|null|undefined} boost
             * @memberof summa.proto.ExactMatchesPromoter
             * @instance
             */
            ExactMatchesPromoter.prototype.boost = null;

            /**
             * ExactMatchesPromoter fields.
             * @member {Array.<string>} fields
             * @memberof summa.proto.ExactMatchesPromoter
             * @instance
             */
            ExactMatchesPromoter.prototype.fields = $util.emptyArray;

            // OneOf field names bound to virtual getters and setters
            let $oneOfFields;

            /**
             * ExactMatchesPromoter _boost.
             * @member {"boost"|undefined} _boost
             * @memberof summa.proto.ExactMatchesPromoter
             * @instance
             */
            Object.defineProperty(ExactMatchesPromoter.prototype, "_boost", {
                get: $util.oneOfGetter($oneOfFields = ["boost"]),
                set: $util.oneOfSetter($oneOfFields)
            });

            /**
             * Creates a new ExactMatchesPromoter instance using the specified properties.
             * @function create
             * @memberof summa.proto.ExactMatchesPromoter
             * @static
             * @param {summa.proto.IExactMatchesPromoter=} [properties] Properties to set
             * @returns {summa.proto.ExactMatchesPromoter} ExactMatchesPromoter instance
             */
            ExactMatchesPromoter.create = function create(properties) {
                return new ExactMatchesPromoter(properties);
            };

            return ExactMatchesPromoter;
        })();

        proto.NerMatchesPromoter = (function() {

            /**
             * Properties of a NerMatchesPromoter.
             * @memberof summa.proto
             * @interface INerMatchesPromoter
             * @property {number|null} [boost] NerMatchesPromoter boost
             * @property {Array.<string>|null} [fields] NerMatchesPromoter fields
             */

            /**
             * Constructs a new NerMatchesPromoter.
             * @memberof summa.proto
             * @classdesc Represents a NerMatchesPromoter.
             * @implements INerMatchesPromoter
             * @constructor
             * @param {summa.proto.INerMatchesPromoter=} [properties] Properties to set
             */
            function NerMatchesPromoter(properties) {
                this.fields = [];
                if (properties)
                    for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }

            /**
             * NerMatchesPromoter boost.
             * @member {number|null|undefined} boost
             * @memberof summa.proto.NerMatchesPromoter
             * @instance
             */
            NerMatchesPromoter.prototype.boost = null;

            /**
             * NerMatchesPromoter fields.
             * @member {Array.<string>} fields
             * @memberof summa.proto.NerMatchesPromoter
             * @instance
             */
            NerMatchesPromoter.prototype.fields = $util.emptyArray;

            // OneOf field names bound to virtual getters and setters
            let $oneOfFields;

            /**
             * NerMatchesPromoter _boost.
             * @member {"boost"|undefined} _boost
             * @memberof summa.proto.NerMatchesPromoter
             * @instance
             */
            Object.defineProperty(NerMatchesPromoter.prototype, "_boost", {
                get: $util.oneOfGetter($oneOfFields = ["boost"]),
                set: $util.oneOfSetter($oneOfFields)
            });

            /**
             * Creates a new NerMatchesPromoter instance using the specified properties.
             * @function create
             * @memberof summa.proto.NerMatchesPromoter
             * @static
             * @param {summa.proto.INerMatchesPromoter=} [properties] Properties to set
             * @returns {summa.proto.NerMatchesPromoter} NerMatchesPromoter instance
             */
            NerMatchesPromoter.create = function create(properties) {
                return new NerMatchesPromoter(properties);
            };

            return NerMatchesPromoter;
        })();

        proto.MorphologyConfig = (function() {

            /**
             * Properties of a MorphologyConfig.
             * @memberof summa.proto
             * @interface IMorphologyConfig
             * @property {number|null} [derive_tenses_coefficient] MorphologyConfig derive_tenses_coefficient
             */

            /**
             * Constructs a new MorphologyConfig.
             * @memberof summa.proto
             * @classdesc Represents a MorphologyConfig.
             * @implements IMorphologyConfig
             * @constructor
             * @param {summa.proto.IMorphologyConfig=} [properties] Properties to set
             */
            function MorphologyConfig(properties) {
                if (properties)
                    for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }

            /**
             * MorphologyConfig derive_tenses_coefficient.
             * @member {number|null|undefined} derive_tenses_coefficient
             * @memberof summa.proto.MorphologyConfig
             * @instance
             */
            MorphologyConfig.prototype.derive_tenses_coefficient = null;

            // OneOf field names bound to virtual getters and setters
            let $oneOfFields;

            /**
             * MorphologyConfig _derive_tenses_coefficient.
             * @member {"derive_tenses_coefficient"|undefined} _derive_tenses_coefficient
             * @memberof summa.proto.MorphologyConfig
             * @instance
             */
            Object.defineProperty(MorphologyConfig.prototype, "_derive_tenses_coefficient", {
                get: $util.oneOfGetter($oneOfFields = ["derive_tenses_coefficient"]),
                set: $util.oneOfSetter($oneOfFields)
            });

            /**
             * Creates a new MorphologyConfig instance using the specified properties.
             * @function create
             * @memberof summa.proto.MorphologyConfig
             * @static
             * @param {summa.proto.IMorphologyConfig=} [properties] Properties to set
             * @returns {summa.proto.MorphologyConfig} MorphologyConfig instance
             */
            MorphologyConfig.create = function create(properties) {
                return new MorphologyConfig(properties);
            };

            return MorphologyConfig;
        })();

        proto.QueryParserConfig = (function() {

            /**
             * Properties of a QueryParserConfig.
             * @memberof summa.proto
             * @interface IQueryParserConfig
             * @property {Object.<string,string>|null} [field_aliases] QueryParserConfig field_aliases
             * @property {Object.<string,number>|null} [field_boosts] QueryParserConfig field_boosts
             * @property {Object.<string,summa.proto.ITermFieldMapperConfig>|null} [term_field_mapper_configs] QueryParserConfig term_field_mapper_configs
             * @property {number|null} [term_limit] QueryParserConfig term_limit
             * @property {Array.<string>|null} [default_fields] QueryParserConfig default_fields
             * @property {summa.proto.IMatchQueryBooleanShouldMode|null} [boolean_should_mode] QueryParserConfig boolean_should_mode
             * @property {summa.proto.IMatchQueryDisjuctionMaxMode|null} [disjuction_max_mode] QueryParserConfig disjuction_max_mode
             * @property {summa.proto.IExactMatchesPromoter|null} [exact_matches_promoter] QueryParserConfig exact_matches_promoter
             * @property {Array.<string>|null} [removed_fields] QueryParserConfig removed_fields
             * @property {Object.<string,summa.proto.IMorphologyConfig>|null} [morphology_configs] QueryParserConfig morphology_configs
             * @property {string|null} [query_language] QueryParserConfig query_language
             */

            /**
             * Constructs a new QueryParserConfig.
             * @memberof summa.proto
             * @classdesc Represents a QueryParserConfig.
             * @implements IQueryParserConfig
             * @constructor
             * @param {summa.proto.IQueryParserConfig=} [properties] Properties to set
             */
            function QueryParserConfig(properties) {
                this.field_aliases = {};
                this.field_boosts = {};
                this.term_field_mapper_configs = {};
                this.default_fields = [];
                this.removed_fields = [];
                this.morphology_configs = {};
                if (properties)
                    for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }

            /**
             * QueryParserConfig field_aliases.
             * @member {Object.<string,string>} field_aliases
             * @memberof summa.proto.QueryParserConfig
             * @instance
             */
            QueryParserConfig.prototype.field_aliases = $util.emptyObject;

            /**
             * QueryParserConfig field_boosts.
             * @member {Object.<string,number>} field_boosts
             * @memberof summa.proto.QueryParserConfig
             * @instance
             */
            QueryParserConfig.prototype.field_boosts = $util.emptyObject;

            /**
             * QueryParserConfig term_field_mapper_configs.
             * @member {Object.<string,summa.proto.ITermFieldMapperConfig>} term_field_mapper_configs
             * @memberof summa.proto.QueryParserConfig
             * @instance
             */
            QueryParserConfig.prototype.term_field_mapper_configs = $util.emptyObject;

            /**
             * QueryParserConfig term_limit.
             * @member {number} term_limit
             * @memberof summa.proto.QueryParserConfig
             * @instance
             */
            QueryParserConfig.prototype.term_limit = 0;

            /**
             * QueryParserConfig default_fields.
             * @member {Array.<string>} default_fields
             * @memberof summa.proto.QueryParserConfig
             * @instance
             */
            QueryParserConfig.prototype.default_fields = $util.emptyArray;

            /**
             * QueryParserConfig boolean_should_mode.
             * @member {summa.proto.IMatchQueryBooleanShouldMode|null|undefined} boolean_should_mode
             * @memberof summa.proto.QueryParserConfig
             * @instance
             */
            QueryParserConfig.prototype.boolean_should_mode = null;

            /**
             * QueryParserConfig disjuction_max_mode.
             * @member {summa.proto.IMatchQueryDisjuctionMaxMode|null|undefined} disjuction_max_mode
             * @memberof summa.proto.QueryParserConfig
             * @instance
             */
            QueryParserConfig.prototype.disjuction_max_mode = null;

            /**
             * QueryParserConfig exact_matches_promoter.
             * @member {summa.proto.IExactMatchesPromoter|null|undefined} exact_matches_promoter
             * @memberof summa.proto.QueryParserConfig
             * @instance
             */
            QueryParserConfig.prototype.exact_matches_promoter = null;

            /**
             * QueryParserConfig removed_fields.
             * @member {Array.<string>} removed_fields
             * @memberof summa.proto.QueryParserConfig
             * @instance
             */
            QueryParserConfig.prototype.removed_fields = $util.emptyArray;

            /**
             * QueryParserConfig morphology_configs.
             * @member {Object.<string,summa.proto.IMorphologyConfig>} morphology_configs
             * @memberof summa.proto.QueryParserConfig
             * @instance
             */
            QueryParserConfig.prototype.morphology_configs = $util.emptyObject;

            /**
             * QueryParserConfig query_language.
             * @member {string|null|undefined} query_language
             * @memberof summa.proto.QueryParserConfig
             * @instance
             */
            QueryParserConfig.prototype.query_language = null;

            // OneOf field names bound to virtual getters and setters
            let $oneOfFields;

            /**
             * QueryParserConfig default_mode.
             * @member {"boolean_should_mode"|"disjuction_max_mode"|undefined} default_mode
             * @memberof summa.proto.QueryParserConfig
             * @instance
             */
            Object.defineProperty(QueryParserConfig.prototype, "default_mode", {
                get: $util.oneOfGetter($oneOfFields = ["boolean_should_mode", "disjuction_max_mode"]),
                set: $util.oneOfSetter($oneOfFields)
            });

            /**
             * QueryParserConfig _query_language.
             * @member {"query_language"|undefined} _query_language
             * @memberof summa.proto.QueryParserConfig
             * @instance
             */
            Object.defineProperty(QueryParserConfig.prototype, "_query_language", {
                get: $util.oneOfGetter($oneOfFields = ["query_language"]),
                set: $util.oneOfSetter($oneOfFields)
            });

            /**
             * Creates a new QueryParserConfig instance using the specified properties.
             * @function create
             * @memberof summa.proto.QueryParserConfig
             * @static
             * @param {summa.proto.IQueryParserConfig=} [properties] Properties to set
             * @returns {summa.proto.QueryParserConfig} QueryParserConfig instance
             */
            QueryParserConfig.create = function create(properties) {
                return new QueryParserConfig(properties);
            };

            return QueryParserConfig;
        })();

        proto.SearchResponse = (function() {

            /**
             * Properties of a SearchResponse.
             * @memberof summa.proto
             * @interface ISearchResponse
             * @property {number|null} [elapsed_secs] SearchResponse elapsed_secs
             * @property {Array.<summa.proto.ICollectorOutput>|null} [collector_outputs] SearchResponse collector_outputs
             */

            /**
             * Constructs a new SearchResponse.
             * @memberof summa.proto
             * @classdesc Represents a SearchResponse.
             * @implements ISearchResponse
             * @constructor
             * @param {summa.proto.ISearchResponse=} [properties] Properties to set
             */
            function SearchResponse(properties) {
                this.collector_outputs = [];
                if (properties)
                    for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }

            /**
             * SearchResponse elapsed_secs.
             * @member {number} elapsed_secs
             * @memberof summa.proto.SearchResponse
             * @instance
             */
            SearchResponse.prototype.elapsed_secs = 0;

            /**
             * SearchResponse collector_outputs.
             * @member {Array.<summa.proto.ICollectorOutput>} collector_outputs
             * @memberof summa.proto.SearchResponse
             * @instance
             */
            SearchResponse.prototype.collector_outputs = $util.emptyArray;

            /**
             * Creates a new SearchResponse instance using the specified properties.
             * @function create
             * @memberof summa.proto.SearchResponse
             * @static
             * @param {summa.proto.ISearchResponse=} [properties] Properties to set
             * @returns {summa.proto.SearchResponse} SearchResponse instance
             */
            SearchResponse.create = function create(properties) {
                return new SearchResponse(properties);
            };

            return SearchResponse;
        })();

        proto.Query = (function() {

            /**
             * Properties of a Query.
             * @memberof summa.proto
             * @interface IQuery
             * @property {summa.proto.IBooleanQuery|null} [boolean] Query boolean
             * @property {summa.proto.IMatchQuery|null} [match] Query match
             * @property {summa.proto.IRegexQuery|null} [regex] Query regex
             * @property {summa.proto.ITermQuery|null} [term] Query term
             * @property {summa.proto.IPhraseQuery|null} [phrase] Query phrase
             * @property {summa.proto.IRangeQuery|null} [range] Query range
             * @property {summa.proto.IAllQuery|null} [all] Query all
             * @property {summa.proto.IMoreLikeThisQuery|null} [more_like_this] Query more_like_this
             * @property {summa.proto.IBoostQuery|null} [boost] Query boost
             * @property {summa.proto.IDisjunctionMaxQuery|null} [disjunction_max] Query disjunction_max
             * @property {summa.proto.IEmptyQuery|null} [empty] Query empty
             * @property {summa.proto.IExistsQuery|null} [exists] Query exists
             */

            /**
             * Constructs a new Query.
             * @memberof summa.proto
             * @classdesc Represents a Query.
             * @implements IQuery
             * @constructor
             * @param {summa.proto.IQuery=} [properties] Properties to set
             */
            function Query(properties) {
                if (properties)
                    for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }

            /**
             * Query boolean.
             * @member {summa.proto.IBooleanQuery|null|undefined} boolean
             * @memberof summa.proto.Query
             * @instance
             */
            Query.prototype.boolean = null;

            /**
             * Query match.
             * @member {summa.proto.IMatchQuery|null|undefined} match
             * @memberof summa.proto.Query
             * @instance
             */
            Query.prototype.match = null;

            /**
             * Query regex.
             * @member {summa.proto.IRegexQuery|null|undefined} regex
             * @memberof summa.proto.Query
             * @instance
             */
            Query.prototype.regex = null;

            /**
             * Query term.
             * @member {summa.proto.ITermQuery|null|undefined} term
             * @memberof summa.proto.Query
             * @instance
             */
            Query.prototype.term = null;

            /**
             * Query phrase.
             * @member {summa.proto.IPhraseQuery|null|undefined} phrase
             * @memberof summa.proto.Query
             * @instance
             */
            Query.prototype.phrase = null;

            /**
             * Query range.
             * @member {summa.proto.IRangeQuery|null|undefined} range
             * @memberof summa.proto.Query
             * @instance
             */
            Query.prototype.range = null;

            /**
             * Query all.
             * @member {summa.proto.IAllQuery|null|undefined} all
             * @memberof summa.proto.Query
             * @instance
             */
            Query.prototype.all = null;

            /**
             * Query more_like_this.
             * @member {summa.proto.IMoreLikeThisQuery|null|undefined} more_like_this
             * @memberof summa.proto.Query
             * @instance
             */
            Query.prototype.more_like_this = null;

            /**
             * Query boost.
             * @member {summa.proto.IBoostQuery|null|undefined} boost
             * @memberof summa.proto.Query
             * @instance
             */
            Query.prototype.boost = null;

            /**
             * Query disjunction_max.
             * @member {summa.proto.IDisjunctionMaxQuery|null|undefined} disjunction_max
             * @memberof summa.proto.Query
             * @instance
             */
            Query.prototype.disjunction_max = null;

            /**
             * Query empty.
             * @member {summa.proto.IEmptyQuery|null|undefined} empty
             * @memberof summa.proto.Query
             * @instance
             */
            Query.prototype.empty = null;

            /**
             * Query exists.
             * @member {summa.proto.IExistsQuery|null|undefined} exists
             * @memberof summa.proto.Query
             * @instance
             */
            Query.prototype.exists = null;

            // OneOf field names bound to virtual getters and setters
            let $oneOfFields;

            /**
             * Query query.
             * @member {"boolean"|"match"|"regex"|"term"|"phrase"|"range"|"all"|"more_like_this"|"boost"|"disjunction_max"|"empty"|"exists"|undefined} query
             * @memberof summa.proto.Query
             * @instance
             */
            Object.defineProperty(Query.prototype, "query", {
                get: $util.oneOfGetter($oneOfFields = ["boolean", "match", "regex", "term", "phrase", "range", "all", "more_like_this", "boost", "disjunction_max", "empty", "exists"]),
                set: $util.oneOfSetter($oneOfFields)
            });

            /**
             * Creates a new Query instance using the specified properties.
             * @function create
             * @memberof summa.proto.Query
             * @static
             * @param {summa.proto.IQuery=} [properties] Properties to set
             * @returns {summa.proto.Query} Query instance
             */
            Query.create = function create(properties) {
                return new Query(properties);
            };

            return Query;
        })();

        proto.AllQuery = (function() {

            /**
             * Properties of an AllQuery.
             * @memberof summa.proto
             * @interface IAllQuery
             */

            /**
             * Constructs a new AllQuery.
             * @memberof summa.proto
             * @classdesc Represents an AllQuery.
             * @implements IAllQuery
             * @constructor
             * @param {summa.proto.IAllQuery=} [properties] Properties to set
             */
            function AllQuery(properties) {
                if (properties)
                    for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }

            /**
             * Creates a new AllQuery instance using the specified properties.
             * @function create
             * @memberof summa.proto.AllQuery
             * @static
             * @param {summa.proto.IAllQuery=} [properties] Properties to set
             * @returns {summa.proto.AllQuery} AllQuery instance
             */
            AllQuery.create = function create(properties) {
                return new AllQuery(properties);
            };

            return AllQuery;
        })();

        proto.EmptyQuery = (function() {

            /**
             * Properties of an EmptyQuery.
             * @memberof summa.proto
             * @interface IEmptyQuery
             */

            /**
             * Constructs a new EmptyQuery.
             * @memberof summa.proto
             * @classdesc Represents an EmptyQuery.
             * @implements IEmptyQuery
             * @constructor
             * @param {summa.proto.IEmptyQuery=} [properties] Properties to set
             */
            function EmptyQuery(properties) {
                if (properties)
                    for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }

            /**
             * Creates a new EmptyQuery instance using the specified properties.
             * @function create
             * @memberof summa.proto.EmptyQuery
             * @static
             * @param {summa.proto.IEmptyQuery=} [properties] Properties to set
             * @returns {summa.proto.EmptyQuery} EmptyQuery instance
             */
            EmptyQuery.create = function create(properties) {
                return new EmptyQuery(properties);
            };

            return EmptyQuery;
        })();

        proto.BoostQuery = (function() {

            /**
             * Properties of a BoostQuery.
             * @memberof summa.proto
             * @interface IBoostQuery
             * @property {summa.proto.IQuery|null} [query] BoostQuery query
             * @property {string|null} [score] BoostQuery score
             */

            /**
             * Constructs a new BoostQuery.
             * @memberof summa.proto
             * @classdesc Represents a BoostQuery.
             * @implements IBoostQuery
             * @constructor
             * @param {summa.proto.IBoostQuery=} [properties] Properties to set
             */
            function BoostQuery(properties) {
                if (properties)
                    for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }

            /**
             * BoostQuery query.
             * @member {summa.proto.IQuery|null|undefined} query
             * @memberof summa.proto.BoostQuery
             * @instance
             */
            BoostQuery.prototype.query = null;

            /**
             * BoostQuery score.
             * @member {string} score
             * @memberof summa.proto.BoostQuery
             * @instance
             */
            BoostQuery.prototype.score = "";

            /**
             * Creates a new BoostQuery instance using the specified properties.
             * @function create
             * @memberof summa.proto.BoostQuery
             * @static
             * @param {summa.proto.IBoostQuery=} [properties] Properties to set
             * @returns {summa.proto.BoostQuery} BoostQuery instance
             */
            BoostQuery.create = function create(properties) {
                return new BoostQuery(properties);
            };

            return BoostQuery;
        })();

        proto.DisjunctionMaxQuery = (function() {

            /**
             * Properties of a DisjunctionMaxQuery.
             * @memberof summa.proto
             * @interface IDisjunctionMaxQuery
             * @property {Array.<summa.proto.IQuery>|null} [disjuncts] DisjunctionMaxQuery disjuncts
             * @property {string|null} [tie_breaker] DisjunctionMaxQuery tie_breaker
             */

            /**
             * Constructs a new DisjunctionMaxQuery.
             * @memberof summa.proto
             * @classdesc Represents a DisjunctionMaxQuery.
             * @implements IDisjunctionMaxQuery
             * @constructor
             * @param {summa.proto.IDisjunctionMaxQuery=} [properties] Properties to set
             */
            function DisjunctionMaxQuery(properties) {
                this.disjuncts = [];
                if (properties)
                    for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }

            /**
             * DisjunctionMaxQuery disjuncts.
             * @member {Array.<summa.proto.IQuery>} disjuncts
             * @memberof summa.proto.DisjunctionMaxQuery
             * @instance
             */
            DisjunctionMaxQuery.prototype.disjuncts = $util.emptyArray;

            /**
             * DisjunctionMaxQuery tie_breaker.
             * @member {string} tie_breaker
             * @memberof summa.proto.DisjunctionMaxQuery
             * @instance
             */
            DisjunctionMaxQuery.prototype.tie_breaker = "";

            /**
             * Creates a new DisjunctionMaxQuery instance using the specified properties.
             * @function create
             * @memberof summa.proto.DisjunctionMaxQuery
             * @static
             * @param {summa.proto.IDisjunctionMaxQuery=} [properties] Properties to set
             * @returns {summa.proto.DisjunctionMaxQuery} DisjunctionMaxQuery instance
             */
            DisjunctionMaxQuery.create = function create(properties) {
                return new DisjunctionMaxQuery(properties);
            };

            return DisjunctionMaxQuery;
        })();

        proto.MoreLikeThisQuery = (function() {

            /**
             * Properties of a MoreLikeThisQuery.
             * @memberof summa.proto
             * @interface IMoreLikeThisQuery
             * @property {string|null} [document] MoreLikeThisQuery document
             * @property {number|Long|null} [min_doc_frequency] MoreLikeThisQuery min_doc_frequency
             * @property {number|Long|null} [max_doc_frequency] MoreLikeThisQuery max_doc_frequency
             * @property {number|Long|null} [min_term_frequency] MoreLikeThisQuery min_term_frequency
             * @property {number|Long|null} [max_query_terms] MoreLikeThisQuery max_query_terms
             * @property {number|Long|null} [min_word_length] MoreLikeThisQuery min_word_length
             * @property {number|Long|null} [max_word_length] MoreLikeThisQuery max_word_length
             * @property {string|null} [boost] MoreLikeThisQuery boost
             * @property {Array.<string>|null} [stop_words] MoreLikeThisQuery stop_words
             */

            /**
             * Constructs a new MoreLikeThisQuery.
             * @memberof summa.proto
             * @classdesc Represents a MoreLikeThisQuery.
             * @implements IMoreLikeThisQuery
             * @constructor
             * @param {summa.proto.IMoreLikeThisQuery=} [properties] Properties to set
             */
            function MoreLikeThisQuery(properties) {
                this.stop_words = [];
                if (properties)
                    for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }

            /**
             * MoreLikeThisQuery document.
             * @member {string} document
             * @memberof summa.proto.MoreLikeThisQuery
             * @instance
             */
            MoreLikeThisQuery.prototype.document = "";

            /**
             * MoreLikeThisQuery min_doc_frequency.
             * @member {number|Long|null|undefined} min_doc_frequency
             * @memberof summa.proto.MoreLikeThisQuery
             * @instance
             */
            MoreLikeThisQuery.prototype.min_doc_frequency = null;

            /**
             * MoreLikeThisQuery max_doc_frequency.
             * @member {number|Long|null|undefined} max_doc_frequency
             * @memberof summa.proto.MoreLikeThisQuery
             * @instance
             */
            MoreLikeThisQuery.prototype.max_doc_frequency = null;

            /**
             * MoreLikeThisQuery min_term_frequency.
             * @member {number|Long|null|undefined} min_term_frequency
             * @memberof summa.proto.MoreLikeThisQuery
             * @instance
             */
            MoreLikeThisQuery.prototype.min_term_frequency = null;

            /**
             * MoreLikeThisQuery max_query_terms.
             * @member {number|Long|null|undefined} max_query_terms
             * @memberof summa.proto.MoreLikeThisQuery
             * @instance
             */
            MoreLikeThisQuery.prototype.max_query_terms = null;

            /**
             * MoreLikeThisQuery min_word_length.
             * @member {number|Long|null|undefined} min_word_length
             * @memberof summa.proto.MoreLikeThisQuery
             * @instance
             */
            MoreLikeThisQuery.prototype.min_word_length = null;

            /**
             * MoreLikeThisQuery max_word_length.
             * @member {number|Long|null|undefined} max_word_length
             * @memberof summa.proto.MoreLikeThisQuery
             * @instance
             */
            MoreLikeThisQuery.prototype.max_word_length = null;

            /**
             * MoreLikeThisQuery boost.
             * @member {string|null|undefined} boost
             * @memberof summa.proto.MoreLikeThisQuery
             * @instance
             */
            MoreLikeThisQuery.prototype.boost = null;

            /**
             * MoreLikeThisQuery stop_words.
             * @member {Array.<string>} stop_words
             * @memberof summa.proto.MoreLikeThisQuery
             * @instance
             */
            MoreLikeThisQuery.prototype.stop_words = $util.emptyArray;

            // OneOf field names bound to virtual getters and setters
            let $oneOfFields;

            /**
             * MoreLikeThisQuery _min_doc_frequency.
             * @member {"min_doc_frequency"|undefined} _min_doc_frequency
             * @memberof summa.proto.MoreLikeThisQuery
             * @instance
             */
            Object.defineProperty(MoreLikeThisQuery.prototype, "_min_doc_frequency", {
                get: $util.oneOfGetter($oneOfFields = ["min_doc_frequency"]),
                set: $util.oneOfSetter($oneOfFields)
            });

            /**
             * MoreLikeThisQuery _max_doc_frequency.
             * @member {"max_doc_frequency"|undefined} _max_doc_frequency
             * @memberof summa.proto.MoreLikeThisQuery
             * @instance
             */
            Object.defineProperty(MoreLikeThisQuery.prototype, "_max_doc_frequency", {
                get: $util.oneOfGetter($oneOfFields = ["max_doc_frequency"]),
                set: $util.oneOfSetter($oneOfFields)
            });

            /**
             * MoreLikeThisQuery _min_term_frequency.
             * @member {"min_term_frequency"|undefined} _min_term_frequency
             * @memberof summa.proto.MoreLikeThisQuery
             * @instance
             */
            Object.defineProperty(MoreLikeThisQuery.prototype, "_min_term_frequency", {
                get: $util.oneOfGetter($oneOfFields = ["min_term_frequency"]),
                set: $util.oneOfSetter($oneOfFields)
            });

            /**
             * MoreLikeThisQuery _max_query_terms.
             * @member {"max_query_terms"|undefined} _max_query_terms
             * @memberof summa.proto.MoreLikeThisQuery
             * @instance
             */
            Object.defineProperty(MoreLikeThisQuery.prototype, "_max_query_terms", {
                get: $util.oneOfGetter($oneOfFields = ["max_query_terms"]),
                set: $util.oneOfSetter($oneOfFields)
            });

            /**
             * MoreLikeThisQuery _min_word_length.
             * @member {"min_word_length"|undefined} _min_word_length
             * @memberof summa.proto.MoreLikeThisQuery
             * @instance
             */
            Object.defineProperty(MoreLikeThisQuery.prototype, "_min_word_length", {
                get: $util.oneOfGetter($oneOfFields = ["min_word_length"]),
                set: $util.oneOfSetter($oneOfFields)
            });

            /**
             * MoreLikeThisQuery _max_word_length.
             * @member {"max_word_length"|undefined} _max_word_length
             * @memberof summa.proto.MoreLikeThisQuery
             * @instance
             */
            Object.defineProperty(MoreLikeThisQuery.prototype, "_max_word_length", {
                get: $util.oneOfGetter($oneOfFields = ["max_word_length"]),
                set: $util.oneOfSetter($oneOfFields)
            });

            /**
             * MoreLikeThisQuery _boost.
             * @member {"boost"|undefined} _boost
             * @memberof summa.proto.MoreLikeThisQuery
             * @instance
             */
            Object.defineProperty(MoreLikeThisQuery.prototype, "_boost", {
                get: $util.oneOfGetter($oneOfFields = ["boost"]),
                set: $util.oneOfSetter($oneOfFields)
            });

            /**
             * Creates a new MoreLikeThisQuery instance using the specified properties.
             * @function create
             * @memberof summa.proto.MoreLikeThisQuery
             * @static
             * @param {summa.proto.IMoreLikeThisQuery=} [properties] Properties to set
             * @returns {summa.proto.MoreLikeThisQuery} MoreLikeThisQuery instance
             */
            MoreLikeThisQuery.create = function create(properties) {
                return new MoreLikeThisQuery(properties);
            };

            return MoreLikeThisQuery;
        })();

        proto.PhraseQuery = (function() {

            /**
             * Properties of a PhraseQuery.
             * @memberof summa.proto
             * @interface IPhraseQuery
             * @property {string|null} [field] PhraseQuery field
             * @property {string|null} [value] PhraseQuery value
             * @property {number|null} [slop] PhraseQuery slop
             */

            /**
             * Constructs a new PhraseQuery.
             * @memberof summa.proto
             * @classdesc Represents a PhraseQuery.
             * @implements IPhraseQuery
             * @constructor
             * @param {summa.proto.IPhraseQuery=} [properties] Properties to set
             */
            function PhraseQuery(properties) {
                if (properties)
                    for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }

            /**
             * PhraseQuery field.
             * @member {string} field
             * @memberof summa.proto.PhraseQuery
             * @instance
             */
            PhraseQuery.prototype.field = "";

            /**
             * PhraseQuery value.
             * @member {string} value
             * @memberof summa.proto.PhraseQuery
             * @instance
             */
            PhraseQuery.prototype.value = "";

            /**
             * PhraseQuery slop.
             * @member {number} slop
             * @memberof summa.proto.PhraseQuery
             * @instance
             */
            PhraseQuery.prototype.slop = 0;

            /**
             * Creates a new PhraseQuery instance using the specified properties.
             * @function create
             * @memberof summa.proto.PhraseQuery
             * @static
             * @param {summa.proto.IPhraseQuery=} [properties] Properties to set
             * @returns {summa.proto.PhraseQuery} PhraseQuery instance
             */
            PhraseQuery.create = function create(properties) {
                return new PhraseQuery(properties);
            };

            return PhraseQuery;
        })();

        proto.RangeQuery = (function() {

            /**
             * Properties of a RangeQuery.
             * @memberof summa.proto
             * @interface IRangeQuery
             * @property {string|null} [field] RangeQuery field
             * @property {summa.proto.IRange|null} [value] RangeQuery value
             */

            /**
             * Constructs a new RangeQuery.
             * @memberof summa.proto
             * @classdesc Represents a RangeQuery.
             * @implements IRangeQuery
             * @constructor
             * @param {summa.proto.IRangeQuery=} [properties] Properties to set
             */
            function RangeQuery(properties) {
                if (properties)
                    for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }

            /**
             * RangeQuery field.
             * @member {string} field
             * @memberof summa.proto.RangeQuery
             * @instance
             */
            RangeQuery.prototype.field = "";

            /**
             * RangeQuery value.
             * @member {summa.proto.IRange|null|undefined} value
             * @memberof summa.proto.RangeQuery
             * @instance
             */
            RangeQuery.prototype.value = null;

            /**
             * Creates a new RangeQuery instance using the specified properties.
             * @function create
             * @memberof summa.proto.RangeQuery
             * @static
             * @param {summa.proto.IRangeQuery=} [properties] Properties to set
             * @returns {summa.proto.RangeQuery} RangeQuery instance
             */
            RangeQuery.create = function create(properties) {
                return new RangeQuery(properties);
            };

            return RangeQuery;
        })();

        proto.MatchQuery = (function() {

            /**
             * Properties of a MatchQuery.
             * @memberof summa.proto
             * @interface IMatchQuery
             * @property {string|null} [value] MatchQuery value
             * @property {summa.proto.IQueryParserConfig|null} [query_parser_config] MatchQuery query_parser_config
             */

            /**
             * Constructs a new MatchQuery.
             * @memberof summa.proto
             * @classdesc Represents a MatchQuery.
             * @implements IMatchQuery
             * @constructor
             * @param {summa.proto.IMatchQuery=} [properties] Properties to set
             */
            function MatchQuery(properties) {
                if (properties)
                    for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }

            /**
             * MatchQuery value.
             * @member {string} value
             * @memberof summa.proto.MatchQuery
             * @instance
             */
            MatchQuery.prototype.value = "";

            /**
             * MatchQuery query_parser_config.
             * @member {summa.proto.IQueryParserConfig|null|undefined} query_parser_config
             * @memberof summa.proto.MatchQuery
             * @instance
             */
            MatchQuery.prototype.query_parser_config = null;

            // OneOf field names bound to virtual getters and setters
            let $oneOfFields;

            /**
             * MatchQuery _query_parser_config.
             * @member {"query_parser_config"|undefined} _query_parser_config
             * @memberof summa.proto.MatchQuery
             * @instance
             */
            Object.defineProperty(MatchQuery.prototype, "_query_parser_config", {
                get: $util.oneOfGetter($oneOfFields = ["query_parser_config"]),
                set: $util.oneOfSetter($oneOfFields)
            });

            /**
             * Creates a new MatchQuery instance using the specified properties.
             * @function create
             * @memberof summa.proto.MatchQuery
             * @static
             * @param {summa.proto.IMatchQuery=} [properties] Properties to set
             * @returns {summa.proto.MatchQuery} MatchQuery instance
             */
            MatchQuery.create = function create(properties) {
                return new MatchQuery(properties);
            };

            return MatchQuery;
        })();

        proto.BooleanSubquery = (function() {

            /**
             * Properties of a BooleanSubquery.
             * @memberof summa.proto
             * @interface IBooleanSubquery
             * @property {summa.proto.Occur|null} [occur] BooleanSubquery occur
             * @property {summa.proto.IQuery|null} [query] BooleanSubquery query
             */

            /**
             * Constructs a new BooleanSubquery.
             * @memberof summa.proto
             * @classdesc Represents a BooleanSubquery.
             * @implements IBooleanSubquery
             * @constructor
             * @param {summa.proto.IBooleanSubquery=} [properties] Properties to set
             */
            function BooleanSubquery(properties) {
                if (properties)
                    for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }

            /**
             * BooleanSubquery occur.
             * @member {summa.proto.Occur} occur
             * @memberof summa.proto.BooleanSubquery
             * @instance
             */
            BooleanSubquery.prototype.occur = 0;

            /**
             * BooleanSubquery query.
             * @member {summa.proto.IQuery|null|undefined} query
             * @memberof summa.proto.BooleanSubquery
             * @instance
             */
            BooleanSubquery.prototype.query = null;

            /**
             * Creates a new BooleanSubquery instance using the specified properties.
             * @function create
             * @memberof summa.proto.BooleanSubquery
             * @static
             * @param {summa.proto.IBooleanSubquery=} [properties] Properties to set
             * @returns {summa.proto.BooleanSubquery} BooleanSubquery instance
             */
            BooleanSubquery.create = function create(properties) {
                return new BooleanSubquery(properties);
            };

            return BooleanSubquery;
        })();

        proto.BooleanQuery = (function() {

            /**
             * Properties of a BooleanQuery.
             * @memberof summa.proto
             * @interface IBooleanQuery
             * @property {Array.<summa.proto.IBooleanSubquery>|null} [subqueries] BooleanQuery subqueries
             */

            /**
             * Constructs a new BooleanQuery.
             * @memberof summa.proto
             * @classdesc Represents a BooleanQuery.
             * @implements IBooleanQuery
             * @constructor
             * @param {summa.proto.IBooleanQuery=} [properties] Properties to set
             */
            function BooleanQuery(properties) {
                this.subqueries = [];
                if (properties)
                    for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }

            /**
             * BooleanQuery subqueries.
             * @member {Array.<summa.proto.IBooleanSubquery>} subqueries
             * @memberof summa.proto.BooleanQuery
             * @instance
             */
            BooleanQuery.prototype.subqueries = $util.emptyArray;

            /**
             * Creates a new BooleanQuery instance using the specified properties.
             * @function create
             * @memberof summa.proto.BooleanQuery
             * @static
             * @param {summa.proto.IBooleanQuery=} [properties] Properties to set
             * @returns {summa.proto.BooleanQuery} BooleanQuery instance
             */
            BooleanQuery.create = function create(properties) {
                return new BooleanQuery(properties);
            };

            return BooleanQuery;
        })();

        proto.RegexQuery = (function() {

            /**
             * Properties of a RegexQuery.
             * @memberof summa.proto
             * @interface IRegexQuery
             * @property {string|null} [field] RegexQuery field
             * @property {string|null} [value] RegexQuery value
             */

            /**
             * Constructs a new RegexQuery.
             * @memberof summa.proto
             * @classdesc Represents a RegexQuery.
             * @implements IRegexQuery
             * @constructor
             * @param {summa.proto.IRegexQuery=} [properties] Properties to set
             */
            function RegexQuery(properties) {
                if (properties)
                    for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }

            /**
             * RegexQuery field.
             * @member {string} field
             * @memberof summa.proto.RegexQuery
             * @instance
             */
            RegexQuery.prototype.field = "";

            /**
             * RegexQuery value.
             * @member {string} value
             * @memberof summa.proto.RegexQuery
             * @instance
             */
            RegexQuery.prototype.value = "";

            /**
             * Creates a new RegexQuery instance using the specified properties.
             * @function create
             * @memberof summa.proto.RegexQuery
             * @static
             * @param {summa.proto.IRegexQuery=} [properties] Properties to set
             * @returns {summa.proto.RegexQuery} RegexQuery instance
             */
            RegexQuery.create = function create(properties) {
                return new RegexQuery(properties);
            };

            return RegexQuery;
        })();

        proto.TermQuery = (function() {

            /**
             * Properties of a TermQuery.
             * @memberof summa.proto
             * @interface ITermQuery
             * @property {string|null} [field] TermQuery field
             * @property {string|null} [value] TermQuery value
             */

            /**
             * Constructs a new TermQuery.
             * @memberof summa.proto
             * @classdesc Represents a TermQuery.
             * @implements ITermQuery
             * @constructor
             * @param {summa.proto.ITermQuery=} [properties] Properties to set
             */
            function TermQuery(properties) {
                if (properties)
                    for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }

            /**
             * TermQuery field.
             * @member {string} field
             * @memberof summa.proto.TermQuery
             * @instance
             */
            TermQuery.prototype.field = "";

            /**
             * TermQuery value.
             * @member {string} value
             * @memberof summa.proto.TermQuery
             * @instance
             */
            TermQuery.prototype.value = "";

            /**
             * Creates a new TermQuery instance using the specified properties.
             * @function create
             * @memberof summa.proto.TermQuery
             * @static
             * @param {summa.proto.ITermQuery=} [properties] Properties to set
             * @returns {summa.proto.TermQuery} TermQuery instance
             */
            TermQuery.create = function create(properties) {
                return new TermQuery(properties);
            };

            return TermQuery;
        })();

        proto.ExistsQuery = (function() {

            /**
             * Properties of an ExistsQuery.
             * @memberof summa.proto
             * @interface IExistsQuery
             * @property {string|null} [field] ExistsQuery field
             */

            /**
             * Constructs a new ExistsQuery.
             * @memberof summa.proto
             * @classdesc Represents an ExistsQuery.
             * @implements IExistsQuery
             * @constructor
             * @param {summa.proto.IExistsQuery=} [properties] Properties to set
             */
            function ExistsQuery(properties) {
                if (properties)
                    for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }

            /**
             * ExistsQuery field.
             * @member {string} field
             * @memberof summa.proto.ExistsQuery
             * @instance
             */
            ExistsQuery.prototype.field = "";

            /**
             * Creates a new ExistsQuery instance using the specified properties.
             * @function create
             * @memberof summa.proto.ExistsQuery
             * @static
             * @param {summa.proto.IExistsQuery=} [properties] Properties to set
             * @returns {summa.proto.ExistsQuery} ExistsQuery instance
             */
            ExistsQuery.create = function create(properties) {
                return new ExistsQuery(properties);
            };

            return ExistsQuery;
        })();

        proto.Aggregation = (function() {

            /**
             * Properties of an Aggregation.
             * @memberof summa.proto
             * @interface IAggregation
             * @property {summa.proto.IBucketAggregation|null} [bucket] Aggregation bucket
             * @property {summa.proto.IMetricAggregation|null} [metric] Aggregation metric
             */

            /**
             * Constructs a new Aggregation.
             * @memberof summa.proto
             * @classdesc Represents an Aggregation.
             * @implements IAggregation
             * @constructor
             * @param {summa.proto.IAggregation=} [properties] Properties to set
             */
            function Aggregation(properties) {
                if (properties)
                    for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }

            /**
             * Aggregation bucket.
             * @member {summa.proto.IBucketAggregation|null|undefined} bucket
             * @memberof summa.proto.Aggregation
             * @instance
             */
            Aggregation.prototype.bucket = null;

            /**
             * Aggregation metric.
             * @member {summa.proto.IMetricAggregation|null|undefined} metric
             * @memberof summa.proto.Aggregation
             * @instance
             */
            Aggregation.prototype.metric = null;

            // OneOf field names bound to virtual getters and setters
            let $oneOfFields;

            /**
             * Aggregation aggregation.
             * @member {"bucket"|"metric"|undefined} aggregation
             * @memberof summa.proto.Aggregation
             * @instance
             */
            Object.defineProperty(Aggregation.prototype, "aggregation", {
                get: $util.oneOfGetter($oneOfFields = ["bucket", "metric"]),
                set: $util.oneOfSetter($oneOfFields)
            });

            /**
             * Creates a new Aggregation instance using the specified properties.
             * @function create
             * @memberof summa.proto.Aggregation
             * @static
             * @param {summa.proto.IAggregation=} [properties] Properties to set
             * @returns {summa.proto.Aggregation} Aggregation instance
             */
            Aggregation.create = function create(properties) {
                return new Aggregation(properties);
            };

            return Aggregation;
        })();

        proto.BucketAggregation = (function() {

            /**
             * Properties of a BucketAggregation.
             * @memberof summa.proto
             * @interface IBucketAggregation
             * @property {summa.proto.IRangeAggregation|null} [range] BucketAggregation range
             * @property {summa.proto.IHistogramAggregation|null} [histogram] BucketAggregation histogram
             * @property {summa.proto.ITermsAggregation|null} [terms] BucketAggregation terms
             * @property {Object.<string,summa.proto.IAggregation>|null} [sub_aggregation] BucketAggregation sub_aggregation
             */

            /**
             * Constructs a new BucketAggregation.
             * @memberof summa.proto
             * @classdesc Represents a BucketAggregation.
             * @implements IBucketAggregation
             * @constructor
             * @param {summa.proto.IBucketAggregation=} [properties] Properties to set
             */
            function BucketAggregation(properties) {
                this.sub_aggregation = {};
                if (properties)
                    for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }

            /**
             * BucketAggregation range.
             * @member {summa.proto.IRangeAggregation|null|undefined} range
             * @memberof summa.proto.BucketAggregation
             * @instance
             */
            BucketAggregation.prototype.range = null;

            /**
             * BucketAggregation histogram.
             * @member {summa.proto.IHistogramAggregation|null|undefined} histogram
             * @memberof summa.proto.BucketAggregation
             * @instance
             */
            BucketAggregation.prototype.histogram = null;

            /**
             * BucketAggregation terms.
             * @member {summa.proto.ITermsAggregation|null|undefined} terms
             * @memberof summa.proto.BucketAggregation
             * @instance
             */
            BucketAggregation.prototype.terms = null;

            /**
             * BucketAggregation sub_aggregation.
             * @member {Object.<string,summa.proto.IAggregation>} sub_aggregation
             * @memberof summa.proto.BucketAggregation
             * @instance
             */
            BucketAggregation.prototype.sub_aggregation = $util.emptyObject;

            // OneOf field names bound to virtual getters and setters
            let $oneOfFields;

            /**
             * BucketAggregation bucket_agg.
             * @member {"range"|"histogram"|"terms"|undefined} bucket_agg
             * @memberof summa.proto.BucketAggregation
             * @instance
             */
            Object.defineProperty(BucketAggregation.prototype, "bucket_agg", {
                get: $util.oneOfGetter($oneOfFields = ["range", "histogram", "terms"]),
                set: $util.oneOfSetter($oneOfFields)
            });

            /**
             * Creates a new BucketAggregation instance using the specified properties.
             * @function create
             * @memberof summa.proto.BucketAggregation
             * @static
             * @param {summa.proto.IBucketAggregation=} [properties] Properties to set
             * @returns {summa.proto.BucketAggregation} BucketAggregation instance
             */
            BucketAggregation.create = function create(properties) {
                return new BucketAggregation(properties);
            };

            return BucketAggregation;
        })();

        proto.RangeAggregation = (function() {

            /**
             * Properties of a RangeAggregation.
             * @memberof summa.proto
             * @interface IRangeAggregation
             * @property {string|null} [field] RangeAggregation field
             * @property {Array.<summa.proto.IRangeAggregationRange>|null} [ranges] RangeAggregation ranges
             */

            /**
             * Constructs a new RangeAggregation.
             * @memberof summa.proto
             * @classdesc Represents a RangeAggregation.
             * @implements IRangeAggregation
             * @constructor
             * @param {summa.proto.IRangeAggregation=} [properties] Properties to set
             */
            function RangeAggregation(properties) {
                this.ranges = [];
                if (properties)
                    for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }

            /**
             * RangeAggregation field.
             * @member {string} field
             * @memberof summa.proto.RangeAggregation
             * @instance
             */
            RangeAggregation.prototype.field = "";

            /**
             * RangeAggregation ranges.
             * @member {Array.<summa.proto.IRangeAggregationRange>} ranges
             * @memberof summa.proto.RangeAggregation
             * @instance
             */
            RangeAggregation.prototype.ranges = $util.emptyArray;

            /**
             * Creates a new RangeAggregation instance using the specified properties.
             * @function create
             * @memberof summa.proto.RangeAggregation
             * @static
             * @param {summa.proto.IRangeAggregation=} [properties] Properties to set
             * @returns {summa.proto.RangeAggregation} RangeAggregation instance
             */
            RangeAggregation.create = function create(properties) {
                return new RangeAggregation(properties);
            };

            return RangeAggregation;
        })();

        proto.RangeAggregationRange = (function() {

            /**
             * Properties of a RangeAggregationRange.
             * @memberof summa.proto
             * @interface IRangeAggregationRange
             * @property {number|null} [from] RangeAggregationRange from
             * @property {number|null} [to] RangeAggregationRange to
             * @property {string|null} [key] RangeAggregationRange key
             */

            /**
             * Constructs a new RangeAggregationRange.
             * @memberof summa.proto
             * @classdesc Represents a RangeAggregationRange.
             * @implements IRangeAggregationRange
             * @constructor
             * @param {summa.proto.IRangeAggregationRange=} [properties] Properties to set
             */
            function RangeAggregationRange(properties) {
                if (properties)
                    for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }

            /**
             * RangeAggregationRange from.
             * @member {number|null|undefined} from
             * @memberof summa.proto.RangeAggregationRange
             * @instance
             */
            RangeAggregationRange.prototype.from = null;

            /**
             * RangeAggregationRange to.
             * @member {number|null|undefined} to
             * @memberof summa.proto.RangeAggregationRange
             * @instance
             */
            RangeAggregationRange.prototype.to = null;

            /**
             * RangeAggregationRange key.
             * @member {string|null|undefined} key
             * @memberof summa.proto.RangeAggregationRange
             * @instance
             */
            RangeAggregationRange.prototype.key = null;

            // OneOf field names bound to virtual getters and setters
            let $oneOfFields;

            /**
             * RangeAggregationRange _from.
             * @member {"from"|undefined} _from
             * @memberof summa.proto.RangeAggregationRange
             * @instance
             */
            Object.defineProperty(RangeAggregationRange.prototype, "_from", {
                get: $util.oneOfGetter($oneOfFields = ["from"]),
                set: $util.oneOfSetter($oneOfFields)
            });

            /**
             * RangeAggregationRange _to.
             * @member {"to"|undefined} _to
             * @memberof summa.proto.RangeAggregationRange
             * @instance
             */
            Object.defineProperty(RangeAggregationRange.prototype, "_to", {
                get: $util.oneOfGetter($oneOfFields = ["to"]),
                set: $util.oneOfSetter($oneOfFields)
            });

            /**
             * RangeAggregationRange _key.
             * @member {"key"|undefined} _key
             * @memberof summa.proto.RangeAggregationRange
             * @instance
             */
            Object.defineProperty(RangeAggregationRange.prototype, "_key", {
                get: $util.oneOfGetter($oneOfFields = ["key"]),
                set: $util.oneOfSetter($oneOfFields)
            });

            /**
             * Creates a new RangeAggregationRange instance using the specified properties.
             * @function create
             * @memberof summa.proto.RangeAggregationRange
             * @static
             * @param {summa.proto.IRangeAggregationRange=} [properties] Properties to set
             * @returns {summa.proto.RangeAggregationRange} RangeAggregationRange instance
             */
            RangeAggregationRange.create = function create(properties) {
                return new RangeAggregationRange(properties);
            };

            return RangeAggregationRange;
        })();

        proto.HistogramAggregation = (function() {

            /**
             * Properties of a HistogramAggregation.
             * @memberof summa.proto
             * @interface IHistogramAggregation
             * @property {string|null} [field] HistogramAggregation field
             * @property {number|null} [interval] HistogramAggregation interval
             * @property {number|null} [offset] HistogramAggregation offset
             * @property {number|Long|null} [min_doc_count] HistogramAggregation min_doc_count
             * @property {summa.proto.IHistogramBounds|null} [hard_bounds] HistogramAggregation hard_bounds
             * @property {summa.proto.IHistogramBounds|null} [extended_bounds] HistogramAggregation extended_bounds
             */

            /**
             * Constructs a new HistogramAggregation.
             * @memberof summa.proto
             * @classdesc Represents a HistogramAggregation.
             * @implements IHistogramAggregation
             * @constructor
             * @param {summa.proto.IHistogramAggregation=} [properties] Properties to set
             */
            function HistogramAggregation(properties) {
                if (properties)
                    for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }

            /**
             * HistogramAggregation field.
             * @member {string} field
             * @memberof summa.proto.HistogramAggregation
             * @instance
             */
            HistogramAggregation.prototype.field = "";

            /**
             * HistogramAggregation interval.
             * @member {number} interval
             * @memberof summa.proto.HistogramAggregation
             * @instance
             */
            HistogramAggregation.prototype.interval = 0;

            /**
             * HistogramAggregation offset.
             * @member {number|null|undefined} offset
             * @memberof summa.proto.HistogramAggregation
             * @instance
             */
            HistogramAggregation.prototype.offset = null;

            /**
             * HistogramAggregation min_doc_count.
             * @member {number|Long|null|undefined} min_doc_count
             * @memberof summa.proto.HistogramAggregation
             * @instance
             */
            HistogramAggregation.prototype.min_doc_count = null;

            /**
             * HistogramAggregation hard_bounds.
             * @member {summa.proto.IHistogramBounds|null|undefined} hard_bounds
             * @memberof summa.proto.HistogramAggregation
             * @instance
             */
            HistogramAggregation.prototype.hard_bounds = null;

            /**
             * HistogramAggregation extended_bounds.
             * @member {summa.proto.IHistogramBounds|null|undefined} extended_bounds
             * @memberof summa.proto.HistogramAggregation
             * @instance
             */
            HistogramAggregation.prototype.extended_bounds = null;

            // OneOf field names bound to virtual getters and setters
            let $oneOfFields;

            /**
             * HistogramAggregation _offset.
             * @member {"offset"|undefined} _offset
             * @memberof summa.proto.HistogramAggregation
             * @instance
             */
            Object.defineProperty(HistogramAggregation.prototype, "_offset", {
                get: $util.oneOfGetter($oneOfFields = ["offset"]),
                set: $util.oneOfSetter($oneOfFields)
            });

            /**
             * HistogramAggregation _min_doc_count.
             * @member {"min_doc_count"|undefined} _min_doc_count
             * @memberof summa.proto.HistogramAggregation
             * @instance
             */
            Object.defineProperty(HistogramAggregation.prototype, "_min_doc_count", {
                get: $util.oneOfGetter($oneOfFields = ["min_doc_count"]),
                set: $util.oneOfSetter($oneOfFields)
            });

            /**
             * HistogramAggregation _hard_bounds.
             * @member {"hard_bounds"|undefined} _hard_bounds
             * @memberof summa.proto.HistogramAggregation
             * @instance
             */
            Object.defineProperty(HistogramAggregation.prototype, "_hard_bounds", {
                get: $util.oneOfGetter($oneOfFields = ["hard_bounds"]),
                set: $util.oneOfSetter($oneOfFields)
            });

            /**
             * HistogramAggregation _extended_bounds.
             * @member {"extended_bounds"|undefined} _extended_bounds
             * @memberof summa.proto.HistogramAggregation
             * @instance
             */
            Object.defineProperty(HistogramAggregation.prototype, "_extended_bounds", {
                get: $util.oneOfGetter($oneOfFields = ["extended_bounds"]),
                set: $util.oneOfSetter($oneOfFields)
            });

            /**
             * Creates a new HistogramAggregation instance using the specified properties.
             * @function create
             * @memberof summa.proto.HistogramAggregation
             * @static
             * @param {summa.proto.IHistogramAggregation=} [properties] Properties to set
             * @returns {summa.proto.HistogramAggregation} HistogramAggregation instance
             */
            HistogramAggregation.create = function create(properties) {
                return new HistogramAggregation(properties);
            };

            return HistogramAggregation;
        })();

        proto.HistogramBounds = (function() {

            /**
             * Properties of a HistogramBounds.
             * @memberof summa.proto
             * @interface IHistogramBounds
             * @property {number|null} [min] HistogramBounds min
             * @property {number|null} [max] HistogramBounds max
             */

            /**
             * Constructs a new HistogramBounds.
             * @memberof summa.proto
             * @classdesc Represents a HistogramBounds.
             * @implements IHistogramBounds
             * @constructor
             * @param {summa.proto.IHistogramBounds=} [properties] Properties to set
             */
            function HistogramBounds(properties) {
                if (properties)
                    for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }

            /**
             * HistogramBounds min.
             * @member {number} min
             * @memberof summa.proto.HistogramBounds
             * @instance
             */
            HistogramBounds.prototype.min = 0;

            /**
             * HistogramBounds max.
             * @member {number} max
             * @memberof summa.proto.HistogramBounds
             * @instance
             */
            HistogramBounds.prototype.max = 0;

            /**
             * Creates a new HistogramBounds instance using the specified properties.
             * @function create
             * @memberof summa.proto.HistogramBounds
             * @static
             * @param {summa.proto.IHistogramBounds=} [properties] Properties to set
             * @returns {summa.proto.HistogramBounds} HistogramBounds instance
             */
            HistogramBounds.create = function create(properties) {
                return new HistogramBounds(properties);
            };

            return HistogramBounds;
        })();

        proto.TermsAggregation = (function() {

            /**
             * Properties of a TermsAggregation.
             * @memberof summa.proto
             * @interface ITermsAggregation
             * @property {string|null} [field] TermsAggregation field
             * @property {number|null} [size] TermsAggregation size
             * @property {number|null} [split_size] TermsAggregation split_size
             * @property {number|null} [segment_size] TermsAggregation segment_size
             * @property {boolean|null} [show_term_doc_count_error] TermsAggregation show_term_doc_count_error
             * @property {number|Long|null} [min_doc_count] TermsAggregation min_doc_count
             * @property {summa.proto.ICustomOrder|null} [order] TermsAggregation order
             */

            /**
             * Constructs a new TermsAggregation.
             * @memberof summa.proto
             * @classdesc Represents a TermsAggregation.
             * @implements ITermsAggregation
             * @constructor
             * @param {summa.proto.ITermsAggregation=} [properties] Properties to set
             */
            function TermsAggregation(properties) {
                if (properties)
                    for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }

            /**
             * TermsAggregation field.
             * @member {string} field
             * @memberof summa.proto.TermsAggregation
             * @instance
             */
            TermsAggregation.prototype.field = "";

            /**
             * TermsAggregation size.
             * @member {number|null|undefined} size
             * @memberof summa.proto.TermsAggregation
             * @instance
             */
            TermsAggregation.prototype.size = null;

            /**
             * TermsAggregation split_size.
             * @member {number|null|undefined} split_size
             * @memberof summa.proto.TermsAggregation
             * @instance
             */
            TermsAggregation.prototype.split_size = null;

            /**
             * TermsAggregation segment_size.
             * @member {number|null|undefined} segment_size
             * @memberof summa.proto.TermsAggregation
             * @instance
             */
            TermsAggregation.prototype.segment_size = null;

            /**
             * TermsAggregation show_term_doc_count_error.
             * @member {boolean|null|undefined} show_term_doc_count_error
             * @memberof summa.proto.TermsAggregation
             * @instance
             */
            TermsAggregation.prototype.show_term_doc_count_error = null;

            /**
             * TermsAggregation min_doc_count.
             * @member {number|Long|null|undefined} min_doc_count
             * @memberof summa.proto.TermsAggregation
             * @instance
             */
            TermsAggregation.prototype.min_doc_count = null;

            /**
             * TermsAggregation order.
             * @member {summa.proto.ICustomOrder|null|undefined} order
             * @memberof summa.proto.TermsAggregation
             * @instance
             */
            TermsAggregation.prototype.order = null;

            // OneOf field names bound to virtual getters and setters
            let $oneOfFields;

            /**
             * TermsAggregation _size.
             * @member {"size"|undefined} _size
             * @memberof summa.proto.TermsAggregation
             * @instance
             */
            Object.defineProperty(TermsAggregation.prototype, "_size", {
                get: $util.oneOfGetter($oneOfFields = ["size"]),
                set: $util.oneOfSetter($oneOfFields)
            });

            /**
             * TermsAggregation _split_size.
             * @member {"split_size"|undefined} _split_size
             * @memberof summa.proto.TermsAggregation
             * @instance
             */
            Object.defineProperty(TermsAggregation.prototype, "_split_size", {
                get: $util.oneOfGetter($oneOfFields = ["split_size"]),
                set: $util.oneOfSetter($oneOfFields)
            });

            /**
             * TermsAggregation _segment_size.
             * @member {"segment_size"|undefined} _segment_size
             * @memberof summa.proto.TermsAggregation
             * @instance
             */
            Object.defineProperty(TermsAggregation.prototype, "_segment_size", {
                get: $util.oneOfGetter($oneOfFields = ["segment_size"]),
                set: $util.oneOfSetter($oneOfFields)
            });

            /**
             * TermsAggregation _show_term_doc_count_error.
             * @member {"show_term_doc_count_error"|undefined} _show_term_doc_count_error
             * @memberof summa.proto.TermsAggregation
             * @instance
             */
            Object.defineProperty(TermsAggregation.prototype, "_show_term_doc_count_error", {
                get: $util.oneOfGetter($oneOfFields = ["show_term_doc_count_error"]),
                set: $util.oneOfSetter($oneOfFields)
            });

            /**
             * TermsAggregation _min_doc_count.
             * @member {"min_doc_count"|undefined} _min_doc_count
             * @memberof summa.proto.TermsAggregation
             * @instance
             */
            Object.defineProperty(TermsAggregation.prototype, "_min_doc_count", {
                get: $util.oneOfGetter($oneOfFields = ["min_doc_count"]),
                set: $util.oneOfSetter($oneOfFields)
            });

            /**
             * TermsAggregation _order.
             * @member {"order"|undefined} _order
             * @memberof summa.proto.TermsAggregation
             * @instance
             */
            Object.defineProperty(TermsAggregation.prototype, "_order", {
                get: $util.oneOfGetter($oneOfFields = ["order"]),
                set: $util.oneOfSetter($oneOfFields)
            });

            /**
             * Creates a new TermsAggregation instance using the specified properties.
             * @function create
             * @memberof summa.proto.TermsAggregation
             * @static
             * @param {summa.proto.ITermsAggregation=} [properties] Properties to set
             * @returns {summa.proto.TermsAggregation} TermsAggregation instance
             */
            TermsAggregation.create = function create(properties) {
                return new TermsAggregation(properties);
            };

            return TermsAggregation;
        })();

        proto.CustomOrder = (function() {

            /**
             * Properties of a CustomOrder.
             * @memberof summa.proto
             * @interface ICustomOrder
             * @property {summa.proto.IEmpty|null} [key] CustomOrder key
             * @property {summa.proto.IEmpty|null} [count] CustomOrder count
             * @property {string|null} [sub_aggregation] CustomOrder sub_aggregation
             * @property {summa.proto.Order|null} [order] CustomOrder order
             */

            /**
             * Constructs a new CustomOrder.
             * @memberof summa.proto
             * @classdesc Represents a CustomOrder.
             * @implements ICustomOrder
             * @constructor
             * @param {summa.proto.ICustomOrder=} [properties] Properties to set
             */
            function CustomOrder(properties) {
                if (properties)
                    for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }

            /**
             * CustomOrder key.
             * @member {summa.proto.IEmpty|null|undefined} key
             * @memberof summa.proto.CustomOrder
             * @instance
             */
            CustomOrder.prototype.key = null;

            /**
             * CustomOrder count.
             * @member {summa.proto.IEmpty|null|undefined} count
             * @memberof summa.proto.CustomOrder
             * @instance
             */
            CustomOrder.prototype.count = null;

            /**
             * CustomOrder sub_aggregation.
             * @member {string|null|undefined} sub_aggregation
             * @memberof summa.proto.CustomOrder
             * @instance
             */
            CustomOrder.prototype.sub_aggregation = null;

            /**
             * CustomOrder order.
             * @member {summa.proto.Order} order
             * @memberof summa.proto.CustomOrder
             * @instance
             */
            CustomOrder.prototype.order = 0;

            // OneOf field names bound to virtual getters and setters
            let $oneOfFields;

            /**
             * CustomOrder order_target.
             * @member {"key"|"count"|"sub_aggregation"|undefined} order_target
             * @memberof summa.proto.CustomOrder
             * @instance
             */
            Object.defineProperty(CustomOrder.prototype, "order_target", {
                get: $util.oneOfGetter($oneOfFields = ["key", "count", "sub_aggregation"]),
                set: $util.oneOfSetter($oneOfFields)
            });

            /**
             * Creates a new CustomOrder instance using the specified properties.
             * @function create
             * @memberof summa.proto.CustomOrder
             * @static
             * @param {summa.proto.ICustomOrder=} [properties] Properties to set
             * @returns {summa.proto.CustomOrder} CustomOrder instance
             */
            CustomOrder.create = function create(properties) {
                return new CustomOrder(properties);
            };

            return CustomOrder;
        })();

        proto.MetricAggregation = (function() {

            /**
             * Properties of a MetricAggregation.
             * @memberof summa.proto
             * @interface IMetricAggregation
             * @property {summa.proto.IAverageAggregation|null} [average] MetricAggregation average
             * @property {summa.proto.IStatsAggregation|null} [stats] MetricAggregation stats
             */

            /**
             * Constructs a new MetricAggregation.
             * @memberof summa.proto
             * @classdesc Represents a MetricAggregation.
             * @implements IMetricAggregation
             * @constructor
             * @param {summa.proto.IMetricAggregation=} [properties] Properties to set
             */
            function MetricAggregation(properties) {
                if (properties)
                    for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }

            /**
             * MetricAggregation average.
             * @member {summa.proto.IAverageAggregation|null|undefined} average
             * @memberof summa.proto.MetricAggregation
             * @instance
             */
            MetricAggregation.prototype.average = null;

            /**
             * MetricAggregation stats.
             * @member {summa.proto.IStatsAggregation|null|undefined} stats
             * @memberof summa.proto.MetricAggregation
             * @instance
             */
            MetricAggregation.prototype.stats = null;

            // OneOf field names bound to virtual getters and setters
            let $oneOfFields;

            /**
             * MetricAggregation metric_aggregation.
             * @member {"average"|"stats"|undefined} metric_aggregation
             * @memberof summa.proto.MetricAggregation
             * @instance
             */
            Object.defineProperty(MetricAggregation.prototype, "metric_aggregation", {
                get: $util.oneOfGetter($oneOfFields = ["average", "stats"]),
                set: $util.oneOfSetter($oneOfFields)
            });

            /**
             * Creates a new MetricAggregation instance using the specified properties.
             * @function create
             * @memberof summa.proto.MetricAggregation
             * @static
             * @param {summa.proto.IMetricAggregation=} [properties] Properties to set
             * @returns {summa.proto.MetricAggregation} MetricAggregation instance
             */
            MetricAggregation.create = function create(properties) {
                return new MetricAggregation(properties);
            };

            return MetricAggregation;
        })();

        proto.AverageAggregation = (function() {

            /**
             * Properties of an AverageAggregation.
             * @memberof summa.proto
             * @interface IAverageAggregation
             * @property {string|null} [field] AverageAggregation field
             */

            /**
             * Constructs a new AverageAggregation.
             * @memberof summa.proto
             * @classdesc Represents an AverageAggregation.
             * @implements IAverageAggregation
             * @constructor
             * @param {summa.proto.IAverageAggregation=} [properties] Properties to set
             */
            function AverageAggregation(properties) {
                if (properties)
                    for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }

            /**
             * AverageAggregation field.
             * @member {string} field
             * @memberof summa.proto.AverageAggregation
             * @instance
             */
            AverageAggregation.prototype.field = "";

            /**
             * Creates a new AverageAggregation instance using the specified properties.
             * @function create
             * @memberof summa.proto.AverageAggregation
             * @static
             * @param {summa.proto.IAverageAggregation=} [properties] Properties to set
             * @returns {summa.proto.AverageAggregation} AverageAggregation instance
             */
            AverageAggregation.create = function create(properties) {
                return new AverageAggregation(properties);
            };

            return AverageAggregation;
        })();

        proto.StatsAggregation = (function() {

            /**
             * Properties of a StatsAggregation.
             * @memberof summa.proto
             * @interface IStatsAggregation
             * @property {string|null} [field] StatsAggregation field
             */

            /**
             * Constructs a new StatsAggregation.
             * @memberof summa.proto
             * @classdesc Represents a StatsAggregation.
             * @implements IStatsAggregation
             * @constructor
             * @param {summa.proto.IStatsAggregation=} [properties] Properties to set
             */
            function StatsAggregation(properties) {
                if (properties)
                    for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }

            /**
             * StatsAggregation field.
             * @member {string} field
             * @memberof summa.proto.StatsAggregation
             * @instance
             */
            StatsAggregation.prototype.field = "";

            /**
             * Creates a new StatsAggregation instance using the specified properties.
             * @function create
             * @memberof summa.proto.StatsAggregation
             * @static
             * @param {summa.proto.IStatsAggregation=} [properties] Properties to set
             * @returns {summa.proto.StatsAggregation} StatsAggregation instance
             */
            StatsAggregation.create = function create(properties) {
                return new StatsAggregation(properties);
            };

            return StatsAggregation;
        })();

        proto.BucketEntry = (function() {

            /**
             * Properties of a BucketEntry.
             * @memberof summa.proto
             * @interface IBucketEntry
             * @property {summa.proto.IKey|null} [key] BucketEntry key
             * @property {number|Long|null} [doc_count] BucketEntry doc_count
             * @property {Object.<string,summa.proto.IAggregationResult>|null} [sub_aggregation] BucketEntry sub_aggregation
             */

            /**
             * Constructs a new BucketEntry.
             * @memberof summa.proto
             * @classdesc Represents a BucketEntry.
             * @implements IBucketEntry
             * @constructor
             * @param {summa.proto.IBucketEntry=} [properties] Properties to set
             */
            function BucketEntry(properties) {
                this.sub_aggregation = {};
                if (properties)
                    for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }

            /**
             * BucketEntry key.
             * @member {summa.proto.IKey|null|undefined} key
             * @memberof summa.proto.BucketEntry
             * @instance
             */
            BucketEntry.prototype.key = null;

            /**
             * BucketEntry doc_count.
             * @member {number|Long} doc_count
             * @memberof summa.proto.BucketEntry
             * @instance
             */
            BucketEntry.prototype.doc_count = $util.Long ? $util.Long.fromBits(0,0,true) : 0;

            /**
             * BucketEntry sub_aggregation.
             * @member {Object.<string,summa.proto.IAggregationResult>} sub_aggregation
             * @memberof summa.proto.BucketEntry
             * @instance
             */
            BucketEntry.prototype.sub_aggregation = $util.emptyObject;

            /**
             * Creates a new BucketEntry instance using the specified properties.
             * @function create
             * @memberof summa.proto.BucketEntry
             * @static
             * @param {summa.proto.IBucketEntry=} [properties] Properties to set
             * @returns {summa.proto.BucketEntry} BucketEntry instance
             */
            BucketEntry.create = function create(properties) {
                return new BucketEntry(properties);
            };

            return BucketEntry;
        })();

        proto.Key = (function() {

            /**
             * Properties of a Key.
             * @memberof summa.proto
             * @interface IKey
             * @property {string|null} [str] Key str
             * @property {number|null} [f64] Key f64
             */

            /**
             * Constructs a new Key.
             * @memberof summa.proto
             * @classdesc Represents a Key.
             * @implements IKey
             * @constructor
             * @param {summa.proto.IKey=} [properties] Properties to set
             */
            function Key(properties) {
                if (properties)
                    for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }

            /**
             * Key str.
             * @member {string|null|undefined} str
             * @memberof summa.proto.Key
             * @instance
             */
            Key.prototype.str = null;

            /**
             * Key f64.
             * @member {number|null|undefined} f64
             * @memberof summa.proto.Key
             * @instance
             */
            Key.prototype.f64 = null;

            // OneOf field names bound to virtual getters and setters
            let $oneOfFields;

            /**
             * Key key.
             * @member {"str"|"f64"|undefined} key
             * @memberof summa.proto.Key
             * @instance
             */
            Object.defineProperty(Key.prototype, "key", {
                get: $util.oneOfGetter($oneOfFields = ["str", "f64"]),
                set: $util.oneOfSetter($oneOfFields)
            });

            /**
             * Creates a new Key instance using the specified properties.
             * @function create
             * @memberof summa.proto.Key
             * @static
             * @param {summa.proto.IKey=} [properties] Properties to set
             * @returns {summa.proto.Key} Key instance
             */
            Key.create = function create(properties) {
                return new Key(properties);
            };

            return Key;
        })();

        /**
         * Occur enum.
         * @name summa.proto.Occur
         * @enum {number}
         * @property {number} should=0 should value
         * @property {number} must=1 must value
         * @property {number} must_not=2 must_not value
         */
        proto.Occur = (function() {
            const valuesById = {}, values = Object.create(valuesById);
            values[valuesById[0] = "should"] = 0;
            values[valuesById[1] = "must"] = 1;
            values[valuesById[2] = "must_not"] = 2;
            return values;
        })();

        proto.Range = (function() {

            /**
             * Properties of a Range.
             * @memberof summa.proto
             * @interface IRange
             * @property {string|null} [left] Range left
             * @property {string|null} [right] Range right
             * @property {boolean|null} [including_left] Range including_left
             * @property {boolean|null} [including_right] Range including_right
             */

            /**
             * Constructs a new Range.
             * @memberof summa.proto
             * @classdesc Represents a Range.
             * @implements IRange
             * @constructor
             * @param {summa.proto.IRange=} [properties] Properties to set
             */
            function Range(properties) {
                if (properties)
                    for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }

            /**
             * Range left.
             * @member {string} left
             * @memberof summa.proto.Range
             * @instance
             */
            Range.prototype.left = "";

            /**
             * Range right.
             * @member {string} right
             * @memberof summa.proto.Range
             * @instance
             */
            Range.prototype.right = "";

            /**
             * Range including_left.
             * @member {boolean} including_left
             * @memberof summa.proto.Range
             * @instance
             */
            Range.prototype.including_left = false;

            /**
             * Range including_right.
             * @member {boolean} including_right
             * @memberof summa.proto.Range
             * @instance
             */
            Range.prototype.including_right = false;

            /**
             * Creates a new Range instance using the specified properties.
             * @function create
             * @memberof summa.proto.Range
             * @static
             * @param {summa.proto.IRange=} [properties] Properties to set
             * @returns {summa.proto.Range} Range instance
             */
            Range.create = function create(properties) {
                return new Range(properties);
            };

            return Range;
        })();

        proto.RangeBucketEntry = (function() {

            /**
             * Properties of a RangeBucketEntry.
             * @memberof summa.proto
             * @interface IRangeBucketEntry
             * @property {summa.proto.IKey|null} [key] RangeBucketEntry key
             * @property {number|Long|null} [doc_count] RangeBucketEntry doc_count
             * @property {Object.<string,summa.proto.IAggregationResult>|null} [sub_aggregation] RangeBucketEntry sub_aggregation
             * @property {number|null} [from] RangeBucketEntry from
             * @property {number|null} [to] RangeBucketEntry to
             */

            /**
             * Constructs a new RangeBucketEntry.
             * @memberof summa.proto
             * @classdesc Represents a RangeBucketEntry.
             * @implements IRangeBucketEntry
             * @constructor
             * @param {summa.proto.IRangeBucketEntry=} [properties] Properties to set
             */
            function RangeBucketEntry(properties) {
                this.sub_aggregation = {};
                if (properties)
                    for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }

            /**
             * RangeBucketEntry key.
             * @member {summa.proto.IKey|null|undefined} key
             * @memberof summa.proto.RangeBucketEntry
             * @instance
             */
            RangeBucketEntry.prototype.key = null;

            /**
             * RangeBucketEntry doc_count.
             * @member {number|Long} doc_count
             * @memberof summa.proto.RangeBucketEntry
             * @instance
             */
            RangeBucketEntry.prototype.doc_count = $util.Long ? $util.Long.fromBits(0,0,true) : 0;

            /**
             * RangeBucketEntry sub_aggregation.
             * @member {Object.<string,summa.proto.IAggregationResult>} sub_aggregation
             * @memberof summa.proto.RangeBucketEntry
             * @instance
             */
            RangeBucketEntry.prototype.sub_aggregation = $util.emptyObject;

            /**
             * RangeBucketEntry from.
             * @member {number|null|undefined} from
             * @memberof summa.proto.RangeBucketEntry
             * @instance
             */
            RangeBucketEntry.prototype.from = null;

            /**
             * RangeBucketEntry to.
             * @member {number|null|undefined} to
             * @memberof summa.proto.RangeBucketEntry
             * @instance
             */
            RangeBucketEntry.prototype.to = null;

            // OneOf field names bound to virtual getters and setters
            let $oneOfFields;

            /**
             * RangeBucketEntry _from.
             * @member {"from"|undefined} _from
             * @memberof summa.proto.RangeBucketEntry
             * @instance
             */
            Object.defineProperty(RangeBucketEntry.prototype, "_from", {
                get: $util.oneOfGetter($oneOfFields = ["from"]),
                set: $util.oneOfSetter($oneOfFields)
            });

            /**
             * RangeBucketEntry _to.
             * @member {"to"|undefined} _to
             * @memberof summa.proto.RangeBucketEntry
             * @instance
             */
            Object.defineProperty(RangeBucketEntry.prototype, "_to", {
                get: $util.oneOfGetter($oneOfFields = ["to"]),
                set: $util.oneOfSetter($oneOfFields)
            });

            /**
             * Creates a new RangeBucketEntry instance using the specified properties.
             * @function create
             * @memberof summa.proto.RangeBucketEntry
             * @static
             * @param {summa.proto.IRangeBucketEntry=} [properties] Properties to set
             * @returns {summa.proto.RangeBucketEntry} RangeBucketEntry instance
             */
            RangeBucketEntry.create = function create(properties) {
                return new RangeBucketEntry(properties);
            };

            return RangeBucketEntry;
        })();

        proto.Score = (function() {

            /**
             * Properties of a Score.
             * @memberof summa.proto
             * @interface IScore
             * @property {number|null} [f64_score] Score f64_score
             * @property {number|Long|null} [u64_score] Score u64_score
             */

            /**
             * Constructs a new Score.
             * @memberof summa.proto
             * @classdesc Represents a Score.
             * @implements IScore
             * @constructor
             * @param {summa.proto.IScore=} [properties] Properties to set
             */
            function Score(properties) {
                if (properties)
                    for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }

            /**
             * Score f64_score.
             * @member {number|null|undefined} f64_score
             * @memberof summa.proto.Score
             * @instance
             */
            Score.prototype.f64_score = null;

            /**
             * Score u64_score.
             * @member {number|Long|null|undefined} u64_score
             * @memberof summa.proto.Score
             * @instance
             */
            Score.prototype.u64_score = null;

            // OneOf field names bound to virtual getters and setters
            let $oneOfFields;

            /**
             * Score score.
             * @member {"f64_score"|"u64_score"|undefined} score
             * @memberof summa.proto.Score
             * @instance
             */
            Object.defineProperty(Score.prototype, "score", {
                get: $util.oneOfGetter($oneOfFields = ["f64_score", "u64_score"]),
                set: $util.oneOfSetter($oneOfFields)
            });

            /**
             * Creates a new Score instance using the specified properties.
             * @function create
             * @memberof summa.proto.Score
             * @static
             * @param {summa.proto.IScore=} [properties] Properties to set
             * @returns {summa.proto.Score} Score instance
             */
            Score.create = function create(properties) {
                return new Score(properties);
            };

            return Score;
        })();

        proto.Highlight = (function() {

            /**
             * Properties of a Highlight.
             * @memberof summa.proto
             * @interface IHighlight
             * @property {number|null} [from] Highlight from
             * @property {number|null} [to] Highlight to
             */

            /**
             * Constructs a new Highlight.
             * @memberof summa.proto
             * @classdesc Represents a Highlight.
             * @implements IHighlight
             * @constructor
             * @param {summa.proto.IHighlight=} [properties] Properties to set
             */
            function Highlight(properties) {
                if (properties)
                    for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }

            /**
             * Highlight from.
             * @member {number} from
             * @memberof summa.proto.Highlight
             * @instance
             */
            Highlight.prototype.from = 0;

            /**
             * Highlight to.
             * @member {number} to
             * @memberof summa.proto.Highlight
             * @instance
             */
            Highlight.prototype.to = 0;

            /**
             * Creates a new Highlight instance using the specified properties.
             * @function create
             * @memberof summa.proto.Highlight
             * @static
             * @param {summa.proto.IHighlight=} [properties] Properties to set
             * @returns {summa.proto.Highlight} Highlight instance
             */
            Highlight.create = function create(properties) {
                return new Highlight(properties);
            };

            return Highlight;
        })();

        proto.Snippet = (function() {

            /**
             * Properties of a Snippet.
             * @memberof summa.proto
             * @interface ISnippet
             * @property {Uint8Array|null} [fragment] Snippet fragment
             * @property {Array.<summa.proto.IHighlight>|null} [highlights] Snippet highlights
             * @property {string|null} [html] Snippet html
             */

            /**
             * Constructs a new Snippet.
             * @memberof summa.proto
             * @classdesc Represents a Snippet.
             * @implements ISnippet
             * @constructor
             * @param {summa.proto.ISnippet=} [properties] Properties to set
             */
            function Snippet(properties) {
                this.highlights = [];
                if (properties)
                    for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }

            /**
             * Snippet fragment.
             * @member {Uint8Array} fragment
             * @memberof summa.proto.Snippet
             * @instance
             */
            Snippet.prototype.fragment = $util.newBuffer([]);

            /**
             * Snippet highlights.
             * @member {Array.<summa.proto.IHighlight>} highlights
             * @memberof summa.proto.Snippet
             * @instance
             */
            Snippet.prototype.highlights = $util.emptyArray;

            /**
             * Snippet html.
             * @member {string} html
             * @memberof summa.proto.Snippet
             * @instance
             */
            Snippet.prototype.html = "";

            /**
             * Creates a new Snippet instance using the specified properties.
             * @function create
             * @memberof summa.proto.Snippet
             * @static
             * @param {summa.proto.ISnippet=} [properties] Properties to set
             * @returns {summa.proto.Snippet} Snippet instance
             */
            Snippet.create = function create(properties) {
                return new Snippet(properties);
            };

            return Snippet;
        })();

        proto.ScoredDocument = (function() {

            /**
             * Properties of a ScoredDocument.
             * @memberof summa.proto
             * @interface IScoredDocument
             * @property {string|null} [document] ScoredDocument document
             * @property {summa.proto.IScore|null} [score] ScoredDocument score
             * @property {number|null} [position] ScoredDocument position
             * @property {Object.<string,summa.proto.ISnippet>|null} [snippets] ScoredDocument snippets
             * @property {string|null} [index_alias] ScoredDocument index_alias
             */

            /**
             * Constructs a new ScoredDocument.
             * @memberof summa.proto
             * @classdesc Represents a ScoredDocument.
             * @implements IScoredDocument
             * @constructor
             * @param {summa.proto.IScoredDocument=} [properties] Properties to set
             */
            function ScoredDocument(properties) {
                this.snippets = {};
                if (properties)
                    for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }

            /**
             * ScoredDocument document.
             * @member {string} document
             * @memberof summa.proto.ScoredDocument
             * @instance
             */
            ScoredDocument.prototype.document = "";

            /**
             * ScoredDocument score.
             * @member {summa.proto.IScore|null|undefined} score
             * @memberof summa.proto.ScoredDocument
             * @instance
             */
            ScoredDocument.prototype.score = null;

            /**
             * ScoredDocument position.
             * @member {number} position
             * @memberof summa.proto.ScoredDocument
             * @instance
             */
            ScoredDocument.prototype.position = 0;

            /**
             * ScoredDocument snippets.
             * @member {Object.<string,summa.proto.ISnippet>} snippets
             * @memberof summa.proto.ScoredDocument
             * @instance
             */
            ScoredDocument.prototype.snippets = $util.emptyObject;

            /**
             * ScoredDocument index_alias.
             * @member {string} index_alias
             * @memberof summa.proto.ScoredDocument
             * @instance
             */
            ScoredDocument.prototype.index_alias = "";

            /**
             * Creates a new ScoredDocument instance using the specified properties.
             * @function create
             * @memberof summa.proto.ScoredDocument
             * @static
             * @param {summa.proto.IScoredDocument=} [properties] Properties to set
             * @returns {summa.proto.ScoredDocument} ScoredDocument instance
             */
            ScoredDocument.create = function create(properties) {
                return new ScoredDocument(properties);
            };

            return ScoredDocument;
        })();

        proto.Scorer = (function() {

            /**
             * Properties of a Scorer.
             * @memberof summa.proto
             * @interface IScorer
             * @property {string|null} [eval_expr] Scorer eval_expr
             * @property {string|null} [order_by] Scorer order_by
             */

            /**
             * Constructs a new Scorer.
             * @memberof summa.proto
             * @classdesc Represents a Scorer.
             * @implements IScorer
             * @constructor
             * @param {summa.proto.IScorer=} [properties] Properties to set
             */
            function Scorer(properties) {
                if (properties)
                    for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }

            /**
             * Scorer eval_expr.
             * @member {string|null|undefined} eval_expr
             * @memberof summa.proto.Scorer
             * @instance
             */
            Scorer.prototype.eval_expr = null;

            /**
             * Scorer order_by.
             * @member {string|null|undefined} order_by
             * @memberof summa.proto.Scorer
             * @instance
             */
            Scorer.prototype.order_by = null;

            // OneOf field names bound to virtual getters and setters
            let $oneOfFields;

            /**
             * Scorer scorer.
             * @member {"eval_expr"|"order_by"|undefined} scorer
             * @memberof summa.proto.Scorer
             * @instance
             */
            Object.defineProperty(Scorer.prototype, "scorer", {
                get: $util.oneOfGetter($oneOfFields = ["eval_expr", "order_by"]),
                set: $util.oneOfSetter($oneOfFields)
            });

            /**
             * Creates a new Scorer instance using the specified properties.
             * @function create
             * @memberof summa.proto.Scorer
             * @static
             * @param {summa.proto.IScorer=} [properties] Properties to set
             * @returns {summa.proto.Scorer} Scorer instance
             */
            Scorer.create = function create(properties) {
                return new Scorer(properties);
            };

            return Scorer;
        })();

        proto.Collector = (function() {

            /**
             * Properties of a Collector.
             * @memberof summa.proto
             * @interface ICollector
             * @property {summa.proto.ITopDocsCollector|null} [top_docs] Collector top_docs
             * @property {summa.proto.IReservoirSamplingCollector|null} [reservoir_sampling] Collector reservoir_sampling
             * @property {summa.proto.ICountCollector|null} [count] Collector count
             * @property {summa.proto.IFacetCollector|null} [facet] Collector facet
             * @property {summa.proto.IAggregationCollector|null} [aggregation] Collector aggregation
             */

            /**
             * Constructs a new Collector.
             * @memberof summa.proto
             * @classdesc Represents a Collector.
             * @implements ICollector
             * @constructor
             * @param {summa.proto.ICollector=} [properties] Properties to set
             */
            function Collector(properties) {
                if (properties)
                    for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }

            /**
             * Collector top_docs.
             * @member {summa.proto.ITopDocsCollector|null|undefined} top_docs
             * @memberof summa.proto.Collector
             * @instance
             */
            Collector.prototype.top_docs = null;

            /**
             * Collector reservoir_sampling.
             * @member {summa.proto.IReservoirSamplingCollector|null|undefined} reservoir_sampling
             * @memberof summa.proto.Collector
             * @instance
             */
            Collector.prototype.reservoir_sampling = null;

            /**
             * Collector count.
             * @member {summa.proto.ICountCollector|null|undefined} count
             * @memberof summa.proto.Collector
             * @instance
             */
            Collector.prototype.count = null;

            /**
             * Collector facet.
             * @member {summa.proto.IFacetCollector|null|undefined} facet
             * @memberof summa.proto.Collector
             * @instance
             */
            Collector.prototype.facet = null;

            /**
             * Collector aggregation.
             * @member {summa.proto.IAggregationCollector|null|undefined} aggregation
             * @memberof summa.proto.Collector
             * @instance
             */
            Collector.prototype.aggregation = null;

            // OneOf field names bound to virtual getters and setters
            let $oneOfFields;

            /**
             * Collector collector.
             * @member {"top_docs"|"reservoir_sampling"|"count"|"facet"|"aggregation"|undefined} collector
             * @memberof summa.proto.Collector
             * @instance
             */
            Object.defineProperty(Collector.prototype, "collector", {
                get: $util.oneOfGetter($oneOfFields = ["top_docs", "reservoir_sampling", "count", "facet", "aggregation"]),
                set: $util.oneOfSetter($oneOfFields)
            });

            /**
             * Creates a new Collector instance using the specified properties.
             * @function create
             * @memberof summa.proto.Collector
             * @static
             * @param {summa.proto.ICollector=} [properties] Properties to set
             * @returns {summa.proto.Collector} Collector instance
             */
            Collector.create = function create(properties) {
                return new Collector(properties);
            };

            return Collector;
        })();

        proto.CollectorOutput = (function() {

            /**
             * Properties of a CollectorOutput.
             * @memberof summa.proto
             * @interface ICollectorOutput
             * @property {summa.proto.IDocumentsCollectorOutput|null} [documents] CollectorOutput documents
             * @property {summa.proto.ICountCollectorOutput|null} [count] CollectorOutput count
             * @property {summa.proto.IFacetCollectorOutput|null} [facet] CollectorOutput facet
             * @property {summa.proto.IAggregationCollectorOutput|null} [aggregation] CollectorOutput aggregation
             */

            /**
             * Constructs a new CollectorOutput.
             * @memberof summa.proto
             * @classdesc Represents a CollectorOutput.
             * @implements ICollectorOutput
             * @constructor
             * @param {summa.proto.ICollectorOutput=} [properties] Properties to set
             */
            function CollectorOutput(properties) {
                if (properties)
                    for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }

            /**
             * CollectorOutput documents.
             * @member {summa.proto.IDocumentsCollectorOutput|null|undefined} documents
             * @memberof summa.proto.CollectorOutput
             * @instance
             */
            CollectorOutput.prototype.documents = null;

            /**
             * CollectorOutput count.
             * @member {summa.proto.ICountCollectorOutput|null|undefined} count
             * @memberof summa.proto.CollectorOutput
             * @instance
             */
            CollectorOutput.prototype.count = null;

            /**
             * CollectorOutput facet.
             * @member {summa.proto.IFacetCollectorOutput|null|undefined} facet
             * @memberof summa.proto.CollectorOutput
             * @instance
             */
            CollectorOutput.prototype.facet = null;

            /**
             * CollectorOutput aggregation.
             * @member {summa.proto.IAggregationCollectorOutput|null|undefined} aggregation
             * @memberof summa.proto.CollectorOutput
             * @instance
             */
            CollectorOutput.prototype.aggregation = null;

            // OneOf field names bound to virtual getters and setters
            let $oneOfFields;

            /**
             * CollectorOutput collector_output.
             * @member {"documents"|"count"|"facet"|"aggregation"|undefined} collector_output
             * @memberof summa.proto.CollectorOutput
             * @instance
             */
            Object.defineProperty(CollectorOutput.prototype, "collector_output", {
                get: $util.oneOfGetter($oneOfFields = ["documents", "count", "facet", "aggregation"]),
                set: $util.oneOfSetter($oneOfFields)
            });

            /**
             * Creates a new CollectorOutput instance using the specified properties.
             * @function create
             * @memberof summa.proto.CollectorOutput
             * @static
             * @param {summa.proto.ICollectorOutput=} [properties] Properties to set
             * @returns {summa.proto.CollectorOutput} CollectorOutput instance
             */
            CollectorOutput.create = function create(properties) {
                return new CollectorOutput(properties);
            };

            return CollectorOutput;
        })();

        proto.CountCollector = (function() {

            /**
             * Properties of a CountCollector.
             * @memberof summa.proto
             * @interface ICountCollector
             */

            /**
             * Constructs a new CountCollector.
             * @memberof summa.proto
             * @classdesc Represents a CountCollector.
             * @implements ICountCollector
             * @constructor
             * @param {summa.proto.ICountCollector=} [properties] Properties to set
             */
            function CountCollector(properties) {
                if (properties)
                    for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }

            /**
             * Creates a new CountCollector instance using the specified properties.
             * @function create
             * @memberof summa.proto.CountCollector
             * @static
             * @param {summa.proto.ICountCollector=} [properties] Properties to set
             * @returns {summa.proto.CountCollector} CountCollector instance
             */
            CountCollector.create = function create(properties) {
                return new CountCollector(properties);
            };

            return CountCollector;
        })();

        proto.CountCollectorOutput = (function() {

            /**
             * Properties of a CountCollectorOutput.
             * @memberof summa.proto
             * @interface ICountCollectorOutput
             * @property {number|null} [count] CountCollectorOutput count
             */

            /**
             * Constructs a new CountCollectorOutput.
             * @memberof summa.proto
             * @classdesc Represents a CountCollectorOutput.
             * @implements ICountCollectorOutput
             * @constructor
             * @param {summa.proto.ICountCollectorOutput=} [properties] Properties to set
             */
            function CountCollectorOutput(properties) {
                if (properties)
                    for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }

            /**
             * CountCollectorOutput count.
             * @member {number} count
             * @memberof summa.proto.CountCollectorOutput
             * @instance
             */
            CountCollectorOutput.prototype.count = 0;

            /**
             * Creates a new CountCollectorOutput instance using the specified properties.
             * @function create
             * @memberof summa.proto.CountCollectorOutput
             * @static
             * @param {summa.proto.ICountCollectorOutput=} [properties] Properties to set
             * @returns {summa.proto.CountCollectorOutput} CountCollectorOutput instance
             */
            CountCollectorOutput.create = function create(properties) {
                return new CountCollectorOutput(properties);
            };

            return CountCollectorOutput;
        })();

        proto.FacetCollector = (function() {

            /**
             * Properties of a FacetCollector.
             * @memberof summa.proto
             * @interface IFacetCollector
             * @property {string|null} [field] FacetCollector field
             * @property {Array.<string>|null} [facets] FacetCollector facets
             */

            /**
             * Constructs a new FacetCollector.
             * @memberof summa.proto
             * @classdesc Represents a FacetCollector.
             * @implements IFacetCollector
             * @constructor
             * @param {summa.proto.IFacetCollector=} [properties] Properties to set
             */
            function FacetCollector(properties) {
                this.facets = [];
                if (properties)
                    for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }

            /**
             * FacetCollector field.
             * @member {string} field
             * @memberof summa.proto.FacetCollector
             * @instance
             */
            FacetCollector.prototype.field = "";

            /**
             * FacetCollector facets.
             * @member {Array.<string>} facets
             * @memberof summa.proto.FacetCollector
             * @instance
             */
            FacetCollector.prototype.facets = $util.emptyArray;

            /**
             * Creates a new FacetCollector instance using the specified properties.
             * @function create
             * @memberof summa.proto.FacetCollector
             * @static
             * @param {summa.proto.IFacetCollector=} [properties] Properties to set
             * @returns {summa.proto.FacetCollector} FacetCollector instance
             */
            FacetCollector.create = function create(properties) {
                return new FacetCollector(properties);
            };

            return FacetCollector;
        })();

        proto.FacetCollectorOutput = (function() {

            /**
             * Properties of a FacetCollectorOutput.
             * @memberof summa.proto
             * @interface IFacetCollectorOutput
             * @property {Object.<string,number|Long>|null} [facet_counts] FacetCollectorOutput facet_counts
             */

            /**
             * Constructs a new FacetCollectorOutput.
             * @memberof summa.proto
             * @classdesc Represents a FacetCollectorOutput.
             * @implements IFacetCollectorOutput
             * @constructor
             * @param {summa.proto.IFacetCollectorOutput=} [properties] Properties to set
             */
            function FacetCollectorOutput(properties) {
                this.facet_counts = {};
                if (properties)
                    for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }

            /**
             * FacetCollectorOutput facet_counts.
             * @member {Object.<string,number|Long>} facet_counts
             * @memberof summa.proto.FacetCollectorOutput
             * @instance
             */
            FacetCollectorOutput.prototype.facet_counts = $util.emptyObject;

            /**
             * Creates a new FacetCollectorOutput instance using the specified properties.
             * @function create
             * @memberof summa.proto.FacetCollectorOutput
             * @static
             * @param {summa.proto.IFacetCollectorOutput=} [properties] Properties to set
             * @returns {summa.proto.FacetCollectorOutput} FacetCollectorOutput instance
             */
            FacetCollectorOutput.create = function create(properties) {
                return new FacetCollectorOutput(properties);
            };

            return FacetCollectorOutput;
        })();

        proto.ReservoirSamplingCollector = (function() {

            /**
             * Properties of a ReservoirSamplingCollector.
             * @memberof summa.proto
             * @interface IReservoirSamplingCollector
             * @property {number|null} [limit] ReservoirSamplingCollector limit
             * @property {Array.<string>|null} [fields] ReservoirSamplingCollector fields
             */

            /**
             * Constructs a new ReservoirSamplingCollector.
             * @memberof summa.proto
             * @classdesc Represents a ReservoirSamplingCollector.
             * @implements IReservoirSamplingCollector
             * @constructor
             * @param {summa.proto.IReservoirSamplingCollector=} [properties] Properties to set
             */
            function ReservoirSamplingCollector(properties) {
                this.fields = [];
                if (properties)
                    for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }

            /**
             * ReservoirSamplingCollector limit.
             * @member {number} limit
             * @memberof summa.proto.ReservoirSamplingCollector
             * @instance
             */
            ReservoirSamplingCollector.prototype.limit = 0;

            /**
             * ReservoirSamplingCollector fields.
             * @member {Array.<string>} fields
             * @memberof summa.proto.ReservoirSamplingCollector
             * @instance
             */
            ReservoirSamplingCollector.prototype.fields = $util.emptyArray;

            /**
             * Creates a new ReservoirSamplingCollector instance using the specified properties.
             * @function create
             * @memberof summa.proto.ReservoirSamplingCollector
             * @static
             * @param {summa.proto.IReservoirSamplingCollector=} [properties] Properties to set
             * @returns {summa.proto.ReservoirSamplingCollector} ReservoirSamplingCollector instance
             */
            ReservoirSamplingCollector.create = function create(properties) {
                return new ReservoirSamplingCollector(properties);
            };

            return ReservoirSamplingCollector;
        })();

        proto.RandomDocument = (function() {

            /**
             * Properties of a RandomDocument.
             * @memberof summa.proto
             * @interface IRandomDocument
             * @property {string|null} [document] RandomDocument document
             * @property {summa.proto.IScore|null} [score] RandomDocument score
             * @property {string|null} [index_alias] RandomDocument index_alias
             */

            /**
             * Constructs a new RandomDocument.
             * @memberof summa.proto
             * @classdesc Represents a RandomDocument.
             * @implements IRandomDocument
             * @constructor
             * @param {summa.proto.IRandomDocument=} [properties] Properties to set
             */
            function RandomDocument(properties) {
                if (properties)
                    for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }

            /**
             * RandomDocument document.
             * @member {string} document
             * @memberof summa.proto.RandomDocument
             * @instance
             */
            RandomDocument.prototype.document = "";

            /**
             * RandomDocument score.
             * @member {summa.proto.IScore|null|undefined} score
             * @memberof summa.proto.RandomDocument
             * @instance
             */
            RandomDocument.prototype.score = null;

            /**
             * RandomDocument index_alias.
             * @member {string} index_alias
             * @memberof summa.proto.RandomDocument
             * @instance
             */
            RandomDocument.prototype.index_alias = "";

            /**
             * Creates a new RandomDocument instance using the specified properties.
             * @function create
             * @memberof summa.proto.RandomDocument
             * @static
             * @param {summa.proto.IRandomDocument=} [properties] Properties to set
             * @returns {summa.proto.RandomDocument} RandomDocument instance
             */
            RandomDocument.create = function create(properties) {
                return new RandomDocument(properties);
            };

            return RandomDocument;
        })();

        proto.ReservoirSamplingCollectorOutput = (function() {

            /**
             * Properties of a ReservoirSamplingCollectorOutput.
             * @memberof summa.proto
             * @interface IReservoirSamplingCollectorOutput
             * @property {Array.<summa.proto.IRandomDocument>|null} [documents] ReservoirSamplingCollectorOutput documents
             */

            /**
             * Constructs a new ReservoirSamplingCollectorOutput.
             * @memberof summa.proto
             * @classdesc Represents a ReservoirSamplingCollectorOutput.
             * @implements IReservoirSamplingCollectorOutput
             * @constructor
             * @param {summa.proto.IReservoirSamplingCollectorOutput=} [properties] Properties to set
             */
            function ReservoirSamplingCollectorOutput(properties) {
                this.documents = [];
                if (properties)
                    for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }

            /**
             * ReservoirSamplingCollectorOutput documents.
             * @member {Array.<summa.proto.IRandomDocument>} documents
             * @memberof summa.proto.ReservoirSamplingCollectorOutput
             * @instance
             */
            ReservoirSamplingCollectorOutput.prototype.documents = $util.emptyArray;

            /**
             * Creates a new ReservoirSamplingCollectorOutput instance using the specified properties.
             * @function create
             * @memberof summa.proto.ReservoirSamplingCollectorOutput
             * @static
             * @param {summa.proto.IReservoirSamplingCollectorOutput=} [properties] Properties to set
             * @returns {summa.proto.ReservoirSamplingCollectorOutput} ReservoirSamplingCollectorOutput instance
             */
            ReservoirSamplingCollectorOutput.create = function create(properties) {
                return new ReservoirSamplingCollectorOutput(properties);
            };

            return ReservoirSamplingCollectorOutput;
        })();

        proto.TopDocsCollector = (function() {

            /**
             * Properties of a TopDocsCollector.
             * @memberof summa.proto
             * @interface ITopDocsCollector
             * @property {number|null} [limit] TopDocsCollector limit
             * @property {number|null} [offset] TopDocsCollector offset
             * @property {summa.proto.IScorer|null} [scorer] TopDocsCollector scorer
             * @property {Object.<string,number>|null} [snippet_configs] TopDocsCollector snippet_configs
             * @property {boolean|null} [explain] TopDocsCollector explain
             * @property {Array.<string>|null} [fields] TopDocsCollector fields
             */

            /**
             * Constructs a new TopDocsCollector.
             * @memberof summa.proto
             * @classdesc Represents a TopDocsCollector.
             * @implements ITopDocsCollector
             * @constructor
             * @param {summa.proto.ITopDocsCollector=} [properties] Properties to set
             */
            function TopDocsCollector(properties) {
                this.snippet_configs = {};
                this.fields = [];
                if (properties)
                    for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }

            /**
             * TopDocsCollector limit.
             * @member {number} limit
             * @memberof summa.proto.TopDocsCollector
             * @instance
             */
            TopDocsCollector.prototype.limit = 0;

            /**
             * TopDocsCollector offset.
             * @member {number} offset
             * @memberof summa.proto.TopDocsCollector
             * @instance
             */
            TopDocsCollector.prototype.offset = 0;

            /**
             * TopDocsCollector scorer.
             * @member {summa.proto.IScorer|null|undefined} scorer
             * @memberof summa.proto.TopDocsCollector
             * @instance
             */
            TopDocsCollector.prototype.scorer = null;

            /**
             * TopDocsCollector snippet_configs.
             * @member {Object.<string,number>} snippet_configs
             * @memberof summa.proto.TopDocsCollector
             * @instance
             */
            TopDocsCollector.prototype.snippet_configs = $util.emptyObject;

            /**
             * TopDocsCollector explain.
             * @member {boolean} explain
             * @memberof summa.proto.TopDocsCollector
             * @instance
             */
            TopDocsCollector.prototype.explain = false;

            /**
             * TopDocsCollector fields.
             * @member {Array.<string>} fields
             * @memberof summa.proto.TopDocsCollector
             * @instance
             */
            TopDocsCollector.prototype.fields = $util.emptyArray;

            // OneOf field names bound to virtual getters and setters
            let $oneOfFields;

            /**
             * TopDocsCollector _scorer.
             * @member {"scorer"|undefined} _scorer
             * @memberof summa.proto.TopDocsCollector
             * @instance
             */
            Object.defineProperty(TopDocsCollector.prototype, "_scorer", {
                get: $util.oneOfGetter($oneOfFields = ["scorer"]),
                set: $util.oneOfSetter($oneOfFields)
            });

            /**
             * Creates a new TopDocsCollector instance using the specified properties.
             * @function create
             * @memberof summa.proto.TopDocsCollector
             * @static
             * @param {summa.proto.ITopDocsCollector=} [properties] Properties to set
             * @returns {summa.proto.TopDocsCollector} TopDocsCollector instance
             */
            TopDocsCollector.create = function create(properties) {
                return new TopDocsCollector(properties);
            };

            return TopDocsCollector;
        })();

        proto.DocumentsCollectorOutput = (function() {

            /**
             * Properties of a DocumentsCollectorOutput.
             * @memberof summa.proto
             * @interface IDocumentsCollectorOutput
             * @property {Array.<summa.proto.IScoredDocument>|null} [scored_documents] DocumentsCollectorOutput scored_documents
             * @property {boolean|null} [has_next] DocumentsCollectorOutput has_next
             */

            /**
             * Constructs a new DocumentsCollectorOutput.
             * @memberof summa.proto
             * @classdesc Represents a DocumentsCollectorOutput.
             * @implements IDocumentsCollectorOutput
             * @constructor
             * @param {summa.proto.IDocumentsCollectorOutput=} [properties] Properties to set
             */
            function DocumentsCollectorOutput(properties) {
                this.scored_documents = [];
                if (properties)
                    for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }

            /**
             * DocumentsCollectorOutput scored_documents.
             * @member {Array.<summa.proto.IScoredDocument>} scored_documents
             * @memberof summa.proto.DocumentsCollectorOutput
             * @instance
             */
            DocumentsCollectorOutput.prototype.scored_documents = $util.emptyArray;

            /**
             * DocumentsCollectorOutput has_next.
             * @member {boolean} has_next
             * @memberof summa.proto.DocumentsCollectorOutput
             * @instance
             */
            DocumentsCollectorOutput.prototype.has_next = false;

            /**
             * Creates a new DocumentsCollectorOutput instance using the specified properties.
             * @function create
             * @memberof summa.proto.DocumentsCollectorOutput
             * @static
             * @param {summa.proto.IDocumentsCollectorOutput=} [properties] Properties to set
             * @returns {summa.proto.DocumentsCollectorOutput} DocumentsCollectorOutput instance
             */
            DocumentsCollectorOutput.create = function create(properties) {
                return new DocumentsCollectorOutput(properties);
            };

            return DocumentsCollectorOutput;
        })();

        proto.AggregationCollector = (function() {

            /**
             * Properties of an AggregationCollector.
             * @memberof summa.proto
             * @interface IAggregationCollector
             * @property {Object.<string,summa.proto.IAggregation>|null} [aggregations] AggregationCollector aggregations
             */

            /**
             * Constructs a new AggregationCollector.
             * @memberof summa.proto
             * @classdesc Represents an AggregationCollector.
             * @implements IAggregationCollector
             * @constructor
             * @param {summa.proto.IAggregationCollector=} [properties] Properties to set
             */
            function AggregationCollector(properties) {
                this.aggregations = {};
                if (properties)
                    for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }

            /**
             * AggregationCollector aggregations.
             * @member {Object.<string,summa.proto.IAggregation>} aggregations
             * @memberof summa.proto.AggregationCollector
             * @instance
             */
            AggregationCollector.prototype.aggregations = $util.emptyObject;

            /**
             * Creates a new AggregationCollector instance using the specified properties.
             * @function create
             * @memberof summa.proto.AggregationCollector
             * @static
             * @param {summa.proto.IAggregationCollector=} [properties] Properties to set
             * @returns {summa.proto.AggregationCollector} AggregationCollector instance
             */
            AggregationCollector.create = function create(properties) {
                return new AggregationCollector(properties);
            };

            return AggregationCollector;
        })();

        proto.AggregationCollectorOutput = (function() {

            /**
             * Properties of an AggregationCollectorOutput.
             * @memberof summa.proto
             * @interface IAggregationCollectorOutput
             * @property {Object.<string,summa.proto.IAggregationResult>|null} [aggregation_results] AggregationCollectorOutput aggregation_results
             */

            /**
             * Constructs a new AggregationCollectorOutput.
             * @memberof summa.proto
             * @classdesc Represents an AggregationCollectorOutput.
             * @implements IAggregationCollectorOutput
             * @constructor
             * @param {summa.proto.IAggregationCollectorOutput=} [properties] Properties to set
             */
            function AggregationCollectorOutput(properties) {
                this.aggregation_results = {};
                if (properties)
                    for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }

            /**
             * AggregationCollectorOutput aggregation_results.
             * @member {Object.<string,summa.proto.IAggregationResult>} aggregation_results
             * @memberof summa.proto.AggregationCollectorOutput
             * @instance
             */
            AggregationCollectorOutput.prototype.aggregation_results = $util.emptyObject;

            /**
             * Creates a new AggregationCollectorOutput instance using the specified properties.
             * @function create
             * @memberof summa.proto.AggregationCollectorOutput
             * @static
             * @param {summa.proto.IAggregationCollectorOutput=} [properties] Properties to set
             * @returns {summa.proto.AggregationCollectorOutput} AggregationCollectorOutput instance
             */
            AggregationCollectorOutput.create = function create(properties) {
                return new AggregationCollectorOutput(properties);
            };

            return AggregationCollectorOutput;
        })();

        proto.AggregationResult = (function() {

            /**
             * Properties of an AggregationResult.
             * @memberof summa.proto
             * @interface IAggregationResult
             * @property {summa.proto.IBucketResult|null} [bucket] AggregationResult bucket
             * @property {summa.proto.IMetricResult|null} [metric] AggregationResult metric
             */

            /**
             * Constructs a new AggregationResult.
             * @memberof summa.proto
             * @classdesc Represents an AggregationResult.
             * @implements IAggregationResult
             * @constructor
             * @param {summa.proto.IAggregationResult=} [properties] Properties to set
             */
            function AggregationResult(properties) {
                if (properties)
                    for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }

            /**
             * AggregationResult bucket.
             * @member {summa.proto.IBucketResult|null|undefined} bucket
             * @memberof summa.proto.AggregationResult
             * @instance
             */
            AggregationResult.prototype.bucket = null;

            /**
             * AggregationResult metric.
             * @member {summa.proto.IMetricResult|null|undefined} metric
             * @memberof summa.proto.AggregationResult
             * @instance
             */
            AggregationResult.prototype.metric = null;

            // OneOf field names bound to virtual getters and setters
            let $oneOfFields;

            /**
             * AggregationResult aggregation_result.
             * @member {"bucket"|"metric"|undefined} aggregation_result
             * @memberof summa.proto.AggregationResult
             * @instance
             */
            Object.defineProperty(AggregationResult.prototype, "aggregation_result", {
                get: $util.oneOfGetter($oneOfFields = ["bucket", "metric"]),
                set: $util.oneOfSetter($oneOfFields)
            });

            /**
             * Creates a new AggregationResult instance using the specified properties.
             * @function create
             * @memberof summa.proto.AggregationResult
             * @static
             * @param {summa.proto.IAggregationResult=} [properties] Properties to set
             * @returns {summa.proto.AggregationResult} AggregationResult instance
             */
            AggregationResult.create = function create(properties) {
                return new AggregationResult(properties);
            };

            return AggregationResult;
        })();

        proto.BucketResult = (function() {

            /**
             * Properties of a BucketResult.
             * @memberof summa.proto
             * @interface IBucketResult
             * @property {summa.proto.IRangeResult|null} [range] BucketResult range
             * @property {summa.proto.IHistogramResult|null} [histogram] BucketResult histogram
             * @property {summa.proto.ITermsResult|null} [terms] BucketResult terms
             */

            /**
             * Constructs a new BucketResult.
             * @memberof summa.proto
             * @classdesc Represents a BucketResult.
             * @implements IBucketResult
             * @constructor
             * @param {summa.proto.IBucketResult=} [properties] Properties to set
             */
            function BucketResult(properties) {
                if (properties)
                    for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }

            /**
             * BucketResult range.
             * @member {summa.proto.IRangeResult|null|undefined} range
             * @memberof summa.proto.BucketResult
             * @instance
             */
            BucketResult.prototype.range = null;

            /**
             * BucketResult histogram.
             * @member {summa.proto.IHistogramResult|null|undefined} histogram
             * @memberof summa.proto.BucketResult
             * @instance
             */
            BucketResult.prototype.histogram = null;

            /**
             * BucketResult terms.
             * @member {summa.proto.ITermsResult|null|undefined} terms
             * @memberof summa.proto.BucketResult
             * @instance
             */
            BucketResult.prototype.terms = null;

            // OneOf field names bound to virtual getters and setters
            let $oneOfFields;

            /**
             * BucketResult bucket_result.
             * @member {"range"|"histogram"|"terms"|undefined} bucket_result
             * @memberof summa.proto.BucketResult
             * @instance
             */
            Object.defineProperty(BucketResult.prototype, "bucket_result", {
                get: $util.oneOfGetter($oneOfFields = ["range", "histogram", "terms"]),
                set: $util.oneOfSetter($oneOfFields)
            });

            /**
             * Creates a new BucketResult instance using the specified properties.
             * @function create
             * @memberof summa.proto.BucketResult
             * @static
             * @param {summa.proto.IBucketResult=} [properties] Properties to set
             * @returns {summa.proto.BucketResult} BucketResult instance
             */
            BucketResult.create = function create(properties) {
                return new BucketResult(properties);
            };

            return BucketResult;
        })();

        proto.RangeResult = (function() {

            /**
             * Properties of a RangeResult.
             * @memberof summa.proto
             * @interface IRangeResult
             * @property {Array.<summa.proto.IRangeBucketEntry>|null} [buckets] RangeResult buckets
             */

            /**
             * Constructs a new RangeResult.
             * @memberof summa.proto
             * @classdesc Represents a RangeResult.
             * @implements IRangeResult
             * @constructor
             * @param {summa.proto.IRangeResult=} [properties] Properties to set
             */
            function RangeResult(properties) {
                this.buckets = [];
                if (properties)
                    for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }

            /**
             * RangeResult buckets.
             * @member {Array.<summa.proto.IRangeBucketEntry>} buckets
             * @memberof summa.proto.RangeResult
             * @instance
             */
            RangeResult.prototype.buckets = $util.emptyArray;

            /**
             * Creates a new RangeResult instance using the specified properties.
             * @function create
             * @memberof summa.proto.RangeResult
             * @static
             * @param {summa.proto.IRangeResult=} [properties] Properties to set
             * @returns {summa.proto.RangeResult} RangeResult instance
             */
            RangeResult.create = function create(properties) {
                return new RangeResult(properties);
            };

            return RangeResult;
        })();

        proto.HistogramResult = (function() {

            /**
             * Properties of a HistogramResult.
             * @memberof summa.proto
             * @interface IHistogramResult
             * @property {Array.<summa.proto.IBucketEntry>|null} [buckets] HistogramResult buckets
             */

            /**
             * Constructs a new HistogramResult.
             * @memberof summa.proto
             * @classdesc Represents a HistogramResult.
             * @implements IHistogramResult
             * @constructor
             * @param {summa.proto.IHistogramResult=} [properties] Properties to set
             */
            function HistogramResult(properties) {
                this.buckets = [];
                if (properties)
                    for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }

            /**
             * HistogramResult buckets.
             * @member {Array.<summa.proto.IBucketEntry>} buckets
             * @memberof summa.proto.HistogramResult
             * @instance
             */
            HistogramResult.prototype.buckets = $util.emptyArray;

            /**
             * Creates a new HistogramResult instance using the specified properties.
             * @function create
             * @memberof summa.proto.HistogramResult
             * @static
             * @param {summa.proto.IHistogramResult=} [properties] Properties to set
             * @returns {summa.proto.HistogramResult} HistogramResult instance
             */
            HistogramResult.create = function create(properties) {
                return new HistogramResult(properties);
            };

            return HistogramResult;
        })();

        proto.TermsResult = (function() {

            /**
             * Properties of a TermsResult.
             * @memberof summa.proto
             * @interface ITermsResult
             * @property {Array.<summa.proto.IBucketEntry>|null} [buckets] TermsResult buckets
             * @property {number|Long|null} [sum_other_doc_count] TermsResult sum_other_doc_count
             * @property {number|Long|null} [doc_count_error_upper_bound] TermsResult doc_count_error_upper_bound
             */

            /**
             * Constructs a new TermsResult.
             * @memberof summa.proto
             * @classdesc Represents a TermsResult.
             * @implements ITermsResult
             * @constructor
             * @param {summa.proto.ITermsResult=} [properties] Properties to set
             */
            function TermsResult(properties) {
                this.buckets = [];
                if (properties)
                    for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }

            /**
             * TermsResult buckets.
             * @member {Array.<summa.proto.IBucketEntry>} buckets
             * @memberof summa.proto.TermsResult
             * @instance
             */
            TermsResult.prototype.buckets = $util.emptyArray;

            /**
             * TermsResult sum_other_doc_count.
             * @member {number|Long} sum_other_doc_count
             * @memberof summa.proto.TermsResult
             * @instance
             */
            TermsResult.prototype.sum_other_doc_count = $util.Long ? $util.Long.fromBits(0,0,true) : 0;

            /**
             * TermsResult doc_count_error_upper_bound.
             * @member {number|Long|null|undefined} doc_count_error_upper_bound
             * @memberof summa.proto.TermsResult
             * @instance
             */
            TermsResult.prototype.doc_count_error_upper_bound = null;

            // OneOf field names bound to virtual getters and setters
            let $oneOfFields;

            /**
             * TermsResult _doc_count_error_upper_bound.
             * @member {"doc_count_error_upper_bound"|undefined} _doc_count_error_upper_bound
             * @memberof summa.proto.TermsResult
             * @instance
             */
            Object.defineProperty(TermsResult.prototype, "_doc_count_error_upper_bound", {
                get: $util.oneOfGetter($oneOfFields = ["doc_count_error_upper_bound"]),
                set: $util.oneOfSetter($oneOfFields)
            });

            /**
             * Creates a new TermsResult instance using the specified properties.
             * @function create
             * @memberof summa.proto.TermsResult
             * @static
             * @param {summa.proto.ITermsResult=} [properties] Properties to set
             * @returns {summa.proto.TermsResult} TermsResult instance
             */
            TermsResult.create = function create(properties) {
                return new TermsResult(properties);
            };

            return TermsResult;
        })();

        proto.MetricResult = (function() {

            /**
             * Properties of a MetricResult.
             * @memberof summa.proto
             * @interface IMetricResult
             * @property {summa.proto.ISingleMetricResult|null} [single_metric] MetricResult single_metric
             * @property {summa.proto.IStatsResult|null} [stats] MetricResult stats
             */

            /**
             * Constructs a new MetricResult.
             * @memberof summa.proto
             * @classdesc Represents a MetricResult.
             * @implements IMetricResult
             * @constructor
             * @param {summa.proto.IMetricResult=} [properties] Properties to set
             */
            function MetricResult(properties) {
                if (properties)
                    for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }

            /**
             * MetricResult single_metric.
             * @member {summa.proto.ISingleMetricResult|null|undefined} single_metric
             * @memberof summa.proto.MetricResult
             * @instance
             */
            MetricResult.prototype.single_metric = null;

            /**
             * MetricResult stats.
             * @member {summa.proto.IStatsResult|null|undefined} stats
             * @memberof summa.proto.MetricResult
             * @instance
             */
            MetricResult.prototype.stats = null;

            // OneOf field names bound to virtual getters and setters
            let $oneOfFields;

            /**
             * MetricResult metric_result.
             * @member {"single_metric"|"stats"|undefined} metric_result
             * @memberof summa.proto.MetricResult
             * @instance
             */
            Object.defineProperty(MetricResult.prototype, "metric_result", {
                get: $util.oneOfGetter($oneOfFields = ["single_metric", "stats"]),
                set: $util.oneOfSetter($oneOfFields)
            });

            /**
             * Creates a new MetricResult instance using the specified properties.
             * @function create
             * @memberof summa.proto.MetricResult
             * @static
             * @param {summa.proto.IMetricResult=} [properties] Properties to set
             * @returns {summa.proto.MetricResult} MetricResult instance
             */
            MetricResult.create = function create(properties) {
                return new MetricResult(properties);
            };

            return MetricResult;
        })();

        proto.SingleMetricResult = (function() {

            /**
             * Properties of a SingleMetricResult.
             * @memberof summa.proto
             * @interface ISingleMetricResult
             * @property {number|null} [value] SingleMetricResult value
             */

            /**
             * Constructs a new SingleMetricResult.
             * @memberof summa.proto
             * @classdesc Represents a SingleMetricResult.
             * @implements ISingleMetricResult
             * @constructor
             * @param {summa.proto.ISingleMetricResult=} [properties] Properties to set
             */
            function SingleMetricResult(properties) {
                if (properties)
                    for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }

            /**
             * SingleMetricResult value.
             * @member {number|null|undefined} value
             * @memberof summa.proto.SingleMetricResult
             * @instance
             */
            SingleMetricResult.prototype.value = null;

            // OneOf field names bound to virtual getters and setters
            let $oneOfFields;

            /**
             * SingleMetricResult _value.
             * @member {"value"|undefined} _value
             * @memberof summa.proto.SingleMetricResult
             * @instance
             */
            Object.defineProperty(SingleMetricResult.prototype, "_value", {
                get: $util.oneOfGetter($oneOfFields = ["value"]),
                set: $util.oneOfSetter($oneOfFields)
            });

            /**
             * Creates a new SingleMetricResult instance using the specified properties.
             * @function create
             * @memberof summa.proto.SingleMetricResult
             * @static
             * @param {summa.proto.ISingleMetricResult=} [properties] Properties to set
             * @returns {summa.proto.SingleMetricResult} SingleMetricResult instance
             */
            SingleMetricResult.create = function create(properties) {
                return new SingleMetricResult(properties);
            };

            return SingleMetricResult;
        })();

        proto.StatsResult = (function() {

            /**
             * Properties of a StatsResult.
             * @memberof summa.proto
             * @interface IStatsResult
             * @property {number|Long|null} [count] StatsResult count
             * @property {number|null} [sum] StatsResult sum
             * @property {number|null} [min] StatsResult min
             * @property {number|null} [max] StatsResult max
             * @property {number|null} [avg] StatsResult avg
             */

            /**
             * Constructs a new StatsResult.
             * @memberof summa.proto
             * @classdesc Represents a StatsResult.
             * @implements IStatsResult
             * @constructor
             * @param {summa.proto.IStatsResult=} [properties] Properties to set
             */
            function StatsResult(properties) {
                if (properties)
                    for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }

            /**
             * StatsResult count.
             * @member {number|Long} count
             * @memberof summa.proto.StatsResult
             * @instance
             */
            StatsResult.prototype.count = $util.Long ? $util.Long.fromBits(0,0,true) : 0;

            /**
             * StatsResult sum.
             * @member {number} sum
             * @memberof summa.proto.StatsResult
             * @instance
             */
            StatsResult.prototype.sum = 0;

            /**
             * StatsResult min.
             * @member {number|null|undefined} min
             * @memberof summa.proto.StatsResult
             * @instance
             */
            StatsResult.prototype.min = null;

            /**
             * StatsResult max.
             * @member {number|null|undefined} max
             * @memberof summa.proto.StatsResult
             * @instance
             */
            StatsResult.prototype.max = null;

            /**
             * StatsResult avg.
             * @member {number|null|undefined} avg
             * @memberof summa.proto.StatsResult
             * @instance
             */
            StatsResult.prototype.avg = null;

            // OneOf field names bound to virtual getters and setters
            let $oneOfFields;

            /**
             * StatsResult _min.
             * @member {"min"|undefined} _min
             * @memberof summa.proto.StatsResult
             * @instance
             */
            Object.defineProperty(StatsResult.prototype, "_min", {
                get: $util.oneOfGetter($oneOfFields = ["min"]),
                set: $util.oneOfSetter($oneOfFields)
            });

            /**
             * StatsResult _max.
             * @member {"max"|undefined} _max
             * @memberof summa.proto.StatsResult
             * @instance
             */
            Object.defineProperty(StatsResult.prototype, "_max", {
                get: $util.oneOfGetter($oneOfFields = ["max"]),
                set: $util.oneOfSetter($oneOfFields)
            });

            /**
             * StatsResult _avg.
             * @member {"avg"|undefined} _avg
             * @memberof summa.proto.StatsResult
             * @instance
             */
            Object.defineProperty(StatsResult.prototype, "_avg", {
                get: $util.oneOfGetter($oneOfFields = ["avg"]),
                set: $util.oneOfSetter($oneOfFields)
            });

            /**
             * Creates a new StatsResult instance using the specified properties.
             * @function create
             * @memberof summa.proto.StatsResult
             * @static
             * @param {summa.proto.IStatsResult=} [properties] Properties to set
             * @returns {summa.proto.StatsResult} StatsResult instance
             */
            StatsResult.create = function create(properties) {
                return new StatsResult(properties);
            };

            return StatsResult;
        })();

        /**
         * Order enum.
         * @name summa.proto.Order
         * @enum {number}
         * @property {number} Asc=0 Asc value
         * @property {number} Desc=1 Desc value
         */
        proto.Order = (function() {
            const valuesById = {}, values = Object.create(valuesById);
            values[valuesById[0] = "Asc"] = 0;
            values[valuesById[1] = "Desc"] = 1;
            return values;
        })();

        proto.Empty = (function() {

            /**
             * Properties of an Empty.
             * @memberof summa.proto
             * @interface IEmpty
             */

            /**
             * Constructs a new Empty.
             * @memberof summa.proto
             * @classdesc Represents an Empty.
             * @implements IEmpty
             * @constructor
             * @param {summa.proto.IEmpty=} [properties] Properties to set
             */
            function Empty(properties) {
                if (properties)
                    for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }

            /**
             * Creates a new Empty instance using the specified properties.
             * @function create
             * @memberof summa.proto.Empty
             * @static
             * @param {summa.proto.IEmpty=} [properties] Properties to set
             * @returns {summa.proto.Empty} Empty instance
             */
            Empty.create = function create(properties) {
                return new Empty(properties);
            };

            return Empty;
        })();

        proto.PrimaryKey = (function() {

            /**
             * Properties of a PrimaryKey.
             * @memberof summa.proto
             * @interface IPrimaryKey
             * @property {string|null} [str] PrimaryKey str
             * @property {number|Long|null} [i64] PrimaryKey i64
             */

            /**
             * Constructs a new PrimaryKey.
             * @memberof summa.proto
             * @classdesc Represents a PrimaryKey.
             * @implements IPrimaryKey
             * @constructor
             * @param {summa.proto.IPrimaryKey=} [properties] Properties to set
             */
            function PrimaryKey(properties) {
                if (properties)
                    for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }

            /**
             * PrimaryKey str.
             * @member {string|null|undefined} str
             * @memberof summa.proto.PrimaryKey
             * @instance
             */
            PrimaryKey.prototype.str = null;

            /**
             * PrimaryKey i64.
             * @member {number|Long|null|undefined} i64
             * @memberof summa.proto.PrimaryKey
             * @instance
             */
            PrimaryKey.prototype.i64 = null;

            // OneOf field names bound to virtual getters and setters
            let $oneOfFields;

            /**
             * PrimaryKey value.
             * @member {"str"|"i64"|undefined} value
             * @memberof summa.proto.PrimaryKey
             * @instance
             */
            Object.defineProperty(PrimaryKey.prototype, "value", {
                get: $util.oneOfGetter($oneOfFields = ["str", "i64"]),
                set: $util.oneOfSetter($oneOfFields)
            });

            /**
             * Creates a new PrimaryKey instance using the specified properties.
             * @function create
             * @memberof summa.proto.PrimaryKey
             * @static
             * @param {summa.proto.IPrimaryKey=} [properties] Properties to set
             * @returns {summa.proto.PrimaryKey} PrimaryKey instance
             */
            PrimaryKey.create = function create(properties) {
                return new PrimaryKey(properties);
            };

            return PrimaryKey;
        })();

        proto.MergePolicy = (function() {

            /**
             * Properties of a MergePolicy.
             * @memberof summa.proto
             * @interface IMergePolicy
             * @property {summa.proto.ILogMergePolicy|null} [log] MergePolicy log
             * @property {summa.proto.ITemporalMergePolicy|null} [temporal] MergePolicy temporal
             */

            /**
             * Constructs a new MergePolicy.
             * @memberof summa.proto
             * @classdesc Represents a MergePolicy.
             * @implements IMergePolicy
             * @constructor
             * @param {summa.proto.IMergePolicy=} [properties] Properties to set
             */
            function MergePolicy(properties) {
                if (properties)
                    for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }

            /**
             * MergePolicy log.
             * @member {summa.proto.ILogMergePolicy|null|undefined} log
             * @memberof summa.proto.MergePolicy
             * @instance
             */
            MergePolicy.prototype.log = null;

            /**
             * MergePolicy temporal.
             * @member {summa.proto.ITemporalMergePolicy|null|undefined} temporal
             * @memberof summa.proto.MergePolicy
             * @instance
             */
            MergePolicy.prototype.temporal = null;

            // OneOf field names bound to virtual getters and setters
            let $oneOfFields;

            /**
             * MergePolicy merge_policy.
             * @member {"log"|"temporal"|undefined} merge_policy
             * @memberof summa.proto.MergePolicy
             * @instance
             */
            Object.defineProperty(MergePolicy.prototype, "merge_policy", {
                get: $util.oneOfGetter($oneOfFields = ["log", "temporal"]),
                set: $util.oneOfSetter($oneOfFields)
            });

            /**
             * Creates a new MergePolicy instance using the specified properties.
             * @function create
             * @memberof summa.proto.MergePolicy
             * @static
             * @param {summa.proto.IMergePolicy=} [properties] Properties to set
             * @returns {summa.proto.MergePolicy} MergePolicy instance
             */
            MergePolicy.create = function create(properties) {
                return new MergePolicy(properties);
            };

            return MergePolicy;
        })();

        proto.AttachFileEngineRequest = (function() {

            /**
             * Properties of an AttachFileEngineRequest.
             * @memberof summa.proto
             * @interface IAttachFileEngineRequest
             */

            /**
             * Constructs a new AttachFileEngineRequest.
             * @memberof summa.proto
             * @classdesc Represents an AttachFileEngineRequest.
             * @implements IAttachFileEngineRequest
             * @constructor
             * @param {summa.proto.IAttachFileEngineRequest=} [properties] Properties to set
             */
            function AttachFileEngineRequest(properties) {
                if (properties)
                    for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }

            /**
             * Creates a new AttachFileEngineRequest instance using the specified properties.
             * @function create
             * @memberof summa.proto.AttachFileEngineRequest
             * @static
             * @param {summa.proto.IAttachFileEngineRequest=} [properties] Properties to set
             * @returns {summa.proto.AttachFileEngineRequest} AttachFileEngineRequest instance
             */
            AttachFileEngineRequest.create = function create(properties) {
                return new AttachFileEngineRequest(properties);
            };

            return AttachFileEngineRequest;
        })();

        proto.AttachRemoteEngineRequest = (function() {

            /**
             * Properties of an AttachRemoteEngineRequest.
             * @memberof summa.proto
             * @interface IAttachRemoteEngineRequest
             * @property {summa.proto.IRemoteEngineConfig|null} [config] AttachRemoteEngineRequest config
             */

            /**
             * Constructs a new AttachRemoteEngineRequest.
             * @memberof summa.proto
             * @classdesc Represents an AttachRemoteEngineRequest.
             * @implements IAttachRemoteEngineRequest
             * @constructor
             * @param {summa.proto.IAttachRemoteEngineRequest=} [properties] Properties to set
             */
            function AttachRemoteEngineRequest(properties) {
                if (properties)
                    for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }

            /**
             * AttachRemoteEngineRequest config.
             * @member {summa.proto.IRemoteEngineConfig|null|undefined} config
             * @memberof summa.proto.AttachRemoteEngineRequest
             * @instance
             */
            AttachRemoteEngineRequest.prototype.config = null;

            /**
             * Creates a new AttachRemoteEngineRequest instance using the specified properties.
             * @function create
             * @memberof summa.proto.AttachRemoteEngineRequest
             * @static
             * @param {summa.proto.IAttachRemoteEngineRequest=} [properties] Properties to set
             * @returns {summa.proto.AttachRemoteEngineRequest} AttachRemoteEngineRequest instance
             */
            AttachRemoteEngineRequest.create = function create(properties) {
                return new AttachRemoteEngineRequest(properties);
            };

            return AttachRemoteEngineRequest;
        })();

        proto.AttachIndexRequest = (function() {

            /**
             * Properties of an AttachIndexRequest.
             * @memberof summa.proto
             * @interface IAttachIndexRequest
             * @property {string|null} [index_name] AttachIndexRequest index_name
             * @property {summa.proto.IAttachFileEngineRequest|null} [file] AttachIndexRequest file
             * @property {summa.proto.IAttachRemoteEngineRequest|null} [remote] AttachIndexRequest remote
             * @property {summa.proto.IMergePolicy|null} [merge_policy] AttachIndexRequest merge_policy
             * @property {summa.proto.IQueryParserConfig|null} [query_parser_config] AttachIndexRequest query_parser_config
             */

            /**
             * Constructs a new AttachIndexRequest.
             * @memberof summa.proto
             * @classdesc Represents an AttachIndexRequest.
             * @implements IAttachIndexRequest
             * @constructor
             * @param {summa.proto.IAttachIndexRequest=} [properties] Properties to set
             */
            function AttachIndexRequest(properties) {
                if (properties)
                    for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }

            /**
             * AttachIndexRequest index_name.
             * @member {string} index_name
             * @memberof summa.proto.AttachIndexRequest
             * @instance
             */
            AttachIndexRequest.prototype.index_name = "";

            /**
             * AttachIndexRequest file.
             * @member {summa.proto.IAttachFileEngineRequest|null|undefined} file
             * @memberof summa.proto.AttachIndexRequest
             * @instance
             */
            AttachIndexRequest.prototype.file = null;

            /**
             * AttachIndexRequest remote.
             * @member {summa.proto.IAttachRemoteEngineRequest|null|undefined} remote
             * @memberof summa.proto.AttachIndexRequest
             * @instance
             */
            AttachIndexRequest.prototype.remote = null;

            /**
             * AttachIndexRequest merge_policy.
             * @member {summa.proto.IMergePolicy|null|undefined} merge_policy
             * @memberof summa.proto.AttachIndexRequest
             * @instance
             */
            AttachIndexRequest.prototype.merge_policy = null;

            /**
             * AttachIndexRequest query_parser_config.
             * @member {summa.proto.IQueryParserConfig|null|undefined} query_parser_config
             * @memberof summa.proto.AttachIndexRequest
             * @instance
             */
            AttachIndexRequest.prototype.query_parser_config = null;

            // OneOf field names bound to virtual getters and setters
            let $oneOfFields;

            /**
             * AttachIndexRequest index_engine.
             * @member {"file"|"remote"|undefined} index_engine
             * @memberof summa.proto.AttachIndexRequest
             * @instance
             */
            Object.defineProperty(AttachIndexRequest.prototype, "index_engine", {
                get: $util.oneOfGetter($oneOfFields = ["file", "remote"]),
                set: $util.oneOfSetter($oneOfFields)
            });

            /**
             * Creates a new AttachIndexRequest instance using the specified properties.
             * @function create
             * @memberof summa.proto.AttachIndexRequest
             * @static
             * @param {summa.proto.IAttachIndexRequest=} [properties] Properties to set
             * @returns {summa.proto.AttachIndexRequest} AttachIndexRequest instance
             */
            AttachIndexRequest.create = function create(properties) {
                return new AttachIndexRequest(properties);
            };

            return AttachIndexRequest;
        })();

        proto.AttachIndexResponse = (function() {

            /**
             * Properties of an AttachIndexResponse.
             * @memberof summa.proto
             * @interface IAttachIndexResponse
             * @property {summa.proto.IIndexDescription|null} [index] AttachIndexResponse index
             */

            /**
             * Constructs a new AttachIndexResponse.
             * @memberof summa.proto
             * @classdesc Represents an AttachIndexResponse.
             * @implements IAttachIndexResponse
             * @constructor
             * @param {summa.proto.IAttachIndexResponse=} [properties] Properties to set
             */
            function AttachIndexResponse(properties) {
                if (properties)
                    for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }

            /**
             * AttachIndexResponse index.
             * @member {summa.proto.IIndexDescription|null|undefined} index
             * @memberof summa.proto.AttachIndexResponse
             * @instance
             */
            AttachIndexResponse.prototype.index = null;

            /**
             * Creates a new AttachIndexResponse instance using the specified properties.
             * @function create
             * @memberof summa.proto.AttachIndexResponse
             * @static
             * @param {summa.proto.IAttachIndexResponse=} [properties] Properties to set
             * @returns {summa.proto.AttachIndexResponse} AttachIndexResponse instance
             */
            AttachIndexResponse.create = function create(properties) {
                return new AttachIndexResponse(properties);
            };

            return AttachIndexResponse;
        })();

        proto.CommitIndexRequest = (function() {

            /**
             * Properties of a CommitIndexRequest.
             * @memberof summa.proto
             * @interface ICommitIndexRequest
             * @property {string|null} [index_name] CommitIndexRequest index_name
             */

            /**
             * Constructs a new CommitIndexRequest.
             * @memberof summa.proto
             * @classdesc Represents a CommitIndexRequest.
             * @implements ICommitIndexRequest
             * @constructor
             * @param {summa.proto.ICommitIndexRequest=} [properties] Properties to set
             */
            function CommitIndexRequest(properties) {
                if (properties)
                    for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }

            /**
             * CommitIndexRequest index_name.
             * @member {string} index_name
             * @memberof summa.proto.CommitIndexRequest
             * @instance
             */
            CommitIndexRequest.prototype.index_name = "";

            /**
             * Creates a new CommitIndexRequest instance using the specified properties.
             * @function create
             * @memberof summa.proto.CommitIndexRequest
             * @static
             * @param {summa.proto.ICommitIndexRequest=} [properties] Properties to set
             * @returns {summa.proto.CommitIndexRequest} CommitIndexRequest instance
             */
            CommitIndexRequest.create = function create(properties) {
                return new CommitIndexRequest(properties);
            };

            return CommitIndexRequest;
        })();

        proto.CommitIndexResponse = (function() {

            /**
             * Properties of a CommitIndexResponse.
             * @memberof summa.proto
             * @interface ICommitIndexResponse
             * @property {number|null} [elapsed_secs] CommitIndexResponse elapsed_secs
             */

            /**
             * Constructs a new CommitIndexResponse.
             * @memberof summa.proto
             * @classdesc Represents a CommitIndexResponse.
             * @implements ICommitIndexResponse
             * @constructor
             * @param {summa.proto.ICommitIndexResponse=} [properties] Properties to set
             */
            function CommitIndexResponse(properties) {
                if (properties)
                    for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }

            /**
             * CommitIndexResponse elapsed_secs.
             * @member {number} elapsed_secs
             * @memberof summa.proto.CommitIndexResponse
             * @instance
             */
            CommitIndexResponse.prototype.elapsed_secs = 0;

            /**
             * Creates a new CommitIndexResponse instance using the specified properties.
             * @function create
             * @memberof summa.proto.CommitIndexResponse
             * @static
             * @param {summa.proto.ICommitIndexResponse=} [properties] Properties to set
             * @returns {summa.proto.CommitIndexResponse} CommitIndexResponse instance
             */
            CommitIndexResponse.create = function create(properties) {
                return new CommitIndexResponse(properties);
            };

            return CommitIndexResponse;
        })();

        proto.CopyDocumentsRequest = (function() {

            /**
             * Properties of a CopyDocumentsRequest.
             * @memberof summa.proto
             * @interface ICopyDocumentsRequest
             * @property {string|null} [source_index_name] CopyDocumentsRequest source_index_name
             * @property {string|null} [target_index_name] CopyDocumentsRequest target_index_name
             * @property {summa.proto.ConflictStrategy|null} [conflict_strategy] CopyDocumentsRequest conflict_strategy
             */

            /**
             * Constructs a new CopyDocumentsRequest.
             * @memberof summa.proto
             * @classdesc Represents a CopyDocumentsRequest.
             * @implements ICopyDocumentsRequest
             * @constructor
             * @param {summa.proto.ICopyDocumentsRequest=} [properties] Properties to set
             */
            function CopyDocumentsRequest(properties) {
                if (properties)
                    for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }

            /**
             * CopyDocumentsRequest source_index_name.
             * @member {string} source_index_name
             * @memberof summa.proto.CopyDocumentsRequest
             * @instance
             */
            CopyDocumentsRequest.prototype.source_index_name = "";

            /**
             * CopyDocumentsRequest target_index_name.
             * @member {string} target_index_name
             * @memberof summa.proto.CopyDocumentsRequest
             * @instance
             */
            CopyDocumentsRequest.prototype.target_index_name = "";

            /**
             * CopyDocumentsRequest conflict_strategy.
             * @member {summa.proto.ConflictStrategy|null|undefined} conflict_strategy
             * @memberof summa.proto.CopyDocumentsRequest
             * @instance
             */
            CopyDocumentsRequest.prototype.conflict_strategy = null;

            // OneOf field names bound to virtual getters and setters
            let $oneOfFields;

            /**
             * CopyDocumentsRequest _conflict_strategy.
             * @member {"conflict_strategy"|undefined} _conflict_strategy
             * @memberof summa.proto.CopyDocumentsRequest
             * @instance
             */
            Object.defineProperty(CopyDocumentsRequest.prototype, "_conflict_strategy", {
                get: $util.oneOfGetter($oneOfFields = ["conflict_strategy"]),
                set: $util.oneOfSetter($oneOfFields)
            });

            /**
             * Creates a new CopyDocumentsRequest instance using the specified properties.
             * @function create
             * @memberof summa.proto.CopyDocumentsRequest
             * @static
             * @param {summa.proto.ICopyDocumentsRequest=} [properties] Properties to set
             * @returns {summa.proto.CopyDocumentsRequest} CopyDocumentsRequest instance
             */
            CopyDocumentsRequest.create = function create(properties) {
                return new CopyDocumentsRequest(properties);
            };

            return CopyDocumentsRequest;
        })();

        proto.CopyDocumentsResponse = (function() {

            /**
             * Properties of a CopyDocumentsResponse.
             * @memberof summa.proto
             * @interface ICopyDocumentsResponse
             * @property {number|null} [elapsed_secs] CopyDocumentsResponse elapsed_secs
             * @property {number|null} [copied_documents] CopyDocumentsResponse copied_documents
             */

            /**
             * Constructs a new CopyDocumentsResponse.
             * @memberof summa.proto
             * @classdesc Represents a CopyDocumentsResponse.
             * @implements ICopyDocumentsResponse
             * @constructor
             * @param {summa.proto.ICopyDocumentsResponse=} [properties] Properties to set
             */
            function CopyDocumentsResponse(properties) {
                if (properties)
                    for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }

            /**
             * CopyDocumentsResponse elapsed_secs.
             * @member {number} elapsed_secs
             * @memberof summa.proto.CopyDocumentsResponse
             * @instance
             */
            CopyDocumentsResponse.prototype.elapsed_secs = 0;

            /**
             * CopyDocumentsResponse copied_documents.
             * @member {number} copied_documents
             * @memberof summa.proto.CopyDocumentsResponse
             * @instance
             */
            CopyDocumentsResponse.prototype.copied_documents = 0;

            /**
             * Creates a new CopyDocumentsResponse instance using the specified properties.
             * @function create
             * @memberof summa.proto.CopyDocumentsResponse
             * @static
             * @param {summa.proto.ICopyDocumentsResponse=} [properties] Properties to set
             * @returns {summa.proto.CopyDocumentsResponse} CopyDocumentsResponse instance
             */
            CopyDocumentsResponse.create = function create(properties) {
                return new CopyDocumentsResponse(properties);
            };

            return CopyDocumentsResponse;
        })();

        proto.CopyIndexRequest = (function() {

            /**
             * Properties of a CopyIndexRequest.
             * @memberof summa.proto
             * @interface ICopyIndexRequest
             * @property {string|null} [source_index_name] CopyIndexRequest source_index_name
             * @property {string|null} [target_index_name] CopyIndexRequest target_index_name
             * @property {summa.proto.ICreateFileEngineRequest|null} [file] CopyIndexRequest file
             * @property {summa.proto.ICreateMemoryEngineRequest|null} [memory] CopyIndexRequest memory
             * @property {summa.proto.IMergePolicy|null} [merge_policy] CopyIndexRequest merge_policy
             */

            /**
             * Constructs a new CopyIndexRequest.
             * @memberof summa.proto
             * @classdesc Represents a CopyIndexRequest.
             * @implements ICopyIndexRequest
             * @constructor
             * @param {summa.proto.ICopyIndexRequest=} [properties] Properties to set
             */
            function CopyIndexRequest(properties) {
                if (properties)
                    for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }

            /**
             * CopyIndexRequest source_index_name.
             * @member {string} source_index_name
             * @memberof summa.proto.CopyIndexRequest
             * @instance
             */
            CopyIndexRequest.prototype.source_index_name = "";

            /**
             * CopyIndexRequest target_index_name.
             * @member {string} target_index_name
             * @memberof summa.proto.CopyIndexRequest
             * @instance
             */
            CopyIndexRequest.prototype.target_index_name = "";

            /**
             * CopyIndexRequest file.
             * @member {summa.proto.ICreateFileEngineRequest|null|undefined} file
             * @memberof summa.proto.CopyIndexRequest
             * @instance
             */
            CopyIndexRequest.prototype.file = null;

            /**
             * CopyIndexRequest memory.
             * @member {summa.proto.ICreateMemoryEngineRequest|null|undefined} memory
             * @memberof summa.proto.CopyIndexRequest
             * @instance
             */
            CopyIndexRequest.prototype.memory = null;

            /**
             * CopyIndexRequest merge_policy.
             * @member {summa.proto.IMergePolicy|null|undefined} merge_policy
             * @memberof summa.proto.CopyIndexRequest
             * @instance
             */
            CopyIndexRequest.prototype.merge_policy = null;

            // OneOf field names bound to virtual getters and setters
            let $oneOfFields;

            /**
             * CopyIndexRequest target_index_engine.
             * @member {"file"|"memory"|undefined} target_index_engine
             * @memberof summa.proto.CopyIndexRequest
             * @instance
             */
            Object.defineProperty(CopyIndexRequest.prototype, "target_index_engine", {
                get: $util.oneOfGetter($oneOfFields = ["file", "memory"]),
                set: $util.oneOfSetter($oneOfFields)
            });

            /**
             * Creates a new CopyIndexRequest instance using the specified properties.
             * @function create
             * @memberof summa.proto.CopyIndexRequest
             * @static
             * @param {summa.proto.ICopyIndexRequest=} [properties] Properties to set
             * @returns {summa.proto.CopyIndexRequest} CopyIndexRequest instance
             */
            CopyIndexRequest.create = function create(properties) {
                return new CopyIndexRequest(properties);
            };

            return CopyIndexRequest;
        })();

        proto.CopyIndexResponse = (function() {

            /**
             * Properties of a CopyIndexResponse.
             * @memberof summa.proto
             * @interface ICopyIndexResponse
             * @property {summa.proto.IIndexDescription|null} [index] CopyIndexResponse index
             */

            /**
             * Constructs a new CopyIndexResponse.
             * @memberof summa.proto
             * @classdesc Represents a CopyIndexResponse.
             * @implements ICopyIndexResponse
             * @constructor
             * @param {summa.proto.ICopyIndexResponse=} [properties] Properties to set
             */
            function CopyIndexResponse(properties) {
                if (properties)
                    for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }

            /**
             * CopyIndexResponse index.
             * @member {summa.proto.IIndexDescription|null|undefined} index
             * @memberof summa.proto.CopyIndexResponse
             * @instance
             */
            CopyIndexResponse.prototype.index = null;

            /**
             * Creates a new CopyIndexResponse instance using the specified properties.
             * @function create
             * @memberof summa.proto.CopyIndexResponse
             * @static
             * @param {summa.proto.ICopyIndexResponse=} [properties] Properties to set
             * @returns {summa.proto.CopyIndexResponse} CopyIndexResponse instance
             */
            CopyIndexResponse.create = function create(properties) {
                return new CopyIndexResponse(properties);
            };

            return CopyIndexResponse;
        })();

        proto.SortByField = (function() {

            /**
             * Properties of a SortByField.
             * @memberof summa.proto
             * @interface ISortByField
             * @property {string|null} [field] SortByField field
             * @property {summa.proto.Order|null} [order] SortByField order
             */

            /**
             * Constructs a new SortByField.
             * @memberof summa.proto
             * @classdesc Represents a SortByField.
             * @implements ISortByField
             * @constructor
             * @param {summa.proto.ISortByField=} [properties] Properties to set
             */
            function SortByField(properties) {
                if (properties)
                    for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }

            /**
             * SortByField field.
             * @member {string} field
             * @memberof summa.proto.SortByField
             * @instance
             */
            SortByField.prototype.field = "";

            /**
             * SortByField order.
             * @member {summa.proto.Order} order
             * @memberof summa.proto.SortByField
             * @instance
             */
            SortByField.prototype.order = 0;

            /**
             * Creates a new SortByField instance using the specified properties.
             * @function create
             * @memberof summa.proto.SortByField
             * @static
             * @param {summa.proto.ISortByField=} [properties] Properties to set
             * @returns {summa.proto.SortByField} SortByField instance
             */
            SortByField.create = function create(properties) {
                return new SortByField(properties);
            };

            return SortByField;
        })();

        proto.CreateFileEngineRequest = (function() {

            /**
             * Properties of a CreateFileEngineRequest.
             * @memberof summa.proto
             * @interface ICreateFileEngineRequest
             */

            /**
             * Constructs a new CreateFileEngineRequest.
             * @memberof summa.proto
             * @classdesc Represents a CreateFileEngineRequest.
             * @implements ICreateFileEngineRequest
             * @constructor
             * @param {summa.proto.ICreateFileEngineRequest=} [properties] Properties to set
             */
            function CreateFileEngineRequest(properties) {
                if (properties)
                    for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }

            /**
             * Creates a new CreateFileEngineRequest instance using the specified properties.
             * @function create
             * @memberof summa.proto.CreateFileEngineRequest
             * @static
             * @param {summa.proto.ICreateFileEngineRequest=} [properties] Properties to set
             * @returns {summa.proto.CreateFileEngineRequest} CreateFileEngineRequest instance
             */
            CreateFileEngineRequest.create = function create(properties) {
                return new CreateFileEngineRequest(properties);
            };

            return CreateFileEngineRequest;
        })();

        proto.CreateMemoryEngineRequest = (function() {

            /**
             * Properties of a CreateMemoryEngineRequest.
             * @memberof summa.proto
             * @interface ICreateMemoryEngineRequest
             */

            /**
             * Constructs a new CreateMemoryEngineRequest.
             * @memberof summa.proto
             * @classdesc Represents a CreateMemoryEngineRequest.
             * @implements ICreateMemoryEngineRequest
             * @constructor
             * @param {summa.proto.ICreateMemoryEngineRequest=} [properties] Properties to set
             */
            function CreateMemoryEngineRequest(properties) {
                if (properties)
                    for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }

            /**
             * Creates a new CreateMemoryEngineRequest instance using the specified properties.
             * @function create
             * @memberof summa.proto.CreateMemoryEngineRequest
             * @static
             * @param {summa.proto.ICreateMemoryEngineRequest=} [properties] Properties to set
             * @returns {summa.proto.CreateMemoryEngineRequest} CreateMemoryEngineRequest instance
             */
            CreateMemoryEngineRequest.create = function create(properties) {
                return new CreateMemoryEngineRequest(properties);
            };

            return CreateMemoryEngineRequest;
        })();

        /**
         * ConflictStrategy enum.
         * @name summa.proto.ConflictStrategy
         * @enum {number}
         * @property {number} DO_NOTHING=0 DO_NOTHING value
         * @property {number} OVERWRITE_ALWAYS=1 OVERWRITE_ALWAYS value
         * @property {number} OVERWRITE=2 OVERWRITE value
         * @property {number} MERGE=3 MERGE value
         */
        proto.ConflictStrategy = (function() {
            const valuesById = {}, values = Object.create(valuesById);
            values[valuesById[0] = "DO_NOTHING"] = 0;
            values[valuesById[1] = "OVERWRITE_ALWAYS"] = 1;
            values[valuesById[2] = "OVERWRITE"] = 2;
            values[valuesById[3] = "MERGE"] = 3;
            return values;
        })();

        proto.MappedField = (function() {

            /**
             * Properties of a MappedField.
             * @memberof summa.proto
             * @interface IMappedField
             * @property {string|null} [source_field] MappedField source_field
             * @property {string|null} [target_field] MappedField target_field
             */

            /**
             * Constructs a new MappedField.
             * @memberof summa.proto
             * @classdesc Represents a MappedField.
             * @implements IMappedField
             * @constructor
             * @param {summa.proto.IMappedField=} [properties] Properties to set
             */
            function MappedField(properties) {
                if (properties)
                    for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }

            /**
             * MappedField source_field.
             * @member {string} source_field
             * @memberof summa.proto.MappedField
             * @instance
             */
            MappedField.prototype.source_field = "";

            /**
             * MappedField target_field.
             * @member {string} target_field
             * @memberof summa.proto.MappedField
             * @instance
             */
            MappedField.prototype.target_field = "";

            /**
             * Creates a new MappedField instance using the specified properties.
             * @function create
             * @memberof summa.proto.MappedField
             * @static
             * @param {summa.proto.IMappedField=} [properties] Properties to set
             * @returns {summa.proto.MappedField} MappedField instance
             */
            MappedField.create = function create(properties) {
                return new MappedField(properties);
            };

            return MappedField;
        })();

        proto.IndexAttributes = (function() {

            /**
             * Properties of an IndexAttributes.
             * @memberof summa.proto
             * @interface IIndexAttributes
             * @property {number|Long|null} [created_at] IndexAttributes created_at
             * @property {Array.<string>|null} [unique_fields] IndexAttributes unique_fields
             * @property {Array.<string>|null} [multi_fields] IndexAttributes multi_fields
             * @property {string|null} [description] IndexAttributes description
             * @property {summa.proto.ConflictStrategy|null} [conflict_strategy] IndexAttributes conflict_strategy
             * @property {Array.<summa.proto.IMappedField>|null} [mapped_fields] IndexAttributes mapped_fields
             */

            /**
             * Constructs a new IndexAttributes.
             * @memberof summa.proto
             * @classdesc Represents an IndexAttributes.
             * @implements IIndexAttributes
             * @constructor
             * @param {summa.proto.IIndexAttributes=} [properties] Properties to set
             */
            function IndexAttributes(properties) {
                this.unique_fields = [];
                this.multi_fields = [];
                this.mapped_fields = [];
                if (properties)
                    for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }

            /**
             * IndexAttributes created_at.
             * @member {number|Long} created_at
             * @memberof summa.proto.IndexAttributes
             * @instance
             */
            IndexAttributes.prototype.created_at = $util.Long ? $util.Long.fromBits(0,0,true) : 0;

            /**
             * IndexAttributes unique_fields.
             * @member {Array.<string>} unique_fields
             * @memberof summa.proto.IndexAttributes
             * @instance
             */
            IndexAttributes.prototype.unique_fields = $util.emptyArray;

            /**
             * IndexAttributes multi_fields.
             * @member {Array.<string>} multi_fields
             * @memberof summa.proto.IndexAttributes
             * @instance
             */
            IndexAttributes.prototype.multi_fields = $util.emptyArray;

            /**
             * IndexAttributes description.
             * @member {string|null|undefined} description
             * @memberof summa.proto.IndexAttributes
             * @instance
             */
            IndexAttributes.prototype.description = null;

            /**
             * IndexAttributes conflict_strategy.
             * @member {summa.proto.ConflictStrategy} conflict_strategy
             * @memberof summa.proto.IndexAttributes
             * @instance
             */
            IndexAttributes.prototype.conflict_strategy = 0;

            /**
             * IndexAttributes mapped_fields.
             * @member {Array.<summa.proto.IMappedField>} mapped_fields
             * @memberof summa.proto.IndexAttributes
             * @instance
             */
            IndexAttributes.prototype.mapped_fields = $util.emptyArray;

            // OneOf field names bound to virtual getters and setters
            let $oneOfFields;

            /**
             * IndexAttributes _description.
             * @member {"description"|undefined} _description
             * @memberof summa.proto.IndexAttributes
             * @instance
             */
            Object.defineProperty(IndexAttributes.prototype, "_description", {
                get: $util.oneOfGetter($oneOfFields = ["description"]),
                set: $util.oneOfSetter($oneOfFields)
            });

            /**
             * Creates a new IndexAttributes instance using the specified properties.
             * @function create
             * @memberof summa.proto.IndexAttributes
             * @static
             * @param {summa.proto.IIndexAttributes=} [properties] Properties to set
             * @returns {summa.proto.IndexAttributes} IndexAttributes instance
             */
            IndexAttributes.create = function create(properties) {
                return new IndexAttributes(properties);
            };

            return IndexAttributes;
        })();

        proto.CreateIndexRequest = (function() {

            /**
             * Properties of a CreateIndexRequest.
             * @memberof summa.proto
             * @interface ICreateIndexRequest
             * @property {string|null} [index_name] CreateIndexRequest index_name
             * @property {summa.proto.ICreateFileEngineRequest|null} [file] CreateIndexRequest file
             * @property {summa.proto.ICreateMemoryEngineRequest|null} [memory] CreateIndexRequest memory
             * @property {string|null} [schema] CreateIndexRequest schema
             * @property {summa.proto.Compression|null} [compression] CreateIndexRequest compression
             * @property {number|null} [blocksize] CreateIndexRequest blocksize
             * @property {summa.proto.ISortByField|null} [sort_by_field] CreateIndexRequest sort_by_field
             * @property {summa.proto.IIndexAttributes|null} [index_attributes] CreateIndexRequest index_attributes
             * @property {summa.proto.IMergePolicy|null} [merge_policy] CreateIndexRequest merge_policy
             * @property {summa.proto.IQueryParserConfig|null} [query_parser_config] CreateIndexRequest query_parser_config
             */

            /**
             * Constructs a new CreateIndexRequest.
             * @memberof summa.proto
             * @classdesc Represents a CreateIndexRequest.
             * @implements ICreateIndexRequest
             * @constructor
             * @param {summa.proto.ICreateIndexRequest=} [properties] Properties to set
             */
            function CreateIndexRequest(properties) {
                if (properties)
                    for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }

            /**
             * CreateIndexRequest index_name.
             * @member {string} index_name
             * @memberof summa.proto.CreateIndexRequest
             * @instance
             */
            CreateIndexRequest.prototype.index_name = "";

            /**
             * CreateIndexRequest file.
             * @member {summa.proto.ICreateFileEngineRequest|null|undefined} file
             * @memberof summa.proto.CreateIndexRequest
             * @instance
             */
            CreateIndexRequest.prototype.file = null;

            /**
             * CreateIndexRequest memory.
             * @member {summa.proto.ICreateMemoryEngineRequest|null|undefined} memory
             * @memberof summa.proto.CreateIndexRequest
             * @instance
             */
            CreateIndexRequest.prototype.memory = null;

            /**
             * CreateIndexRequest schema.
             * @member {string} schema
             * @memberof summa.proto.CreateIndexRequest
             * @instance
             */
            CreateIndexRequest.prototype.schema = "";

            /**
             * CreateIndexRequest compression.
             * @member {summa.proto.Compression} compression
             * @memberof summa.proto.CreateIndexRequest
             * @instance
             */
            CreateIndexRequest.prototype.compression = 0;

            /**
             * CreateIndexRequest blocksize.
             * @member {number|null|undefined} blocksize
             * @memberof summa.proto.CreateIndexRequest
             * @instance
             */
            CreateIndexRequest.prototype.blocksize = null;

            /**
             * CreateIndexRequest sort_by_field.
             * @member {summa.proto.ISortByField|null|undefined} sort_by_field
             * @memberof summa.proto.CreateIndexRequest
             * @instance
             */
            CreateIndexRequest.prototype.sort_by_field = null;

            /**
             * CreateIndexRequest index_attributes.
             * @member {summa.proto.IIndexAttributes|null|undefined} index_attributes
             * @memberof summa.proto.CreateIndexRequest
             * @instance
             */
            CreateIndexRequest.prototype.index_attributes = null;

            /**
             * CreateIndexRequest merge_policy.
             * @member {summa.proto.IMergePolicy|null|undefined} merge_policy
             * @memberof summa.proto.CreateIndexRequest
             * @instance
             */
            CreateIndexRequest.prototype.merge_policy = null;

            /**
             * CreateIndexRequest query_parser_config.
             * @member {summa.proto.IQueryParserConfig|null|undefined} query_parser_config
             * @memberof summa.proto.CreateIndexRequest
             * @instance
             */
            CreateIndexRequest.prototype.query_parser_config = null;

            // OneOf field names bound to virtual getters and setters
            let $oneOfFields;

            /**
             * CreateIndexRequest index_engine.
             * @member {"file"|"memory"|undefined} index_engine
             * @memberof summa.proto.CreateIndexRequest
             * @instance
             */
            Object.defineProperty(CreateIndexRequest.prototype, "index_engine", {
                get: $util.oneOfGetter($oneOfFields = ["file", "memory"]),
                set: $util.oneOfSetter($oneOfFields)
            });

            /**
             * CreateIndexRequest _blocksize.
             * @member {"blocksize"|undefined} _blocksize
             * @memberof summa.proto.CreateIndexRequest
             * @instance
             */
            Object.defineProperty(CreateIndexRequest.prototype, "_blocksize", {
                get: $util.oneOfGetter($oneOfFields = ["blocksize"]),
                set: $util.oneOfSetter($oneOfFields)
            });

            /**
             * CreateIndexRequest _sort_by_field.
             * @member {"sort_by_field"|undefined} _sort_by_field
             * @memberof summa.proto.CreateIndexRequest
             * @instance
             */
            Object.defineProperty(CreateIndexRequest.prototype, "_sort_by_field", {
                get: $util.oneOfGetter($oneOfFields = ["sort_by_field"]),
                set: $util.oneOfSetter($oneOfFields)
            });

            /**
             * Creates a new CreateIndexRequest instance using the specified properties.
             * @function create
             * @memberof summa.proto.CreateIndexRequest
             * @static
             * @param {summa.proto.ICreateIndexRequest=} [properties] Properties to set
             * @returns {summa.proto.CreateIndexRequest} CreateIndexRequest instance
             */
            CreateIndexRequest.create = function create(properties) {
                return new CreateIndexRequest(properties);
            };

            return CreateIndexRequest;
        })();

        proto.CreateIndexResponse = (function() {

            /**
             * Properties of a CreateIndexResponse.
             * @memberof summa.proto
             * @interface ICreateIndexResponse
             * @property {summa.proto.IIndexDescription|null} [index] CreateIndexResponse index
             */

            /**
             * Constructs a new CreateIndexResponse.
             * @memberof summa.proto
             * @classdesc Represents a CreateIndexResponse.
             * @implements ICreateIndexResponse
             * @constructor
             * @param {summa.proto.ICreateIndexResponse=} [properties] Properties to set
             */
            function CreateIndexResponse(properties) {
                if (properties)
                    for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }

            /**
             * CreateIndexResponse index.
             * @member {summa.proto.IIndexDescription|null|undefined} index
             * @memberof summa.proto.CreateIndexResponse
             * @instance
             */
            CreateIndexResponse.prototype.index = null;

            /**
             * Creates a new CreateIndexResponse instance using the specified properties.
             * @function create
             * @memberof summa.proto.CreateIndexResponse
             * @static
             * @param {summa.proto.ICreateIndexResponse=} [properties] Properties to set
             * @returns {summa.proto.CreateIndexResponse} CreateIndexResponse instance
             */
            CreateIndexResponse.create = function create(properties) {
                return new CreateIndexResponse(properties);
            };

            return CreateIndexResponse;
        })();

        proto.DeleteDocumentsRequest = (function() {

            /**
             * Properties of a DeleteDocumentsRequest.
             * @memberof summa.proto
             * @interface IDeleteDocumentsRequest
             * @property {string|null} [index_name] DeleteDocumentsRequest index_name
             * @property {summa.proto.IQuery|null} [query] DeleteDocumentsRequest query
             */

            /**
             * Constructs a new DeleteDocumentsRequest.
             * @memberof summa.proto
             * @classdesc Represents a DeleteDocumentsRequest.
             * @implements IDeleteDocumentsRequest
             * @constructor
             * @param {summa.proto.IDeleteDocumentsRequest=} [properties] Properties to set
             */
            function DeleteDocumentsRequest(properties) {
                if (properties)
                    for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }

            /**
             * DeleteDocumentsRequest index_name.
             * @member {string} index_name
             * @memberof summa.proto.DeleteDocumentsRequest
             * @instance
             */
            DeleteDocumentsRequest.prototype.index_name = "";

            /**
             * DeleteDocumentsRequest query.
             * @member {summa.proto.IQuery|null|undefined} query
             * @memberof summa.proto.DeleteDocumentsRequest
             * @instance
             */
            DeleteDocumentsRequest.prototype.query = null;

            /**
             * Creates a new DeleteDocumentsRequest instance using the specified properties.
             * @function create
             * @memberof summa.proto.DeleteDocumentsRequest
             * @static
             * @param {summa.proto.IDeleteDocumentsRequest=} [properties] Properties to set
             * @returns {summa.proto.DeleteDocumentsRequest} DeleteDocumentsRequest instance
             */
            DeleteDocumentsRequest.create = function create(properties) {
                return new DeleteDocumentsRequest(properties);
            };

            return DeleteDocumentsRequest;
        })();

        proto.DeleteDocumentsResponse = (function() {

            /**
             * Properties of a DeleteDocumentsResponse.
             * @memberof summa.proto
             * @interface IDeleteDocumentsResponse
             * @property {number|Long|null} [deleted_documents] DeleteDocumentsResponse deleted_documents
             */

            /**
             * Constructs a new DeleteDocumentsResponse.
             * @memberof summa.proto
             * @classdesc Represents a DeleteDocumentsResponse.
             * @implements IDeleteDocumentsResponse
             * @constructor
             * @param {summa.proto.IDeleteDocumentsResponse=} [properties] Properties to set
             */
            function DeleteDocumentsResponse(properties) {
                if (properties)
                    for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }

            /**
             * DeleteDocumentsResponse deleted_documents.
             * @member {number|Long} deleted_documents
             * @memberof summa.proto.DeleteDocumentsResponse
             * @instance
             */
            DeleteDocumentsResponse.prototype.deleted_documents = $util.Long ? $util.Long.fromBits(0,0,true) : 0;

            /**
             * Creates a new DeleteDocumentsResponse instance using the specified properties.
             * @function create
             * @memberof summa.proto.DeleteDocumentsResponse
             * @static
             * @param {summa.proto.IDeleteDocumentsResponse=} [properties] Properties to set
             * @returns {summa.proto.DeleteDocumentsResponse} DeleteDocumentsResponse instance
             */
            DeleteDocumentsResponse.create = function create(properties) {
                return new DeleteDocumentsResponse(properties);
            };

            return DeleteDocumentsResponse;
        })();

        proto.DeleteIndexRequest = (function() {

            /**
             * Properties of a DeleteIndexRequest.
             * @memberof summa.proto
             * @interface IDeleteIndexRequest
             * @property {string|null} [index_name] DeleteIndexRequest index_name
             */

            /**
             * Constructs a new DeleteIndexRequest.
             * @memberof summa.proto
             * @classdesc Represents a DeleteIndexRequest.
             * @implements IDeleteIndexRequest
             * @constructor
             * @param {summa.proto.IDeleteIndexRequest=} [properties] Properties to set
             */
            function DeleteIndexRequest(properties) {
                if (properties)
                    for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }

            /**
             * DeleteIndexRequest index_name.
             * @member {string} index_name
             * @memberof summa.proto.DeleteIndexRequest
             * @instance
             */
            DeleteIndexRequest.prototype.index_name = "";

            /**
             * Creates a new DeleteIndexRequest instance using the specified properties.
             * @function create
             * @memberof summa.proto.DeleteIndexRequest
             * @static
             * @param {summa.proto.IDeleteIndexRequest=} [properties] Properties to set
             * @returns {summa.proto.DeleteIndexRequest} DeleteIndexRequest instance
             */
            DeleteIndexRequest.create = function create(properties) {
                return new DeleteIndexRequest(properties);
            };

            return DeleteIndexRequest;
        })();

        proto.DeleteIndexResponse = (function() {

            /**
             * Properties of a DeleteIndexResponse.
             * @memberof summa.proto
             * @interface IDeleteIndexResponse
             * @property {string|null} [deleted_index_name] DeleteIndexResponse deleted_index_name
             */

            /**
             * Constructs a new DeleteIndexResponse.
             * @memberof summa.proto
             * @classdesc Represents a DeleteIndexResponse.
             * @implements IDeleteIndexResponse
             * @constructor
             * @param {summa.proto.IDeleteIndexResponse=} [properties] Properties to set
             */
            function DeleteIndexResponse(properties) {
                if (properties)
                    for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }

            /**
             * DeleteIndexResponse deleted_index_name.
             * @member {string} deleted_index_name
             * @memberof summa.proto.DeleteIndexResponse
             * @instance
             */
            DeleteIndexResponse.prototype.deleted_index_name = "";

            /**
             * Creates a new DeleteIndexResponse instance using the specified properties.
             * @function create
             * @memberof summa.proto.DeleteIndexResponse
             * @static
             * @param {summa.proto.IDeleteIndexResponse=} [properties] Properties to set
             * @returns {summa.proto.DeleteIndexResponse} DeleteIndexResponse instance
             */
            DeleteIndexResponse.create = function create(properties) {
                return new DeleteIndexResponse(properties);
            };

            return DeleteIndexResponse;
        })();

        proto.GetIndicesAliasesRequest = (function() {

            /**
             * Properties of a GetIndicesAliasesRequest.
             * @memberof summa.proto
             * @interface IGetIndicesAliasesRequest
             */

            /**
             * Constructs a new GetIndicesAliasesRequest.
             * @memberof summa.proto
             * @classdesc Represents a GetIndicesAliasesRequest.
             * @implements IGetIndicesAliasesRequest
             * @constructor
             * @param {summa.proto.IGetIndicesAliasesRequest=} [properties] Properties to set
             */
            function GetIndicesAliasesRequest(properties) {
                if (properties)
                    for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }

            /**
             * Creates a new GetIndicesAliasesRequest instance using the specified properties.
             * @function create
             * @memberof summa.proto.GetIndicesAliasesRequest
             * @static
             * @param {summa.proto.IGetIndicesAliasesRequest=} [properties] Properties to set
             * @returns {summa.proto.GetIndicesAliasesRequest} GetIndicesAliasesRequest instance
             */
            GetIndicesAliasesRequest.create = function create(properties) {
                return new GetIndicesAliasesRequest(properties);
            };

            return GetIndicesAliasesRequest;
        })();

        proto.GetIndicesAliasesResponse = (function() {

            /**
             * Properties of a GetIndicesAliasesResponse.
             * @memberof summa.proto
             * @interface IGetIndicesAliasesResponse
             * @property {Object.<string,string>|null} [indices_aliases] GetIndicesAliasesResponse indices_aliases
             */

            /**
             * Constructs a new GetIndicesAliasesResponse.
             * @memberof summa.proto
             * @classdesc Represents a GetIndicesAliasesResponse.
             * @implements IGetIndicesAliasesResponse
             * @constructor
             * @param {summa.proto.IGetIndicesAliasesResponse=} [properties] Properties to set
             */
            function GetIndicesAliasesResponse(properties) {
                this.indices_aliases = {};
                if (properties)
                    for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }

            /**
             * GetIndicesAliasesResponse indices_aliases.
             * @member {Object.<string,string>} indices_aliases
             * @memberof summa.proto.GetIndicesAliasesResponse
             * @instance
             */
            GetIndicesAliasesResponse.prototype.indices_aliases = $util.emptyObject;

            /**
             * Creates a new GetIndicesAliasesResponse instance using the specified properties.
             * @function create
             * @memberof summa.proto.GetIndicesAliasesResponse
             * @static
             * @param {summa.proto.IGetIndicesAliasesResponse=} [properties] Properties to set
             * @returns {summa.proto.GetIndicesAliasesResponse} GetIndicesAliasesResponse instance
             */
            GetIndicesAliasesResponse.create = function create(properties) {
                return new GetIndicesAliasesResponse(properties);
            };

            return GetIndicesAliasesResponse;
        })();

        proto.GetIndexRequest = (function() {

            /**
             * Properties of a GetIndexRequest.
             * @memberof summa.proto
             * @interface IGetIndexRequest
             * @property {string|null} [index_name] GetIndexRequest index_name
             */

            /**
             * Constructs a new GetIndexRequest.
             * @memberof summa.proto
             * @classdesc Represents a GetIndexRequest.
             * @implements IGetIndexRequest
             * @constructor
             * @param {summa.proto.IGetIndexRequest=} [properties] Properties to set
             */
            function GetIndexRequest(properties) {
                if (properties)
                    for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }

            /**
             * GetIndexRequest index_name.
             * @member {string} index_name
             * @memberof summa.proto.GetIndexRequest
             * @instance
             */
            GetIndexRequest.prototype.index_name = "";

            /**
             * Creates a new GetIndexRequest instance using the specified properties.
             * @function create
             * @memberof summa.proto.GetIndexRequest
             * @static
             * @param {summa.proto.IGetIndexRequest=} [properties] Properties to set
             * @returns {summa.proto.GetIndexRequest} GetIndexRequest instance
             */
            GetIndexRequest.create = function create(properties) {
                return new GetIndexRequest(properties);
            };

            return GetIndexRequest;
        })();

        proto.GetIndexResponse = (function() {

            /**
             * Properties of a GetIndexResponse.
             * @memberof summa.proto
             * @interface IGetIndexResponse
             * @property {summa.proto.IIndexDescription|null} [index] GetIndexResponse index
             */

            /**
             * Constructs a new GetIndexResponse.
             * @memberof summa.proto
             * @classdesc Represents a GetIndexResponse.
             * @implements IGetIndexResponse
             * @constructor
             * @param {summa.proto.IGetIndexResponse=} [properties] Properties to set
             */
            function GetIndexResponse(properties) {
                if (properties)
                    for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }

            /**
             * GetIndexResponse index.
             * @member {summa.proto.IIndexDescription|null|undefined} index
             * @memberof summa.proto.GetIndexResponse
             * @instance
             */
            GetIndexResponse.prototype.index = null;

            /**
             * Creates a new GetIndexResponse instance using the specified properties.
             * @function create
             * @memberof summa.proto.GetIndexResponse
             * @static
             * @param {summa.proto.IGetIndexResponse=} [properties] Properties to set
             * @returns {summa.proto.GetIndexResponse} GetIndexResponse instance
             */
            GetIndexResponse.create = function create(properties) {
                return new GetIndexResponse(properties);
            };

            return GetIndexResponse;
        })();

        proto.GetIndicesRequest = (function() {

            /**
             * Properties of a GetIndicesRequest.
             * @memberof summa.proto
             * @interface IGetIndicesRequest
             */

            /**
             * Constructs a new GetIndicesRequest.
             * @memberof summa.proto
             * @classdesc Represents a GetIndicesRequest.
             * @implements IGetIndicesRequest
             * @constructor
             * @param {summa.proto.IGetIndicesRequest=} [properties] Properties to set
             */
            function GetIndicesRequest(properties) {
                if (properties)
                    for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }

            /**
             * Creates a new GetIndicesRequest instance using the specified properties.
             * @function create
             * @memberof summa.proto.GetIndicesRequest
             * @static
             * @param {summa.proto.IGetIndicesRequest=} [properties] Properties to set
             * @returns {summa.proto.GetIndicesRequest} GetIndicesRequest instance
             */
            GetIndicesRequest.create = function create(properties) {
                return new GetIndicesRequest(properties);
            };

            return GetIndicesRequest;
        })();

        proto.GetIndicesResponse = (function() {

            /**
             * Properties of a GetIndicesResponse.
             * @memberof summa.proto
             * @interface IGetIndicesResponse
             * @property {Array.<string>|null} [index_names] GetIndicesResponse index_names
             */

            /**
             * Constructs a new GetIndicesResponse.
             * @memberof summa.proto
             * @classdesc Represents a GetIndicesResponse.
             * @implements IGetIndicesResponse
             * @constructor
             * @param {summa.proto.IGetIndicesResponse=} [properties] Properties to set
             */
            function GetIndicesResponse(properties) {
                this.index_names = [];
                if (properties)
                    for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }

            /**
             * GetIndicesResponse index_names.
             * @member {Array.<string>} index_names
             * @memberof summa.proto.GetIndicesResponse
             * @instance
             */
            GetIndicesResponse.prototype.index_names = $util.emptyArray;

            /**
             * Creates a new GetIndicesResponse instance using the specified properties.
             * @function create
             * @memberof summa.proto.GetIndicesResponse
             * @static
             * @param {summa.proto.IGetIndicesResponse=} [properties] Properties to set
             * @returns {summa.proto.GetIndicesResponse} GetIndicesResponse instance
             */
            GetIndicesResponse.create = function create(properties) {
                return new GetIndicesResponse(properties);
            };

            return GetIndicesResponse;
        })();

        proto.IndexDocumentStreamRequest = (function() {

            /**
             * Properties of an IndexDocumentStreamRequest.
             * @memberof summa.proto
             * @interface IIndexDocumentStreamRequest
             * @property {string|null} [index_name] IndexDocumentStreamRequest index_name
             * @property {Array.<Uint8Array>|null} [documents] IndexDocumentStreamRequest documents
             * @property {summa.proto.ConflictStrategy|null} [conflict_strategy] IndexDocumentStreamRequest conflict_strategy
             */

            /**
             * Constructs a new IndexDocumentStreamRequest.
             * @memberof summa.proto
             * @classdesc Represents an IndexDocumentStreamRequest.
             * @implements IIndexDocumentStreamRequest
             * @constructor
             * @param {summa.proto.IIndexDocumentStreamRequest=} [properties] Properties to set
             */
            function IndexDocumentStreamRequest(properties) {
                this.documents = [];
                if (properties)
                    for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }

            /**
             * IndexDocumentStreamRequest index_name.
             * @member {string} index_name
             * @memberof summa.proto.IndexDocumentStreamRequest
             * @instance
             */
            IndexDocumentStreamRequest.prototype.index_name = "";

            /**
             * IndexDocumentStreamRequest documents.
             * @member {Array.<Uint8Array>} documents
             * @memberof summa.proto.IndexDocumentStreamRequest
             * @instance
             */
            IndexDocumentStreamRequest.prototype.documents = $util.emptyArray;

            /**
             * IndexDocumentStreamRequest conflict_strategy.
             * @member {summa.proto.ConflictStrategy|null|undefined} conflict_strategy
             * @memberof summa.proto.IndexDocumentStreamRequest
             * @instance
             */
            IndexDocumentStreamRequest.prototype.conflict_strategy = null;

            // OneOf field names bound to virtual getters and setters
            let $oneOfFields;

            /**
             * IndexDocumentStreamRequest _conflict_strategy.
             * @member {"conflict_strategy"|undefined} _conflict_strategy
             * @memberof summa.proto.IndexDocumentStreamRequest
             * @instance
             */
            Object.defineProperty(IndexDocumentStreamRequest.prototype, "_conflict_strategy", {
                get: $util.oneOfGetter($oneOfFields = ["conflict_strategy"]),
                set: $util.oneOfSetter($oneOfFields)
            });

            /**
             * Creates a new IndexDocumentStreamRequest instance using the specified properties.
             * @function create
             * @memberof summa.proto.IndexDocumentStreamRequest
             * @static
             * @param {summa.proto.IIndexDocumentStreamRequest=} [properties] Properties to set
             * @returns {summa.proto.IndexDocumentStreamRequest} IndexDocumentStreamRequest instance
             */
            IndexDocumentStreamRequest.create = function create(properties) {
                return new IndexDocumentStreamRequest(properties);
            };

            return IndexDocumentStreamRequest;
        })();

        proto.IndexDocumentStreamResponse = (function() {

            /**
             * Properties of an IndexDocumentStreamResponse.
             * @memberof summa.proto
             * @interface IIndexDocumentStreamResponse
             * @property {number|null} [elapsed_secs] IndexDocumentStreamResponse elapsed_secs
             * @property {number|Long|null} [success_docs] IndexDocumentStreamResponse success_docs
             * @property {number|Long|null} [failed_docs] IndexDocumentStreamResponse failed_docs
             */

            /**
             * Constructs a new IndexDocumentStreamResponse.
             * @memberof summa.proto
             * @classdesc Represents an IndexDocumentStreamResponse.
             * @implements IIndexDocumentStreamResponse
             * @constructor
             * @param {summa.proto.IIndexDocumentStreamResponse=} [properties] Properties to set
             */
            function IndexDocumentStreamResponse(properties) {
                if (properties)
                    for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }

            /**
             * IndexDocumentStreamResponse elapsed_secs.
             * @member {number} elapsed_secs
             * @memberof summa.proto.IndexDocumentStreamResponse
             * @instance
             */
            IndexDocumentStreamResponse.prototype.elapsed_secs = 0;

            /**
             * IndexDocumentStreamResponse success_docs.
             * @member {number|Long} success_docs
             * @memberof summa.proto.IndexDocumentStreamResponse
             * @instance
             */
            IndexDocumentStreamResponse.prototype.success_docs = $util.Long ? $util.Long.fromBits(0,0,true) : 0;

            /**
             * IndexDocumentStreamResponse failed_docs.
             * @member {number|Long} failed_docs
             * @memberof summa.proto.IndexDocumentStreamResponse
             * @instance
             */
            IndexDocumentStreamResponse.prototype.failed_docs = $util.Long ? $util.Long.fromBits(0,0,true) : 0;

            /**
             * Creates a new IndexDocumentStreamResponse instance using the specified properties.
             * @function create
             * @memberof summa.proto.IndexDocumentStreamResponse
             * @static
             * @param {summa.proto.IIndexDocumentStreamResponse=} [properties] Properties to set
             * @returns {summa.proto.IndexDocumentStreamResponse} IndexDocumentStreamResponse instance
             */
            IndexDocumentStreamResponse.create = function create(properties) {
                return new IndexDocumentStreamResponse(properties);
            };

            return IndexDocumentStreamResponse;
        })();

        proto.IndexDocumentRequest = (function() {

            /**
             * Properties of an IndexDocumentRequest.
             * @memberof summa.proto
             * @interface IIndexDocumentRequest
             * @property {string|null} [index_name] IndexDocumentRequest index_name
             * @property {Uint8Array|null} [document] IndexDocumentRequest document
             */

            /**
             * Constructs a new IndexDocumentRequest.
             * @memberof summa.proto
             * @classdesc Represents an IndexDocumentRequest.
             * @implements IIndexDocumentRequest
             * @constructor
             * @param {summa.proto.IIndexDocumentRequest=} [properties] Properties to set
             */
            function IndexDocumentRequest(properties) {
                if (properties)
                    for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }

            /**
             * IndexDocumentRequest index_name.
             * @member {string} index_name
             * @memberof summa.proto.IndexDocumentRequest
             * @instance
             */
            IndexDocumentRequest.prototype.index_name = "";

            /**
             * IndexDocumentRequest document.
             * @member {Uint8Array} document
             * @memberof summa.proto.IndexDocumentRequest
             * @instance
             */
            IndexDocumentRequest.prototype.document = $util.newBuffer([]);

            /**
             * Creates a new IndexDocumentRequest instance using the specified properties.
             * @function create
             * @memberof summa.proto.IndexDocumentRequest
             * @static
             * @param {summa.proto.IIndexDocumentRequest=} [properties] Properties to set
             * @returns {summa.proto.IndexDocumentRequest} IndexDocumentRequest instance
             */
            IndexDocumentRequest.create = function create(properties) {
                return new IndexDocumentRequest(properties);
            };

            return IndexDocumentRequest;
        })();

        proto.IndexDocumentResponse = (function() {

            /**
             * Properties of an IndexDocumentResponse.
             * @memberof summa.proto
             * @interface IIndexDocumentResponse
             */

            /**
             * Constructs a new IndexDocumentResponse.
             * @memberof summa.proto
             * @classdesc Represents an IndexDocumentResponse.
             * @implements IIndexDocumentResponse
             * @constructor
             * @param {summa.proto.IIndexDocumentResponse=} [properties] Properties to set
             */
            function IndexDocumentResponse(properties) {
                if (properties)
                    for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }

            /**
             * Creates a new IndexDocumentResponse instance using the specified properties.
             * @function create
             * @memberof summa.proto.IndexDocumentResponse
             * @static
             * @param {summa.proto.IIndexDocumentResponse=} [properties] Properties to set
             * @returns {summa.proto.IndexDocumentResponse} IndexDocumentResponse instance
             */
            IndexDocumentResponse.create = function create(properties) {
                return new IndexDocumentResponse(properties);
            };

            return IndexDocumentResponse;
        })();

        proto.MergeSegmentsRequest = (function() {

            /**
             * Properties of a MergeSegmentsRequest.
             * @memberof summa.proto
             * @interface IMergeSegmentsRequest
             * @property {string|null} [index_name] MergeSegmentsRequest index_name
             * @property {Array.<string>|null} [segment_ids] MergeSegmentsRequest segment_ids
             */

            /**
             * Constructs a new MergeSegmentsRequest.
             * @memberof summa.proto
             * @classdesc Represents a MergeSegmentsRequest.
             * @implements IMergeSegmentsRequest
             * @constructor
             * @param {summa.proto.IMergeSegmentsRequest=} [properties] Properties to set
             */
            function MergeSegmentsRequest(properties) {
                this.segment_ids = [];
                if (properties)
                    for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }

            /**
             * MergeSegmentsRequest index_name.
             * @member {string} index_name
             * @memberof summa.proto.MergeSegmentsRequest
             * @instance
             */
            MergeSegmentsRequest.prototype.index_name = "";

            /**
             * MergeSegmentsRequest segment_ids.
             * @member {Array.<string>} segment_ids
             * @memberof summa.proto.MergeSegmentsRequest
             * @instance
             */
            MergeSegmentsRequest.prototype.segment_ids = $util.emptyArray;

            /**
             * Creates a new MergeSegmentsRequest instance using the specified properties.
             * @function create
             * @memberof summa.proto.MergeSegmentsRequest
             * @static
             * @param {summa.proto.IMergeSegmentsRequest=} [properties] Properties to set
             * @returns {summa.proto.MergeSegmentsRequest} MergeSegmentsRequest instance
             */
            MergeSegmentsRequest.create = function create(properties) {
                return new MergeSegmentsRequest(properties);
            };

            return MergeSegmentsRequest;
        })();

        proto.MergeSegmentsResponse = (function() {

            /**
             * Properties of a MergeSegmentsResponse.
             * @memberof summa.proto
             * @interface IMergeSegmentsResponse
             * @property {string|null} [segment_id] MergeSegmentsResponse segment_id
             */

            /**
             * Constructs a new MergeSegmentsResponse.
             * @memberof summa.proto
             * @classdesc Represents a MergeSegmentsResponse.
             * @implements IMergeSegmentsResponse
             * @constructor
             * @param {summa.proto.IMergeSegmentsResponse=} [properties] Properties to set
             */
            function MergeSegmentsResponse(properties) {
                if (properties)
                    for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }

            /**
             * MergeSegmentsResponse segment_id.
             * @member {string|null|undefined} segment_id
             * @memberof summa.proto.MergeSegmentsResponse
             * @instance
             */
            MergeSegmentsResponse.prototype.segment_id = null;

            // OneOf field names bound to virtual getters and setters
            let $oneOfFields;

            /**
             * MergeSegmentsResponse _segment_id.
             * @member {"segment_id"|undefined} _segment_id
             * @memberof summa.proto.MergeSegmentsResponse
             * @instance
             */
            Object.defineProperty(MergeSegmentsResponse.prototype, "_segment_id", {
                get: $util.oneOfGetter($oneOfFields = ["segment_id"]),
                set: $util.oneOfSetter($oneOfFields)
            });

            /**
             * Creates a new MergeSegmentsResponse instance using the specified properties.
             * @function create
             * @memberof summa.proto.MergeSegmentsResponse
             * @static
             * @param {summa.proto.IMergeSegmentsResponse=} [properties] Properties to set
             * @returns {summa.proto.MergeSegmentsResponse} MergeSegmentsResponse instance
             */
            MergeSegmentsResponse.create = function create(properties) {
                return new MergeSegmentsResponse(properties);
            };

            return MergeSegmentsResponse;
        })();

        proto.SetIndexAliasRequest = (function() {

            /**
             * Properties of a SetIndexAliasRequest.
             * @memberof summa.proto
             * @interface ISetIndexAliasRequest
             * @property {string|null} [index_alias] SetIndexAliasRequest index_alias
             * @property {string|null} [index_name] SetIndexAliasRequest index_name
             */

            /**
             * Constructs a new SetIndexAliasRequest.
             * @memberof summa.proto
             * @classdesc Represents a SetIndexAliasRequest.
             * @implements ISetIndexAliasRequest
             * @constructor
             * @param {summa.proto.ISetIndexAliasRequest=} [properties] Properties to set
             */
            function SetIndexAliasRequest(properties) {
                if (properties)
                    for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }

            /**
             * SetIndexAliasRequest index_alias.
             * @member {string} index_alias
             * @memberof summa.proto.SetIndexAliasRequest
             * @instance
             */
            SetIndexAliasRequest.prototype.index_alias = "";

            /**
             * SetIndexAliasRequest index_name.
             * @member {string} index_name
             * @memberof summa.proto.SetIndexAliasRequest
             * @instance
             */
            SetIndexAliasRequest.prototype.index_name = "";

            /**
             * Creates a new SetIndexAliasRequest instance using the specified properties.
             * @function create
             * @memberof summa.proto.SetIndexAliasRequest
             * @static
             * @param {summa.proto.ISetIndexAliasRequest=} [properties] Properties to set
             * @returns {summa.proto.SetIndexAliasRequest} SetIndexAliasRequest instance
             */
            SetIndexAliasRequest.create = function create(properties) {
                return new SetIndexAliasRequest(properties);
            };

            return SetIndexAliasRequest;
        })();

        proto.SetIndexAliasResponse = (function() {

            /**
             * Properties of a SetIndexAliasResponse.
             * @memberof summa.proto
             * @interface ISetIndexAliasResponse
             * @property {string|null} [old_index_name] SetIndexAliasResponse old_index_name
             */

            /**
             * Constructs a new SetIndexAliasResponse.
             * @memberof summa.proto
             * @classdesc Represents a SetIndexAliasResponse.
             * @implements ISetIndexAliasResponse
             * @constructor
             * @param {summa.proto.ISetIndexAliasResponse=} [properties] Properties to set
             */
            function SetIndexAliasResponse(properties) {
                if (properties)
                    for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }

            /**
             * SetIndexAliasResponse old_index_name.
             * @member {string|null|undefined} old_index_name
             * @memberof summa.proto.SetIndexAliasResponse
             * @instance
             */
            SetIndexAliasResponse.prototype.old_index_name = null;

            // OneOf field names bound to virtual getters and setters
            let $oneOfFields;

            /**
             * SetIndexAliasResponse _old_index_name.
             * @member {"old_index_name"|undefined} _old_index_name
             * @memberof summa.proto.SetIndexAliasResponse
             * @instance
             */
            Object.defineProperty(SetIndexAliasResponse.prototype, "_old_index_name", {
                get: $util.oneOfGetter($oneOfFields = ["old_index_name"]),
                set: $util.oneOfSetter($oneOfFields)
            });

            /**
             * Creates a new SetIndexAliasResponse instance using the specified properties.
             * @function create
             * @memberof summa.proto.SetIndexAliasResponse
             * @static
             * @param {summa.proto.ISetIndexAliasResponse=} [properties] Properties to set
             * @returns {summa.proto.SetIndexAliasResponse} SetIndexAliasResponse instance
             */
            SetIndexAliasResponse.create = function create(properties) {
                return new SetIndexAliasResponse(properties);
            };

            return SetIndexAliasResponse;
        })();

        proto.DocumentsRequest = (function() {

            /**
             * Properties of a DocumentsRequest.
             * @memberof summa.proto
             * @interface IDocumentsRequest
             * @property {string|null} [index_name] DocumentsRequest index_name
             * @property {Array.<string>|null} [fields] DocumentsRequest fields
             */

            /**
             * Constructs a new DocumentsRequest.
             * @memberof summa.proto
             * @classdesc Represents a DocumentsRequest.
             * @implements IDocumentsRequest
             * @constructor
             * @param {summa.proto.IDocumentsRequest=} [properties] Properties to set
             */
            function DocumentsRequest(properties) {
                this.fields = [];
                if (properties)
                    for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }

            /**
             * DocumentsRequest index_name.
             * @member {string} index_name
             * @memberof summa.proto.DocumentsRequest
             * @instance
             */
            DocumentsRequest.prototype.index_name = "";

            /**
             * DocumentsRequest fields.
             * @member {Array.<string>} fields
             * @memberof summa.proto.DocumentsRequest
             * @instance
             */
            DocumentsRequest.prototype.fields = $util.emptyArray;

            /**
             * Creates a new DocumentsRequest instance using the specified properties.
             * @function create
             * @memberof summa.proto.DocumentsRequest
             * @static
             * @param {summa.proto.IDocumentsRequest=} [properties] Properties to set
             * @returns {summa.proto.DocumentsRequest} DocumentsRequest instance
             */
            DocumentsRequest.create = function create(properties) {
                return new DocumentsRequest(properties);
            };

            return DocumentsRequest;
        })();

        proto.DocumentsResponse = (function() {

            /**
             * Properties of a DocumentsResponse.
             * @memberof summa.proto
             * @interface IDocumentsResponse
             * @property {string|null} [document] DocumentsResponse document
             */

            /**
             * Constructs a new DocumentsResponse.
             * @memberof summa.proto
             * @classdesc Represents a DocumentsResponse.
             * @implements IDocumentsResponse
             * @constructor
             * @param {summa.proto.IDocumentsResponse=} [properties] Properties to set
             */
            function DocumentsResponse(properties) {
                if (properties)
                    for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }

            /**
             * DocumentsResponse document.
             * @member {string} document
             * @memberof summa.proto.DocumentsResponse
             * @instance
             */
            DocumentsResponse.prototype.document = "";

            /**
             * Creates a new DocumentsResponse instance using the specified properties.
             * @function create
             * @memberof summa.proto.DocumentsResponse
             * @static
             * @param {summa.proto.IDocumentsResponse=} [properties] Properties to set
             * @returns {summa.proto.DocumentsResponse} DocumentsResponse instance
             */
            DocumentsResponse.create = function create(properties) {
                return new DocumentsResponse(properties);
            };

            return DocumentsResponse;
        })();

        proto.VacuumIndexRequest = (function() {

            /**
             * Properties of a VacuumIndexRequest.
             * @memberof summa.proto
             * @interface IVacuumIndexRequest
             * @property {string|null} [index_name] VacuumIndexRequest index_name
             * @property {Array.<string>|null} [excluded_segments] VacuumIndexRequest excluded_segments
             */

            /**
             * Constructs a new VacuumIndexRequest.
             * @memberof summa.proto
             * @classdesc Represents a VacuumIndexRequest.
             * @implements IVacuumIndexRequest
             * @constructor
             * @param {summa.proto.IVacuumIndexRequest=} [properties] Properties to set
             */
            function VacuumIndexRequest(properties) {
                this.excluded_segments = [];
                if (properties)
                    for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }

            /**
             * VacuumIndexRequest index_name.
             * @member {string} index_name
             * @memberof summa.proto.VacuumIndexRequest
             * @instance
             */
            VacuumIndexRequest.prototype.index_name = "";

            /**
             * VacuumIndexRequest excluded_segments.
             * @member {Array.<string>} excluded_segments
             * @memberof summa.proto.VacuumIndexRequest
             * @instance
             */
            VacuumIndexRequest.prototype.excluded_segments = $util.emptyArray;

            /**
             * Creates a new VacuumIndexRequest instance using the specified properties.
             * @function create
             * @memberof summa.proto.VacuumIndexRequest
             * @static
             * @param {summa.proto.IVacuumIndexRequest=} [properties] Properties to set
             * @returns {summa.proto.VacuumIndexRequest} VacuumIndexRequest instance
             */
            VacuumIndexRequest.create = function create(properties) {
                return new VacuumIndexRequest(properties);
            };

            return VacuumIndexRequest;
        })();

        proto.VacuumIndexResponse = (function() {

            /**
             * Properties of a VacuumIndexResponse.
             * @memberof summa.proto
             * @interface IVacuumIndexResponse
             * @property {number|Long|null} [freed_space_bytes] VacuumIndexResponse freed_space_bytes
             */

            /**
             * Constructs a new VacuumIndexResponse.
             * @memberof summa.proto
             * @classdesc Represents a VacuumIndexResponse.
             * @implements IVacuumIndexResponse
             * @constructor
             * @param {summa.proto.IVacuumIndexResponse=} [properties] Properties to set
             */
            function VacuumIndexResponse(properties) {
                if (properties)
                    for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }

            /**
             * VacuumIndexResponse freed_space_bytes.
             * @member {number|Long} freed_space_bytes
             * @memberof summa.proto.VacuumIndexResponse
             * @instance
             */
            VacuumIndexResponse.prototype.freed_space_bytes = $util.Long ? $util.Long.fromBits(0,0,true) : 0;

            /**
             * Creates a new VacuumIndexResponse instance using the specified properties.
             * @function create
             * @memberof summa.proto.VacuumIndexResponse
             * @static
             * @param {summa.proto.IVacuumIndexResponse=} [properties] Properties to set
             * @returns {summa.proto.VacuumIndexResponse} VacuumIndexResponse instance
             */
            VacuumIndexResponse.create = function create(properties) {
                return new VacuumIndexResponse(properties);
            };

            return VacuumIndexResponse;
        })();

        proto.WarmupIndexRequest = (function() {

            /**
             * Properties of a WarmupIndexRequest.
             * @memberof summa.proto
             * @interface IWarmupIndexRequest
             * @property {string|null} [index_name] WarmupIndexRequest index_name
             * @property {boolean|null} [is_full] WarmupIndexRequest is_full
             */

            /**
             * Constructs a new WarmupIndexRequest.
             * @memberof summa.proto
             * @classdesc Represents a WarmupIndexRequest.
             * @implements IWarmupIndexRequest
             * @constructor
             * @param {summa.proto.IWarmupIndexRequest=} [properties] Properties to set
             */
            function WarmupIndexRequest(properties) {
                if (properties)
                    for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }

            /**
             * WarmupIndexRequest index_name.
             * @member {string} index_name
             * @memberof summa.proto.WarmupIndexRequest
             * @instance
             */
            WarmupIndexRequest.prototype.index_name = "";

            /**
             * WarmupIndexRequest is_full.
             * @member {boolean} is_full
             * @memberof summa.proto.WarmupIndexRequest
             * @instance
             */
            WarmupIndexRequest.prototype.is_full = false;

            /**
             * Creates a new WarmupIndexRequest instance using the specified properties.
             * @function create
             * @memberof summa.proto.WarmupIndexRequest
             * @static
             * @param {summa.proto.IWarmupIndexRequest=} [properties] Properties to set
             * @returns {summa.proto.WarmupIndexRequest} WarmupIndexRequest instance
             */
            WarmupIndexRequest.create = function create(properties) {
                return new WarmupIndexRequest(properties);
            };

            return WarmupIndexRequest;
        })();

        proto.WarmupIndexResponse = (function() {

            /**
             * Properties of a WarmupIndexResponse.
             * @memberof summa.proto
             * @interface IWarmupIndexResponse
             * @property {number|null} [elapsed_secs] WarmupIndexResponse elapsed_secs
             */

            /**
             * Constructs a new WarmupIndexResponse.
             * @memberof summa.proto
             * @classdesc Represents a WarmupIndexResponse.
             * @implements IWarmupIndexResponse
             * @constructor
             * @param {summa.proto.IWarmupIndexResponse=} [properties] Properties to set
             */
            function WarmupIndexResponse(properties) {
                if (properties)
                    for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }

            /**
             * WarmupIndexResponse elapsed_secs.
             * @member {number} elapsed_secs
             * @memberof summa.proto.WarmupIndexResponse
             * @instance
             */
            WarmupIndexResponse.prototype.elapsed_secs = 0;

            /**
             * Creates a new WarmupIndexResponse instance using the specified properties.
             * @function create
             * @memberof summa.proto.WarmupIndexResponse
             * @static
             * @param {summa.proto.IWarmupIndexResponse=} [properties] Properties to set
             * @returns {summa.proto.WarmupIndexResponse} WarmupIndexResponse instance
             */
            WarmupIndexResponse.create = function create(properties) {
                return new WarmupIndexResponse(properties);
            };

            return WarmupIndexResponse;
        })();

        /**
         * Compression enum.
         * @name summa.proto.Compression
         * @enum {number}
         * @property {number} None=0 None value
         * @property {number} Brotli=1 Brotli value
         * @property {number} Lz4=2 Lz4 value
         * @property {number} Snappy=3 Snappy value
         * @property {number} Zstd=4 Zstd value
         * @property {number} Zstd7=5 Zstd7 value
         * @property {number} Zstd9=6 Zstd9 value
         * @property {number} Zstd14=7 Zstd14 value
         * @property {number} Zstd19=8 Zstd19 value
         * @property {number} Zstd22=9 Zstd22 value
         */
        proto.Compression = (function() {
            const valuesById = {}, values = Object.create(valuesById);
            values[valuesById[0] = "None"] = 0;
            values[valuesById[4] = "Zstd"] = 4;
            values[valuesById[5] = "Zstd7"] = 5;
            values[valuesById[6] = "Zstd9"] = 6;
            values[valuesById[7] = "Zstd14"] = 7;
            values[valuesById[8] = "Zstd19"] = 8;
            values[valuesById[9] = "Zstd22"] = 9;
            return values;
        })();

        proto.FileEngineConfig = (function() {

            /**
             * Properties of a FileEngineConfig.
             * @memberof summa.proto
             * @interface IFileEngineConfig
             * @property {string|null} [path] FileEngineConfig path
             */

            /**
             * Constructs a new FileEngineConfig.
             * @memberof summa.proto
             * @classdesc Represents a FileEngineConfig.
             * @implements IFileEngineConfig
             * @constructor
             * @param {summa.proto.IFileEngineConfig=} [properties] Properties to set
             */
            function FileEngineConfig(properties) {
                if (properties)
                    for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }

            /**
             * FileEngineConfig path.
             * @member {string} path
             * @memberof summa.proto.FileEngineConfig
             * @instance
             */
            FileEngineConfig.prototype.path = "";

            /**
             * Creates a new FileEngineConfig instance using the specified properties.
             * @function create
             * @memberof summa.proto.FileEngineConfig
             * @static
             * @param {summa.proto.IFileEngineConfig=} [properties] Properties to set
             * @returns {summa.proto.FileEngineConfig} FileEngineConfig instance
             */
            FileEngineConfig.create = function create(properties) {
                return new FileEngineConfig(properties);
            };

            return FileEngineConfig;
        })();

        proto.MemoryEngineConfig = (function() {

            /**
             * Properties of a MemoryEngineConfig.
             * @memberof summa.proto
             * @interface IMemoryEngineConfig
             * @property {string|null} [schema] MemoryEngineConfig schema
             */

            /**
             * Constructs a new MemoryEngineConfig.
             * @memberof summa.proto
             * @classdesc Represents a MemoryEngineConfig.
             * @implements IMemoryEngineConfig
             * @constructor
             * @param {summa.proto.IMemoryEngineConfig=} [properties] Properties to set
             */
            function MemoryEngineConfig(properties) {
                if (properties)
                    for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }

            /**
             * MemoryEngineConfig schema.
             * @member {string} schema
             * @memberof summa.proto.MemoryEngineConfig
             * @instance
             */
            MemoryEngineConfig.prototype.schema = "";

            /**
             * Creates a new MemoryEngineConfig instance using the specified properties.
             * @function create
             * @memberof summa.proto.MemoryEngineConfig
             * @static
             * @param {summa.proto.IMemoryEngineConfig=} [properties] Properties to set
             * @returns {summa.proto.MemoryEngineConfig} MemoryEngineConfig instance
             */
            MemoryEngineConfig.create = function create(properties) {
                return new MemoryEngineConfig(properties);
            };

            return MemoryEngineConfig;
        })();

        proto.CacheConfig = (function() {

            /**
             * Properties of a CacheConfig.
             * @memberof summa.proto
             * @interface ICacheConfig
             * @property {number|Long|null} [cache_size] CacheConfig cache_size
             */

            /**
             * Constructs a new CacheConfig.
             * @memberof summa.proto
             * @classdesc Represents a CacheConfig.
             * @implements ICacheConfig
             * @constructor
             * @param {summa.proto.ICacheConfig=} [properties] Properties to set
             */
            function CacheConfig(properties) {
                if (properties)
                    for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }

            /**
             * CacheConfig cache_size.
             * @member {number|Long} cache_size
             * @memberof summa.proto.CacheConfig
             * @instance
             */
            CacheConfig.prototype.cache_size = $util.Long ? $util.Long.fromBits(0,0,true) : 0;

            /**
             * Creates a new CacheConfig instance using the specified properties.
             * @function create
             * @memberof summa.proto.CacheConfig
             * @static
             * @param {summa.proto.ICacheConfig=} [properties] Properties to set
             * @returns {summa.proto.CacheConfig} CacheConfig instance
             */
            CacheConfig.create = function create(properties) {
                return new CacheConfig(properties);
            };

            return CacheConfig;
        })();

        proto.RemoteEngineConfig = (function() {

            /**
             * Properties of a RemoteEngineConfig.
             * @memberof summa.proto
             * @interface IRemoteEngineConfig
             * @property {string|null} [method] RemoteEngineConfig method
             * @property {string|null} [url_template] RemoteEngineConfig url_template
             * @property {Object.<string,string>|null} [headers_template] RemoteEngineConfig headers_template
             * @property {summa.proto.ICacheConfig|null} [cache_config] RemoteEngineConfig cache_config
             * @property {number|null} [timeout_ms] RemoteEngineConfig timeout_ms
             */

            /**
             * Constructs a new RemoteEngineConfig.
             * @memberof summa.proto
             * @classdesc Represents a RemoteEngineConfig.
             * @implements IRemoteEngineConfig
             * @constructor
             * @param {summa.proto.IRemoteEngineConfig=} [properties] Properties to set
             */
            function RemoteEngineConfig(properties) {
                this.headers_template = {};
                if (properties)
                    for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }

            /**
             * RemoteEngineConfig method.
             * @member {string} method
             * @memberof summa.proto.RemoteEngineConfig
             * @instance
             */
            RemoteEngineConfig.prototype.method = "";

            /**
             * RemoteEngineConfig url_template.
             * @member {string} url_template
             * @memberof summa.proto.RemoteEngineConfig
             * @instance
             */
            RemoteEngineConfig.prototype.url_template = "";

            /**
             * RemoteEngineConfig headers_template.
             * @member {Object.<string,string>} headers_template
             * @memberof summa.proto.RemoteEngineConfig
             * @instance
             */
            RemoteEngineConfig.prototype.headers_template = $util.emptyObject;

            /**
             * RemoteEngineConfig cache_config.
             * @member {summa.proto.ICacheConfig|null|undefined} cache_config
             * @memberof summa.proto.RemoteEngineConfig
             * @instance
             */
            RemoteEngineConfig.prototype.cache_config = null;

            /**
             * RemoteEngineConfig timeout_ms.
             * @member {number|null|undefined} timeout_ms
             * @memberof summa.proto.RemoteEngineConfig
             * @instance
             */
            RemoteEngineConfig.prototype.timeout_ms = null;

            // OneOf field names bound to virtual getters and setters
            let $oneOfFields;

            /**
             * RemoteEngineConfig _timeout_ms.
             * @member {"timeout_ms"|undefined} _timeout_ms
             * @memberof summa.proto.RemoteEngineConfig
             * @instance
             */
            Object.defineProperty(RemoteEngineConfig.prototype, "_timeout_ms", {
                get: $util.oneOfGetter($oneOfFields = ["timeout_ms"]),
                set: $util.oneOfSetter($oneOfFields)
            });

            /**
             * Creates a new RemoteEngineConfig instance using the specified properties.
             * @function create
             * @memberof summa.proto.RemoteEngineConfig
             * @static
             * @param {summa.proto.IRemoteEngineConfig=} [properties] Properties to set
             * @returns {summa.proto.RemoteEngineConfig} RemoteEngineConfig instance
             */
            RemoteEngineConfig.create = function create(properties) {
                return new RemoteEngineConfig(properties);
            };

            return RemoteEngineConfig;
        })();

        proto.LogMergePolicy = (function() {

            /**
             * Properties of a LogMergePolicy.
             * @memberof summa.proto
             * @interface ILogMergePolicy
             * @property {boolean|null} [is_frozen] LogMergePolicy is_frozen
             */

            /**
             * Constructs a new LogMergePolicy.
             * @memberof summa.proto
             * @classdesc Represents a LogMergePolicy.
             * @implements ILogMergePolicy
             * @constructor
             * @param {summa.proto.ILogMergePolicy=} [properties] Properties to set
             */
            function LogMergePolicy(properties) {
                if (properties)
                    for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }

            /**
             * LogMergePolicy is_frozen.
             * @member {boolean} is_frozen
             * @memberof summa.proto.LogMergePolicy
             * @instance
             */
            LogMergePolicy.prototype.is_frozen = false;

            /**
             * Creates a new LogMergePolicy instance using the specified properties.
             * @function create
             * @memberof summa.proto.LogMergePolicy
             * @static
             * @param {summa.proto.ILogMergePolicy=} [properties] Properties to set
             * @returns {summa.proto.LogMergePolicy} LogMergePolicy instance
             */
            LogMergePolicy.create = function create(properties) {
                return new LogMergePolicy(properties);
            };

            return LogMergePolicy;
        })();

        proto.TemporalMergePolicy = (function() {

            /**
             * Properties of a TemporalMergePolicy.
             * @memberof summa.proto
             * @interface ITemporalMergePolicy
             * @property {number|Long|null} [merge_older_then_secs] TemporalMergePolicy merge_older_then_secs
             */

            /**
             * Constructs a new TemporalMergePolicy.
             * @memberof summa.proto
             * @classdesc Represents a TemporalMergePolicy.
             * @implements ITemporalMergePolicy
             * @constructor
             * @param {summa.proto.ITemporalMergePolicy=} [properties] Properties to set
             */
            function TemporalMergePolicy(properties) {
                if (properties)
                    for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }

            /**
             * TemporalMergePolicy merge_older_then_secs.
             * @member {number|Long} merge_older_then_secs
             * @memberof summa.proto.TemporalMergePolicy
             * @instance
             */
            TemporalMergePolicy.prototype.merge_older_then_secs = $util.Long ? $util.Long.fromBits(0,0,true) : 0;

            /**
             * Creates a new TemporalMergePolicy instance using the specified properties.
             * @function create
             * @memberof summa.proto.TemporalMergePolicy
             * @static
             * @param {summa.proto.ITemporalMergePolicy=} [properties] Properties to set
             * @returns {summa.proto.TemporalMergePolicy} TemporalMergePolicy instance
             */
            TemporalMergePolicy.create = function create(properties) {
                return new TemporalMergePolicy(properties);
            };

            return TemporalMergePolicy;
        })();

        proto.IndexEngineConfig = (function() {

            /**
             * Properties of an IndexEngineConfig.
             * @memberof summa.proto
             * @interface IIndexEngineConfig
             * @property {summa.proto.IFileEngineConfig|null} [file] IndexEngineConfig file
             * @property {summa.proto.IMemoryEngineConfig|null} [memory] IndexEngineConfig memory
             * @property {summa.proto.IRemoteEngineConfig|null} [remote] IndexEngineConfig remote
             * @property {summa.proto.IMergePolicy|null} [merge_policy] IndexEngineConfig merge_policy
             * @property {summa.proto.IQueryParserConfig|null} [query_parser_config] IndexEngineConfig query_parser_config
             */

            /**
             * Constructs a new IndexEngineConfig.
             * @memberof summa.proto
             * @classdesc Represents an IndexEngineConfig.
             * @implements IIndexEngineConfig
             * @constructor
             * @param {summa.proto.IIndexEngineConfig=} [properties] Properties to set
             */
            function IndexEngineConfig(properties) {
                if (properties)
                    for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }

            /**
             * IndexEngineConfig file.
             * @member {summa.proto.IFileEngineConfig|null|undefined} file
             * @memberof summa.proto.IndexEngineConfig
             * @instance
             */
            IndexEngineConfig.prototype.file = null;

            /**
             * IndexEngineConfig memory.
             * @member {summa.proto.IMemoryEngineConfig|null|undefined} memory
             * @memberof summa.proto.IndexEngineConfig
             * @instance
             */
            IndexEngineConfig.prototype.memory = null;

            /**
             * IndexEngineConfig remote.
             * @member {summa.proto.IRemoteEngineConfig|null|undefined} remote
             * @memberof summa.proto.IndexEngineConfig
             * @instance
             */
            IndexEngineConfig.prototype.remote = null;

            /**
             * IndexEngineConfig merge_policy.
             * @member {summa.proto.IMergePolicy|null|undefined} merge_policy
             * @memberof summa.proto.IndexEngineConfig
             * @instance
             */
            IndexEngineConfig.prototype.merge_policy = null;

            /**
             * IndexEngineConfig query_parser_config.
             * @member {summa.proto.IQueryParserConfig|null|undefined} query_parser_config
             * @memberof summa.proto.IndexEngineConfig
             * @instance
             */
            IndexEngineConfig.prototype.query_parser_config = null;

            // OneOf field names bound to virtual getters and setters
            let $oneOfFields;

            /**
             * IndexEngineConfig config.
             * @member {"file"|"memory"|"remote"|undefined} config
             * @memberof summa.proto.IndexEngineConfig
             * @instance
             */
            Object.defineProperty(IndexEngineConfig.prototype, "config", {
                get: $util.oneOfGetter($oneOfFields = ["file", "memory", "remote"]),
                set: $util.oneOfSetter($oneOfFields)
            });

            /**
             * Creates a new IndexEngineConfig instance using the specified properties.
             * @function create
             * @memberof summa.proto.IndexEngineConfig
             * @static
             * @param {summa.proto.IIndexEngineConfig=} [properties] Properties to set
             * @returns {summa.proto.IndexEngineConfig} IndexEngineConfig instance
             */
            IndexEngineConfig.create = function create(properties) {
                return new IndexEngineConfig(properties);
            };

            return IndexEngineConfig;
        })();

        proto.IndexDescription = (function() {

            /**
             * Properties of an IndexDescription.
             * @memberof summa.proto
             * @interface IIndexDescription
             * @property {string|null} [index_name] IndexDescription index_name
             * @property {Array.<string>|null} [index_aliases] IndexDescription index_aliases
             * @property {summa.proto.IIndexEngineConfig|null} [index_engine] IndexDescription index_engine
             * @property {number|Long|null} [num_docs] IndexDescription num_docs
             * @property {summa.proto.Compression|null} [compression] IndexDescription compression
             * @property {summa.proto.IIndexAttributes|null} [index_attributes] IndexDescription index_attributes
             */

            /**
             * Constructs a new IndexDescription.
             * @memberof summa.proto
             * @classdesc Represents an IndexDescription.
             * @implements IIndexDescription
             * @constructor
             * @param {summa.proto.IIndexDescription=} [properties] Properties to set
             */
            function IndexDescription(properties) {
                this.index_aliases = [];
                if (properties)
                    for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }

            /**
             * IndexDescription index_name.
             * @member {string} index_name
             * @memberof summa.proto.IndexDescription
             * @instance
             */
            IndexDescription.prototype.index_name = "";

            /**
             * IndexDescription index_aliases.
             * @member {Array.<string>} index_aliases
             * @memberof summa.proto.IndexDescription
             * @instance
             */
            IndexDescription.prototype.index_aliases = $util.emptyArray;

            /**
             * IndexDescription index_engine.
             * @member {summa.proto.IIndexEngineConfig|null|undefined} index_engine
             * @memberof summa.proto.IndexDescription
             * @instance
             */
            IndexDescription.prototype.index_engine = null;

            /**
             * IndexDescription num_docs.
             * @member {number|Long} num_docs
             * @memberof summa.proto.IndexDescription
             * @instance
             */
            IndexDescription.prototype.num_docs = $util.Long ? $util.Long.fromBits(0,0,true) : 0;

            /**
             * IndexDescription compression.
             * @member {summa.proto.Compression} compression
             * @memberof summa.proto.IndexDescription
             * @instance
             */
            IndexDescription.prototype.compression = 0;

            /**
             * IndexDescription index_attributes.
             * @member {summa.proto.IIndexAttributes|null|undefined} index_attributes
             * @memberof summa.proto.IndexDescription
             * @instance
             */
            IndexDescription.prototype.index_attributes = null;

            /**
             * Creates a new IndexDescription instance using the specified properties.
             * @function create
             * @memberof summa.proto.IndexDescription
             * @static
             * @param {summa.proto.IIndexDescription=} [properties] Properties to set
             * @returns {summa.proto.IndexDescription} IndexDescription instance
             */
            IndexDescription.create = function create(properties) {
                return new IndexDescription(properties);
            };

            return IndexDescription;
        })();

        proto.IndexDocumentOperation = (function() {

            /**
             * Properties of an IndexDocumentOperation.
             * @memberof summa.proto
             * @interface IIndexDocumentOperation
             * @property {Uint8Array|null} [document] IndexDocumentOperation document
             */

            /**
             * Constructs a new IndexDocumentOperation.
             * @memberof summa.proto
             * @classdesc Represents an IndexDocumentOperation.
             * @implements IIndexDocumentOperation
             * @constructor
             * @param {summa.proto.IIndexDocumentOperation=} [properties] Properties to set
             */
            function IndexDocumentOperation(properties) {
                if (properties)
                    for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }

            /**
             * IndexDocumentOperation document.
             * @member {Uint8Array} document
             * @memberof summa.proto.IndexDocumentOperation
             * @instance
             */
            IndexDocumentOperation.prototype.document = $util.newBuffer([]);

            /**
             * Creates a new IndexDocumentOperation instance using the specified properties.
             * @function create
             * @memberof summa.proto.IndexDocumentOperation
             * @static
             * @param {summa.proto.IIndexDocumentOperation=} [properties] Properties to set
             * @returns {summa.proto.IndexDocumentOperation} IndexDocumentOperation instance
             */
            IndexDocumentOperation.create = function create(properties) {
                return new IndexDocumentOperation(properties);
            };

            return IndexDocumentOperation;
        })();

        proto.IndexOperation = (function() {

            /**
             * Properties of an IndexOperation.
             * @memberof summa.proto
             * @interface IIndexOperation
             * @property {summa.proto.IIndexDocumentOperation|null} [index_document] IndexOperation index_document
             */

            /**
             * Constructs a new IndexOperation.
             * @memberof summa.proto
             * @classdesc Represents an IndexOperation.
             * @implements IIndexOperation
             * @constructor
             * @param {summa.proto.IIndexOperation=} [properties] Properties to set
             */
            function IndexOperation(properties) {
                if (properties)
                    for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }

            /**
             * IndexOperation index_document.
             * @member {summa.proto.IIndexDocumentOperation|null|undefined} index_document
             * @memberof summa.proto.IndexOperation
             * @instance
             */
            IndexOperation.prototype.index_document = null;

            // OneOf field names bound to virtual getters and setters
            let $oneOfFields;

            /**
             * IndexOperation operation.
             * @member {"index_document"|undefined} operation
             * @memberof summa.proto.IndexOperation
             * @instance
             */
            Object.defineProperty(IndexOperation.prototype, "operation", {
                get: $util.oneOfGetter($oneOfFields = ["index_document"]),
                set: $util.oneOfSetter($oneOfFields)
            });

            /**
             * Creates a new IndexOperation instance using the specified properties.
             * @function create
             * @memberof summa.proto.IndexOperation
             * @static
             * @param {summa.proto.IIndexOperation=} [properties] Properties to set
             * @returns {summa.proto.IndexOperation} IndexOperation instance
             */
            IndexOperation.create = function create(properties) {
                return new IndexOperation(properties);
            };

            return IndexOperation;
        })();

        return proto;
    })();

    return summa;
})();

export { $root as default };
