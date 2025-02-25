// =============================================================================
// Response types
// =============================================================================

/** A document with field values. */
export interface Document {
  fields: Record<string, any>;
}

/** Unique document address: segment + local doc_id. */
export interface DocAddress {
  segmentId: string;
  docId: number;
}

/** Score contribution from a specific ordinal in a multi-valued field. */
export interface OrdinalScore {
  ordinal: number;
  score: number;
}

/** A single search result. */
export interface SearchHit {
  address: DocAddress;
  score: number;
  fields: Record<string, any>;
  ordinalScores: OrdinalScore[];
}

/** Detailed timing breakdown for search phases (all values in microseconds). */
export interface SearchTimings {
  searchUs: number;
  rerankUs: number;
  loadUs: number;
  totalUs: number;
}

/** Search response with hits and metadata. */
export interface SearchResponse {
  hits: SearchHit[];
  totalHits: number;
  tookMs: number;
  timings?: SearchTimings;
}

/** Per-field vector statistics. */
export interface VectorFieldStats {
  fieldName: string;
  vectorType: string;
  totalVectors: number;
  dimension: number;
}

/** Information about an index. */
export interface IndexInfo {
  indexName: string;
  numDocs: number;
  numSegments: number;
  schema: string;
  vectorStats: VectorFieldStats[];
}

// =============================================================================
// Multi-value score combiner (mirrors proto MultiValueCombiner)
// =============================================================================

export type Combiner = "log_sum_exp" | "max" | "avg" | "sum" | "weighted_top_k";

// =============================================================================
// Query types (mirrors proto Query oneof)
// =============================================================================

export interface TermQuery {
  field: string;
  term: string;
}

export interface MatchQuery {
  field: string;
  text: string;
}

export interface BooleanQuery {
  must?: Query[];
  should?: Query[];
  mustNot?: Query[];
}

export interface BoostQuery {
  query: Query;
  boost: number;
}

export interface AllQuery {}

export interface SparseVectorQuery {
  field: string;
  /** Pre-computed token indices */
  indices?: number[];
  /** Pre-computed token values */
  values?: number[];
  /** Raw text (tokenized server-side if tokenizer configured) */
  text?: string;
  combiner?: Combiner;
  /** Approximate search factor (1.0 = exact, 0.8 = ~20% faster) */
  heapFactor?: number;
  /** Temperature for LogSumExp combiner (default: 1.5) */
  combinerTemperature?: number;
  /** K for WeightedTopK combiner (default: 5) */
  combinerTopK?: number;
  /** Decay for WeightedTopK combiner (default: 0.7) */
  combinerDecay?: number;
  /** Min abs(weight) for query dims (0 = no filtering) */
  weightThreshold?: number;
  /** Max query dimensions to process (0 = all) */
  maxQueryDims?: number;
  /** Fraction of query dims to keep (0-1, e.g. 0.1 = top 10%) */
  pruning?: number;
}

export interface DenseVectorQuery {
  field: string;
  vector: number[];
  /** Number of clusters to probe (for IVF indexes) */
  nprobe?: number;
  /** Re-ranking factor (multiplied by k) */
  rerankFactor?: number;
  combiner?: Combiner;
  /** Temperature for LogSumExp combiner (default: 1.5) */
  combinerTemperature?: number;
  /** K for WeightedTopK combiner (default: 5) */
  combinerTopK?: number;
  /** Decay for WeightedTopK combiner (default: 0.7) */
  combinerDecay?: number;
}

export interface RangeQuery {
  field: string;
  /** u64 bounds (inclusive) */
  minU64?: number;
  maxU64?: number;
  /** i64 bounds (inclusive) */
  minI64?: number;
  maxI64?: number;
  /** f64 bounds (inclusive) */
  minF64?: number;
  maxF64?: number;
}

export interface PrefixQuery {
  field: string;
  prefix: string;
}

/** Discriminated union matching proto Query oneof. Exactly one key must be set. */
export type Query =
  | { term: TermQuery }
  | { match: MatchQuery }
  | { boolean: BooleanQuery }
  | { sparseVector: SparseVectorQuery }
  | { denseVector: DenseVectorQuery }
  | { boost: BoostQuery }
  | { all: AllQuery }
  | { range: RangeQuery }
  | { prefix: PrefixQuery };

// =============================================================================
// Reranker (mirrors proto Reranker)
// =============================================================================

export interface Reranker {
  field: string;
  vector: number[];
  /** L1 candidate count (0 = 10x final limit) */
  limit?: number;
  combiner?: Combiner;
  combinerTemperature?: number;
  combinerTopK?: number;
  combinerDecay?: number;
  /** Matryoshka pre-filter dims (0 = disabled) */
  matryoshkaDims?: number;
}

// =============================================================================
// SearchRequest (mirrors proto SearchRequest, minus index_name)
// =============================================================================

export interface SearchRequest {
  query: Query;
  limit?: number;
  offset?: number;
  fieldsToLoad?: string[];
  reranker?: Reranker;
}
