import type { CandidateScores, L1Ranking } from "./generated/hermes";
export type { CandidateScores, FeatureTransform, L1Ranking, PassageScores } from "./generated/hermes";
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
  candidateScores?: CandidateScores;
}

/** Detailed timing breakdown for search phases (all values in microseconds). */
export interface SearchTimings {
  searchUs: number;
  rerankUs: number;
  loadUs: number;
  totalUs: number;
  candidateScoringUs?: number;
}

/** Search response with hits and metadata. */
export interface SearchResponse {
  hits: SearchHit[];
  totalHits: number;
  tookMs: number;
  timings?: SearchTimings;
  rankingMethod?: string;
  truncated?: boolean;
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
  candidateScoringVersion?: number;
  unpreparedCandidateFields?: string[];
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
  /** Passed to the field's tokenizer; a dynamic stemmer reads a language list ("ru,en"). */
  tokenizerHint?: string;
}

export interface MatchQuery {
  field: string;
  text: string;
  /** Passed to the field's tokenizer; a dynamic stemmer reads a language list ("ru,en"). */
  tokenizerHint?: string;
  /** Proximity rescoring weight for the top BM25 candidates (0 = off). */
  proximityWeight?: number;
  /** Unordered proximity window size (0 = 8). */
  proximityWindow?: number;
  /** Approximate text top-k: skip candidates below threshold / heapFactor (<= 1 = exact). */
  heapFactor?: number;
  /** Keep only the highest-idf terms of a long query (0 = all). */
  maxTerms?: number;
}

/** Consecutive-terms query on a field indexed with token positions. */
export interface PhraseQuery {
  field: string;
  text: string;
  /** Max positions between terms; 0 = exact phrase. */
  slop?: number;
  tokenizerHint?: string;
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
  /** LSP/0 gamma (unset = depth-derived, 0 = exhaustive) */
  lspGamma?: number;
}

export interface DenseVectorQuery {
  field: string;
  vector: number[];
  /** Number of clusters to probe (for IVF indexes) */
  nprobe?: number;
  combiner?: Combiner;
  /** Temperature for LogSumExp combiner (default: 1.5) */
  combinerTemperature?: number;
  /** K for WeightedTopK combiner (default: 5) */
  combinerTopK?: number;
  /** Decay for WeightedTopK combiner (default: 0.7) */
  combinerDecay?: number;
}

export interface BinaryDenseVectorQuery {
  field: string;
  /** Packed-bit query vector (ceil(dim/8) bytes) */
  vector: Uint8Array;
  combiner?: Combiner;
  combinerTemperature?: number;
  combinerTopK?: number;
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

/** Weighted sub-query for hybrid fusion */
export interface WeightedQuery {
  /** Required for L1/export; unique coefficient identity. */
  name?: string;
  scope?: "document" | "chunk";
  scoreOnly?: boolean;
  query: Query;
  /** Contribution scale (default: 1.0) */
  weight?: number;
}

/**
 * Union fusion of independently-executed sub-queries (e.g. sparse + dense).
 * Unlike the reranker, keeps documents found by ANY sub-query.
 * Only valid at the top level of a search request.
 */
export interface FusionQuery {
  filters?: Query[];
  candidateDepth?: number;
  queries: WeightedQuery[];
  /** Fusion method (default: "rrf") */
  method?: "rrf" | "normalized_weighted_sum";
  /** RRF rank constant (default: 60) */
  rrfK?: number;
  /**
   * Combiner for fused per-chunk (ordinal) scores into a document score.
   * Fusion runs at chunk granularity: same-chunk hits across sub-queries
   * compound. Default: "max" (recommended for RRF score magnitudes).
   */
  combiner?: Combiner;
}

/** Discriminated union matching proto Query oneof. Exactly one key must be set. */
export type Query =
  | { term: TermQuery }
  | { match: MatchQuery }
  | { phrase: PhraseQuery }
  | { boolean: BooleanQuery }
  | { sparseVector: SparseVectorQuery }
  | { denseVector: DenseVectorQuery }
  | { binaryDenseVector: BinaryDenseVectorQuery }
  | { boost: BoostQuery }
  | { all: AllQuery }
  | { range: RangeQuery }
  | { prefix: PrefixQuery }
  | { fusion: FusionQuery };

// =============================================================================
// Reranker (mirrors proto Reranker)
// =============================================================================

export interface Reranker {
  field: string;
  /** Query vector (f32, for dense fields) */
  vector?: number[];
  combiner?: Combiner;
  combinerTemperature?: number;
  combinerTopK?: number;
  combinerDecay?: number;
  /** Matryoshka pre-filter dims (0 = disabled) */
  matryoshkaDims?: number;
  /** Query vector (packed bits, for binary dense fields) */
  binaryVector?: Uint8Array;
  /** Reciprocal Rank Fusion k (0 = disabled, typical: 60) */
  rrfK?: number;
}

// =============================================================================
// SearchRequest (mirrors proto SearchRequest, minus index_name)
// =============================================================================

export interface SearchRequest {
  l1?: Pick<L1Ranking, "weights"> & Partial<Omit<L1Ranking, "weights">>;
  scoreExport?: { passagesPerDocument?: number; allPassages?: boolean };
  query: Query;
  limit?: number;
  offset?: number;
  fieldsToLoad?: string[];
  reranker?: Reranker;
  /** Shared first-stage candidate pool (0 = result window, maximum: 2x). */
  candidateLimit?: number;
  /** Anytime mode: wall-clock budget of the scoring phase in ms (0 = exact). */
  timeBudgetMs?: number;
}
