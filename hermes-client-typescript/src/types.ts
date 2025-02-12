/** A document with field values. */
export interface Document {
  fields: Record<string, any>;
}

/** Unique document address: segment + local doc_id. */
export interface DocAddress {
  segmentId: string;
  docId: number;
}

/** A single search result. */
export interface SearchHit {
  address: DocAddress;
  score: number;
  fields: Record<string, any>;
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
  vectorType: string; // "dense" or "sparse"
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

/** Term query: (field, term) */
export type TermQuery = [string, string];

/** Boolean query with must/should/mustNot clauses */
export interface BooleanQueryDef {
  must?: TermQuery[];
  should?: TermQuery[];
  mustNot?: TermQuery[];
}

/** Sparse vector query: (field, indices, values) */
export type SparseVectorQueryDef = [string, number[], number[]];

/** Sparse text query: (field, text) */
export type SparseTextQueryDef = [string, string];

/** Dense vector query: (field, vector) */
export type DenseVectorQueryDef = [string, number[]];

/** Reranker: (field, queryVector, l1Limit) */
export type RerankerDef = [string, number[], number];

/** Score combiner for multi-value fields */
export type Combiner = "sum" | "max" | "avg";

/** Reranker combiner */
export type RerankerCombiner =
  | "log_sum_exp"
  | "max"
  | "avg"
  | "sum"
  | "weighted_top_k";

/** Fast-field filter condition */
export interface FilterDef {
  field: string;
  eq_u64?: number;
  eq_i64?: number;
  eq_f64?: number;
  eq_text?: string;
  range?: { min?: number; max?: number };
  in_text?: string[];
  in_u64?: number[];
  in_i64?: number[];
}

/** Search options */
export interface SearchOptions {
  term?: TermQuery;
  boolean?: BooleanQueryDef;
  sparseVector?: SparseVectorQueryDef;
  sparseText?: SparseTextQueryDef;
  denseVector?: DenseVectorQueryDef;
  nprobe?: number;
  rerankFactor?: number;
  heapFactor?: number;
  combiner?: Combiner;
  limit?: number;
  offset?: number;
  fieldsToLoad?: string[];
  reranker?: RerankerDef;
  rerankerCombiner?: RerankerCombiner;
  /** Matryoshka pre-filter: number of leading dimensions for cheap approximate
   *  scoring before full-dimension exact reranking (0 or undefined = disabled). */
  matryoshkaDims?: number;
  /** Fast-field filters for efficient document filtering. */
  filters?: FilterDef[];
}
