/** A document with field values. */
export interface Document {
  fields: Record<string, any>;
}

/** A single search result. */
export interface SearchHit {
  docId: number;
  score: number;
  fields: Record<string, any>;
}

/** Search response with hits and metadata. */
export interface SearchResponse {
  hits: SearchHit[];
  totalHits: number;
  tookMs: number;
}

/** Information about an index. */
export interface IndexInfo {
  indexName: string;
  numDocs: number;
  numSegments: number;
  schema: string;
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
}
