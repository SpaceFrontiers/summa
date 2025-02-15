/**
 * Async Hermes client implementation.
 *
 * All search types mirror the proto API structure exactly.
 * See types.ts for Query, Reranker, SearchRequest definitions.
 */

import { ChannelCredentials } from "@grpc/grpc-js";
import { createChannel, createClient, Channel, Client } from "nice-grpc";

import {
  SearchServiceDefinition,
  IndexServiceDefinition,
  FieldValue as PbFieldValue,
  FieldValueList as PbFieldValueList,
  FieldEntry as PbFieldEntry,
  Query as PbQuery,
  MultiValueCombiner,
} from "./generated/hermes";

import type {
  DocAddress,
  Document,
  SearchHit,
  SearchResponse,
  SearchTimings,
  IndexInfo,
  SearchRequest,
  Query,
  Combiner,
  Reranker,
} from "./types";

type SearchClient = Client<typeof SearchServiceDefinition>;
type IndexClient = Client<typeof IndexServiceDefinition>;

export class HermesClient {
  private address: string;
  private channel: Channel | null = null;
  private indexClient: IndexClient | null = null;
  private searchClient: SearchClient | null = null;

  constructor(address: string = "localhost:50051") {
    this.address = address;
  }

  /** Connect to the server. */
  connect(): void {
    this.channel = createChannel(this.address, ChannelCredentials.createInsecure());
    this.indexClient = createClient(IndexServiceDefinition, this.channel);
    this.searchClient = createClient(SearchServiceDefinition, this.channel);
  }

  /** Close the connection. */
  close(): void {
    if (this.channel) {
      this.channel.close();
      this.channel = null;
      this.indexClient = null;
      this.searchClient = null;
    }
  }

  private ensureConnected(): void {
    if (!this.indexClient || !this.searchClient) {
      throw new Error("Client not connected. Call connect() first.");
    }
  }

  // =========================================================================
  // Index Management
  // =========================================================================

  /** Create a new index. */
  async createIndex(indexName: string, schema: string): Promise<boolean> {
    this.ensureConnected();
    const response = await this.indexClient!.createIndex({ indexName, schema });
    return response.success;
  }

  /** Delete an index. */
  async deleteIndex(indexName: string): Promise<boolean> {
    this.ensureConnected();
    const response = await this.indexClient!.deleteIndex({ indexName });
    return response.success;
  }

  /** List all indexes on the server. */
  async listIndexes(): Promise<string[]> {
    this.ensureConnected();
    const response = await this.indexClient!.listIndexes({});
    return response.indexNames;
  }

  /** Get information about an index. */
  async getIndexInfo(indexName: string): Promise<IndexInfo> {
    this.ensureConnected();
    const response = await this.searchClient!.getIndexInfo({ indexName });
    return {
      indexName: response.indexName,
      numDocs: response.numDocs,
      numSegments: response.numSegments,
      schema: response.schema,
      vectorStats: (response.vectorStats || []).map((vs) => ({
        fieldName: vs.fieldName,
        vectorType: vs.vectorType,
        totalVectors: vs.totalVectors,
        dimension: vs.dimension,
      })),
    };
  }

  // =========================================================================
  // Document Indexing
  // =========================================================================

  /** Index multiple documents in batch. Returns [indexedCount, errorCount, errors]. */
  async indexDocuments(
    indexName: string,
    documents: Record<string, any>[]
  ): Promise<[number, number, Array<{ index: number; error: string }>]> {
    this.ensureConnected();

    const namedDocs = documents.map((doc) => ({
      fields: toFieldEntries(doc),
    }));

    const response = await this.indexClient!.batchIndexDocuments({
      indexName,
      documents: namedDocs,
    });
    const errors = (response.errors ?? []).map((e) => ({
      index: e.index,
      error: e.error,
    }));
    return [response.indexedCount, response.errorCount, errors];
  }

  /** Index a single document. */
  async indexDocument(
    indexName: string,
    document: Record<string, any>
  ): Promise<void> {
    await this.indexDocuments(indexName, [document]);
  }

  /** Stream documents for indexing. Returns number of indexed documents. */
  async indexDocumentsStream(
    indexName: string,
    documents: AsyncIterable<Record<string, any>>
  ): Promise<number> {
    this.ensureConnected();

    async function* requestIterator() {
      for await (const doc of documents) {
        yield {
          indexName,
          fields: toFieldEntries(doc),
        };
      }
    }

    const response = await this.indexClient!.indexDocuments(requestIterator());
    return response.indexedCount;
  }

  /** Commit pending changes. Returns total number of documents. */
  async commit(indexName: string): Promise<number> {
    this.ensureConnected();
    const response = await this.indexClient!.commit({ indexName });
    return response.numDocs;
  }

  /** Force merge all segments. Returns number of segments after merge. */
  async forceMerge(indexName: string): Promise<number> {
    this.ensureConnected();
    const response = await this.indexClient!.forceMerge({ indexName });
    return response.numSegments;
  }

  /** Retrain vector index centroids/codebooks from current data. */
  async retrainVectorIndex(indexName: string): Promise<boolean> {
    this.ensureConnected();
    const response = await this.indexClient!.retrainVectorIndex({ indexName });
    return response.success;
  }

  // =========================================================================
  // Search
  // =========================================================================

  /**
   * Search for documents.
   *
   * @example
   * // Term query
   * await client.search("articles", {
   *   query: { term: { field: "title", term: "hello" } },
   * });
   *
   * // Match query (full-text, tokenized server-side)
   * await client.search("articles", {
   *   query: { match: { field: "title", text: "what is hemoglobin" } },
   * });
   *
   * // Sparse vector query with server-side tokenization + pruning
   * await client.search("docs", {
   *   query: { sparseVector: { field: "embedding", text: "machine learning", pruning: 0.5 } },
   *   fieldsToLoad: ["title"],
   * });
   *
   * // Dense vector query
   * await client.search("docs", {
   *   query: { denseVector: { field: "embedding", vector: [0.1, 0.2, ...], nprobe: 10 } },
   *   reranker: { field: "embedding", vector: [0.1, 0.2, ...], limit: 100 },
   * });
   *
   * // Boolean query
   * await client.search("articles", {
   *   query: { boolean: {
   *     must: [{ match: { field: "title", text: "hello" } }],
   *     should: [{ match: { field: "body", text: "world" } }],
   *   }},
   * });
   */
  async search(indexName: string, request: SearchRequest): Promise<SearchResponse> {
    this.ensureConnected();

    const query = buildQuery(request.query);
    const reranker = request.reranker ? buildReranker(request.reranker) : undefined;

    const response = await this.searchClient!.search({
      indexName,
      query,
      limit: request.limit ?? 10,
      offset: request.offset ?? 0,
      fieldsToLoad: request.fieldsToLoad ?? [],
      reranker,
    });

    const hits: SearchHit[] = response.hits.map((hit) => ({
      address: {
        segmentId: hit.address?.segmentId ?? "",
        docId: hit.address?.docId ?? 0,
      },
      score: hit.score,
      fields: Object.fromEntries(
        Object.entries(hit.fields).map(([k, v]) => [k, fromFieldValueList(v)])
      ),
      ordinalScores: (hit.ordinalScores ?? []).map((os) => ({
        ordinal: os.ordinal,
        score: os.score,
      })),
    }));

    const timings: SearchTimings | undefined = response.timings
      ? {
          searchUs: Number(response.timings.searchUs),
          rerankUs: Number(response.timings.rerankUs),
          loadUs: Number(response.timings.loadUs),
          totalUs: Number(response.timings.totalUs),
        }
      : undefined;

    return {
      hits,
      totalHits: response.totalHits,
      tookMs: response.tookMs,
      timings,
    };
  }

  /** Get a document by address. Returns null if not found. */
  async getDocument(indexName: string, address: DocAddress): Promise<Document | null> {
    this.ensureConnected();
    try {
      const response = await this.searchClient!.getDocument({
        indexName,
        address: {
          segmentId: address.segmentId,
          docId: address.docId,
        },
      });
      const fields = Object.fromEntries(
        Object.entries(response.fields).map(([k, v]) => [k, fromFieldValueList(v)])
      );
      return { fields };
    } catch (err: any) {
      // gRPC NOT_FOUND status code
      const GRPC_NOT_FOUND = 5;
      if (err?.code === GRPC_NOT_FOUND) {
        return null;
      }
      throw err;
    }
  }
}

// =============================================================================
// Proto conversion helpers
// =============================================================================

const COMBINER_MAP: Record<string, MultiValueCombiner> = {
  log_sum_exp: MultiValueCombiner.COMBINER_LOG_SUM_EXP,
  max: MultiValueCombiner.COMBINER_MAX,
  avg: MultiValueCombiner.COMBINER_AVG,
  sum: MultiValueCombiner.COMBINER_SUM,
  weighted_top_k: MultiValueCombiner.COMBINER_WEIGHTED_TOP_K,
};

function combinerToProto(combiner?: Combiner): MultiValueCombiner {
  return combiner ? (COMBINER_MAP[combiner] ?? MultiValueCombiner.COMBINER_LOG_SUM_EXP) : MultiValueCombiner.COMBINER_LOG_SUM_EXP;
}

function buildQuery(q: Query): PbQuery {
  if ("term" in q) {
    return { term: { field: q.term.field, term: q.term.term } };
  }
  if ("match" in q) {
    return { match: { field: q.match.field, text: q.match.text } };
  }
  if ("boolean" in q) {
    return {
      boolean: {
        must: (q.boolean.must ?? []).map(buildQuery),
        should: (q.boolean.should ?? []).map(buildQuery),
        mustNot: (q.boolean.mustNot ?? []).map(buildQuery),
      },
    };
  }
  if ("sparseVector" in q) {
    const sv = q.sparseVector;
    return {
      sparseVector: {
        field: sv.field,
        indices: sv.indices ?? [],
        values: sv.values ?? [],
        text: sv.text ?? "",
        combiner: combinerToProto(sv.combiner),
        heapFactor: sv.heapFactor ?? 0,
        combinerTemperature: sv.combinerTemperature ?? 0,
        combinerTopK: sv.combinerTopK ?? 0,
        combinerDecay: sv.combinerDecay ?? 0,
        weightThreshold: sv.weightThreshold ?? 0,
        maxQueryDims: sv.maxQueryDims ?? 0,
        pruning: sv.pruning ?? 0,
      },
    };
  }
  if ("denseVector" in q) {
    const dv = q.denseVector;
    return {
      denseVector: {
        field: dv.field,
        vector: dv.vector,
        nprobe: dv.nprobe ?? 0,
        rerankFactor: dv.rerankFactor ?? 0,
        combiner: combinerToProto(dv.combiner),
        combinerTemperature: dv.combinerTemperature ?? 0,
        combinerTopK: dv.combinerTopK ?? 0,
        combinerDecay: dv.combinerDecay ?? 0,
      },
    };
  }
  if ("boost" in q) {
    return { boost: { query: buildQuery(q.boost.query), boost: q.boost.boost } };
  }
  if ("range" in q) {
    const r = q.range;
    return {
      range: {
        field: r.field,
        minU64: r.minU64,
        maxU64: r.maxU64,
        minI64: r.minI64,
        maxI64: r.maxI64,
        minF64: r.minF64,
        maxF64: r.maxF64,
      },
    };
  }
  if ("all" in q) {
    return { all: {} };
  }
  const validKeys = ["term", "match", "boolean", "sparseVector", "denseVector", "boost", "range", "all"];
  const keys = Object.keys(q);
  throw new Error(
    `Unrecognized query key(s): ${keys.join(", ")}. Valid keys: ${validKeys.join(", ")}`
  );
}

function buildReranker(r: Reranker): any {
  return {
    field: r.field,
    vector: r.vector,
    limit: r.limit ?? 0,
    combiner: combinerToProto(r.combiner),
    combinerTemperature: r.combinerTemperature ?? 0,
    combinerTopK: r.combinerTopK ?? 0,
    combinerDecay: r.combinerDecay ?? 0,
    matryoshkaDims: r.matryoshkaDims ?? 0,
  };
}

// =============================================================================
// Document field helpers
// =============================================================================

function isSparseVector(value: any[]): boolean {
  if (value.length === 0) return false;
  return value.every(
    (item) =>
      Array.isArray(item) &&
      item.length === 2 &&
      typeof item[0] === "number" &&
      typeof item[1] === "number" &&
      Number.isInteger(item[0])
  );
}

function isMultiSparseVector(value: any[]): boolean {
  if (value.length === 0) return false;
  return value.every((item) => Array.isArray(item) && isSparseVector(item));
}

function isDenseVector(value: any[]): boolean {
  if (value.length === 0) return false;
  return value.every(
    (v) => typeof v === "number" && typeof v !== "boolean"
  );
}

function isMultiDenseVector(value: any[]): boolean {
  if (value.length === 0) return false;
  return value.every((item) => Array.isArray(item) && isDenseVector(item));
}

function toFieldEntries(doc: Record<string, any>): PbFieldEntry[] {
  const entries: PbFieldEntry[] = [];
  for (const [name, value] of Object.entries(doc)) {
    if (Array.isArray(value)) {
      if (isMultiSparseVector(value)) {
        for (const sv of value) {
          const indices = sv.map((item: any) => item[0]);
          const values = sv.map((item: any) => item[1]);
          entries.push({
            name,
            value: { sparseVector: { indices, values } },
          });
        }
        continue;
      }
      if (isMultiDenseVector(value)) {
        for (const dv of value) {
          entries.push({
            name,
            value: { denseVector: { values: dv.map(Number) } },
          });
        }
        continue;
      }
      // Multi-value plain field: ["val1", "val2", ...] -> separate entries
      for (const item of value) {
        entries.push({ name, value: toFieldValue(item) });
      }
      continue;
    }
    entries.push({ name, value: toFieldValue(value) });
  }
  return entries;
}

function toFieldValue(value: any): PbFieldValue {
  if (typeof value === "string") {
    return { text: value };
  }
  if (typeof value === "boolean") {
    return { u64: value ? 1 : 0 };
  }
  if (typeof value === "number") {
    if (Number.isInteger(value)) {
      return value >= 0 ? { u64: value } : { i64: value };
    }
    return { f64: value };
  }
  if (value instanceof Uint8Array || Buffer.isBuffer(value)) {
    return { bytesValue: value instanceof Uint8Array ? value : new Uint8Array(value) };
  }
  if (Array.isArray(value)) {
    if (isSparseVector(value)) {
      const indices = value.map((item) => item[0]);
      const values = value.map((item) => item[1]);
      return { sparseVector: { indices, values } };
    }
    if (isDenseVector(value)) {
      return { denseVector: { values: value.map(Number) } };
    }
    return { jsonValue: JSON.stringify(value) };
  }
  if (typeof value === "object" && value !== null) {
    return { jsonValue: JSON.stringify(value) };
  }
  return { text: String(value) };
}

function fromFieldValue(fv: PbFieldValue): any {
  if (fv.text !== undefined) return fv.text;
  if (fv.u64 !== undefined) return fv.u64;
  if (fv.i64 !== undefined) return fv.i64;
  if (fv.f64 !== undefined) return fv.f64;
  if (fv.bytesValue !== undefined) return fv.bytesValue;
  if (fv.jsonValue !== undefined) return JSON.parse(fv.jsonValue);
  if (fv.sparseVector !== undefined) {
    return {
      indices: Array.from(fv.sparseVector.indices),
      values: Array.from(fv.sparseVector.values),
    };
  }
  if (fv.denseVector !== undefined) {
    return Array.from(fv.denseVector.values);
  }
  return null;
}

function fromFieldValueList(fvl: PbFieldValueList): any {
  const values = fvl.values.map(fromFieldValue);
  if (values.length === 1) return values[0];
  return values;
}
