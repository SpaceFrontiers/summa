/**
 * Async Hermes client implementation.
 */

import { ChannelCredentials } from "@grpc/grpc-js";
import { createChannel, createClient, Channel, Client } from "nice-grpc";

import {
  SearchServiceDefinition,
  IndexServiceDefinition,
  FieldValue as PbFieldValue,
  FieldEntry as PbFieldEntry,
  Query as PbQuery,
  Reranker as PbReranker,
  MultiValueCombiner,
} from "./generated/hermes";

import {
  Document,
  SearchHit,
  SearchResponse,
  IndexInfo,
  SearchOptions,
  Combiner,
  RerankerCombiner,
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
    };
  }

  // =========================================================================
  // Document Indexing
  // =========================================================================

  /** Index multiple documents in batch. Returns [indexedCount, errorCount]. */
  async indexDocuments(
    indexName: string,
    documents: Record<string, any>[]
  ): Promise<[number, number]> {
    this.ensureConnected();

    const namedDocs = documents.map((doc) => ({
      fields: toFieldEntries(doc),
    }));

    const response = await this.indexClient!.batchIndexDocuments({
      indexName,
      documents: namedDocs,
    });
    return [response.indexedCount, response.errorCount];
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

  /** Search for documents. */
  async search(indexName: string, options: SearchOptions = {}): Promise<SearchResponse> {
    this.ensureConnected();

    const query = buildQuery(options);

    let reranker: PbReranker | undefined;
    if (options.reranker) {
      const [field, vector, limit] = options.reranker;
      reranker = {
        field,
        vector,
        limit,
        combiner: rerankerCombinerToProto(options.rerankerCombiner ?? "weighted_top_k"),
        combinerTemperature: 0,
        combinerTopK: 0,
        combinerDecay: 0,
      };
    }

    const response = await this.searchClient!.search({
      indexName,
      query,
      limit: options.limit ?? 10,
      offset: options.offset ?? 0,
      fieldsToLoad: options.fieldsToLoad ?? [],
      reranker,
    });

    const hits: SearchHit[] = response.hits.map((hit) => ({
      docId: hit.docId,
      score: hit.score,
      fields: Object.fromEntries(
        Object.entries(hit.fields).map(([k, v]) => [k, fromFieldValue(v)])
      ),
    }));

    return {
      hits,
      totalHits: response.totalHits,
      tookMs: response.tookMs,
    };
  }

  /** Get a document by ID. Returns null if not found. */
  async getDocument(indexName: string, docId: number): Promise<Document | null> {
    this.ensureConnected();
    try {
      const response = await this.searchClient!.getDocument({
        indexName,
        docId,
      });
      const fields = Object.fromEntries(
        Object.entries(response.fields).map(([k, v]) => [k, fromFieldValue(v)])
      );
      return { fields };
    } catch (err: any) {
      if (err?.code === 5) {
        // NOT_FOUND
        return null;
      }
      throw err;
    }
  }
}

// =============================================================================
// Helper functions
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

function combinerToProto(combiner: Combiner): number {
  const map: Record<string, number> = { sum: 0, max: 1, avg: 2 };
  return map[combiner] ?? 0;
}

function rerankerCombinerToProto(combiner: RerankerCombiner): MultiValueCombiner {
  const map: Record<string, MultiValueCombiner> = {
    log_sum_exp: MultiValueCombiner.COMBINER_LOG_SUM_EXP,
    max: MultiValueCombiner.COMBINER_MAX,
    avg: MultiValueCombiner.COMBINER_AVG,
    sum: MultiValueCombiner.COMBINER_SUM,
    weighted_top_k: MultiValueCombiner.COMBINER_WEIGHTED_TOP_K,
  };
  return map[combiner] ?? MultiValueCombiner.COMBINER_LOG_SUM_EXP;
}

function buildQuery(options: SearchOptions): PbQuery {
  if (options.term) {
    const [field, term] = options.term;
    return { term: { field, term } };
  }

  if (options.boolean) {
    const must = (options.boolean.must ?? []).map(([f, t]) => ({
      term: { field: f, term: t },
    }));
    const should = (options.boolean.should ?? []).map(([f, t]) => ({
      term: { field: f, term: t },
    }));
    const mustNot = (options.boolean.mustNot ?? []).map(([f, t]) => ({
      term: { field: f, term: t },
    }));
    return { boolean: { must, should, mustNot } };
  }

  const combiner = combinerToProto(options.combiner ?? "sum");

  if (options.sparseVector) {
    const [field, indices, values] = options.sparseVector;
    return {
      sparseVector: {
        field,
        indices,
        values,
        text: "",
        combiner,
        heapFactor: options.heapFactor ?? 1.0,
        combinerTemperature: 0,
        combinerTopK: 0,
        combinerDecay: 0,
      },
    };
  }

  if (options.sparseText) {
    const [field, text] = options.sparseText;
    return {
      sparseVector: {
        field,
        indices: [],
        values: [],
        text,
        combiner,
        heapFactor: options.heapFactor ?? 1.0,
        combinerTemperature: 0,
        combinerTopK: 0,
        combinerDecay: 0,
      },
    };
  }

  if (options.denseVector) {
    const [field, vector] = options.denseVector;
    return {
      denseVector: {
        field,
        vector,
        nprobe: options.nprobe ?? 0,
        rerankFactor: options.rerankFactor ?? 0,
        combiner,
        combinerTemperature: 0,
        combinerTopK: 0,
        combinerDecay: 0,
      },
    };
  }

  // Default: match all (empty boolean query)
  return { boolean: { must: [], should: [], mustNot: [] } };
}
