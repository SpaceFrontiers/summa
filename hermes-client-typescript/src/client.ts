/**
 * Async Hermes client implementation.
 *
 * Search types mirror the protobuf API. Serialization details live in
 * converters.ts so this class remains focused on connection and RPC lifecycle.
 */

import { ChannelCredentials } from "@grpc/grpc-js";
import { Channel, Client, createChannel, createClientFactory } from "nice-grpc";
import {
  DeadlineOptions,
  deadlineMiddleware,
} from "nice-grpc-client-middleware-deadline";

import {
  buildQuery,
  buildReranker,
  fromFieldValueList,
  toFieldEntries,
} from "./converters";
import {
  IndexServiceDefinition,
  SearchServiceDefinition,
} from "./generated/hermes";

import type {
  DocAddress,
  Document,
  IndexInfo,
  SearchHit,
  SearchRequest,
  SearchResponse,
  SearchTimings,
} from "./types";

type SearchClient = Client<typeof SearchServiceDefinition, DeadlineOptions>;
type IndexClient = Client<typeof IndexServiceDefinition, DeadlineOptions>;

export interface HermesClientOptions {
  /**
   * Default per-RPC deadline in milliseconds, applied to every call unless
   * overridden by the call's `timeoutMs` argument. Undefined means no
   * deadline. Expired calls reject with gRPC DEADLINE_EXCEEDED.
   */
  defaultTimeoutMs?: number;
}

export class HermesClient {
  private readonly address: string;
  private readonly defaultTimeoutMs?: number;
  private channel: Channel | null = null;
  private indexClient: IndexClient | null = null;
  private searchClient: SearchClient | null = null;

  constructor(
    address: string = "localhost:50051",
    options: HermesClientOptions = {},
  ) {
    this.address = address;
    this.defaultTimeoutMs = options.defaultTimeoutMs;
  }

  /** Connect to the server. */
  connect(): void {
    this.channel = createChannel(
      this.address,
      ChannelCredentials.createInsecure(),
    );
    const factory = createClientFactory().use(deadlineMiddleware);
    this.indexClient = factory.create(IndexServiceDefinition, this.channel);
    this.searchClient = factory.create(SearchServiceDefinition, this.channel);
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

  /** Per-call options with the effective deadline (call override > default). */
  private callOptions(timeoutMs?: number): DeadlineOptions {
    const milliseconds = timeoutMs ?? this.defaultTimeoutMs;
    return milliseconds !== undefined && milliseconds > 0
      ? { deadline: new Date(Date.now() + milliseconds) }
      : {};
  }

  private ensureConnected(): void {
    if (!this.indexClient || !this.searchClient) {
      throw new Error("Client not connected. Call connect() first.");
    }
  }

  /** Create a new index. */
  async createIndex(
    indexName: string,
    schema: string,
    timeoutMs?: number,
  ): Promise<boolean> {
    this.ensureConnected();
    const response = await this.indexClient!.createIndex(
      { indexName, schema },
      this.callOptions(timeoutMs),
    );
    return response.success;
  }

  /** Delete an index. */
  async deleteIndex(
    indexName: string,
    timeoutMs?: number,
  ): Promise<boolean> {
    this.ensureConnected();
    const response = await this.indexClient!.deleteIndex(
      { indexName },
      this.callOptions(timeoutMs),
    );
    return response.success;
  }

  /** List all indexes on the server. */
  async listIndexes(timeoutMs?: number): Promise<string[]> {
    this.ensureConnected();
    const response = await this.indexClient!.listIndexes(
      {},
      this.callOptions(timeoutMs),
    );
    return response.indexNames;
  }

  /** Get information about an index. */
  async getIndexInfo(
    indexName: string,
    timeoutMs?: number,
  ): Promise<IndexInfo> {
    this.ensureConnected();
    const response = await this.searchClient!.getIndexInfo(
      { indexName },
      this.callOptions(timeoutMs),
    );
    return {
      indexName: response.indexName,
      numDocs: response.numDocs,
      numSegments: response.numSegments,
      schema: response.schema,
      candidateScoringVersion: response.candidateScoringVersion,
      unpreparedCandidateFields: response.unpreparedCandidateFields,
      vectorStats: (response.vectorStats ?? []).map((stats) => ({
        fieldName: stats.fieldName,
        vectorType: stats.vectorType,
        totalVectors: stats.totalVectors,
        dimension: stats.dimension,
      })),
    };
  }

  /** Index multiple documents. Returns [indexedCount, errorCount, errors]. */
  async indexDocuments(
    indexName: string,
    documents: Record<string, unknown>[],
    timeoutMs?: number,
  ): Promise<[number, number, Array<{ index: number; error: string }>]> {
    this.ensureConnected();
    const response = await this.indexClient!.batchIndexDocuments(
      {
        indexName,
        documents: documents.map((document) => ({
          fields: toFieldEntries(document),
        })),
      },
      this.callOptions(timeoutMs),
    );
    const errors = (response.errors ?? []).map((error) => ({
      index: error.index,
      error: error.error,
    }));
    return [response.indexedCount, response.errorCount, errors];
  }

  /** Index a single document. */
  async indexDocument(
    indexName: string,
    document: Record<string, unknown>,
    timeoutMs?: number,
  ): Promise<void> {
    await this.indexDocuments(indexName, [document], timeoutMs);
  }

  /** Stream documents for indexing. Returns number of indexed documents. */
  async indexDocumentsStream(
    indexName: string,
    documents: AsyncIterable<Record<string, unknown>>,
    timeoutMs?: number,
  ): Promise<number> {
    this.ensureConnected();

    async function* requestIterator() {
      for await (const document of documents) {
        yield {
          indexName,
          fields: toFieldEntries(document),
        };
      }
    }

    const response = await this.indexClient!.indexDocuments(
      requestIterator(),
      this.callOptions(timeoutMs),
    );
    return response.indexedCount;
  }

  /** Commit pending changes. Returns total number of documents. */
  async commit(indexName: string, timeoutMs?: number): Promise<number> {
    this.ensureConnected();
    const response = await this.indexClient!.commit(
      { indexName },
      this.callOptions(timeoutMs),
    );
    return response.numDocs;
  }

  /** Force merge all segments. Returns number of segments after merge. */
  async forceMerge(indexName: string, timeoutMs?: number): Promise<number> {
    this.ensureConnected();
    const response = await this.indexClient!.forceMerge(
      { indexName },
      this.callOptions(timeoutMs),
    );
    return response.numSegments;
  }

  /** Retrain vector index centroids/codebooks from current data. */
  async retrainVectorIndex(
    indexName: string,
    timeoutMs?: number,
  ): Promise<boolean> {
    this.ensureConnected();
    const response = await this.indexClient!.retrainVectorIndex(
      { indexName },
      this.callOptions(timeoutMs),
    );
    return response.success;
  }

  /** Reorder BMP blocks by SimHash similarity. */
  async reorder(indexName: string, timeoutMs?: number): Promise<number> {
    this.ensureConnected();
    const response = await this.indexClient!.reorder(
      { indexName },
      this.callOptions(timeoutMs),
    );
    return response.numSegments;
  }

  /**
   * Search for documents.
   *
   * @example
   * await client.search("articles", {
   *   query: { match: { field: "title", text: "search engine" } },
   *   fieldsToLoad: ["title"],
   * });
   */
  async search(
    indexName: string,
    request: SearchRequest,
    timeoutMs?: number,
  ): Promise<SearchResponse> {
    this.ensureConnected();
    const response = await this.searchClient!.search(
      {
        indexName,
        query: buildQuery(request.query),
        limit: request.limit ?? 10,
        offset: request.offset ?? 0,
        fieldsToLoad: request.fieldsToLoad ?? [],
        reranker: request.reranker
          ? buildReranker(request.reranker)
          : undefined,
        candidateLimit: request.candidateLimit ?? 0,
        timeBudgetMs: request.timeBudgetMs ?? 0,
        textStats: undefined,
        l1: request.l1 ? { weights: request.l1.weights, bias: request.l1.bias ?? 0, transforms: request.l1.transforms ?? {} } : undefined,
        scoreExport: request.scoreExport ? { passagesPerDocument: request.scoreExport.passagesPerDocument ?? 0, allPassages: request.scoreExport.allPassages ?? false } : undefined,
      },
      this.callOptions(timeoutMs),
    );

    const hits: SearchHit[] = response.hits.map((hit) => ({
      address: {
        segmentId: hit.address?.segmentId ?? "",
        docId: hit.address?.docId ?? 0,
      },
      score: hit.score,
      candidateScores: hit.candidateScores,
      fields: Object.fromEntries(
        Object.entries(hit.fields).map(([name, value]) => [
          name,
          fromFieldValueList(value),
        ]),
      ),
      ordinalScores: (hit.ordinalScores ?? []).map((score) => ({
        ordinal: score.ordinal,
        score: score.score,
      })),
    }));

    const timings: SearchTimings | undefined = response.timings
      ? {
          searchUs: Number(response.timings.searchUs),
          rerankUs: Number(response.timings.rerankUs),
          loadUs: Number(response.timings.loadUs),
          totalUs: Number(response.timings.totalUs),
          candidateScoringUs: Number(response.timings.candidateScoringUs),
        }
      : undefined;

    return {
      hits,
      totalHits: response.totalHits,
      rankingMethod: response.rankingMethod,
      fusionCandidates: response.fusionCandidates.map(branch => ({ queryIndex: branch.queryIndex,
        candidates: branch.candidates.map(hit => ({ address: { segmentId: hit.address?.segmentId ?? "", docId: hit.address?.docId ?? 0 },
          score: hit.score, ordinalScores: hit.ordinalScores })) })),
      truncated: response.truncated,
      tookMs: response.tookMs,
      timings,
    };
  }

  /** Get a document by address. Returns null if not found. */
  async getDocument(
    indexName: string,
    address: DocAddress,
    timeoutMs?: number,
  ): Promise<Document | null> {
    this.ensureConnected();
    try {
      const response = await this.searchClient!.getDocument(
        {
          indexName,
          address: {
            segmentId: address.segmentId,
            docId: address.docId,
          },
        },
        this.callOptions(timeoutMs),
      );
      return {
        fields: Object.fromEntries(
          Object.entries(response.fields).map(([name, value]) => [
            name,
            fromFieldValueList(value),
          ]),
        ),
      };
    } catch (error: unknown) {
      // gRPC NOT_FOUND status code.
      if (
        typeof error === "object" &&
        error !== null &&
        "code" in error &&
        error.code === 5
      ) {
        return null;
      }
      throw error;
    }
  }
}
