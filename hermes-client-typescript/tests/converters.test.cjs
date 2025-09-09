const assert = require("node:assert/strict");
const test = require("node:test");

const {
  buildQuery,
  fromFieldValueList,
  toFieldEntries,
} = require("../dist/converters.js");
const {
  FusionMethod,
  MultiValueCombiner,
} = require("../dist/generated/hermes.js");

test("document conversion preserves repeated fields and vector shapes", () => {
  const entries = toFieldEntries({
    tags: ["rust", "search"],
    sparse: [
      [[1, 0.5]],
      [[2, 0.25]],
    ],
    dense: [
      [1, 2.5],
      [3, 4],
    ],
  });

  const values = (name) =>
    entries.filter((entry) => entry.name === name).map((entry) => entry.value);

  assert.deepEqual(
    values("tags").map((value) => value.text),
    ["rust", "search"],
  );
  assert.deepEqual(
    values("sparse").map((value) => value.sparseVector),
    [
      { indices: [1], values: [0.5] },
      { indices: [2], values: [0.25] },
    ],
  );
  assert.deepEqual(
    values("dense").map((value) => value.denseVector),
    [{ values: [1, 2.5] }, { values: [3, 4] }],
  );
});

test("field value lists retain scalar unwrapping", () => {
  assert.deepEqual(fromFieldValueList({ values: [] }), []);
  assert.equal(fromFieldValueList({ values: [{ text: "one" }] }), "one");
  assert.deepEqual(
    fromFieldValueList({ values: [{ text: "one" }, { text: "two" }] }),
    ["one", "two"],
  );
});

test("query conversion retains recursive fusion configuration", () => {
  const query = buildQuery({
    fusion: {
      method: "normalized_weighted_sum",
      rrfK: 42,
      combiner: "max",
      queries: [
        {
          query: {
            boolean: {
              must: [
                { match: { field: "title", text: "search engine" } },
              ],
            },
          },
          weight: 0.75,
        },
        { query: { all: {} } },
      ],
    },
  });

  assert.equal(
    query.fusion.method,
    FusionMethod.FUSION_NORMALIZED_WEIGHTED_SUM,
  );
  assert.equal(query.fusion.combiner, MultiValueCombiner.COMBINER_MAX);
  assert.equal(query.fusion.rrfK, 42);
  assert.equal(query.fusion.queries[0].weight, 0.75);
  assert.equal(
    query.fusion.queries[0].query.boolean.must[0].match.text,
    "search engine",
  );
});

test("sparse query conversion preserves optional LSP gamma presence", () => {
  const unset = buildQuery({
    sparseVector: { field: "embedding" },
  });
  assert.equal(unset.sparseVector.lspGamma, undefined);

  const exhaustive = buildQuery({
    sparseVector: { field: "embedding", lspGamma: 0 },
  });
  assert.equal(exhaustive.sparseVector.lspGamma, 0);
});

test("named scoring branches retain scopes, eligibility and omission of RRF weights", () => {
  const { ScoreScope } = require("../dist/generated/hermes.js");
  const result = buildQuery({ fusion: {
    queries: [{ name: "body", scope: "chunk", query: { match: { field: "body", text: "hemoglobin" } } },
              { name: "title", scope: "document", scoreOnly: true, query: { match: { field: "title", text: "hemoglobin" } } }],
    candidateDepth: 42,
    filters: [{ phrase: { field: "body", text: "red blood cells" } }],
  } }).fusion;
  assert.equal(result.queries[0].weight, 0);
  assert.equal(result.queries[0].name, "body");
  assert.equal(result.queries[0].scope, ScoreScope.SCORE_SCOPE_CHUNK);
  assert.equal(result.queries[1].scope, ScoreScope.SCORE_SCOPE_DOCUMENT);
  assert.equal(result.queries[1].scoreOnly, true);
  assert.equal(result.filters[0].phrase.text, "red blood cells");
  assert.equal(result.candidateDepth, 42);
});


test("candidate export preserves method, depth and per-branch wire results", () => {
  const { SearchResponse } = require("../dist/generated/hermes.js");
  const query = buildQuery({ fusion: { method: "candidates", candidateDepth: 12,
    queries: [{ query: { match: { field: "body", text: "hemoglobin" } } }] } });
  assert.equal(query.fusion.method, FusionMethod.FUSION_CANDIDATES);
  assert.equal(query.fusion.candidateDepth, 12);
  const original = SearchResponse.fromPartial({ rankingMethod: "fusion_candidates_v1", fusionCandidates: [
    { queryIndex: 0, candidates: [{ address: { segmentId: "abc", docId: 2 }, score: -0.5,
      ordinalScores: [{ ordinal: 7, score: -0.25 }] }] }
  ] });
  const decoded = SearchResponse.decode(SearchResponse.encode(original).finish());
  assert.deepEqual(decoded.fusionCandidates, original.fusionCandidates);
});


test("formula options preserve explicit disabled backfill and learned defaults", () => {
  const { SearchRequest } = require("../dist/generated/hermes.js");
  for (const backfill of [undefined, false, true]) {
    const request = SearchRequest.fromPartial({ l1: {
      formula: "2 * dense", backfill, missingValues: { dense: -0.75 },
    } });
    const decoded = SearchRequest.decode(SearchRequest.encode(request).finish());
    assert.equal(decoded.l1.backfill, backfill);
    assert.deepEqual(decoded.l1.missingValues, { dense: -0.75 });
  }
});

test("client search forwards disabled backfill and learned missing defaults", async () => {
  const { HermesClient } = require("../dist/client.js");
  const { SearchResponse } = require("../dist/generated/hermes.js");
  const client = new HermesClient();
  client.indexClient = {};
  let sent;
  client.searchClient = { search: async (request) => {
    sent = request;
    return SearchResponse.fromPartial({ rankingMethod: "formula_v1" });
  } };
  await client.search("docs", { query: { all: {} },
    l1: { formula: "dense", backfill: false, missingValues: { dense: -0.75 } },
  });
  assert.equal(sent.l1.backfill, false);
  assert.deepEqual(sent.l1.missingValues, { dense: -0.75 });
});

test("formula client request rejects a backend with legacy ranking semantics", async () => {
  const { HermesClient } = require("../dist/client.js");
  const { SearchResponse } = require("../dist/generated/hermes.js");
  const client = new HermesClient();
  client.indexClient = {};
  client.searchClient = { search: async () => SearchResponse.fromPartial({ rankingMethod: "linear_v1" }) };
  await assert.rejects(client.search("docs", { query: { all: {} }, l1: { formula: "dense" } }), /formula_v1/);
});

test("document passage seeding survives the client and protobuf boundary", async () => {
  const { HermesClient } = require("../dist/client.js");
  const { SearchRequest, SearchResponse } = require("../dist/generated/hermes.js");
  const client = new HermesClient();
  client.indexClient = {};
  let sent;
  client.searchClient = { search: async (request) => {
    sent = SearchRequest.decode(SearchRequest.encode(request).finish());
    return SearchResponse.fromPartial({ rankingMethod: "feature_export_v2", seededDocumentPassages: true });
  } };
  await client.search("docs", { query: { all: {} }, scoreExport: { seedDocumentPassages: true } });
  assert.equal(sent.scoreExport.seedDocumentPassages, true);
  assert.equal(sent.scoreExport.allPassages, false);
  await client.search("docs", { query: { all: {} }, scoreExport: {} });
  assert.equal(sent.scoreExport.seedDocumentPassages, false);
  client.searchClient = { search: async () => SearchResponse.fromPartial({ rankingMethod: "feature_export_v2" }) };
  await assert.rejects(client.search("docs", { query: { all: {} }, scoreExport: { seedDocumentPassages: true } }), /acknowledge document passage seeding/);
});


test("RRF and trace preserve zero presence and discarded nominations through client and wire", async () => {
  const { HermesClient } = require("../dist/client.js");
  const { SearchResponse, SearchRequest } = require("../dist/generated/hermes.js");
  const client = new HermesClient();
  client.indexClient = {};
  let sent;
  let wire = SearchResponse.fromPartial({ hits: [{ score: -3, rrfScore: 0, rrfContributions: [
    { queryIndex: 0, queryName: "title", rank: 1, score: 0 },
    { queryIndex: 1, queryName: "body", rank: 2, score: 0, ordinal: 0 },
  ] }], trace: { shards: [{ shardId: "s", backendId: "b", queries: [{ queryName: "body",
    query: { term: { field: "body", term: "rust" } }, candidateDepth: 2, totalSeen: 100,
    candidates: [{ address: { segmentId: "abc", docId: 9 }, score: -0.5 }],
  }], filters: [{ all: {} }] }] } });
  client.searchClient = { search: async (request) => {
    sent = SearchRequest.decode(SearchRequest.encode(SearchRequest.fromPartial(request)).finish());
    return SearchResponse.decode(SearchResponse.encode(wire).finish());
  } };
  const result = await client.search("docs", { query: { all: {} }, includeRrfScores: true, tracing: true });
  assert.equal(sent.includeRrfScores, true);
  assert.equal(sent.tracing, true);
  assert.equal(result.hits[0].score, -3);
  assert.equal(result.hits[0].rrfScore, 0);
  assert.deepEqual(result.hits[0].rrfContributions.map(v => v.ordinal), [undefined, 0]);
  assert.deepEqual(result.trace, wire.trace);
  wire = SearchResponse.fromPartial({ hits: [{}] });
  const plain = await client.search("docs", { query: { all: {} } });
  assert.equal(sent.tracing, false);
  assert.equal(sent.includeRrfScores, false);
  assert.equal(plain.trace, undefined);
  assert.equal(plain.hits[0].rrfScore, undefined);
  await assert.rejects(client.search("docs", { query: { all: {} }, tracing: true }), /trace/);
  await assert.rejects(client.search("docs", { query: { all: {} }, includeRrfScores: true }), /RRF/);
});


test("symbolic formula roundtrips and legacy coefficients are rejected", async () => {
  const { HermesClient } = require("../dist/client.js");
  const { SearchResponse, SearchRequest } = require("../dist/generated/hermes.js");
  const client = new HermesClient();
  client.indexClient = {};
  let sent;
  let rankingMethod = "formula_v1";
  client.searchClient = { search: async request => {
    sent = SearchRequest.decode(SearchRequest.encode(SearchRequest.fromPartial(request)).finish());
    return SearchResponse.fromPartial({ rankingMethod });
  } };
  const formula = "log1p(title) - 1000 * rrf";
  await client.search("docs", { query: { all: {} }, l1: { formula } });
  assert.equal(sent.l1.formula, formula);
  for (const legacy of ["weights", "bias", "transforms", "rrfWeight"]) {
    await assert.rejects(client.search("docs", { query: { all: {} }, l1: { formula, [legacy]: {} } }), /only formula/);
  }
  rankingMethod = "linear_v2";
  await assert.rejects(client.search("docs", { query: { all: {} }, l1: { formula } }), /formula_v1/);
});

test("compaction is opt-in and index info exposes deletion counts", async () => {
  const { HermesClient } = require("../dist/client.js");
  const { ForceMergeRequest } = require("../dist/generated/hermes.js");
  const calls = [];
  const client = new HermesClient();
  client.indexClient = { forceMerge: async (request) => {
    calls.push(ForceMergeRequest.decode(ForceMergeRequest.encode(request).finish()));
    return { numSegments: 1 };
  }};
  client.searchClient = { getIndexInfo: async () => ({
    numDocs: 3, physicalNumDocs: 4, numDeletedDocs: 1, deletedRatio: 0.25,
  }) };
  await client.forceMerge("test");
  await client.forceMerge("test", undefined, true);
  assert.deepEqual(calls.map((call) => call.compact), [false, true]);
  const info = await client.getIndexInfo("test");
  assert.deepEqual([info.numDocs, info.physicalNumDocs, info.numDeletedDocs, info.deletedRatio], [3, 4, 1, 0.25]);
});

test("mutations preserve keys, chunks, error positions and explicit commit", async () => {
  const { HermesClient } = require("../dist/client.js");
  const client = new HermesClient("localhost:50051", { defaultTimeoutMs: 5000 });
  client.ensureConnected = () => {};
  let sent;
  client.indexClient = {
    deleteDocuments: async (request, options) => {
      sent = { request, options };
      return { acceptedCount: 2, errors: [{ index: 1, error: "invalid key" }] };
    },
    upsertDocuments: async (request, options) => {
      sent = { request, options };
      return { acceptedCount: 1, errors: [] };
    },
    commit: () => assert.fail("mutations must not commit automatically"),
  };
  const result = await client.deleteDocuments("docs", ["Á", "", "missing"], 250);
  assert.deepEqual(result, { acceptedCount: 2, errors: [{ index: 1, error: "invalid key" }] });
  assert.deepEqual(sent.request.primaryKeys, ["Á", "", "missing"]);
  await client.upsertDocument("docs", { id: "Á", body: ["one", "two"] });
  assert.deepEqual(sent.request.documents[0].fields.map((entry) => [entry.name, entry.value.text]), [
    ["id", "Á"], ["body", "one"], ["body", "two"],
  ]);
  client.indexClient.upsertDocuments = async () => ({ acceptedCount: 0, errors: [{ index: 0, error: "pending insertion" }] });
  await assert.rejects(client.upsertDocument("docs", { id: "Á" }), /pending insertion/);
  client.indexClient.deleteDocuments = async () => ({ acceptedCount: 0, errors: [{ index: 0, error: "invalid key" }] });
  await assert.rejects(client.deleteDocument("docs", ""), /invalid key/);
  await assert.rejects(client.upsertDocuments("docs", new Array(1001).fill({})), /1000 documents/);
});

test("maximum deletion batches receive every error over real gRPC transport", async () => {
  const { createServer } = require("nice-grpc");
  const { HermesClient, IndexServiceDefinition } = require("../dist/index.js");
  const server = createServer();
  server.add({ ...IndexServiceDefinition, methods: { deleteDocuments: IndexServiceDefinition.methods.deleteDocuments } }, {
    deleteDocuments: async (request) => ({
      acceptedCount: 0,
      errors: request.primaryKeys.map((_, index) => ({ index, error: "commit the pending insertion before deleting or upserting this key again" })),
    }),
  });
  const port = await server.listen("127.0.0.1:0");
  const client = new HermesClient(`127.0.0.1:${port}`);
  client.connect();
  try {
    const result = await client.deleteDocuments("docs", new Array(100000).fill("a"));
    assert.equal(result.acceptedCount, 0);
    assert.equal(result.errors.length, 100000);
    assert.equal(result.errors[99999].index, 99999);
  } finally {
    client.close();
    await server.shutdown();
  }
});
