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
