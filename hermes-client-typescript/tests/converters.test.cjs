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
