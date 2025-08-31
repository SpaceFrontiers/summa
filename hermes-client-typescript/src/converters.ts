/**
 * Pure conversion helpers between the ergonomic client API and generated
 * protobuf types. Keeping them free of channel state makes them easy to test
 * and keeps HermesClient focused on RPC orchestration.
 */

import {
  DenseVector as PbDenseVector,
  FieldEntry as PbFieldEntry,
  FieldValue as PbFieldValue,
  FieldValueList as PbFieldValueList,
  FusionMethod as PbFusionMethod,
  MultiValueCombiner,
  Query as PbQuery,
  Reranker as PbReranker,
  SparseVector as PbSparseVector,
} from "./generated/hermes";

import type { Combiner, Query, Reranker } from "./types";

const COMBINER_MAP: Record<Combiner, MultiValueCombiner> = {
  log_sum_exp: MultiValueCombiner.COMBINER_LOG_SUM_EXP,
  max: MultiValueCombiner.COMBINER_MAX,
  avg: MultiValueCombiner.COMBINER_AVG,
  sum: MultiValueCombiner.COMBINER_SUM,
  weighted_top_k: MultiValueCombiner.COMBINER_WEIGHTED_TOP_K,
};

function combinerToProto(combiner?: Combiner): MultiValueCombiner {
  return combiner
    ? (COMBINER_MAP[combiner] ??
        MultiValueCombiner.COMBINER_LOG_SUM_EXP)
    : MultiValueCombiner.COMBINER_LOG_SUM_EXP;
}

export function buildQuery(q: Query): PbQuery {
  if ("term" in q) {
    return {
      term: {
        field: q.term.field,
        term: q.term.term,
        tokenizerHint: q.term.tokenizerHint ?? "",
      },
    };
  }
  if ("match" in q) {
    return {
      match: {
        field: q.match.field,
        text: q.match.text,
        tokenizerHint: q.match.tokenizerHint ?? "",
      },
    };
  }
  if ("phrase" in q) {
    return {
      phrase: {
        field: q.phrase.field,
        text: q.phrase.text,
        slop: q.phrase.slop ?? 0,
        tokenizerHint: q.phrase.tokenizerHint ?? "",
      },
    };
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
        lspGamma: sv.lspGamma,
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
        combiner: combinerToProto(dv.combiner),
        combinerTemperature: dv.combinerTemperature ?? 0,
        combinerTopK: dv.combinerTopK ?? 0,
        combinerDecay: dv.combinerDecay ?? 0,
      },
    };
  }
  if ("binaryDenseVector" in q) {
    const bv = q.binaryDenseVector;
    return {
      binaryDenseVector: {
        field: bv.field,
        vector: bv.vector,
        combiner: combinerToProto(bv.combiner),
        combinerTemperature: bv.combinerTemperature ?? 0,
        combinerTopK: bv.combinerTopK ?? 0,
        combinerDecay: bv.combinerDecay ?? 0,
      },
    };
  }
  if ("boost" in q) {
    return {
      boost: { query: buildQuery(q.boost.query), boost: q.boost.boost },
    };
  }
  if ("range" in q) {
    const range = q.range;
    return {
      range: {
        field: range.field,
        minU64: range.minU64,
        maxU64: range.maxU64,
        minI64: range.minI64,
        maxI64: range.maxI64,
        minF64: range.minF64,
        maxF64: range.maxF64,
      },
    };
  }
  if ("prefix" in q) {
    return {
      prefix: { field: q.prefix.field, prefix: q.prefix.prefix },
    };
  }
  if ("all" in q) {
    return { all: {} };
  }
  if ("fusion" in q) {
    const fusion = q.fusion;
    return {
      fusion: {
        queries: fusion.queries.map((weightedQuery) => ({
          query: buildQuery(weightedQuery.query),
          weight: weightedQuery.weight ?? 1.0,
        })),
        method:
          fusion.method === "normalized_weighted_sum"
            ? PbFusionMethod.FUSION_NORMALIZED_WEIGHTED_SUM
            : PbFusionMethod.FUSION_RRF,
        rrfK: fusion.rrfK ?? 0,
        combiner: combinerToProto(fusion.combiner),
      },
    };
  }

  const validKeys = [
    "term",
    "match",
    "phrase",
    "boolean",
    "sparseVector",
    "denseVector",
    "binaryDenseVector",
    "boost",
    "range",
    "prefix",
    "all",
    "fusion",
  ];
  throw new Error(
    `Unrecognized query key(s): ${Object.keys(q).join(", ")}. ` +
      `Valid keys: ${validKeys.join(", ")}`,
  );
}

export function buildReranker(reranker: Reranker): PbReranker {
  return {
    field: reranker.field,
    vector: reranker.vector ?? [],
    combiner: combinerToProto(reranker.combiner),
    combinerTemperature: reranker.combinerTemperature ?? 0,
    combinerTopK: reranker.combinerTopK ?? 0,
    combinerDecay: reranker.combinerDecay ?? 0,
    matryoshkaDims: reranker.matryoshkaDims ?? 0,
    binaryVector: reranker.binaryVector ?? new Uint8Array(0),
    rrfK: reranker.rrfK ?? 0,
  };
}

type SparseVectorInput = Array<[number, number]>;

function isSparseVector(value: unknown[]): value is SparseVectorInput {
  if (value.length === 0) return false;
  return value.every(
    (item) =>
      Array.isArray(item) &&
      item.length === 2 &&
      Number.isInteger(item[0]) &&
      typeof item[1] === "number",
  );
}

function isMultiSparseVector(
  value: unknown[],
): value is SparseVectorInput[] {
  if (value.length === 0) return false;
  return value.every(
    (item) => Array.isArray(item) && isSparseVector(item),
  );
}

function isDenseVector(value: unknown[]): value is number[] {
  if (value.length === 0) return false;
  return value.every((item) => typeof item === "number");
}

function isMultiDenseVector(value: unknown[]): value is number[][] {
  if (value.length === 0) return false;
  return value.every(
    (item) => Array.isArray(item) && isDenseVector(item),
  );
}

function sparseVectorToProto(value: SparseVectorInput): PbSparseVector {
  return {
    indices: value.map(([index]) => index),
    values: value.map(([, weight]) => weight),
  };
}

function denseVectorToProto(value: number[]): PbDenseVector {
  return { values: value.map(Number) };
}

export function toFieldEntries(
  document: Record<string, unknown>,
): PbFieldEntry[] {
  const entries: PbFieldEntry[] = [];

  for (const [name, value] of Object.entries(document)) {
    if (Array.isArray(value)) {
      if (isMultiSparseVector(value)) {
        for (const vector of value) {
          entries.push({
            name,
            value: { sparseVector: sparseVectorToProto(vector) },
          });
        }
        continue;
      }
      if (isMultiDenseVector(value)) {
        for (const vector of value) {
          entries.push({
            name,
            value: { denseVector: denseVectorToProto(vector) },
          });
        }
        continue;
      }
      for (const item of value) {
        entries.push({ name, value: toFieldValue(item) });
      }
      continue;
    }
    entries.push({ name, value: toFieldValue(value) });
  }

  return entries;
}

function toFieldValue(value: unknown): PbFieldValue {
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
    return {
      bytesValue:
        value instanceof Uint8Array ? value : new Uint8Array(value),
    };
  }
  if (Array.isArray(value)) {
    if (isSparseVector(value)) {
      return { sparseVector: sparseVectorToProto(value) };
    }
    if (isDenseVector(value)) {
      return { denseVector: denseVectorToProto(value) };
    }
    return { jsonValue: JSON.stringify(value) };
  }
  if (typeof value === "object" && value !== null) {
    return { jsonValue: JSON.stringify(value) };
  }
  return { text: String(value) };
}

function fromFieldValue(value: PbFieldValue): unknown {
  if (value.text !== undefined) return value.text;
  if (value.u64 !== undefined) return value.u64;
  if (value.i64 !== undefined) return value.i64;
  if (value.f64 !== undefined) return value.f64;
  if (value.bytesValue !== undefined) return value.bytesValue;
  if (value.jsonValue !== undefined) return JSON.parse(value.jsonValue);
  if (value.sparseVector !== undefined) {
    return {
      indices: Array.from(value.sparseVector.indices),
      values: Array.from(value.sparseVector.values),
    };
  }
  if (value.denseVector !== undefined) {
    return Array.from(value.denseVector.values);
  }
  if (value.binaryDenseVector !== undefined) {
    return value.binaryDenseVector;
  }
  return null;
}

export function fromFieldValueList(valueList: PbFieldValueList): unknown {
  const values = valueList.values.map(fromFieldValue);
  return values.length === 1 ? values[0] : values;
}
