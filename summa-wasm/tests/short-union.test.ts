import { test, expect } from "vitest";
import init, { LocalIndex } from "../pkg/summa_wasm";

test("two-term browser unions preserve independent scores across dense blocks and late hits", async () => {
  await init();
  const index = await LocalIndex.create("index short_union { field text: text }");
  try {
    const count = 2049;
    await index.addDocuments(Array.from({ length: count }, (_, doc) => ({
      text: [
        ...Array(1 + doc % 17).fill("padding"),
        ...(doc % 2 === 0 ? Array(doc === count - 1 ? 20 : 1).fill("alpha") : []),
        ...(doc % 3 === 0 || doc === count - 1 ? Array(doc === count - 1 ? 19 : 1).fill("beta") : []),
      ].join(" "),
    })));
    await index.commit();
    const terms = new Map<string, any[]>();
    for (const term of ["alpha", "beta"]) {
      const response = await index.search(term, count);
      terms.set(term, response.hits);
    }
    for (const clauses of [["alpha", "beta"], ["beta", "alpha"], ["alpha", "alpha"]]) {
      const expected = new Map<string, { address: any; score: number }>();
      for (const term of clauses) {
        for (const hit of terms.get(term)!) {
          const key = `${hit.address.segment_id}:${hit.address.doc_id}`;
          const previous = expected.get(key);
          expected.set(key, { address: hit.address, score: Math.fround((previous?.score ?? 0) + hit.score) });
        }
      }
      const ordered = [...expected.values()].sort((a, b) => b.score - a.score || a.address.doc_id - b.address.doc_id);
      expect(ordered.length).toBeGreaterThan(1000);
      for (const limit of [1, 10, 1000, count]) {
        const result = await index.search(clauses.join(" OR "), limit);
        expect(result.hits.map((hit: any) => ({ address: hit.address, score: hit.score }))).toEqual(ordered.slice(0, limit));
      }
    }
  } finally {
    index.free();
  }
});
