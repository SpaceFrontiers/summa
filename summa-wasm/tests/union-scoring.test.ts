import { test, expect } from "vitest";
import init, { LocalIndex } from "../pkg/summa_wasm";

test("sparse text unions preserve browser scores from independent term scorers", async () => {
  await init();
  const index = await LocalIndex.create("index sparse_union { field text: text }");
  try {
    const count = 4097;
    await index.addDocuments(Array.from({ length: count }, (_, doc) => ({
      text: [
        "padding",
        ...([13, 1031, 4096].includes(doc) ? Array(doc === 4096 ? 8 : 1).fill("rareone") : []),
        ...([71, 2048, 4096].includes(doc) ? Array(doc === 4096 ? 7 : 1).fill("raretwo") : []),
      ].join(" "),
    })));
    await index.commit();
    const expected = new Map<string, { address: any; score: number }>();
    for (const term of ["rareone", "raretwo"]) {
      const response = await index.search(term, count);
      expect(response.hits).toHaveLength(3);
      for (const hit of response.hits) {
        const key = `${hit.address.segment_id}:${hit.address.doc_id}`;
        const previous = expected.get(key);
        expected.set(key, { address: hit.address, score: Math.fround((previous?.score ?? 0) + hit.score) });
      }
    }
    const ordered = [...expected.values()].sort((a, b) => b.score - a.score || a.address.doc_id - b.address.doc_id);
    expect(ordered).toHaveLength(5);
    for (const query of ["rareone OR raretwo", "rareone OR missingterm OR raretwo"]) {
      for (const limit of [1, 3, 10, count]) {
        const result = await index.search(query, limit);
        expect(result.hits.map((hit: any) => ({ address: hit.address, score: hit.score }))).toEqual(ordered.slice(0, limit));
      }
    }
  } finally {
    index.free();
  }
});
